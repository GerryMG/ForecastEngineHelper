# -*- coding: utf-8 -*-
"""
Configuración y entrada/salida Oracle del motor de vigilancia.

Este es el ÚNICO archivo que se toca: qué se vigila (cada vigilancia con su propia
consulta), con qué umbrales y a dónde va la notificación. El cálculo está en
vig_engine.py.

  1. CONEXIONES       origen (lectura) y destino (escritura), por variables de entorno
  2. QUÉ SE VIGILA    una entrada por métrica, cada una con su SQL
  3. PARÁMETROS       ventanas, umbrales, gravedad y notificación
  4. De acá para abajo   no hace falta tocar nada

Tablas:
    VIG_SERIE         una fila por serie vigilada, con su historia (CLOB). Se reemplaza.
    VIG_EVENTO        una fila por evento, con identidad estable. Se actualiza y acumula.
    VIG_CAUSA         la llenan ustedes: qué causó cada cosa. El motor sólo la lee.
    VIG_NOTIFICACION  lo que habría que avisar. Se acumula.
    VIG_RESUMEN       conteo por vigilancia, grano y nivel de cada corrida. Se acumula.

Variables de entorno:
    ORA_USER / ORA_PASSWORD / ORA_DSN                 origen
    ORA_DEST_USER / ORA_DEST_PASSWORD / ORA_DEST_DSN  destino
    VG_FECHA_EJECUCION, VG_MODO, VG_DRY_RUN           opcionales
    VG_WEBHOOK_URL                                    si se usa el canal "webhook"
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import oracledb

import vig_engine
from vig_engine import (Fechas, NIVELES, Resultado, VigConfig, Vigilancia, VigEngine,
                        catalogo_detectores)

# ═══════════════════════════════════════════════════════════════════════════
#  1. CONEXIONES
# ═══════════════════════════════════════════════════════════════════════════
ORA_USER = os.getenv("ORA_USER", "APP_LECTURA")
ORA_PASSWORD = os.getenv("ORA_PASSWORD", "")
ORA_DSN = os.getenv("ORA_DSN", "srv-origen.midominio.com:1521/DWH")

ORA_DEST_USER = os.getenv("ORA_DEST_USER", "")
ORA_DEST_PASSWORD = os.getenv("ORA_DEST_PASSWORD", "")
ORA_DEST_DSN = os.getenv("ORA_DEST_DSN") or ORA_DSN

ORACLE_CLIENT_LIB = os.getenv("ORACLE_CLIENT_LIB")

# ═══════════════════════════════════════════════════════════════════════════
#  2. QUÉ SE VIGILA
# ═══════════════════════════════════════════════════════════════════════════
# Cada vigilancia trae su propia consulta. Tiene que filtrar por :desde y :hasta,
# devolver la fecha, las categorías y la métrica. Si querés explicar quién causó una
# anomalía, traé también la columna de atribución (el hijo: cliente, producto...).
#
# granos: "dia" para errores de carga, "mes" para tendencia, "semana" si te sirve.
# direccion: "ambas", "baja" (sólo caídas) o "sube" (sólo subas, ej. devoluciones).

VIGILANCIAS = [
    Vigilancia(
        nombre="VENTA_CANAL",
        descripcion="Venta diaria por canal.",
        sql="""
            SELECT TRUNC(v.FECHA) AS FECHA,
                   v.BD_CANAL,
                   v.SK_CLIENTE,
                   SUM(v.VENTA)   AS MT_VENTA
              FROM VENTAS v
             WHERE v.FECHA >= :desde
               AND v.FECHA <  :hasta
             GROUP BY TRUNC(v.FECHA), v.BD_CANAL, v.SK_CLIENTE
        """,
        categorias=["BD_CANAL"],
        metrica="MT_VENTA",
        unidad="USD",
        granos=("dia", "mes"),
        atribucion="SK_CLIENTE",
        materialidad_minima=500.0,
        # con asuetos por país: la categoría BD_PAIS tiene que venir en el SQL
        # calendario_por=["BD_PAIS"],
        # dias_sin_operacion=("domingo",),                        # para todos
        # dias_sin_operacion_por={"SV": ["sabado", "domingo"]},  # distinto por país
    ),
    Vigilancia(
        nombre="TASA_DEVOLUCION",
        descripcion="Devoluciones sobre venta, por canal.",
        sql="""
            SELECT TRUNC(v.FECHA)   AS FECHA,
                   v.BD_CANAL,
                   v.SK_CLIENTE,
                   SUM(v.DEVOLUCION) AS MT_DEVOLUCION,
                   SUM(v.VENTA)      AS MT_VENTA
              FROM VENTAS v
             WHERE v.FECHA >= :desde
               AND v.FECHA <  :hasta
             GROUP BY TRUNC(v.FECHA), v.BD_CANAL, v.SK_CLIENTE
        """,
        categorias=["BD_CANAL"],
        metrica="MT_DEVOLUCION",
        denominador="MT_VENTA",
        agregacion="ratio",
        unidad="%",
        granos=("mes",),
        atribucion="SK_CLIENTE",
        direccion="sube",
    ),
]

COL_FECHA = "FECHA"

TABLA_SERIE = "VIG_SERIE"
TABLA_EVENTO = "VIG_EVENTO"
TABLA_CAUSA = "VIG_CAUSA"
TABLA_NOTIFICACION = "VIG_NOTIFICACION"
TABLA_RESUMEN = "VIG_RESUMEN"

# ═══════════════════════════════════════════════════════════════════════════
#  3. PARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════
DIAS_HISTORIA = 1095            # cuánta historia se guarda y sirve de referencia
DIAS_RELECTURA = 45             # cuántos días se releen de la fuente en cada corrida

# ── Calendario de asuetos ─────────────────────────────────────────────────────
# La historia de asuetos que ya tienen, en una consulta. Tiene que devolver FECHA y las
# columnas que dicen a quién aplica (las mismas que pongas en `calendario_por` de cada
# vigilancia, por ejemplo BD_PAIS). Opcional: BD_MOTIVO, que aparece en la explicación.
# Un "*" (o vacío) en BD_PAIS vale para todos los países. :desde y :hasta los completa el
# pipeline con toda la historia que se analiza. None = sin calendario.
#
#     CALENDARIO_SQL = """
#         SELECT c.BD_PAIS, c.FECHA, c.BD_MOTIVO
#           FROM CAL_ASUETOS c
#          WHERE c.FECHA >= :desde AND c.FECHA < :hasta
#     """
CALENDARIO_SQL: Optional[str] = None
CALENDARIO_CONEXION = "origen"      # dónde está esa tabla: "origen" o "destino"
# En semana y mes, lo esperado se ajusta a los días operados de cada período: una semana con
# asueto vende menos sin que nada ande mal, y febrero tiene tres días menos que marzo. Sólo
# para sumas y conteos. Sin esto, cada marzo llegan alertas falsas de febrero.
AJUSTAR_DIAS_OPERADOS = True
MODO_CORRIDA = "auto"           # auto | completo | incremental

PERIODOS_EVALUADOS = {"dia": 30, "semana": 8, "mes": 6}     # qué tan atrás se revisa
PERIODOS_BASE = {"dia": 91, "semana": 26, "mes": 18}        # referencia de lo normal
MIN_PERIODOS = 6                # menos datos que esto: el detector no opina

# congelado: el dato repite exactamente el mismo valor (medidor trabado, ETL que copia el
#            último). dia_cerrado: actividad en un día que normalmente está cerrado.
DETECTORES = ("hueco", "salto", "escalon", "tendencia", "estacional", "racha", "nueva",
              "congelado", "dia_cerrado")
# Nivel máximo por detector. dia_cerrado queda en ATENCION: en ventas, abrir un domingo no
# es grave. En una vigilancia de consumo (agua, energía), subilo con sus `ajustes`:
#     ajustes={"nivel_maximo": {"dia_cerrado": "CRITICO"}}
NIVEL_MAXIMO = {"dia_cerrado": "ATENCION"}
UMBRAL_Z = {"ATENCION": 3.0, "ALERTA": 5.0, "CRITICO": 8.0}
PERSISTENCIA_SUBE_NIVEL = 3     # períodos seguidos que suben un nivel (detectores de punto)
PESO_RELATIVO_MINIMO = 0.2      # serie que pesa menos que 0,2 veces la serie promedio: no pasa de ATENCION
DESVIO_RELATIVO_MINIMO = 0.05   # diferencia mínima contra lo esperado para que sea un evento
PISO_SIGMA_RELATIVO = 0.02      # piso del ruido: nunca menos del 2% del nivel de la serie

# ── Cómo se decide si algo es una anomalía (comprobable mirando los datos) ─────────
# Un día se compara contra los mismos días de la semana de las N semanas anteriores; un
# tramo, por su total, contra los tramos anteriores de igual largo; una semana contra las
# N anteriores; un mes contra los N anteriores. Es anomalía sólo si queda FUERA de todo ese
# rango, descontada la temporada (lo que pasó en las mismas fechas hace un año).
VENTANAS_REFERENCIA = {"dia": 8, "semana": 8, "mes": 12}
MIN_VENTANAS_REFERENCIA = 4     # con menos, no se puede comprobar: queda en ATENCION, sin avisar
# Algo que a ESA serie le pasa seguido no es anomalía para ella: si en el último año tuvo
# más de esta cantidad de días así (o de tramos sin movimiento así de largos), no se marca.
# Es lo que separa a una tienda errática, que cierra días sueltos, de una que nunca cierra.
VECES_POR_ANIO = 1
# Sólo se avisa lo que sigue pasando en los últimos N períodos cerrados. Lo anterior queda
# guardado en VIG_EVENTO, pero una alerta sobre algo de hace semanas no sirve para actuar.
NOTIFICAR_ULTIMOS_PERIODOS = {"dia": 3, "semana": 1, "mes": 1}

# Que un CRÍTICO valga la pena. La calibración ajusta cuántas alertas salen, pero no mira
# cuánto se movió de verdad ni si el dato tiene sentido: eso lo deciden estas tres.
CRITICO_DESVIO_RELATIVO_MINIMO = 0.20   # para ser CRÍTICO, apartarse al menos 20% de lo esperado
                                        # (un 3% puede ser rarísimo y no importar). Bajalo por
                                        # vigilancia con `ajustes` en métricas muy estables.
FACTOR_SOSPECHA_DATO = 20.0     # 20 veces lo esperado en un período: casi seguro una carga mal
                                # hecha. Se avisa igual, con "OJO: posible error de carga". 0 = apagado
DIA_CERRADO_RELATIVO = 0.05     # grano día: un día de la semana que casi siempre es cero (el domingo
                                # de un B2B) está cerrado y no se evalúa. Sin esto, cada domingo
                                # cerrado era un CRÍTICO "hueco" falso. 0 = apagado

NIVEL_NOTIFICACION = "ALERTA"   # desde qué nivel se notifica
NIVEL_MINIMO_EVENTO = "ATENCION"    # desde qué nivel se guarda un evento ("INFO" = todo)
CANALES_NOTIFICACION = ("tabla",)   # "tabla", "webhook", y los que agregues abajo


# En las vigilancias con agregacion="ratio", `min_denominador` corta el período cuyo
# denominador es real pero despreciable: 25 sobre 0,0000064 da 390.000.000 y dispara una
# alerta crítica que no significa nada. Va en cada Vigilancia, porque cada métrica tiene su
# propia unidad. Por defecto 0 = sin mínimo.

# El dinero se suma en su escala decimal (centavos), igual que hace Oracle con NUMBER:
# los enteros se suman sin error, así que un importe y su reverso dan cero EXACTO y un
# ratio con denominador anulado queda nulo en vez de dispararse a 1e11.
# Es lo que resuelve el problema; no cuesta nada y no hay umbrales de por medio.
SUMA_EXACTA = True
MAX_DECIMALES = 6       # hasta cuántos decimales busca esa escala; más allá, redondea

# Plan B, sólo si no hay escala decimal usable (importes con muchísimos decimales, o un
# bruto que desborda 2^53 ≈ 90 billones en centavos). Una suma cuenta como cero cuando no
# llega a esta fracción de lo que pasó por ella. 0 lo apaga.
TOLERANCIA_CERO = 1e-9


def build_config(fecha_ejecucion: str | None = None) -> VigConfig:
    return VigConfig(
        vigilancias=VIGILANCIAS,
        col_fecha=COL_FECHA,
        fecha_ejecucion=fecha_ejecucion,
        dias_historia=DIAS_HISTORIA,
        periodos_evaluados=PERIODOS_EVALUADOS,
        periodos_base=PERIODOS_BASE,
        min_periodos=MIN_PERIODOS,
        detectores=DETECTORES,
        nivel_maximo=NIVEL_MAXIMO,
        umbral_z=UMBRAL_Z,
        persistencia_sube_nivel=PERSISTENCIA_SUBE_NIVEL,
        peso_relativo_minimo=PESO_RELATIVO_MINIMO,
        desvio_relativo_minimo=DESVIO_RELATIVO_MINIMO,
        piso_sigma_relativo=PISO_SIGMA_RELATIVO,
        nivel_notificacion=NIVEL_NOTIFICACION,
        nivel_minimo_evento=NIVEL_MINIMO_EVENTO,
        critico_desvio_relativo_minimo=CRITICO_DESVIO_RELATIVO_MINIMO,
        factor_sospecha_dato=FACTOR_SOSPECHA_DATO,
        dia_cerrado_relativo=DIA_CERRADO_RELATIVO,
        ajustar_dias_operados=AJUSTAR_DIAS_OPERADOS,
        ventanas_referencia=VENTANAS_REFERENCIA,
        min_ventanas_referencia=MIN_VENTANAS_REFERENCIA,
        veces_por_anio=VECES_POR_ANIO,
        notificar_ultimos_periodos=NOTIFICAR_ULTIMOS_PERIODOS,
        suma_exacta=SUMA_EXACTA,
        max_decimales=MAX_DECIMALES,
        tolerancia_cero=TOLERANCIA_CERO,
    )


# ── Notificadores ──────────────────────────────────────────────────────────
# Cada canal es una función que recibe la lista de notificaciones y devuelve el
# estado del envío. "tabla" siempre se escribe. Para sumar correo o Teams, escribí
# la función y agregá su nombre a CANALES_NOTIFICACION.
def enviar_webhook(mensajes: List[dict]) -> str:
    """POST con el resumen a VG_WEBHOOK_URL. Si no hay URL, no hace nada."""
    url = os.getenv("VG_WEBHOOK_URL", "")
    if not url:
        raise RuntimeError("Falta VG_WEBHOOK_URL para el canal webhook")
    import urllib.request
    cuerpo = json.dumps({"origen": "vigilancia", "cantidad": len(mensajes),
                         "notificaciones": mensajes}, ensure_ascii=False).encode("utf-8")
    pedido = urllib.request.Request(url, data=cuerpo, method="POST",
                                    headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(pedido, timeout=30) as r:
        return f"HTTP {r.status}"


def enviar_correo(mensajes: List[dict]) -> str:
    """Enganchá acá tu SMTP. Ejemplo de lo que recibís en cada mensaje:
    {"id": ..., "vigilancia": ..., "clave": ..., "nivel": ..., "mensaje": ...}"""
    raise NotImplementedError("Escribí enviar_correo() en vig_oracle.py y agregá 'correo' "
                              "a CANALES_NOTIFICACION")


def enviar_teams(mensajes: List[dict]) -> str:
    """Igual que el webhook, pero con el formato de tarjetas de Teams."""
    raise NotImplementedError("Escribí enviar_teams() en vig_oracle.py y agregá 'teams' "
                              "a CANALES_NOTIFICACION")


NOTIFICADORES: Dict[str, Callable[[List[dict]], str]] = {
    "webhook": enviar_webhook,
    "correo": enviar_correo,
    "teams": enviar_teams,
}


# ═══════════════════════════════════════════════════════════════════════════
#  De acá para abajo no hace falta tocar nada
# ═══════════════════════════════════════════════════════════════════════════
ARRAYSIZE = 100_000
BATCH_ROWS = 5_000
ORACLE_NUMBER_MIN = 1e-130
ORACLE_NUMBER_MAX = 9.99999999e125

log = logging.getLogger("vig")


def configurar_logging(nombre: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], force=True,
                        format=f"%(asctime)s {nombre} [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    vig_engine.LOGGER.handlers.clear()
    vig_engine.LOGGER.propagate = True
    return logging.getLogger(nombre)


# ─── conexiones ─────────────────────────────────────────────────────────────
_ROL = "_vig_rol"


def _conectar(usuario: str, password: str, dsn: str, rol: str):
    if ORACLE_CLIENT_LIB and not getattr(_conectar, "_thick", False):
        oracledb.init_oracle_client(lib_dir=ORACLE_CLIENT_LIB)
        _conectar._thick = True
    sufijo = "" if rol == "origen" else "_DEST"
    if not usuario:
        raise RuntimeError(f"Falta el usuario de {rol} (ORA{sufijo}_USER)")
    if not password:
        raise RuntimeError(f"Falta la password de {rol} (ORA{sufijo}_PASSWORD)")
    oracledb.defaults.fetch_decimals = False
    oracledb.defaults.fetch_lobs = False
    conn = oracledb.connect(user=usuario, password=password, dsn=dsn)
    conn.autocommit = False
    try:
        setattr(conn, _ROL, rol)
    except AttributeError:
        pass
    return conn


def _exigir(conn, rol: str, que: str) -> None:
    actual = getattr(conn, _ROL, None)
    if actual is not None and actual != rol:
        raise RuntimeError(f"{que} necesita la conexión de {rol} y recibió la de {actual}.")


def conexion_origen():
    return _conectar(ORA_USER, ORA_PASSWORD, ORA_DSN, "origen")


def conexion_destino():
    return _conectar(ORA_DEST_USER, ORA_DEST_PASSWORD, ORA_DEST_DSN, "destino")


def _fetch_df(conn, sql: str, params: dict | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.arraysize = ARRAYSIZE
        cur.prefetchrows = ARRAYSIZE + 1
        cur.execute(sql, params or {})
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


# ─── esquema ────────────────────────────────────────────────────────────────
def huella_config(cfg: VigConfig) -> str:
    """Si cambia lo que define el contenido de las series, la historia guardada no sirve."""
    base = json.dumps({v.nombre: {"categorias": list(v.categorias), "metrica": v.metrica,
                                  "denominador": v.denominador, "agregacion": v.agregacion,
                                  "granos": list(v.granos), "sql": " ".join(v.sql.split())}
                       for v in cfg.vigilancias}, sort_keys=True)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


COLUMNAS_SERIE = [
    ("BD_VIGILANCIA", "VARCHAR2(60)", "Qué métrica se vigila (nombre de la vigilancia)."),
    ("BD_GRANO", "VARCHAR2(10)", "dia, semana o mes."),
    ("BD_CLAVE", "VARCHAR2(400)", "Valores de las categorías que identifican la serie, separados por |."),
    ("BD_CATEGORIAS", "VARCHAR2(400)", "Nombres de las categorías que forman BD_CLAVE."),
    ("BD_UNIDAD", "VARCHAR2(20)", "Unidad de la métrica: USD, unidades, %, ..."),
    ("MT_ULTIMO", "NUMBER", "Valor del último período cerrado."),
    ("MT_ESPERADO", "NUMBER", "Lo que se esperaba: mediana móvil de la ventana de referencia."),
    ("MT_Z", "NUMBER", "Desvío robusto del último período contra lo esperado."),
    ("BD_NIVEL", "VARCHAR2(10)", "Nivel del último período: INFO, ATENCION, ALERTA o CRITICO."),
    ("MT_PERIODOS_CON_DATOS", "NUMBER", "Cuántos períodos de la historia tienen dato."),
    ("MT_PARTICIPACION", "NUMBER", "Cuánto pesa esta serie sobre el total de su vigilancia."),
    ("FECHA_ULTIMO_PERIODO", "DATE", "Primer día del último período evaluado."),
    ("BD_HISTORIAL", "CLOB", "Historia completa de la serie: {\"p0\": primer período, \"v\": valores} "
                             "con null en los períodos sin dato."),
    ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo."),
    ("HUELLA_CONFIG", "VARCHAR2(40)", "Resumen de la configuración con la que se armó la serie."),
    ("FECHA_CARGA", "DATE", "Fecha y hora de la carga."),
]

COLUMNAS_EVENTO = [
    ("BD_ID_EVENTO", "VARCHAR2(40)", "Identidad estable del evento: el mismo problema conserva su id "
                                     "mientras siga abierto."),
    ("BD_VIGILANCIA", "VARCHAR2(60)", "Qué métrica se vigila."),
    ("BD_GRANO", "VARCHAR2(10)", "dia, semana o mes."),
    ("BD_CLAVE", "VARCHAR2(400)", "Serie afectada."),
    ("BD_CATEGORIAS", "VARCHAR2(400)", "Nombres de las categorías que forman BD_CLAVE."),
    ("BD_DETECTOR", "VARCHAR2(20)", "Detector que lo encontró."),
    ("BD_CONFIRMAN", "VARCHAR2(200)", "Otros detectores que marcaron lo mismo."),
    ("BD_PRINCIPAL", "VARCHAR2(5)", "SI cuando es el evento que representa el problema; NO cuando "
                                    "es la misma anomalía vista por otro detector."),
    ("FECHA_INICIO", "DATE", "Primer período con la anomalía."),
    ("FECHA_FIN", "DATE", "Último período con la anomalía."),
    ("MT_PERIODOS", "NUMBER", "Cuántos períodos lleva."),
    ("MT_Z", "NUMBER", "Peor desvío robusto del evento. Negativo = por debajo de lo esperado."),
    ("MT_OBSERVADO", "NUMBER", "Lo observado: el valor del período si el evento es de un período; "
                                "el total (o el promedio, en métricas que no se suman) del tramo si "
                                "es de varios."),
    ("MT_ESPERADO", "NUMBER", "Lo esperado para eso mismo: la mediana de los mismos períodos "
                               "anteriores (ver BD_EXPLICACION, que los lista), descontada la temporada."),
    ("MT_MATERIALIDAD", "NUMBER", "Lo que está en juego: la diferencia entre lo observado y lo esperado, "
                                   "en la unidad de la métrica."),
    ("BD_UNIDAD", "VARCHAR2(20)", "Unidad de la métrica."),
    ("MT_PARTICIPACION", "NUMBER", "Cuánto pesa la serie sobre el total de su vigilancia."),
    ("BD_NIVEL", "VARCHAR2(10)", "Gravedad: INFO, ATENCION, ALERTA o CRITICO."),
    ("BD_NIVEL_ANTERIOR", "VARCHAR2(10)", "Nivel que tenía en la corrida anterior."),
    ("BD_MOTIVO_NIVEL", "VARCHAR2(400)", "Por qué ese nivel: desvío, persistencia y peso."),
    ("BD_EXPLICACION", "VARCHAR2(2000)", "La razón exacta del evento: contra qué se lo comparó, cuánto "
                                          "se movió y cuánto se mueve normalmente la serie, qué días no se "
                                          "evaluaron y por qué (asuetos), cuánto está en juego y por qué "
                                          "quedó en ese nivel."),
    ("BD_ESTADO", "VARCHAR2(10)", "NUEVO, EN_CURSO o CERRADO."),
    ("BD_ATRIBUCION", "VARCHAR2(1000)", "Quiénes explican el cambio, calculado por el motor."),
    ("BD_CAUSA", "VARCHAR2(1000)", "Causa escrita en VIG_CAUSA que cae dentro del evento."),
    ("BD_HISTORIA_CAUSAS", "VARCHAR2(1000)", "Si ya había pasado antes, cuántas veces y con qué causa."),
    ("FECHA_DETECCION", "DATE", "Cuándo se detectó por primera vez."),
    ("FECHA_CIERRE", "DATE", "Cuándo volvió a la normalidad."),
    ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo."),
    ("HUELLA_CONFIG", "VARCHAR2(40)", "Resumen de la configuración."),
    ("FECHA_CARGA", "DATE", "Fecha y hora de la carga."),
]

COLUMNAS_CAUSA = [
    ("BD_VIGILANCIA", "VARCHAR2(60)", "A qué vigilancia aplica."),
    ("BD_CLAVE", "VARCHAR2(400)", "Serie afectada, o * para todas las de la vigilancia."),
    ("BD_DETECTOR", "VARCHAR2(20)", "Detector al que aplica, o vacío para cualquiera."),
    ("FECHA_DESDE", "DATE", "Desde cuándo vale la explicación."),
    ("FECHA_HASTA", "DATE", "Hasta cuándo. Vacío = sigue vigente."),
    ("BD_CAUSA", "VARCHAR2(1000)", "Qué pasó, en palabras."),
    ("BD_ACCION", "VARCHAR2(1000)", "Qué se hizo o hay que hacer."),
    ("BD_AUTOR", "VARCHAR2(100)", "Quién lo anotó."),
    ("FECHA_CARGA", "DATE", "Fecha y hora de la anotación."),
]

COLUMNAS_NOTIFICACION = [
    ("BD_ID_EVENTO", "VARCHAR2(40)", "Evento que la origina."),
    ("BD_VIGILANCIA", "VARCHAR2(60)", "Qué métrica se vigila."),
    ("BD_GRANO", "VARCHAR2(10)", "dia, semana o mes."),
    ("BD_CLAVE", "VARCHAR2(400)", "Serie afectada."),
    ("BD_NIVEL", "VARCHAR2(10)", "Gravedad."),
    ("BD_ESTADO", "VARCHAR2(10)", "Estado del evento cuando se notificó."),
    ("BD_MOTIVO", "VARCHAR2(200)", "Por qué se notifica: evento nuevo o empeoró de nivel."),
    ("BD_MENSAJE", "VARCHAR2(2000)", "Texto listo para mandar."),
    ("BD_CANALES", "VARCHAR2(200)", "Canales por los que se envió."),
    ("BD_ESTADO_ENVIO", "VARCHAR2(200)", "Resultado del envío por canal."),
    ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo."),
    ("FECHA_CARGA", "DATE", "Fecha y hora de la carga."),
]

COLUMNAS_RESUMEN = [
    ("BD_VIGILANCIA", "VARCHAR2(60)", "Qué métrica se vigila."),
    ("BD_GRANO", "VARCHAR2(10)", "dia, semana o mes."),
    ("BD_NIVEL", "VARCHAR2(10)", "Gravedad."),
    ("BD_ESTADO", "VARCHAR2(10)", "NUEVO, EN_CURSO o CERRADO."),
    ("MT_EVENTOS", "NUMBER", "Cuántos eventos."),
    ("MT_MATERIALIDAD", "NUMBER", "Suma de la diferencia contra lo esperado."),
    ("MT_SERIES_VIGILADAS", "NUMBER", "Cuántas series se vigilan en esa vigilancia y grano."),
    ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo."),
    ("HUELLA_CONFIG", "VARCHAR2(40)", "Resumen de la configuración."),
    ("FECHA_CARGA", "DATE", "Fecha y hora de la carga."),
]

TABLAS = [(TABLA_SERIE, COLUMNAS_SERIE, "Series vigiladas con su historia. Se reemplaza en cada corrida."),
          (TABLA_EVENTO, COLUMNAS_EVENTO, "Eventos detectados, con identidad estable. Se actualiza y acumula."),
          (TABLA_CAUSA, COLUMNAS_CAUSA, "Causas escritas a mano. El motor sólo la lee."),
          (TABLA_NOTIFICACION, COLUMNAS_NOTIFICACION, "Notificaciones emitidas. Se acumula."),
          (TABLA_RESUMEN, COLUMNAS_RESUMEN, "Resumen por corrida. Se acumula.")]


def _ddl(tabla: str, cols, comentario: str) -> str:
    largos = [c for c, _, _ in cols if len(c) > 30]
    ancho = max(len(c) for c, _, _ in cols)
    cuerpo = ",\n".join(f"    {c.ljust(ancho)}  {t}" + ("  DEFAULT SYSDATE" if c == "FECHA_CARGA" else "")
                        for c, t, _ in cols)
    comentarios = "\n".join(f"COMMENT ON COLUMN {tabla}.{c} IS '{d.replace(chr(39), chr(39) * 2)}';"
                            for c, _, d in cols)
    aviso = f"-- ATENCIÓN: columnas de más de 30 caracteres (Oracle < 12.2): {largos}\n" if largos else ""
    return (f"{aviso}CREATE TABLE {tabla} (\n{cuerpo}\n);\n\n"
            f"COMMENT ON TABLE {tabla} IS '{comentario}';\n\n{comentarios}\n")


def ddl_sugerido(cfg: Optional[VigConfig] = None) -> str:
    return "\n".join(_ddl(t, c, d) for t, c, d in TABLAS)


def validar_tablas(conn, cfg: VigConfig) -> None:
    _exigir(conn, "destino", "validar_tablas()")
    for tabla, cols, _ in TABLAS:
        owner, _, nombre = tabla.rpartition(".")
        sql = "SELECT OWNER, COLUMN_NAME FROM ALL_TAB_COLUMNS WHERE TABLE_NAME = :t"
        params = {"t": nombre.upper()}
        if owner:
            sql += " AND OWNER = :o"
            params["o"] = owner.upper()
        enc = _fetch_df(conn, sql, params)
        if enc.empty:
            raise RuntimeError(f"El usuario de destino ({ORA_DEST_USER}) no ve la tabla {tabla}.\n"
                               f"Si hay que crearla:\n\n{ddl_sugerido(cfg)}")
        existentes = {c.upper() for c in enc["COLUMN_NAME"]}
        faltan = [(c, t) for c, t, _ in cols if c not in existentes]
        if faltan:
            raise RuntimeError(f"A {tabla} le faltan columnas: {', '.join(c for c, _ in faltan)}\n\n"
                               f"ALTER TABLE {tabla} ADD ({', '.join(f'{c} {t}' for c, t in faltan)});")
        log.info("tabla %s validada: %d columnas", tabla, len(cols))


# ─── plan y lectura ─────────────────────────────────────────────────────────
def planificar(conn, cfg: VigConfig, modo: str = MODO_CORRIDA) -> Tuple[pd.Timestamp, bool, str]:
    """Desde cuándo leer la fuente y si se puede reusar la historia guardada."""
    _exigir(conn, "destino", "planificar()")
    modo = str(modo).strip().lower()
    if modo not in ("auto", "completo", "incremental"):
        raise ValueError("VG_MODO debe ser auto, completo o incremental")
    f = Fechas.desde(cfg.fecha_ejecucion)
    completa = (f.ayer - pd.Timedelta(days=cfg.dias_historia)).normalize()
    if modo == "completo":
        return completa, False, "pedida con VG_MODO=completo"

    e = _fetch_df(conn, f"""
        SELECT COUNT(*) AS FILAS,
               COUNT(DISTINCT HUELLA_CONFIG) AS N_HUELLAS,
               MIN(HUELLA_CONFIG) AS HUELLA,
               MAX(FECHA_CORTE) AS CORTE,
               SUM(CASE WHEN BD_HISTORIAL IS NULL THEN 1 ELSE 0 END) AS SIN_HISTORIAL
          FROM {TABLA_SERIE}""").iloc[0]
    problema = None
    if int(e["FILAS"]) == 0:
        problema = f"{TABLA_SERIE} está vacía"
    elif int(e["N_HUELLAS"]) != 1 or e["HUELLA"] != huella_config(cfg):
        problema = "cambió la configuración de las vigilancias (categorías, métrica o SQL)"
    elif int(e["SIN_HISTORIAL"] or 0) > 0:
        problema = f"{int(e['SIN_HISTORIAL']):,} series sin historial"
    if problema:
        if modo == "incremental":
            raise RuntimeError(f"No se puede correr incremental: {problema}. Usá VG_MODO=completo.")
        log.warning("corrida COMPLETA: %s", problema)
        return completa, False, problema

    corte = pd.Timestamp(e["CORTE"]).normalize()
    desde = (f.hoy - pd.Timedelta(days=DIAS_RELECTURA)).normalize()
    siguiente = corte + pd.Timedelta(days=1)
    motivo = f"relee los últimos {DIAS_RELECTURA} días"
    if siguiente < desde:
        desde = siguiente
        motivo = f"la última corrida llegó hasta {corte.date()}: se relee desde el día siguiente"
    # La relectura arranca al inicio de la semana y del mes donde cae: una semana o un mes
    # releídos a medias se rearmarían con días de menos (y pisarían al valor guardado).
    lunes = desde - pd.Timedelta(days=desde.dayofweek)
    inicio = min(lunes, desde.replace(day=1))
    if inicio < desde:
        motivo += f", desde el {inicio.date()} para tomar semanas y meses enteros"
    return inicio, True, motivo


def leer_fuente(conn, cfg: VigConfig, desde: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """Una consulta por vigilancia. Devuelve {nombre: DataFrame}."""
    _exigir(conn, "origen", "leer_fuente()")
    f = Fechas.desde(cfg.fecha_ejecucion)
    datos: Dict[str, pd.DataFrame] = {}
    for v in cfg.vigilancias:
        if not v.activa:
            continue
        for bind in (":desde", ":hasta"):
            if bind not in v.sql:
                raise RuntimeError(f"{v.nombre}: el SQL tiene que filtrar la fecha con {bind}")
        t0 = time.time()
        df = _fetch_df(conn, v.sql, {"desde": desde.to_pydatetime(), "hasta": f.hoy.to_pydatetime()})
        if df.empty:
            log.warning("[%s] la fuente no devolvió filas entre %s y %s", v.nombre, desde.date(),
                        f.ayer.date())
            datos[v.nombre] = df
            continue
        df.columns = [c.upper() for c in df.columns]
        df[cfg.col_fecha] = pd.to_datetime(df[cfg.col_fecha]).dt.normalize()
        datos[v.nombre] = df
        log.info("[%s] fuente: %s filas, %s a %s (%.1fs)", v.nombre, f"{len(df):,}", desde.date(),
                 df[cfg.col_fecha].max().date(), time.time() - t0)
    return datos


def leer_estado(conn, cfg: VigConfig) -> pd.DataFrame:
    """La historia guardada de cada serie."""
    _exigir(conn, "destino", "leer_estado()")
    t0 = time.time()
    df = _fetch_df(conn, f"SELECT BD_VIGILANCIA, BD_GRANO, BD_CLAVE, BD_HISTORIAL FROM {TABLA_SERIE}")
    log.info("estado: %s series guardadas (%.1fs)", f"{len(df):,}", time.time() - t0)
    return df


def leer_eventos_abiertos(conn, cfg: VigConfig) -> pd.DataFrame:
    """Los eventos que quedaron abiertos, con los nombres que usa el motor."""
    _exigir(conn, "destino", "leer_eventos_abiertos()")
    df = _fetch_df(conn, f"""
        SELECT BD_ID_EVENTO, BD_VIGILANCIA, BD_GRANO, BD_CLAVE, BD_DETECTOR, BD_NIVEL, BD_ESTADO,
               FECHA_INICIO, FECHA_FIN, MT_PERIODOS, MT_Z, MT_OBSERVADO, MT_ESPERADO,
               MT_MATERIALIDAD, MT_PARTICIPACION, BD_UNIDAD, BD_CATEGORIAS, BD_MOTIVO_NIVEL,
               FECHA_DETECCION, BD_EXPLICACION
          FROM {TABLA_EVENTO}
         WHERE BD_ESTADO <> 'CERRADO'""")
    if df.empty:
        return df
    f = Fechas.desde(cfg.fecha_ejecucion)
    out = pd.DataFrame({
        "id_evento": df["BD_ID_EVENTO"], "vigilancia": df["BD_VIGILANCIA"], "grano": df["BD_GRANO"],
        "clave": df["BD_CLAVE"], "detector": df["BD_DETECTOR"], "nivel": df["BD_NIVEL"],
        "estado": df["BD_ESTADO"], "categorias": df["BD_CATEGORIAS"], "unidad": df["BD_UNIDAD"],
        "fecha_inicio": pd.to_datetime(df["FECHA_INICIO"]), "fecha_fin": pd.to_datetime(df["FECHA_FIN"]),
        "periodos": df["MT_PERIODOS"], "z": df["MT_Z"], "observado": df["MT_OBSERVADO"],
        "esperado": df["MT_ESPERADO"], "materialidad": df["MT_MATERIALIDAD"],
        "participacion": df["MT_PARTICIPACION"], "motivo_nivel": df["BD_MOTIVO_NIVEL"],
        "fecha_deteccion": pd.to_datetime(df["FECHA_DETECCION"]),
        "explicacion": df["BD_EXPLICACION"].fillna("").astype(str),
    })
    out["periodo_inicio"] = [int(vig_engine.indice_periodo(pd.Series([d]), g)[0])
                             for d, g in zip(out["fecha_inicio"], out["grano"])]
    out["periodo_fin"] = [int(vig_engine.indice_periodo(pd.Series([d]), g)[0])
                          for d, g in zip(out["fecha_fin"], out["grano"])]
    out["principal"] = True
    log.info("eventos abiertos de corridas anteriores: %s", f"{len(out):,}")
    return out


def leer_calendario(cfg: VigConfig) -> Optional[pd.DataFrame]:
    """Tu historia de asuetos (CALENDARIO_SQL), de toda la ventana que se analiza.

    Abre su propia conexión (CALENDARIO_CONEXION), así la tabla puede estar en origen o
    en destino. Sin CALENDARIO_SQL devuelve None: no hay calendario.
    """
    if not CALENDARIO_SQL:
        return None
    for bind in (":desde", ":hasta"):
        if bind not in CALENDARIO_SQL:
            raise RuntimeError(f"CALENDARIO_SQL tiene que filtrar la fecha con {bind}")
    f = Fechas.desde(cfg.fecha_ejecucion)
    desde = f.hoy - pd.Timedelta(days=int(cfg.dias_historia) + 400)   # toda la historia y un año más
    abrir = conexion_destino if CALENDARIO_CONEXION == "destino" else conexion_origen
    with abrir() as conn:
        df = _fetch_df(conn, CALENDARIO_SQL, {"desde": desde.to_pydatetime(),
                                              "hasta": (f.hoy + pd.Timedelta(days=1)).to_pydatetime()})
    df.columns = [c.upper() for c in df.columns]
    if df.empty:
        log.warning("CALENDARIO_SQL no devolvió asuetos entre %s y %s", desde.date(), f.hoy.date())
        return None
    log.info("calendario de asuetos: %s días (%s a %s)", f"{len(df):,}",
             pd.to_datetime(df["FECHA"]).min().date(), pd.to_datetime(df["FECHA"]).max().date())
    return df


def leer_causas(conn, cfg: VigConfig) -> pd.DataFrame:
    """Lo que ustedes anotaron en VIG_CAUSA."""
    _exigir(conn, "destino", "leer_causas()")
    df = _fetch_df(conn, f"""SELECT BD_VIGILANCIA, BD_CLAVE, BD_DETECTOR, FECHA_DESDE, FECHA_HASTA,
                                    BD_CAUSA, BD_ACCION, BD_AUTOR FROM {TABLA_CAUSA}""")
    if df.empty:
        return df
    out = pd.DataFrame({"vigilancia": df["BD_VIGILANCIA"], "clave": df["BD_CLAVE"],
                        "detector": df["BD_DETECTOR"].fillna(""),
                        "desde": pd.to_datetime(df["FECHA_DESDE"]),
                        "hasta": pd.to_datetime(df["FECHA_HASTA"]),
                        "causa": df["BD_CAUSA"], "accion": df["BD_ACCION"], "autor": df["BD_AUTOR"]})
    log.info("causas anotadas: %s", f"{len(out):,}")
    return out


# ─── escritura ──────────────────────────────────────────────────────────────
def _rango_oracle(a: np.ndarray) -> np.ndarray:
    abs_ = np.abs(a)
    chicos = (abs_ < ORACLE_NUMBER_MIN) & (abs_ > 0)
    grandes = np.isfinite(a) & (abs_ > ORACLE_NUMBER_MAX)
    if chicos.any() or grandes.any():
        a = a.copy()
        a[chicos] = 0.0
        a[grandes] = np.sign(a[grandes]) * ORACLE_NUMBER_MAX
    return a


def _ancho(tipo: str) -> Optional[int]:
    if tipo.upper().startswith("VARCHAR") and "(" in tipo:
        try:
            return int(tipo[tipo.index("(") + 1:tipo.index(")")].split()[0])
        except ValueError:
            return None
    return None


def _a_python(arr: np.ndarray, tipo_completo: str) -> list:
    tipo = tipo_completo.split("(")[0]
    if tipo == "NUMBER":
        if arr.dtype.kind in "iu":
            return arr.astype(object).tolist()
        a = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(float)
        with np.errstate(invalid="ignore"):
            a = _rango_oracle(a)
        o = a.astype(object)
        o[~np.isfinite(a)] = None
        return o.tolist()
    if tipo == "DATE":
        fechas = pd.to_datetime(pd.Series(arr), errors="coerce")
        return [None if pd.isna(x) else x.to_pydatetime() for x in fechas]
    ancho = _ancho(tipo_completo)
    salida = []
    for x in arr:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            salida.append(None)
        else:
            t = str(x)
            salida.append(t[:ancho] if ancho else t)
    return salida


def _tipo_bind(tipo_completo: str):
    tipo = tipo_completo.split("(")[0]
    if tipo == "NUMBER":
        return oracledb.DB_TYPE_NUMBER
    if tipo == "DATE":
        return oracledb.DB_TYPE_DATE
    if tipo == "CLOB":
        return oracledb.DB_TYPE_LONG
    return oracledb.DB_TYPE_VARCHAR


def _insertar(conn, tabla: str, df: pd.DataFrame, cols) -> int:
    nombres = [c for c, _, _ in cols if c != "FECHA_CARGA"]
    tipos = [t for c, t, _ in cols if c != "FECHA_CARGA"]
    faltan = [c for c in nombres if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas {faltan} para {tabla}")
    sql = (f"INSERT INTO {tabla} ({', '.join(nombres)}, FECHA_CARGA)\n"
           f"VALUES ({', '.join(f':{i + 1}' for i in range(len(nombres)))}, SYSDATE)")
    datos = [df[c].to_numpy() for c in nombres]
    with conn.cursor() as cur:
        cur.setinputsizes(*[_tipo_bind(t) for t in tipos])
        for i in range(0, len(df), BATCH_ROWS):
            lote = [_a_python(a[i:i + BATCH_ROWS], t) for a, t in zip(datos, tipos)]
            cur.executemany(sql, list(zip(*lote)))
    return len(df)


def _texto(s, ancho=None):
    return s.astype(object).where(pd.notna(s), "").astype(str)


def preparar_tablas(res: Resultado, cfg: VigConfig) -> Dict[str, pd.DataFrame]:
    """Del resultado del motor a las columnas de cada tabla."""
    f = Fechas.desde(cfg.fecha_ejecucion)
    huella = huella_config(cfg)
    salida: Dict[str, pd.DataFrame] = {}

    s = res.series
    salida[TABLA_SERIE] = pd.DataFrame() if s.empty else pd.DataFrame({
        "BD_VIGILANCIA": s["vigilancia"], "BD_GRANO": s["grano"], "BD_CLAVE": s["clave"],
        "BD_CATEGORIAS": s["categorias"], "BD_UNIDAD": s["unidad"],
        "MT_ULTIMO": s["ultimo"], "MT_ESPERADO": s["esperado"], "MT_Z": s["z"],
        "BD_NIVEL": s["nivel"], "MT_PERIODOS_CON_DATOS": s["periodos_con_datos"],
        "MT_PARTICIPACION": s["participacion"], "FECHA_ULTIMO_PERIODO": s["fecha_ultimo_periodo"],
        "BD_HISTORIAL": s["historial"], "FECHA_CORTE": f.ayer, "HUELLA_CONFIG": huella})

    e = res.eventos
    salida[TABLA_EVENTO] = pd.DataFrame() if e.empty else pd.DataFrame({
        "BD_ID_EVENTO": e["id_evento"], "BD_VIGILANCIA": e["vigilancia"], "BD_GRANO": e["grano"],
        "BD_CLAVE": e["clave"], "BD_CATEGORIAS": e.get("categorias", ""), "BD_DETECTOR": e["detector"],
        "BD_CONFIRMAN": _texto(e.get("confirman", pd.Series([""] * len(e)))),
        "BD_PRINCIPAL": np.where(e.get("principal", True), "SI", "NO"),
        "FECHA_INICIO": pd.to_datetime(e["fecha_inicio"]), "FECHA_FIN": pd.to_datetime(e["fecha_fin"]),
        "MT_PERIODOS": e["periodos"], "MT_Z": e["z"], "MT_OBSERVADO": e["observado"],
        "MT_ESPERADO": e["esperado"], "MT_MATERIALIDAD": e["materialidad"],
        "BD_UNIDAD": e.get("unidad", ""), "MT_PARTICIPACION": e["participacion"],
        "BD_NIVEL": e["nivel"], "BD_NIVEL_ANTERIOR": _texto(e.get("nivel_anterior", pd.Series([""] * len(e)))),
        "BD_MOTIVO_NIVEL": e["motivo_nivel"], "BD_ESTADO": e["estado"],
        "BD_EXPLICACION": _texto(e.get("explicacion", pd.Series([""] * len(e)))),
        "BD_ATRIBUCION": _texto(e.get("atribucion", pd.Series([""] * len(e)))),
        "BD_CAUSA": _texto(e.get("causa", pd.Series([""] * len(e)))),
        "BD_HISTORIA_CAUSAS": _texto(e.get("historia_causas", pd.Series([""] * len(e)))),
        "FECHA_DETECCION": pd.to_datetime(e["fecha_deteccion"]),
        "FECHA_CIERRE": pd.to_datetime(e["fecha_cierre"]),
        "FECHA_CORTE": f.ayer, "HUELLA_CONFIG": huella})

    n = res.notificaciones
    salida[TABLA_NOTIFICACION] = pd.DataFrame() if n.empty else pd.DataFrame({
        "BD_ID_EVENTO": n["id_evento"], "BD_VIGILANCIA": n["vigilancia"], "BD_GRANO": n["grano"],
        "BD_CLAVE": n["clave"], "BD_NIVEL": n["nivel"], "BD_ESTADO": n["estado"],
        "BD_MOTIVO": n["motivo_notificacion"], "BD_MENSAJE": [mensaje(r) for _, r in n.iterrows()],
        "BD_CANALES": ", ".join(CANALES_NOTIFICACION), "BD_ESTADO_ENVIO": "PENDIENTE",
        "FECHA_CORTE": f.ayer})

    r = res.resumen
    salida[TABLA_RESUMEN] = pd.DataFrame() if r.empty else pd.DataFrame({
        "BD_VIGILANCIA": r["vigilancia"], "BD_GRANO": r["grano"], "BD_NIVEL": r["nivel"],
        "BD_ESTADO": r["estado"], "MT_EVENTOS": r["eventos"], "MT_MATERIALIDAD": r["materialidad"],
        "MT_SERIES_VIGILADAS": r.get("series_vigiladas", 0), "FECHA_CORTE": f.ayer,
        "HUELLA_CONFIG": huella})
    return salida


def mensaje(evento) -> str:
    """El texto que se manda. Si el motor armó la explicación (casi siempre), va esa: dice la
    razón exacta. Alrededor: si puede ser un error de carga, si empeoró, quién lo explica
    y si ya hay una causa anotada. Sin explicación (un evento viejo), el texto de antes."""
    motivo = str(evento.get("motivo_nivel", "") or "")
    expl = str(evento.get("explicacion", "") or "").strip()
    if expl and expl.lower() != "nan":
        partes = []
        if "error de dato" in motivo:
            partes.append("OJO: posible error de carga, revisar la fuente antes de actuar.")
        partes.append(f"[{evento['nivel']}] {evento['vigilancia']} / {evento['clave']} "
                      f"({evento['grano']}).")
        # "empeoró" sólo si el evento ya estaba abierto y SUBIÓ de nivel (no si bajó, y
        # nunca en un evento nuevo)
        anterior = str(evento.get("nivel_anterior", "") or "")
        if (str(evento.get("estado", "")) == "EN_CURSO" and anterior in vig_engine.NIVELES
                and vig_engine.NIVELES.index(evento["nivel"]) > vig_engine.NIVELES.index(anterior)):
            partes.append(f"Empeoró: en la corrida anterior estaba en {anterior}.")
        partes.append(expl)
        if str(evento.get("atribucion", "") or ""):
            partes.append(f"Quién lo explica: {evento['atribucion']}")
        if str(evento.get("causa", "") or ""):
            partes.append(f"Causa anotada: {evento['causa']}")
        elif str(evento.get("historia_causas", "") or ""):
            partes.append(f"Antecedente: {evento['historia_causas']}")
        return " ".join(partes)[:2000]
    return _mensaje_simple(evento)


def _mensaje_simple(evento) -> str:
    """El texto que se manda: cuándo, qué pasó, cuánto pesa, quién y si ya sabemos por qué.

    El "cuándo" va completo: el rango del evento y, aparte, la fecha del período del que
    sale el valor observado (que es siempre el último del evento). Sin la fecha del final,
    un "esperado 1.200, observado 0" no dice si el cero fue ayer o hace tres semanas.
    """
    signo = "cayó" if float(evento["z"]) < 0 else "subió"
    ini, fin = str(evento["fecha_inicio"])[:10], str(evento["fecha_fin"])[:10]
    n = int(evento["periodos"])
    cuando = (f"el {fin}" if n <= 1 or ini == fin
              else f"{n} período(s), del {ini} al {fin}")
    unidad = evento.get("unidad", "")
    detector = str(evento.get("detector", ""))
    if detector == "congelado":          # no es un desvío: es un dato que dejó de moverse
        partes = [f"[{evento['nivel']}] {evento['vigilancia']} / {evento['clave']} ({evento['grano']}) "
                  f"desde el {ini}: el valor NO cambia. {int(abs(float(evento['z'])))} período(s) "
                  f"seguidos con exactamente el mismo valor ({float(evento['observado']):,.2f} "
                  f"{unidad}), algo que en esta serie casi nunca pasa: el dato no se está "
                  f"actualizando. Revisar el medidor o la carga."]
        if str(evento.get("motivo_nivel", "")):
            partes.append(f"Por qué {evento['nivel']}: {evento['motivo_nivel']}.")
        return " ".join(partes)[:2000]
    if detector == "dia_cerrado":
        esperado_txt = (f"{float(evento['esperado']):,.2f}" if pd.notna(evento["esperado"])
                        else "casi nada")
        partes = [f"[{evento['nivel']}] {evento['vigilancia']} / {evento['clave']} ({evento['grano']}) "
                  f"el {fin}: hubo actividad en un día que normalmente está cerrado: "
                  f"{float(evento['observado']):,.2f} {unidad} contra {esperado_txt} habituales. "
                  f"En consumo (agua, energía) suele ser una fuga o algo que quedó prendido."]
        if str(evento.get("motivo_nivel", "")):
            partes.append(f"Por qué {evento['nivel']}: {evento['motivo_nivel']}.")
        return " ".join(partes)[:2000]
    esperado = float(evento["esperado"]) if pd.notna(evento["esperado"]) else None
    contra = (f"contra {esperado:,.2f} esperados, desvío {abs(float(evento['z'])):.1f}"
              if esperado is not None else "sin referencia previa (serie nueva)")
    partes = [f"[{evento['nivel']}] {evento['vigilancia']} / {evento['clave']} ({evento['grano']}) "
              f"{cuando}: {signo} contra lo esperado. En {fin}: {evento['observado']:,.2f} "
              f"{unidad} {contra}.",
              f"En juego: {float(evento['materialidad']):,.0f} {unidad}."]
    motivo = str(evento.get("motivo_nivel", "") or "")
    if "error de dato" in motivo:        # lo primero que hay que mirar: puede no ser negocio
        partes.insert(0, "OJO: posible error de carga, revisar la fuente antes de actuar.")
    if motivo:
        partes.append(f"Por qué {evento['nivel']}: {motivo}.")
    if str(evento.get("atribucion", "")):
        partes.append(f"Quién: {evento['atribucion']}")
    if str(evento.get("causa", "")):
        partes.append(f"Causa anotada: {evento['causa']}")
    elif str(evento.get("historia_causas", "")):
        partes.append(f"Antecedente: {evento['historia_causas']}")
    return " ".join(partes)[:2000]


def enviar(tabla_notificaciones: pd.DataFrame) -> pd.DataFrame:
    """Manda por los canales configurados. 'tabla' siempre queda escrito."""
    if tabla_notificaciones.empty:
        return tabla_notificaciones
    df = tabla_notificaciones.copy()
    estados = ["tabla: OK"]
    mensajes = [{"id": r["BD_ID_EVENTO"], "vigilancia": r["BD_VIGILANCIA"], "clave": r["BD_CLAVE"],
                 "nivel": r["BD_NIVEL"], "mensaje": r["BD_MENSAJE"]} for _, r in df.iterrows()]
    for canal in CANALES_NOTIFICACION:
        if canal == "tabla":
            continue
        funcion = NOTIFICADORES.get(canal)
        if funcion is None:
            estados.append(f"{canal}: no hay notificador")
            continue
        try:
            estados.append(f"{canal}: {funcion(mensajes)}")
        except Exception as err:                      # un canal caído no puede tumbar la corrida
            log.error("no se pudo notificar por %s: %s", canal, err)
            estados.append(f"{canal}: ERROR {err}")
    df["BD_ESTADO_ENVIO"] = "; ".join(estados)[:200]
    return df


def guardar(conn, tablas: Dict[str, pd.DataFrame], cfg: VigConfig) -> Dict[str, int]:
    """Series: se reemplazan. Eventos: se pisan los que se recalcularon y se conserva el resto.
    Notificaciones y resumen: se acumulan (el resumen, uno por fecha de corte)."""
    _exigir(conn, "destino", "guardar()")
    f = Fechas.desde(cfg.fecha_ejecucion)
    escritas: Dict[str, int] = {}
    t0 = time.time()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLA_SERIE}")
            log.info("borradas %s filas de %s", f"{cur.rowcount:,}", TABLA_SERIE)
            eventos = tablas.get(TABLA_EVENTO, pd.DataFrame())
            if len(eventos):
                ids = eventos["BD_ID_EVENTO"].tolist()
                cur.executemany(f"DELETE FROM {TABLA_EVENTO} WHERE BD_ID_EVENTO = :1",
                                [(x,) for x in ids])
                log.info("actualizados %s eventos ya existentes", f"{len(ids):,}")
            cur.execute(f"DELETE FROM {TABLA_RESUMEN} WHERE FECHA_CORTE = :1", (f.ayer.to_pydatetime(),))
        for tabla, cols in ((TABLA_SERIE, COLUMNAS_SERIE), (TABLA_EVENTO, COLUMNAS_EVENTO),
                            (TABLA_NOTIFICACION, COLUMNAS_NOTIFICACION), (TABLA_RESUMEN, COLUMNAS_RESUMEN)):
            df = tablas.get(tabla, pd.DataFrame())
            escritas[tabla] = _insertar(conn, tabla, df, cols) if len(df) else 0
        conn.commit()
    except Exception:
        conn.rollback()
        log.error("falló la carga: las tablas quedaron como estaban")
        raise
    log.info("guardado: %s (%.1fs)", {k: f"{v:,}" for k, v in escritas.items()}, time.time() - t0)
    return escritas


# ─── control ────────────────────────────────────────────────────────────────
def resumen(motor: VigEngine, res: Resultado) -> None:
    log.info("corte %s | %s series vigiladas", motor.fechas.ayer.date(), f"{len(res.series):,}")
    if res.eventos.empty:
        log.info("sin eventos")
        return
    e = res.eventos
    principales = e[e.get("principal", True) & (e["estado"] != "CERRADO")]
    log.info("eventos: %s (principales abiertos %s) | por nivel %s | por estado %s",
             f"{len(e):,}", f"{len(principales):,}",
             e["nivel"].value_counts().to_dict(), e["estado"].value_counts().to_dict())
    for _, n in res.notificaciones.iterrows():
        log.warning("%s", mensaje(n))
    if len(principales):
        top = principales.sort_values("materialidad", ascending=False).head(5)
        log.info("lo que más pesa: %s", [(r["clave"], r["detector"], f"{r['materialidad']:,.0f} {r['unidad']}")
                                         for _, r in top.iterrows()])
    log.info("tiempos: %s", {k: f"{v:.1f}s" for k, v in motor.tiempos_.items()})
