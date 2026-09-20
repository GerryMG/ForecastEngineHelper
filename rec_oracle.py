# -*- coding: utf-8 -*-
"""
Configuración y entrada/salida Oracle del motor de recomendación.

Este es el ÚNICO archivo que se toca para poner a andar un análisis nuevo:
qué se recomienda, a quién, con qué segmentación y con qué reglas. El cálculo
está en rec_engine.py.

  1. CONEXIONES          origen (lectura) y destino (escritura), por variables de entorno
  2. QUÉ SE RECOMIENDA   entidad, ítem, segmentos y el SQL de la fuente
  3. PARÁMETROS          ventanas, recortes y la batería de algoritmos
  4. De acá para abajo   no hace falta tocar nada

Variables de entorno:
    ORA_USER / ORA_PASSWORD / ORA_DSN                 origen
    ORA_DEST_USER / ORA_DEST_PASSWORD / ORA_DEST_DSN  destino
    RC_FECHA_EJECUCION, RC_SELECCION, RC_DRY_RUN      opcionales
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import oracledb

import rec_engine
from rec_engine import RecConfig, RecEngine, Fechas, catalogo

# ═══════════════════════════════════════════════════════════════════════════
#  1. CONEXIONES  (origen y destino son distintos)
# ═══════════════════════════════════════════════════════════════════════════
ORA_USER = os.getenv("ORA_USER", "APP_LECTURA")
ORA_PASSWORD = os.getenv("ORA_PASSWORD", "")
ORA_DSN = os.getenv("ORA_DSN", "srv-origen.midominio.com:1521/DWH")

ORA_DEST_USER = os.getenv("ORA_DEST_USER", "")
ORA_DEST_PASSWORD = os.getenv("ORA_DEST_PASSWORD", "")
ORA_DEST_DSN = os.getenv("ORA_DEST_DSN") or ORA_DSN

ORACLE_CLIENT_LIB = os.getenv("ORACLE_CLIENT_LIB")  # sólo modo thick

# ═══════════════════════════════════════════════════════════════════════════
#  2. QUÉ SE RECOMIENDA
# ═══════════════════════════════════════════════════════════════════════════
# Convención: SK_/BK_ agrupan (claves), BD_ se arrastran con su valor más reciente.

# A quién se le recomienda.
ENTIDAD = ["SK_CLIENTE", "BD_CLIENTE"]

# Qué se le recomienda. Es el nivel al que se arma la matriz: submarca, familia,
# servicio, lo que sea. Más fino = más específico pero más ralo; si el cliente
# medio compra 2 de 5.000 ítems, subí de nivel.
ITEM = ["BK_SUBMARCA", "BD_SUBMARCA"]

# Con quién se compara: la jerarquía del nivel MÁS GRUESO al MÁS FINO. Se prueba primero
# la combinación completa (SEGMENTO | SUBSEGMENTO) y, si a ese grupo no le llegan
# MIN_ENTIDADES_SEGMENTO entidades con compras, se suelta el último nivel, y así hasta
# GLOBAL. [] = un solo grupo para todos.
SEGMENTOS = ["BD_SEGMENTO", "BD_SUBSEGMENTO"]

# Tipo Oracle de cada categoría. Por defecto: SK_/BK_ -> NUMBER, BD_ -> VARCHAR2(200).
TIPOS_CATEGORIA: dict = {}

COL_FECHA = "FECHA"
COL_VALOR = "MT_VENTA"            # importe en USD
COL_MARGEN = "MT_MARGEN"          # margen bruto en USD (venta - costo)

# ── Tipo de documento ──────────────────────────────────────────────────────
# Si tu fuente distingue facturas, notas de crédito, devoluciones, fletes, ajustes...
# traé esa columna y decidí acá qué hace cada tipo. Dejá COL_TIPO_DOCUMENTO = None
# si ya filtrás todo en el SQL.
COL_TIPO_DOCUMENTO = "BD_TIPO_DOCUMENTO"
TIPOS_VENTA = ["FACTURA"]              # sólo estos cuentan como compra ([] = todos los no excluidos)
TIPOS_DEVOLUCION = ["NOTA_CREDITO"]    # estos restan
DEVOLUCION_YA_NEGATIVA = True          # False si en la fuente vienen en positivo
TIPOS_EXCLUIDOS = ["FLETE", "AJUSTE"]  # estos se ignoran por completo
EXCLUIR_NETOS_NO_POSITIVOS = True      # comprado y devuelto entero no es una compra

TABLA_DESTINO = "REC_CLIENTE_ITEM"        # una fila por entidad e ítem recomendado
TABLA_DIAGNOSTICO = "REC_DIAGNOSTICO"     # qué algoritmo ganó en cada segmento y con qué números
GUARDAR_DIAGNOSTICO = True

# "delete"   : DELETE + INSERT en una transacción. Si algo falla, la tabla queda como estaba.
# "truncate" : TRUNCATE + INSERT. Más rápido, pero si el INSERT falla la tabla queda VACÍA.
MODO_CARGA = "delete"

# La fuente se lee al grano en que quieras: el motor agrega solo. :desde y :hasta los
# completa el pipeline con la ventana que necesita (afinidad + backtest): NO los quites.
# Filtrá acá lo que no querés recomendar (discontinuados, fletes, notas de crédito).
SQL_FUENTE = f"""
    SELECT v.SK_CLIENTE,
           v.BD_CLIENTE,
           v.BD_SUBSEGMENTO,
           v.BD_SEGMENTO,
           v.BK_SUBMARCA,
           v.BD_SUBMARCA,
           v.BD_TIPO_DOCUMENTO,
           TRUNC(v.FECHA)   AS {COL_FECHA},
           SUM(v.VENTA)     AS {COL_VALOR},
           SUM(v.MARGEN)    AS {COL_MARGEN}
      FROM VENTAS v
     WHERE v.FECHA >= :desde
       AND v.FECHA <  :hasta
     GROUP BY v.SK_CLIENTE, v.BD_CLIENTE, v.BD_SUBSEGMENTO, v.BD_SEGMENTO,
              v.BK_SUBMARCA, v.BD_SUBMARCA, v.BD_TIPO_DOCUMENTO, TRUNC(v.FECHA)
"""

# ── Cómo se mide la afinidad ───────────────────────────────────────────────
# "canasta"     : lo que se compra JUNTO. Sin número de documento, la canasta es el día:
#                 sirve cuando el cliente compra una vez por día.
# "repertorio"  : todo lo que el cliente compra en la ventana, junto o no.
# Si tu fuente trae el documento, poné COL_DOCUMENTO y la canasta pasa a ser el documento.
AFINIDAD = "canasta"
COL_DOCUMENTO = None

# ── Tamaño del cliente ─────────────────────────────────────────────────────
# Agrega un nivel de segmentación por tamaño, para no comparar a un cliente mediano
# contra uno que compra veinte veces más. Son cuantiles de compra en la ventana.
# [] = no segmentar por tamaño.
CORTES_TAMANO = [0.5, 0.8, 0.95]
ETIQUETAS_TAMANO = ["CHICO", "MEDIANO", "GRANDE", "TOP"]

# ═══════════════════════════════════════════════════════════════════════════
#  3. PARÁMETROS DEL CÁLCULO
# ═══════════════════════════════════════════════════════════════════════════
DIAS_AFINIDAD = 365            # ventana de la matriz de compras. 0 = TODA la historia
FECHA_INICIO_FUENTE = "1900-01-01"   # desde dónde leer cuando DIAS_AFINIDAD = 0
DIAS_BACKTEST = 90             # tramo final que se reserva para medir aciertos
MIN_ENTIDADES_SEGMENTO = 200   # menos que esto y el segmento sube de nivel
MIN_SOPORTE = 5                # entidades del segmento que tienen que comprar el ítem
MIN_PENETRACION = 0.02         # y la fracción mínima del segmento
MAX_ITEMS_RECO = 10            # recomendaciones por entidad

# La batería. Se miden todas con backtest y gana la mejor en cada segmento.
ALGORITMOS = ("popularidad", "coseno_item", "coseno_entidad", "svd", "kmeans_valor", "reglas")
SELECCION = "backtest"         # backtest | rrf | ponderado | el nombre de un algoritmo
METRICA_SELECCION = "precision"     # precision | usd | recall
PESOS: dict = {}               # sólo para seleccion="ponderado", por ejemplo {"coseno_item": 2}

TIPOS_RECOMENDACION = ("CRUZADA", "REPOSICION", "BRECHA")
FACTOR_REPOSICION = 1.5        # silencio mayor a esto x su intervalo típico = atrasado
BRECHA_RATIO = 0.5             # compra menos de la mitad de lo que le dedican sus pares

# ── Cómo se estima el valor ────────────────────────────────────────────────
# Los tres tipos se miden igual: USD esperados en los próximos HORIZONTE_DIAS. Así el
# ranking compara lo mismo y no mezcla "lo que gasta un par al año" con "lo que dejó de
# comprar en 300 días".
HORIZONTE_DIAS = 90
# Cuánta evidencia de los pares se le presta al cliente con poca historia propia.
# 3 significa "sus datos valen tanto como los del segmento cuando tiene 3 intervalos".
# 0 = no prestar nada.
PESO_PRIOR_PARES = 3.0
# Multiplicar por la probabilidad de que la compra ocurra (recompra en reposición, tasa de
# adopción medida por el backtest en cruzada).
USAR_PROBABILIDAD = True
# Pesar la afinidad por recencia: lo de hace N días pesa la mitad. 0 = todo pesa igual.
VIDA_MEDIA_AFINIDAD_DIAS = 0
# Por qué se ordena la lista de cada cliente:
#   "esperado" : USD por probabilidad. Asigna bien el esfuerzo del vendedor.
#   "bruto"    : el tamaño de la oportunidad sin descontar la probabilidad. Deja arriba a
#                los clientes muy atrasados, que son campañas de recuperación.
ORDENAR_POR = "esperado"
PISO_PROB = 0.0       # piso de la probabilidad; 0,05 le deja una chance mínima a lo muy atrasado
MIN_CASOS_RECUPERACION = 30   # casos para creerle a la curva de un ítem; con menos, la del panel

# ── Cuánta evidencia se exige ──────────────────────────────────────────────
MIN_COMPRAS_REPOSICION = 2     # días de compra del par. Con 1 no hay ritmo propio pero el
                               # segmento lo presta; subilo a 3 si querés ser conservador
MAX_CV_INTERVALO = 1.0         # qué tan irregular puede ser el ritmo PROPIO. None = no filtrar
MIN_DIAS_COMPRA_ENTIDAD = 3    # días de compra de la entidad para recomendarle algo
TOPE_POTENCIAL_POR_HISTORICO = 1.5   # veces el propio ritmo de compra del ítem. 0 = sin tope
TOPE_POTENCIAL_RELATIVO = 1.0        # veces su compra total en el mismo lapso. 0 = sin tope


def build_config(fecha_ejecucion: str | None = None, seleccion: str | None = None) -> RecConfig:
    return RecConfig(
        entidad=ENTIDAD,
        item=ITEM,
        segmentos=SEGMENTOS,
        col_fecha=COL_FECHA,
        col_valor=COL_VALOR,
        col_margen=COL_MARGEN,
        fecha_ejecucion=fecha_ejecucion,        # None = hoy

        col_tipo_documento=COL_TIPO_DOCUMENTO,
        tipos_venta=TIPOS_VENTA,
        tipos_devolucion=TIPOS_DEVOLUCION,
        devolucion_ya_negativa=DEVOLUCION_YA_NEGATIVA,
        tipos_excluidos=TIPOS_EXCLUIDOS,
        excluir_netos_no_positivos=EXCLUIR_NETOS_NO_POSITIVOS,

        afinidad=AFINIDAD,
        col_documento=COL_DOCUMENTO,
        cortes_tamano=CORTES_TAMANO,
        etiquetas_tamano=ETIQUETAS_TAMANO,

        dias_afinidad=DIAS_AFINIDAD,
        dias_backtest=DIAS_BACKTEST,
        min_entidades_segmento=MIN_ENTIDADES_SEGMENTO,
        min_soporte=MIN_SOPORTE,
        min_penetracion=MIN_PENETRACION,
        max_items_reco=MAX_ITEMS_RECO,

        algoritmos=ALGORITMOS,
        seleccion=seleccion or SELECCION,
        metrica_seleccion=METRICA_SELECCION,
        pesos=PESOS,

        incluir_tipos=TIPOS_RECOMENDACION,
        factor_reposicion=FACTOR_REPOSICION,
        brecha_ratio=BRECHA_RATIO,
        min_compras_reposicion=MIN_COMPRAS_REPOSICION,
        max_cv_intervalo=MAX_CV_INTERVALO,
        min_dias_compra_entidad=MIN_DIAS_COMPRA_ENTIDAD,
        horizonte_dias=HORIZONTE_DIAS,
        peso_prior_pares=PESO_PRIOR_PARES,
        usar_probabilidad=USAR_PROBABILIDAD,
        vida_media_afinidad_dias=VIDA_MEDIA_AFINIDAD_DIAS,
        ordenar_por=ORDENAR_POR,
        piso_prob=PISO_PROB,
        min_casos_recuperacion=MIN_CASOS_RECUPERACION,
        tope_potencial_por_historico=TOPE_POTENCIAL_POR_HISTORICO,
        tope_potencial_relativo=TOPE_POTENCIAL_RELATIVO,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  De acá para abajo no hace falta tocar nada
# ═══════════════════════════════════════════════════════════════════════════
ARRAYSIZE = 100_000
BATCH_ROWS = 20_000
ORACLE_NUMBER_MIN = 1e-130
ORACLE_NUMBER_MAX = 9.99999999e125

log = logging.getLogger("rec")


def configurar_logging(nombre: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], force=True,
                        format=f"%(asctime)s {nombre} [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    rec_engine.LOGGER.handlers.clear()
    rec_engine.LOGGER.propagate = True
    return logging.getLogger(nombre)


# ─── conexiones ─────────────────────────────────────────────────────────────
_ROL = "_rec_rol"


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


# ─── esquema de las tablas destino ──────────────────────────────────────────
def tipo_categoria(col: str) -> str:
    if col in TIPOS_CATEGORIA:
        return TIPOS_CATEGORIA[col]
    return "VARCHAR2(200)" if col.upper().startswith("BD_") else "NUMBER"


def huella_config(cfg: RecConfig) -> str:
    """Resumen de la configuración, para saber con qué reglas se escribió cada fila."""
    base = json.dumps({"entidad": [c.upper() for c in cfg.entidad],
                       "item": [c.upper() for c in cfg.item],
                       "segmentos": [c.upper() for c in cfg.segmentos],
                       "algoritmos": list(cfg.algoritmos), "seleccion": cfg.seleccion,
                       "afinidad": cfg.dias_afinidad, "backtest": cfg.dias_backtest,
                       "sql": " ".join(SQL_FUENTE.split())}, sort_keys=True)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


COLUMNAS_CONTROL = [
    ("HUELLA_CONFIG", "VARCHAR2(40)", "Resumen de la configuración con la que se generó la fila "
                                      "(entidad, ítem, segmentos, algoritmos, SQL de la fuente)."),
    ("FECHA_CARGA", "DATE", "Fecha y hora en que se cargó la fila."),
]

COLUMNAS_DIAGNOSTICO = [
    ("BD_SEGMENTO", "VARCHAR2(400)", "Segmento evaluado."),
    ("BD_NIVEL_SEGMENTO", "VARCHAR2(100)", "Nivel de segmentación usado, o GLOBAL."),
    ("BD_ALGORITMO", "VARCHAR2(40)", "Algoritmo medido."),
    ("MT_ENTIDADES", "NUMBER", "Entidades del segmento."),
    ("MT_ITEMS_CANDIDATOS", "NUMBER", "Ítems que pasaron los mínimos de soporte y penetración."),
    ("MT_RECOMENDADOS", "NUMBER", "Recomendaciones cruzadas que produjo en el backtest."),
    ("MT_ACIERTOS", "NUMBER", "Cuántas de esas se compraron después."),
    ("MT_PRECISION", "NUMBER", "Aciertos sobre recomendados."),
    ("MT_ADOPCIONES", "NUMBER", "Ítems nuevos que el segmento compró en el tramo evaluado."),
    ("MT_ENTIDADES_QUE_ADOPTAN", "NUMBER", "Entidades que compraron al menos un ítem nuevo."),
    ("MT_RECALL", "NUMBER", "Aciertos sobre adopciones."),
    ("MT_USD_ACERTADO", "NUMBER", "USD comprados en el tramo evaluado de los ítems acertados."),
    ("MT_SEGUNDOS", "NUMBER", "Lo que tardó el algoritmo en ese segmento."),
    ("BD_ELEGIDO", "VARCHAR2(5)", "SI si es el algoritmo que se usó en la corrida final."),
    ("BD_MOTIVO_SELECCION", "VARCHAR2(200)", "Por qué se eligió: el que más acertó en el segmento, o el "
                                             "ganador del panel cuando el segmento no tenía adopciones suficientes."),
    ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo."),
    ("HUELLA_CONFIG", "VARCHAR2(40)", "Resumen de la configuración."),
    ("FECHA_CARGA", "DATE", "Fecha y hora en que se cargó la fila."),
]


def columnas_destino(cfg: RecConfig) -> List[Tuple[str, str, str]]:
    cols = [(c, t, d) for c, t, d in catalogo(cfg)]
    salida = []
    for c, t, d in cols:
        if c in [x.upper() for x in list(cfg.entidad) + list(cfg.item)]:
            t = tipo_categoria(c)
        salida.append((c, t, d))
    return salida + COLUMNAS_CONTROL


def _ddl(tabla: str, cols: List[Tuple[str, str, str]], comentario: str) -> str:
    largos = [c for c, _, _ in cols if len(c) > 30]
    ancho = max(len(c) for c, _, _ in cols)
    cuerpo = ",\n".join(f"    {c.ljust(ancho)}  {t}" + ("  DEFAULT SYSDATE" if c == "FECHA_CARGA" else "")
                        for c, t, _ in cols)
    comentarios = "\n".join(f"COMMENT ON COLUMN {tabla}.{c} IS '{d.replace(chr(39), chr(39) * 2)}';"
                            for c, _, d in cols)
    aviso = f"-- ATENCIÓN: columnas de más de 30 caracteres (Oracle < 12.2): {largos}\n" if largos else ""
    return (f"{aviso}CREATE TABLE {tabla} (\n{cuerpo}\n);\n\n"
            f"COMMENT ON TABLE {tabla} IS '{comentario}';\n\n{comentarios}\n")


def ddl_sugerido(cfg: RecConfig) -> str:
    """CREATE TABLE de las dos tablas, sin constraints y con la descripción de cada columna."""
    texto = _ddl(TABLA_DESTINO, columnas_destino(cfg),
                 f"Recomendaciones por {', '.join(cfg.claves_entidad())}. Se reemplaza completa en cada ejecución.")
    if GUARDAR_DIAGNOSTICO:
        texto += "\n" + _ddl(TABLA_DIAGNOSTICO, COLUMNAS_DIAGNOSTICO,
                             "Backtest: qué algoritmo acertó más en cada segmento. Se reemplaza completa.")
    return texto


def validar_tabla(conn, cfg: RecConfig) -> None:
    """Corta antes de leer nada si falta alguna columna, e imprime el ALTER exacto."""
    _exigir(conn, "destino", "validar_tabla()")
    objetivo = [(TABLA_DESTINO, columnas_destino(cfg))]
    if GUARDAR_DIAGNOSTICO:
        objetivo.append((TABLA_DIAGNOSTICO, COLUMNAS_DIAGNOSTICO))
    for tabla, cols in objetivo:
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
        duenos = sorted(set(enc["OWNER"]))
        if len(duenos) > 1:
            propio = [d for d in duenos if d.upper() == ORA_DEST_USER.upper()]
            enc = enc[enc["OWNER"] == (propio[0] if propio else duenos[0])]
        existentes = {c.upper() for c in enc["COLUMN_NAME"]}
        faltan = [(c, t) for c, t, _ in cols if c not in existentes]
        if faltan:
            raise RuntimeError(f"A {tabla} le faltan columnas: {', '.join(c for c, _ in faltan)}\n\n"
                               f"ALTER TABLE {tabla} ADD ({', '.join(f'{c} {t}' for c, t in faltan)});")
        log.info("tabla %s validada: %d columnas", tabla, len(cols))


# ─── lectura ────────────────────────────────────────────────────────────────
def ventana(cfg: RecConfig) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Desde cuándo hay que leer la fuente: la afinidad más el tramo del backtest.
    Con DIAS_AFINIDAD = 0 se lee toda la historia desde FECHA_INICIO_FUENTE."""
    f = Fechas.desde(cfg.fecha_ejecucion)
    if not cfg.dias_afinidad:
        return pd.Timestamp(FECHA_INICIO_FUENTE).normalize(), f.hoy
    dias = cfg.dias_afinidad + (cfg.dias_backtest if cfg.seleccion == "backtest" else 0)
    return (f.ayer - pd.Timedelta(days=dias - 1)).normalize(), f.hoy


def leer_fuente(conn, cfg: RecConfig) -> pd.DataFrame:
    _exigir(conn, "origen", "leer_fuente()")
    for bind in (":desde", ":hasta"):
        if bind not in SQL_FUENTE:
            raise RuntimeError(f"SQL_FUENTE tiene que filtrar la fecha con {bind} (ver rec_oracle.py)")
    desde, hasta = ventana(cfg)
    t0 = time.time()
    df = _fetch_df(conn, SQL_FUENTE, {"desde": desde.to_pydatetime(), "hasta": hasta.to_pydatetime()})
    if df.empty:
        raise RuntimeError(f"La fuente no devolvió filas entre {desde.date()} y {hasta.date()}. "
                           f"¿Se cargó la fuente?")
    df.columns = [c.upper() for c in df.columns]
    for c in list(cfg.entidad) + list(cfg.item):
        if not c.upper().startswith("BD_") and tipo_categoria(c) == "NUMBER":
            df[c] = pd.to_numeric(df[c], errors="raise")
    df[cfg.col_fecha] = pd.to_datetime(df[cfg.col_fecha]).dt.normalize()
    for c in (cfg.col_valor, cfg.col_margen):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log.info("fuente: %s filas, %s a %s (%.1fs)", f"{len(df):,}", desde.date(),
             df[cfg.col_fecha].max().date(), time.time() - t0)
    return df


# ─── escritura ──────────────────────────────────────────────────────────────
def _rango_oracle(a: np.ndarray) -> np.ndarray:
    """Lleva los floats al rango de NUMBER: casi cero -> 0, enormes -> ±máximo."""
    abs_ = np.abs(a)
    chicos = (abs_ < ORACLE_NUMBER_MIN) & (abs_ > 0)
    grandes = np.isfinite(a) & (abs_ > ORACLE_NUMBER_MAX)
    if chicos.any() or grandes.any():
        a = a.copy()
        a[chicos] = 0.0
        a[grandes] = np.sign(a[grandes]) * ORACLE_NUMBER_MAX
    return a


def _ancho(tipo: str) -> Optional[int]:
    """Ancho declarado de un VARCHAR2(n), si lo tiene."""
    if "(" in tipo and tipo.upper().startswith("VARCHAR"):
        try:
            return int(tipo[tipo.index("(") + 1:tipo.index(")")].split()[0])
        except ValueError:
            return None
    return None


def _a_python(arr: np.ndarray, tipo_completo: str) -> list:
    """Columna -> valores que oracledb entiende. NaN/inf -> None; fuera de rango -> dentro;
    textos recortados al ancho declarado (si no, Oracle rechaza la fila entera)."""
    tipo = tipo_completo.split("(")[0]
    if tipo != "NUMBER" and tipo != "DATE":
        ancho = _ancho(tipo_completo)
        if ancho:
            return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)[:ancho]
                    for x in arr]
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
        return list(pd.DatetimeIndex(arr).to_pydatetime())
    return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x) for x in arr]


def _tipo_bind(tipo_completo: str):
    tipo = tipo_completo.split("(")[0]
    if tipo == "NUMBER":
        return oracledb.DB_TYPE_NUMBER
    if tipo == "DATE":
        return oracledb.DB_TYPE_DATE
    return oracledb.DB_TYPE_VARCHAR


def _escribir(conn, tabla: str, df: pd.DataFrame, cols: List[Tuple[str, str, str]]) -> int:
    nombres = [c for c, _, _ in cols if c != "FECHA_CARGA"]
    tipos = [t for c, t, _ in cols if c != "FECHA_CARGA"]
    faltan = [c for c in nombres if c not in df.columns]
    if faltan:
        raise KeyError(f"El resultado no tiene las columnas {faltan} para {tabla}")
    sql = (f"INSERT INTO {tabla} ({', '.join(nombres)}, FECHA_CARGA)\n"
           f"VALUES ({', '.join(f':{i + 1}' for i in range(len(nombres)))}, SYSDATE)")
    datos = [df[c].to_numpy() for c in nombres]
    with conn.cursor() as cur:
        if MODO_CARGA == "truncate":
            cur.execute(f"TRUNCATE TABLE {tabla}")
            log.info("tabla %s truncada", tabla)
        else:
            cur.execute(f"DELETE FROM {tabla}")
            log.info("borradas %s filas de %s", f"{cur.rowcount:,}", tabla)
    with conn.cursor() as cur:
        cur.setinputsizes(*[_tipo_bind(t) for t in tipos])
        for i in range(0, len(df), BATCH_ROWS):
            lote = [_a_python(a[i:i + BATCH_ROWS], t) for a, t in zip(datos, tipos)]
            cur.executemany(sql, list(zip(*lote)))
    return len(df)


def anotar(df: pd.DataFrame, cfg: RecConfig) -> pd.DataFrame:
    df = df.copy()
    df["HUELLA_CONFIG"] = huella_config(cfg)
    return df


def preparar_diagnostico(diag: pd.DataFrame, cfg: RecConfig) -> pd.DataFrame:
    """La tabla del backtest, con los nombres de columna del destino."""
    if diag is None or diag.empty:
        return pd.DataFrame(columns=[c for c, _, _ in COLUMNAS_DIAGNOSTICO])
    f = Fechas.desde(cfg.fecha_ejecucion)
    return pd.DataFrame({
        "BD_SEGMENTO": diag["segmento"].to_numpy(dtype=object),
        "BD_NIVEL_SEGMENTO": diag["nivel_segmento"].to_numpy(dtype=object),
        "BD_ALGORITMO": diag["algoritmo"].to_numpy(dtype=object),
        "MT_ENTIDADES": diag["entidades"].to_numpy(float),
        "MT_ITEMS_CANDIDATOS": diag["items_candidatos"].to_numpy(float),
        "MT_RECOMENDADOS": diag["recomendados"].to_numpy(float),
        "MT_ACIERTOS": diag["aciertos"].to_numpy(float),
        "MT_PRECISION": np.round(diag["precision"].to_numpy(float), 6),
        "MT_ADOPCIONES": diag["adopciones"].to_numpy(float),
        "MT_ENTIDADES_QUE_ADOPTAN": diag["entidades_que_adoptan"].to_numpy(float),
        "MT_RECALL": np.round(diag["recall"].to_numpy(float), 6),
        "MT_USD_ACERTADO": diag["usd_acertado"].to_numpy(float),
        "MT_SEGUNDOS": diag["segundos"].to_numpy(float),
        "BD_ELEGIDO": np.where(diag["elegido"].to_numpy(), "SI", "NO"),
        "BD_MOTIVO_SELECCION": diag.get("motivo_seleccion", pd.Series([""] * len(diag))).to_numpy(dtype=object),
        "FECHA_CORTE": np.full(len(diag), f.ayer),
        "HUELLA_CONFIG": huella_config(cfg),
    })


def guardar(conn, df: pd.DataFrame, cfg: RecConfig, diagnostico: Optional[pd.DataFrame] = None) -> int:
    """Reemplaza las dos tablas en una sola transacción."""
    _exigir(conn, "destino", f"guardar() en {TABLA_DESTINO}")
    if df.empty:
        log.warning("no hay nada para guardar")
        return 0
    t0 = time.time()
    try:
        n = _escribir(conn, TABLA_DESTINO, df, columnas_destino(cfg))
        if GUARDAR_DIAGNOSTICO and diagnostico is not None and len(diagnostico):
            _escribir(conn, TABLA_DIAGNOSTICO, diagnostico, COLUMNAS_DIAGNOSTICO)
        conn.commit()
    except Exception:
        conn.rollback()
        log.error("falló la carga%s", ": las tablas quedaron como estaban" if MODO_CARGA == "delete"
                  else ": con TRUNCATE pueden haber quedado vacías")
        raise
    log.info("insertadas %s filas en %s (%.1fs)", f"{n:,}", TABLA_DESTINO, time.time() - t0)
    return n


# ─── control ────────────────────────────────────────────────────────────────
def resumen(motor: RecEngine, df: pd.DataFrame) -> None:
    cfg = motor.cfg
    clave = cfg.claves_entidad()[0].upper()
    log.info("%s recomendaciones | %s entidades | corte %s", f"{len(df):,}",
             f"{df[clave].nunique():,}", motor.fechas.ayer.date())
    log.info("por tipo: %s", df["BD_TIPO"].value_counts().to_dict())
    log.info("USD en juego: %s | margen: %s", f"{df['MT_USD_POTENCIAL'].sum():,.0f}",
             f"{df['MT_MARGEN_POTENCIAL'].sum():,.0f}")
    d = motor.diagnostico
    if len(d):
        ganadores = d[d["elegido"]]
        log.info("algoritmo elegido por segmento: %s",
                 dict(zip(ganadores["segmento"], ganadores["algoritmo"])))
        pop = d[d["algoritmo"] == "popularidad"].set_index("segmento")["precision"]
        for _, g in ganadores.iterrows():
            base = float(pop.get(g["segmento"], 0.0))
            if g["adopciones"] < motor.cfg.min_adopciones_backtest:
                log.info("  [%s] %s: %s adopciones en el tramo evaluado, no alcanza para medir (%s)",
                         g["segmento"], g["algoritmo"], f"{int(g['adopciones']):,}",
                         g.get("motivo_seleccion", ""))
            else:
                log.info("  [%s] %s: precisión %.3f contra %.3f de popularidad%s", g["segmento"],
                         g["algoritmo"], g["precision"], base,
                         f" ({g['precision'] / base:.1f}x)" if base else "")
    log.info("tiempos: %s", {k: f"{v:.1f}s" for k, v in motor.tiempos_.items()})
