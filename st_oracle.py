"""
st_oracle — configuración y acceso a Oracle para el motor de estadísticas.

ESTE ES EL ÚNICO ARCHIVO QUE HAY QUE EDITAR. El notebook del pipeline
(`run_estadisticas.ipynb`) no tiene nada configurable adentro.

Dependencias:
    pip install pandas numpy scipy oracledb

Variables de entorno (las mismas que el forecast):
    ORA_USER, ORA_PASSWORD, ORA_DSN                    origen (lee la venta)
    ORA_DEST_USER, ORA_DEST_PASSWORD, ORA_DEST_DSN     destino (tabla de estadísticas)
    ST_MODO              auto | completo | incremental            (opcional)
    ST_RELEER_MESES      meses a releer en la incremental          (opcional)
    ST_RELEER_DESDE      releer desde esta fecha, YYYY-MM-DD       (opcional)
    ST_FECHA_EJECUCION   "hoy" para el cálculo, YYYY-MM-DD         (opcional)
    ST_DRY_RUN           "1" = calcula y no escribe                (opcional)

MODOS DE CORRIDA
----------------
    auto         incremental si la tabla destino está sana; si no, completa y dice por qué.
    completo     relee toda la fuente y recalcula todo.
    incremental  exige incremental: si la tabla no sirve, corta con error en vez de
                 pasar a completa.

CÓMO FUNCIONA LA INCREMENTAL
----------------------------
1. Valida la tabla destino con una consulta agregada, sin traer los CLOB:
   que tenga filas, una sola fecha de corte, una sola "fecha de datos desde",
   la misma configuración que la actual (HUELLA_CONFIG) y historial en todas.
2. Decide desde cuándo releer: el 1° del mes RELEER_MESES antes del mes en curso.
   Si la última carga quedó más atrás (el pipeline no corrió unos días), relee
   desde el día siguiente a esa carga: nunca quedan huecos.
3. Arma la historia: BD_HISTORIAL guardado ANTES de esa fecha + la fuente DESDE esa
   fecha. Lo releído reemplaza a lo guardado, así que las correcciones dentro de la
   ventana se aplican solas, incluidas las bajas.
4. Informa qué cambió en el tramo releído: días nuevos, corregidos, eliminados y
   la diferencia en USD.
5. Recalcula y reemplaza la tabla completa. Hace falta igual: días sin compra,
   ventanas y probabilidad cambian para todos los clientes cada día.

Una corrección ANTERIOR a la ventana no se ve: para eso ST_RELEER_DESDE=fecha o
ST_MODO=completo.

Dónde está cada cosa:
    qué columnas salen y su descripción   -> stats_engine.catalogo()
    cómo se calcula cada una              -> la función indicada en el catálogo
    de dónde salen los datos              -> SQL_FUENTE, acá abajo
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import oracledb

import stats_engine
from stats_engine import Fechas, StatsConfig, StatsEngine, catalogo, historial_a_filas, MODELOS_ACTIVIDAD

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
#  2. FUENTE Y DESTINO
# ═══════════════════════════════════════════════════════════════════════════
# Convención: SK_/BK_ agrupan (claves), BD_ se arrastran con su valor más reciente.
CATEGORIAS = ["SK_CLIENTE", "BD_CLIENTE"]

# Tipo Oracle de cada categoría. Por defecto: SK_/BK_ -> NUMBER, BD_ -> VARCHAR2(200).
TIPOS_CATEGORIA: dict = {}

COL_FECHA = "FECHA"
COL_VENTA = "MT_VENTA"
COL_MARGEN = "MT_MARGEN"          # margen bruto en USD (venta - costo)

TABLA_DESTINO = "EST_CLIENTE"     # podés calificarla: "ESQUEMA.EST_CLIENTE"

# "delete"   : DELETE + INSERT en una transacción. Si algo falla, la tabla queda como estaba.
# "truncate" : TRUNCATE + INSERT. Más rápido con tablas grandes, pero TRUNCATE hace
#              commit implícito: si el INSERT falla, la tabla queda VACÍA.
MODO_CARGA = "delete"

# Grano diario ya agregado en la base. :desde y :hasta los completa el pipeline
# según el modo de corrida: NO los quites (la incremental depende de ellos).
SQL_FUENTE = f"""
    SELECT v.SK_CLIENTE,
           v.BD_CLIENTE,
           TRUNC(v.FECHA)   AS {COL_FECHA},
           SUM(v.VENTA)     AS {COL_VENTA},
           SUM(v.MARGEN)    AS {COL_MARGEN}
      FROM VENTAS v
     WHERE v.FECHA >= :desde
       AND v.FECHA <  :hasta
     GROUP BY v.SK_CLIENTE, v.BD_CLIENTE, TRUNC(v.FECHA)
"""

# ── Lectura incremental ─────────────────────────────────────────────────────
MODO_CORRIDA = "auto"                # auto | completo | incremental
RELEER_MESES = 1                     # 1 = desde el 1° del mes anterior al mes en curso
MAX_DIAS_SIN_DATOS = 3               # la fuente debe traer datos hasta ayer - N días; None = no validar
FECHA_INICIO_FUENTE = "1900-01-01"   # desde dónde lee la corrida completa

# ═══════════════════════════════════════════════════════════════════════════
#  3. PARÁMETROS DEL CÁLCULO
# ═══════════════════════════════════════════════════════════════════════════
def build_config(fecha_ejecucion: str | None = None) -> StatsConfig:
    return StatsConfig(
        categorias=CATEGORIAS,
        col_fecha=COL_FECHA,
        col_venta=COL_VENTA,
        col_margen=COL_MARGEN,
        fecha_ejecucion=fecha_ejecucion,     # None = hoy

        dias_ventana={"R6": 183, "R12": 365, "R24": 730, "R36": 1095},   # "hasta ayer"
        meses_ventana={"R6": 6, "R12": 12, "R24": 24, "R36": 36},        # "a mes cerrado"
        variantes=("R12", "R24", "R36"),

        ddof=1,                   # desvío muestral
        idd_unidad="gon",         # 0 plano, 100 creciente, -100 decreciente
        idd_normalizar=False,     # pendiente cruda, como se acordó
        margen_escala=100.0,      # márgenes en %

        modelo_actividad="pareto",     # el mejor en las pruebas; alternativas: mbgnbd, bgnbd
        max_clientes_ajuste=20_000,    # los 4 parámetros se estiman sobre esta muestra estable

        incluir_historial=True,   # la incremental lo necesita
        decimales_historial=2,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  De acá para abajo no hace falta tocar nada
# ═══════════════════════════════════════════════════════════════════════════
ARRAYSIZE = 100_000
BATCH_ROWS = 5_000        # filas por executemany (el CLOB pesa)

log = logging.getLogger("st")


def configurar_logging(nombre: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], force=True,
                        format=f"%(asctime)s {nombre} [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    stats_engine.LOGGER.handlers.clear()
    stats_engine.LOGGER.propagate = True
    return logging.getLogger(nombre)


# ─── conexiones ─────────────────────────────────────────────────────────────
_ROL = "_st_rol"


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
    oracledb.defaults.fetch_lobs = False          # CLOB -> str directo, sin locators
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


# ─── esquema de la tabla destino ────────────────────────────────────────────
def tipo_categoria(col: str) -> str:
    if col in TIPOS_CATEGORIA:
        return TIPOS_CATEGORIA[col]
    return "VARCHAR2(200)" if col.upper().startswith("BD_") else "NUMBER"


def huella_config(cfg: StatsConfig) -> str:
    """Resume lo que define el contenido del historial. Si cambia, la incremental
    no puede reutilizar lo guardado y se hace una corrida completa."""
    base = json.dumps({"categorias": [c.upper() for c in cfg.categorias], "fecha": cfg.col_fecha,
                       "venta": cfg.col_venta, "margen": cfg.col_margen,
                       "decimales": cfg.decimales_historial, "sql": " ".join(SQL_FUENTE.split())},
                      sort_keys=True)
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


COLUMNAS_CONTROL = [
    ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo (el día anterior a la ejecución)."),
    ("FECHA_DATOS_DESDE", "DATE", "Primer día de la fuente contenido en la historia: desde cuándo hay datos."),
    ("FECHA_RELECTURA_DESDE", "DATE", "Desde qué día se releyó la fuente en la corrida que escribió la fila. "
                                      "En una corrida completa coincide con FECHA_DATOS_DESDE."),
    ("TIPO_CORRIDA", "VARCHAR2(20)", "COMPLETA o INCREMENTAL."),
    ("HUELLA_CONFIG", "VARCHAR2(40)", "Resumen de la configuración (categorías, SQL de la fuente, historial). "
                                      "Si cambia, la próxima corrida es completa."),
]


def columnas_destino(cfg: StatsConfig) -> List[Tuple[str, str, str]]:
    """(columna, tipo Oracle, descripción) en el orden de la tabla."""
    cols = []
    for c in cfg.categorias:
        if c.upper().startswith("BD_"):
            desc = f"Descripción de categoría ({c}): valor de la fila más reciente de la fuente."
        else:
            desc = f"Clave de categoría ({c}). Junto con las demás claves define una fila."
        cols.append((c.upper(), tipo_categoria(c), desc))
    cols += [(m.nombre, m.tipo, m.descripcion) for m in catalogo(cfg)]
    cols += COLUMNAS_CONTROL
    cols.append(("FECHA_CARGA", "DATE", "Fecha y hora en que se cargó la fila."))
    return cols


def ddl_sugerido(cfg: StatsConfig) -> str:
    """CREATE TABLE sin constraints + COMMENT de cada columna. No lo ejecuta."""
    cols = columnas_destino(cfg)
    largos = [c for c, _, _ in cols if len(c) > 30]
    ancho = max(len(c) for c, _, _ in cols)
    cuerpo = ",\n".join(f"    {c.ljust(ancho)}  {t}" + ("  DEFAULT SYSDATE" if c == "FECHA_CARGA" else "")
                        for c, t, _ in cols)
    comentarios = "\n".join(
        f"COMMENT ON COLUMN {TABLA_DESTINO}.{c} IS '{d.replace(chr(39), chr(39) * 2)}';"
        for c, _, d in cols)
    aviso = (f"-- ATENCIÓN: columnas de más de 30 caracteres (Oracle < 12.2): {largos}\n"
             if largos else "")
    return (f"{aviso}CREATE TABLE {TABLA_DESTINO} (\n{cuerpo}\n);\n\n"
            f"COMMENT ON TABLE {TABLA_DESTINO} IS 'Estadísticas descriptivas por "
            f"{', '.join(cfg.claves())}. Se reemplaza completa en cada ejecución.';\n\n"
            f"{comentarios}\n\n"
            f"-- In-Memory: si la usás, conviene excluir el historial del column store\n"
            f"-- ALTER TABLE {TABLA_DESTINO} INMEMORY NO INMEMORY (BD_HISTORIAL);\n")


def validar_tabla(conn, cfg: StatsConfig) -> None:
    _exigir(conn, "destino", f"validar_tabla({TABLA_DESTINO})")
    owner, _, tabla = TABLA_DESTINO.rpartition(".")
    sql = "SELECT OWNER, COLUMN_NAME FROM ALL_TAB_COLUMNS WHERE TABLE_NAME = :t"
    params = {"t": tabla.upper()}
    if owner:
        sql += " AND OWNER = :o"
        params["o"] = owner.upper()
    enc = _fetch_df(conn, sql, params)
    if enc.empty:
        raise RuntimeError(f"El usuario de destino ({ORA_DEST_USER}) no ve la tabla {TABLA_DESTINO}.\n"
                           f"Si hay que crearla:\n\n{ddl_sugerido(cfg)}")
    duenos = sorted(set(enc["OWNER"]))
    if len(duenos) > 1:
        propio = [d for d in duenos if d.upper() == ORA_DEST_USER.upper()]
        enc = enc[enc["OWNER"] == (propio[0] if propio else duenos[0])]
    existentes = {c.upper() for c in enc["COLUMN_NAME"]}
    faltan = [(c, t) for c, t, _ in columnas_destino(cfg) if c not in existentes]
    if faltan:
        raise RuntimeError(f"A {TABLA_DESTINO} le faltan columnas: {', '.join(c for c, _ in faltan)}\n\n"
                           f"ALTER TABLE {TABLA_DESTINO} ADD ({', '.join(f'{c} {t}' for c, t in faltan)});")
    log.info("tabla %s.%s validada: %d columnas", enc["OWNER"].iloc[0], tabla.upper(),
             len(columnas_destino(cfg)))


# ─── plan de la corrida ─────────────────────────────────────────────────────
@dataclass
class Plan:
    tipo: str                                  # COMPLETA | INCREMENTAL
    releer_desde: pd.Timestamp                 # primer día que se lee de la fuente
    datos_desde: Optional[pd.Timestamp]        # incremental: heredado de la tabla
    corte_anterior: Optional[pd.Timestamp]     # incremental: hasta dónde llegaba lo guardado
    motivo: str

    def __str__(self) -> str:
        if self.tipo == "COMPLETA":
            return f"COMPLETA: lee toda la fuente ({self.motivo})"
        return (f"INCREMENTAL: guardado hasta {self.corte_anterior.date()}, datos desde "
                f"{self.datos_desde.date()}, relee la fuente desde {self.releer_desde.date()} ({self.motivo})")


def _diagnosticar(conn, cfg: StatsConfig):
    """(problema, estado). Si hay problema la incremental no es segura."""
    if not cfg.incluir_historial:
        return "el historial está desactivado y la incremental lo necesita", None
    e = _fetch_df(conn, f"""
        SELECT COUNT(*)                          AS FILAS,
               COUNT(DISTINCT FECHA_CORTE)       AS N_CORTES,
               MIN(FECHA_CORTE)                  AS CORTE,
               COUNT(DISTINCT FECHA_DATOS_DESDE) AS N_DESDE,
               MIN(FECHA_DATOS_DESDE)            AS DATOS_DESDE,
               COUNT(DISTINCT HUELLA_CONFIG)     AS N_HUELLAS,
               MIN(HUELLA_CONFIG)                AS HUELLA,
               SUM(CASE WHEN BD_HISTORIAL IS NULL THEN 1 ELSE 0 END) AS SIN_HISTORIAL
          FROM {TABLA_DESTINO}""").iloc[0]
    if int(e["FILAS"]) == 0:
        return "la tabla destino está vacía", None
    if int(e["N_CORTES"]) != 1:
        return f"la tabla tiene {int(e['N_CORTES'])} fechas de corte distintas (carga incompleta o mezclada)", None
    if int(e["N_DESDE"]) != 1:
        return "no se puede saber desde cuándo hay datos (FECHA_DATOS_DESDE vacía o mezclada)", None
    if int(e["N_HUELLAS"]) != 1 or e["HUELLA"] != huella_config(cfg):
        return ("la configuración cambió desde la última carga (categorías, SQL de la fuente o formato "
                "del historial)"), None
    if int(e["SIN_HISTORIAL"] or 0) > 0:
        return f"{int(e['SIN_HISTORIAL']):,} filas sin BD_HISTORIAL", None
    return None, {"filas": int(e["FILAS"]), "corte": pd.Timestamp(e["CORTE"]).normalize(),
                  "datos_desde": pd.Timestamp(e["DATOS_DESDE"]).normalize()}


def planificar(conn, cfg: StatsConfig, modo: str = MODO_CORRIDA, releer_meses: int = RELEER_MESES,
               releer_desde: str | None = None) -> Plan:
    """Decide si la corrida es completa o incremental y desde cuándo leer la fuente."""
    _exigir(conn, "destino", "planificar()")
    modo = str(modo).strip().lower()
    if modo not in ("auto", "completo", "incremental"):
        raise ValueError("ST_MODO debe ser auto, completo o incremental")
    for bind in (":desde", ":hasta"):
        if bind not in SQL_FUENTE:
            raise RuntimeError(f"SQL_FUENTE tiene que filtrar la fecha con {bind} (ver st_oracle.py)")

    inicio = pd.Timestamp(FECHA_INICIO_FUENTE)
    if modo == "completo":
        return Plan("COMPLETA", inicio, None, None, "pedida con ST_MODO=completo")

    problema, estado = _diagnosticar(conn, cfg)
    if problema:
        if modo == "incremental":
            raise RuntimeError(f"No se puede correr incremental: {problema}. Usá ST_MODO=completo.")
        log.warning("corrida COMPLETA: %s", problema)
        return Plan("COMPLETA", inicio, None, None, problema)

    f = Fechas.desde(cfg.fecha_ejecucion)
    if releer_desde:
        desde = pd.Timestamp(releer_desde).normalize()
        motivo = "relectura pedida con ST_RELEER_DESDE"
    else:
        desde = pd.Timestamp(f.hoy.year, f.hoy.month, 1) - pd.DateOffset(months=max(int(releer_meses), 0))
        motivo = f"relee {int(releer_meses)} mes(es) cerrado(s) más el mes en curso"
    siguiente = estado["corte"] + pd.Timedelta(days=1)
    if siguiente < desde:
        desde = siguiente
        motivo = (f"la última carga llegó hasta {estado['corte'].date()}: se relee desde el día "
                  f"siguiente para no dejar huecos")
    if desde <= estado["datos_desde"]:
        return Plan("COMPLETA", inicio, None, None,
                    f"la relectura ({desde.date()}) ya cubre desde el inicio de los datos "
                    f"({estado['datos_desde'].date()})")
    return Plan("INCREMENTAL", desde, estado["datos_desde"], estado["corte"], motivo)


# ─── lectura ────────────────────────────────────────────────────────────────
def _tipar(df: pd.DataFrame, cfg: StatsConfig) -> pd.DataFrame:
    df.columns = [c.upper() for c in df.columns]
    for c in cfg.categorias:
        if not c.upper().startswith("BD_") and tipo_categoria(c) == "NUMBER":
            df[c] = pd.to_numeric(df[c], errors="raise")
    return df


def leer_fuente(conn, cfg: StatsConfig, plan: Plan) -> pd.DataFrame:
    """Lee la fuente desde plan.releer_desde hasta ayer y valida que esté al día."""
    _exigir(conn, "origen", "leer_fuente()")
    f = Fechas.desde(cfg.fecha_ejecucion)
    t0 = time.time()
    df = _fetch_df(conn, SQL_FUENTE, {"desde": plan.releer_desde.to_pydatetime(),
                                      "hasta": f.hoy.to_pydatetime()})
    if df.empty:
        raise RuntimeError(f"La fuente no devolvió filas entre {plan.releer_desde.date()} y "
                           f"{f.ayer.date()}. ¿Se cargó la fuente?")
    df = _tipar(df, cfg)
    df[cfg.col_fecha] = pd.to_datetime(df[cfg.col_fecha]).dt.normalize()
    for c in (cfg.col_venta, cfg.col_margen):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ultimo = df[cfg.col_fecha].max()
    if MAX_DIAS_SIN_DATOS is not None and (f.ayer - ultimo).days > MAX_DIAS_SIN_DATOS:
        raise RuntimeError(
            f"La fuente tiene datos hasta {ultimo.date()} y se esperaban hasta {f.ayer.date()} "
            f"(tolerancia {MAX_DIAS_SIN_DATOS} días). Si la fuente se carga con más atraso, "
            f"subí MAX_DIAS_SIN_DATOS en st_oracle.py.")
    log.info("fuente: %s filas, %s grupos, %s a %s (%.1fs)", f"{len(df):,}",
             f"{df.groupby(cfg.claves()).ngroups:,}", df[cfg.col_fecha].min().date(),
             ultimo.date(), time.time() - t0)
    return df


def leer_estado(conn, cfg: StatsConfig) -> pd.DataFrame:
    """Reconstruye, desde BD_HISTORIAL, la historia diaria que está guardada."""
    _exigir(conn, "destino", "leer_estado()")
    t0 = time.time()
    cats = [c.upper() for c in cfg.categorias]
    df = _tipar(_fetch_df(conn, f"SELECT {', '.join(cats)}, BD_HISTORIAL FROM {TABLA_DESTINO}"), cfg)
    filas = historial_a_filas(df, cfg)
    log.info("estado guardado: %s grupos, %s días-cliente (%.1fs)", f"{len(df):,}",
             f"{len(filas):,}", time.time() - t0)
    return filas


def combinar(estado: Optional[pd.DataFrame], fuente: pd.DataFrame, cfg: StatsConfig,
             plan: Plan) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Historia guardada antes de releer + fuente desde releer. Devuelve también
    qué cambió en el tramo que se volvió a leer."""
    if plan.tipo == "COMPLETA" or estado is None:
        return fuente, None
    f, v = cfg.col_fecha, cfg.col_venta
    claves = cfg.claves()

    def por_dia(d):
        return d.groupby(claves + [f], dropna=False)[v].sum()

    antes = por_dia(estado[(estado[f] >= plan.releer_desde) & (estado[f] <= plan.corte_anterior)])
    ahora = por_dia(fuente[fuente[f] <= plan.corte_anterior])
    m = pd.concat([antes.rename("antes"), ahora.rename("ahora")], axis=1)
    auditoria = {
        "tramo": f"{plan.releer_desde.date()} a {plan.corte_anterior.date()}",
        "dias_nuevos": int((m["antes"].isna() & m["ahora"].notna()).sum()),
        "dias_eliminados": int((m["antes"].notna() & m["ahora"].isna()).sum()),
        "dias_corregidos": int((m["antes"].notna() & m["ahora"].notna()
                                & ((m["antes"] - m["ahora"]).abs() > 0.005)).sum()),
        "diferencia_usd": round(float(m["ahora"].fillna(0).sum() - m["antes"].fillna(0).sum()), 2),
        "dias_posteriores": int((fuente[f] > plan.corte_anterior).sum()),
    }
    columnas = list(cfg.categorias) + [f, v, cfg.col_margen]
    historia = pd.concat([estado.loc[estado[f] < plan.releer_desde, columnas], fuente[columnas]],
                         ignore_index=True)
    if plan.releer_desde > plan.corte_anterior:
        auditoria["tramo"] = "ninguno"
        log.info("relectura: no se volvió a leer nada ya guardado (se lee desde %s y lo guardado llega "
                 "hasta %s) | %s días-cliente nuevos desde la última carga", plan.releer_desde.date(),
                 plan.corte_anterior.date(), f"{auditoria['dias_posteriores']:,}")
    else:
        log.info("relectura %s: %s días-cliente nuevos, %s corregidos, %s eliminados, diferencia USD %s | "
                 "%s días-cliente posteriores a la última carga", auditoria["tramo"],
                 f"{auditoria['dias_nuevos']:,}", f"{auditoria['dias_corregidos']:,}",
                 f"{auditoria['dias_eliminados']:,}", f"{auditoria['diferencia_usd']:,.2f}",
                 f"{auditoria['dias_posteriores']:,}")
    return historia, auditoria


def anotar(out: pd.DataFrame, cfg: StatsConfig, plan: Plan, fuente: pd.DataFrame) -> pd.DataFrame:
    """Columnas de control que permiten validar la próxima corrida."""
    if plan.tipo == "COMPLETA":
        desde = fuente[cfg.col_fecha].min().normalize()
        releida = desde
    else:
        desde, releida = plan.datos_desde, plan.releer_desde
    out["FECHA_DATOS_DESDE"] = desde
    out["FECHA_RELECTURA_DESDE"] = releida
    out["TIPO_CORRIDA"] = plan.tipo
    out["HUELLA_CONFIG"] = huella_config(cfg)
    return out


# ─── escritura ──────────────────────────────────────────────────────────────
def _a_python(arr: np.ndarray, tipo: str) -> list:
    """Columna -> valores que oracledb entiende. NaN/inf -> None."""
    if tipo == "NUMBER":
        if arr.dtype.kind in "iu":
            return arr.astype(object).tolist()
        a = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(float)
        o = a.astype(object)
        o[~np.isfinite(a)] = None
        return o.tolist()
    if tipo == "DATE":
        return list(pd.DatetimeIndex(arr).to_pydatetime())
    return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x) for x in arr]


def _tipo_bind(tipo: str):
    if tipo == "NUMBER":
        return oracledb.DB_TYPE_NUMBER
    if tipo == "DATE":
        return oracledb.DB_TYPE_DATE
    if tipo == "CLOB":
        return oracledb.DB_TYPE_LONG      # strings largos a CLOB en array binding
    return oracledb.DB_TYPE_VARCHAR


def guardar(conn, df: pd.DataFrame, cfg: StatsConfig) -> int:
    """Borra la tabla destino e inserta todo, en lotes."""
    _exigir(conn, "destino", f"guardar() en {TABLA_DESTINO}")
    if df.empty:
        log.warning("no hay nada para guardar")
        return 0
    cols = [(c, t) for c, t, _ in columnas_destino(cfg) if c != "FECHA_CARGA"]
    faltan = [c for c, _ in cols if c not in df.columns]
    if faltan:
        raise KeyError(f"El resultado no tiene las columnas {faltan} (¿falta llamar a anotar()?)")
    nombres = [c for c, _ in cols]
    tipos = [t.split("(")[0] for _, t in cols]
    sql = (f"INSERT INTO {TABLA_DESTINO} ({', '.join(nombres)}, FECHA_CARGA)\n"
           f"VALUES ({', '.join(f':{i + 1}' for i in range(len(nombres)))}, SYSDATE)")
    datos = [df[c].to_numpy() for c in nombres]

    t0 = time.time()
    try:
        with conn.cursor() as cur:
            if MODO_CARGA == "truncate":
                cur.execute(f"TRUNCATE TABLE {TABLA_DESTINO}")
                log.info("tabla %s truncada", TABLA_DESTINO)
            else:
                cur.execute(f"DELETE FROM {TABLA_DESTINO}")
                log.info("borradas %s filas de %s", f"{cur.rowcount:,}", TABLA_DESTINO)
        with conn.cursor() as cur:
            cur.setinputsizes(*[_tipo_bind(t) for t in tipos])
            for i in range(0, len(df), BATCH_ROWS):
                lote = [_a_python(a[i:i + BATCH_ROWS], t) for a, t in zip(datos, tipos)]
                cur.executemany(sql, list(zip(*lote)))
        conn.commit()
    except Exception:
        conn.rollback()
        log.error("falló la carga%s", ": la tabla quedó como estaba" if MODO_CARGA == "delete"
                  else ": con TRUNCATE la tabla puede haber quedado vacía")
        raise
    log.info("insertadas %s filas x %d columnas en %s (%.1fs)", f"{len(df):,}",
             len(nombres) + 1, TABLA_DESTINO, time.time() - t0)
    return len(df)


# ─── control ────────────────────────────────────────────────────────────────
def resumen(motor: StatsEngine, df: pd.DataFrame) -> None:
    log.info("%s filas x %d columnas | corte %s", f"{len(df):,}", df.shape[1],
             motor.fechas.ayer.date())
    params = getattr(motor.ctx, "parametros_actividad", None)
    if params:
        log.info("modelo de actividad %s: %s", MODELOS_ACTIVIDAD[motor.cfg.modelo_actividad],
                 ", ".join(f"{k}={v:.4f}" for k, v in params.items()))
    nulos = df.isna().mean().sort_values(ascending=False)
    nulos = nulos[nulos > 0].head(8)
    if len(nulos):
        log.info("columnas con más nulos: %s", {k: f"{v:.0%}" for k, v in nulos.items()})
    log.info("más lentas: %s", {k: f"{v:.2f}s" for k, v in motor.tiempos(5).items()})
