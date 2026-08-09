"""
fc_oracle — configuración y acceso a Oracle para el pipeline de forecast.

ESTE ES EL ÚNICO ARCHIVO QUE HAY QUE EDITAR. Los notebooks del pipeline
(`run_inicial.ipynb` y `run_mensual.ipynb`) no tienen nada configurable adentro.

Dependencias:
    pip install pandas statsmodels joblib psutil oracledb prophet

Variables de entorno:
    ORA_USER, ORA_PASSWORD, ORA_DSN                    origen (lectura de la métrica real)
    ORA_DEST_USER, ORA_DEST_PASSWORD, ORA_DEST_DSN     destino (tabla de forecast)
    FC_CUTOFF                           último período cerrado, YYYY-MM-DD (opcional)
    FC_REESCRIBIR_DESDE                 desde dónde reescribir la salida (opcional)
    FC_DRY_RUN                          "1" = calcula y no escribe (opcional)
    FC_MAX_WORKERS                      tope de procesos, para el límite del pod (opcional)
    FC_LOG_FILE                         archivo de log además de stdout (opcional)

Origen y destino son dos conexiones distintas: pueden ser otro usuario, otro
esquema u otra base. Si no definís las variables del destino, se usan las del
origen y queda avisado en el log.

FORMATO DE LA TABLA DE SALIDA
-----------------------------
Ancha: una fila por (categorías, período) y una columna por modelo, más la
métrica real y el valor del modelo ganador. Es la misma tabla que se relee como
entrada en la corrida mensual.

    SK_CLIENTE | AT_FECHA   | Y_REAL | F_ETS | F_THETA | ... | BEST_MODEL | YHAT
    900001     | 2026-07-01 | 481.3  | 463.9 | 461.1   | ... | ets        | 463.9
    900001     | 2026-08-01 | (null) | 407.0 | 414.2   | ... | ets        | 407.0

Los nombres de las columnas se definen en COLUMNAS_MODELO y COLUMNAS_FIJAS más
abajo: cambiás el diccionario y cambia el DDL, el MERGE y la lectura.

La tabla NO se crea desde acá. `print(ddl_sugerido(cfg))` te imprime el CREATE
TABLE exacto que corresponde a tu configuración, y `validar_tabla(conn, cfg)`
avisa —antes de calcular nada— si falta alguna columna, con el ALTER que hace falta.

CÓMO SE ESCRIBE
---------------
DELETE de un rango + INSERT, todo en una transacción. El rango lo decide
`reescribir_desde`:

    "auto"        desde el corte (el mes que acaba de cerrar). Es lo normal en la
                  corrida mensual: se borran las 12 proyecciones viejas y se
                  insertan las 12 nuevas. Los meses anteriores no se tocan.
    "todo"        borra todo (con FILTRO_DESTINO si lo definiste) y reinserta.
                  Es lo que usa la corrida inicial y lo que querés si cambiaste
                  de modelos y necesitás rehacer la historia.
    6             seis períodos hacia atrás desde el corte.
    "2025-01-01"  desde esa fecha.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import oracledb

import forecast_engine
from forecast_engine import ForecastConfig, available_models, floor_to_freq, shift_period

# ═══════════════════════════════════════════════════════════════════════════
#  1. CONEXIONES  (origen y destino son distintos)
# ═══════════════════════════════════════════════════════════════════════════
ORA_USER = os.getenv("ORA_USER", "APP_LECTURA")
ORA_PASSWORD = os.getenv("ORA_PASSWORD", "")
ORA_DSN = os.getenv("ORA_DSN", "srv-origen.midominio.com:1521/DWH")

# Destino: si no están las variables, cae al origen y avisa por log.
ORA_DEST_USER = os.getenv("ORA_DEST_USER") or ORA_USER
ORA_DEST_PASSWORD = os.getenv("ORA_DEST_PASSWORD") or ORA_PASSWORD
ORA_DEST_DSN = os.getenv("ORA_DEST_DSN") or ORA_DSN

# Sólo si necesitás modo thick (base 11g, wallet, Kerberos):
ORACLE_CLIENT_LIB = os.getenv("ORACLE_CLIENT_LIB")  # ej. C:\oracle\instantclient_21_13

# ═══════════════════════════════════════════════════════════════════════════
#  2. TABLAS Y COLUMNAS
# ═══════════════════════════════════════════════════════════════════════════
CATEGORIAS = ["SK_CLIENTE"]                # las que identifican la serie
CATEGORIAS_TIPO = {"SK_CLIENTE": "int"}    # "int" si en Oracle es NUMBER, "str" si es VARCHAR2
COL_FECHA = "AT_FECHA"
COL_METRICA = "VENTA"

TABLA_SALIDA = "FORECAST_VENTA"            # podés calificarla: "ESQUEMA.FORECAST_VENTA"
MESES_HISTORIA = 60                        # cuánta historia leer de la fuente
MESES_PREVIOS = 36                         # cuánta salida anterior releer en la corrida mensual

# El DELETE nunca va más atrás que la fila más vieja que esta corrida va a
# insertar. Es la red de seguridad para que un "todo" no te borre historia que
# la corrida ya no tiene en memoria (todo lo anterior a MESES_PREVIOS).
# Ponelo en False sólo si querés un borrado real de toda la tabla.
PROTEGER_HISTORIA = True

# Predicado extra para acotar el DELETE y la relectura, por si la tabla la
# comparten varios procesos (ej. "PAIS = 'AR'"). Vacío = toda la tabla.
FILTRO_DESTINO = ""

SQL_FUENTE = f"""
    SELECT v.SK_CLIENTE,
           TRUNC(v.AT_FECHA, 'MM')  AS {COL_FECHA},
           SUM(v.VENTA)             AS {COL_METRICA}
      FROM VENTAS_HIST v
     WHERE v.AT_FECHA >= ADD_MONTHS(TRUNC(SYSDATE, 'MM'), -{MESES_HISTORIA})
       AND v.AT_FECHA <  TRUNC(SYSDATE, 'MM')
     GROUP BY v.SK_CLIENTE, TRUNC(v.AT_FECHA, 'MM')
"""

# ── Una columna por modelo ──────────────────────────────────────────────────
# {nombre del modelo en el motor: nombre de la columna en Oracle}
# Para sacar un modelo: quitalo de acá Y de `models` en build_config().
# Para agregar uno: sumalo acá, hacé el ALTER TABLE y agregalo a `models`.
# Un forecast externo (de otro sistema) se declara igual, con su nombre de columna.
COLUMNAS_MODELO = {
    "snaive":       "F_SNAIVE",
    "drift":        "F_DRIFT",
    "seasonal_ols": "F_SEASONAL_OLS",
    "croston_sba":  "F_CROSTON_SBA",
    "tsb":          "F_TSB",
    "ses":          "F_SES",
    "holt":         "F_HOLT",
    "ets":          "F_ETS",
    "theta":        "F_THETA",
    "stl_ets":      "F_STL_ETS",
    "sarima":       "F_SARIMA",
    "prophet":      "F_PROPHET",
    "combo_median": "F_COMBO",
}

# ── El resto de las columnas ────────────────────────────────────────────────
# Poné None en las que no quieras guardar.
COLUMNAS_FIJAS = {
    "real":           "Y_REAL",       # métrica real (NULL en las filas futuras)
    "ganador":        "YHAT",         # valor del modelo elegido
    "modelo_ganador": "BEST_MODEL",   # nombre del modelo elegido
    "score":          "BEST_SCORE",   # su error en la ventana de validación
    "futuro":         "IS_FUTURE",    # 0 = mes validado, 1 = proyección
    "corte":          "CUTOFF_DATE",  # último mes cerrado de la corrida
    "run":            "RUN_ID",       # id de la corrida, para auditar
    "actualizado":    "UPDATED_AT",   # se completa con SYSDATE
}

# ═══════════════════════════════════════════════════════════════════════════
#  3. PARÁMETROS DEL FORECAST
# ═══════════════════════════════════════════════════════════════════════════
def build_config(cutoff: str | None = None) -> ForecastConfig:
    """Config de la corrida. `cutoff` (YYYY-MM-DD) fuerza el último período cerrado;
    si es None se toma el mes anterior al actual."""
    return ForecastConfig(
        category_cols=CATEGORIAS,
        date_col=COL_FECHA,
        target_col=COL_METRICA,

        freq="MS",                   # mensual, primero de mes
        agg="sum",
        fill_value=0.0,              # meses sin movimiento -> 0
        cutoff=cutoff,               # None = último mes cerrado

        horizon=12,                  # meses a proyectar
        backtest_horizon=12,         # meses a validar hacia atrás
        refit_step=3,                # reentrena cada 3 meses en el backtest

        # Ventana MÓVIL con la que se elige el modelo: siempre los últimos 12
        # meses cerrados. Cada primero de mes entra el que cerró y sale el más
        # viejo — de la evaluación, no de la tabla: la fila queda guardada.
        score_window=12,

        # tienen que ser los mismos que están en COLUMNAS_MODELO
        models=[m for m in COLUMNAS_MODELO if m != "combo_median"],
        seasonal=True, season_length=12,

        metric="wmape", bias_weight=0.25, recency_half_life="auto",

        n_jobs=-1,                   # -1 todos los cores, -2 todos menos uno
        worker_memory_gb=0.9,
        memory_fraction=0.75,
        reserve_memory_gb=2.0,

        non_negative=True, round_to=2, verbose=1,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  De acá para abajo no hace falta tocar nada
# ═══════════════════════════════════════════════════════════════════════════
ARRAYSIZE = 50_000
BATCH_ROWS = 20_000
FECHA_DB = COL_FECHA.upper()

log = logging.getLogger("fc")


def configurar_logging(nombre: str) -> logging.Logger:
    """Log a stdout (lo que captura el orquestador) y, si está seteado, a FC_LOG_FILE."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    archivo = os.getenv("FC_LOG_FILE")
    if archivo:
        handlers.append(logging.FileHandler(archivo, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO, handlers=handlers, force=True,
        format=f"%(asctime)s {nombre} [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # el motor tiene su propio handler: lo sacamos para no duplicar líneas
    forecast_engine.LOGGER.handlers.clear()
    forecast_engine.LOGGER.propagate = True
    return logging.getLogger(nombre)


def _conectar(usuario: str, password: str, dsn: str, rol: str):
    if ORACLE_CLIENT_LIB and not getattr(_conectar, "_thick", False):
        oracledb.init_oracle_client(lib_dir=ORACLE_CLIENT_LIB)
        _conectar._thick = True
    if not password:
        raise RuntimeError(f"Falta la password del {rol} "
                           f"({'ORA_PASSWORD' if rol == 'origen' else 'ORA_DEST_PASSWORD'})")
    # con fetch_decimals en False, NUMBER(n,0) vuelve como int y el resto como float:
    # así una SK numérica conserva su valor exacto en las dos lecturas.
    oracledb.defaults.fetch_decimals = False
    conn = oracledb.connect(user=usuario, password=password, dsn=dsn)
    conn.autocommit = False
    return conn


def conexion_origen():
    """Base de donde se lee la métrica real."""
    return _conectar(ORA_USER, ORA_PASSWORD, ORA_DSN, "origen")


def conexion_destino():
    """Base donde vive la tabla de forecast. Puede ser otro usuario u otra base."""
    if (ORA_DEST_USER, ORA_DEST_DSN) == (ORA_USER, ORA_DSN):
        log.warning("no hay conexión de destino configurada (ORA_DEST_*): se usa la de origen")
    return _conectar(ORA_DEST_USER, ORA_DEST_PASSWORD, ORA_DEST_DSN, "destino")


# ─── esquema de la tabla ────────────────────────────────────────────────────
def _tipo_categoria(col: str) -> str:
    return "NUMBER" if CATEGORIAS_TIPO.get(col, "str") == "int" else "VARCHAR2(100)"


def columnas_requeridas(cfg: ForecastConfig) -> list[tuple[str, str]]:
    """(columna, tipo) que la tabla de salida tiene que tener, en orden."""
    cols = [(c.upper(), _tipo_categoria(c)) for c in cfg.category_cols]
    cols.append((FECHA_DB, "DATE"))
    if COLUMNAS_FIJAS.get("real"):
        cols.append((COLUMNAS_FIJAS["real"], "NUMBER"))
    for modelo in _modelos_esperados(cfg):
        cols.append((COLUMNAS_MODELO[modelo], "NUMBER"))
    for clave, tipo in (("ganador", "NUMBER"), ("modelo_ganador", "VARCHAR2(40)"),
                        ("score", "NUMBER"), ("futuro", "NUMBER(1)"),
                        ("corte", "DATE"), ("run", "VARCHAR2(32)"),
                        ("actualizado", "DATE")):
        if COLUMNAS_FIJAS.get(clave):
            cols.append((COLUMNAS_FIJAS[clave], tipo))
    return cols


def _modelos_esperados(cfg: ForecastConfig) -> list[str]:
    """Modelos que van a producir una columna: los del motor más los externos."""
    disponibles = available_models(cfg.models) + list(cfg.external_forecast_cols)
    faltan = [m for m in disponibles if m not in COLUMNAS_MODELO]
    if faltan:
        raise RuntimeError(
            "Estos modelos no tienen columna en COLUMNAS_MODELO: " + ", ".join(faltan) +
            ". Agregalos al diccionario (y la columna a la tabla) o sacalos de "
            "`models` en build_config(); si no, la corrida mensual no puede reutilizar "
            "su historia y rehace el backtest completo todos los meses.")
    return [m for m in COLUMNAS_MODELO if m in disponibles]


def ddl_sugerido(cfg: ForecastConfig) -> str:
    """CREATE TABLE que corresponde a la configuración actual. No lo ejecuta."""
    cols = columnas_requeridas(cfg)
    ancho = max(len(c) for c, _ in cols)
    cuerpo = ",\n".join(f"    {c.ljust(ancho)}  {t}" +
                        ("  DEFAULT SYSDATE" if c == COLUMNAS_FIJAS.get("actualizado") else "") +
                        ("  NOT NULL" if c in {x.upper() for x in cfg.category_cols} | {FECHA_DB} else "")
                        for c, t in cols)
    pk = ", ".join([c.upper() for c in cfg.category_cols] + [FECHA_DB])
    return (f"CREATE TABLE {TABLA_SALIDA} (\n{cuerpo},\n"
            f"    CONSTRAINT PK_{TABLA_SALIDA[:26]} PRIMARY KEY ({pk})\n);\n"
            f"CREATE INDEX IX_{TABLA_SALIDA[:24]}_F ON {TABLA_SALIDA} "
            f"({FECHA_DB}{', ' + COLUMNAS_FIJAS['futuro'] if COLUMNAS_FIJAS.get('futuro') else ''});")


def validar_tabla(conn, cfg: ForecastConfig) -> None:
    """Chequea contra el diccionario de datos que no falte ninguna columna.

    Se llama al principio de la corrida: es mucho mejor enterarse acá que
    después de media hora de cálculo con un ORA-00904.
    """
    owner, _, tabla = TABLA_SALIDA.rpartition(".")
    if owner:
        sql = ("SELECT COLUMN_NAME FROM ALL_TAB_COLUMNS "
               "WHERE OWNER = :o AND TABLE_NAME = :t")
        params = {"o": owner.upper(), "t": tabla.upper()}
    else:
        sql = "SELECT COLUMN_NAME FROM USER_TAB_COLUMNS WHERE TABLE_NAME = :t"
        params = {"t": tabla.upper()}
    existentes = {r[0].upper() for r in _fetch_df(conn, sql, params).itertuples(index=False)}
    if not existentes:
        raise RuntimeError(f"La tabla {TABLA_SALIDA} no existe o no es visible para {ORA_USER}.\n"
                           f"Creala con:\n\n{ddl_sugerido(cfg)}")
    faltan = [(c, t) for c, t in columnas_requeridas(cfg) if c.upper() not in existentes]
    if faltan:
        alter = ", ".join(f"{c} {t}" for c, t in faltan)
        raise RuntimeError(f"A {TABLA_SALIDA} le faltan columnas: "
                           f"{', '.join(c for c, _ in faltan)}\n\n"
                           f"ALTER TABLE {TABLA_SALIDA} ADD ({alter});")
    log.info("tabla %s validada: %d columnas requeridas presentes",
             TABLA_SALIDA, len(columnas_requeridas(cfg)))


# ─── lectura ────────────────────────────────────────────────────────────────
def _fetch_df(conn, sql: str, params: dict | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.arraysize = ARRAYSIZE
        cur.prefetchrows = ARRAYSIZE + 1
        cur.execute(sql, params or {})
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def _normalizar_categorias(df: pd.DataFrame, cfg: ForecastConfig) -> pd.DataFrame:
    """Deja las categorías con el mismo tipo vengan de donde vengan.

    Sin esto, una SK que la fuente devuelve como int y la tabla de salida como
    float no cruzaría, y la corrida mensual no encontraría el forecast anterior.
    """
    for c in cfg.category_cols:
        if CATEGORIAS_TIPO.get(c, "str") == "int":
            df[c] = pd.to_numeric(df[c], errors="raise").round().astype("int64")
        else:
            df[c] = df[c].astype(str)
    return df


def leer_fuente(conn, cfg: ForecastConfig) -> pd.DataFrame:
    """Trae la métrica real desde la tabla de origen."""
    t0 = time.time()
    df = _fetch_df(conn, SQL_FUENTE)
    if df.empty:
        return df
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = _normalizar_categorias(df, cfg)
    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce")
    log.info("fuente: %s filas, %s series, hasta %s (%.1fs)", f"{len(df):,}",
             f"{df.groupby(list(cfg.category_cols)).ngroups:,}",
             df[cfg.date_col].max().date(), time.time() - t0)
    return df


def _col_motor(cfg: ForecastConfig, modelo: str) -> str:
    """Nombre que usa el motor para la columna de un modelo."""
    if modelo in cfg.external_forecast_cols:
        return modelo
    return f"{cfg.model_col_prefix}{modelo}"


def _renombres(cfg: ForecastConfig) -> dict[str, str]:
    """{columna en Oracle: nombre que usa el motor}."""
    r = {COLUMNAS_MODELO[m]: _col_motor(cfg, m) for m in _modelos_esperados(cfg)}
    r.update({v: k for k, v in {
        cfg.actual_col: COLUMNAS_FIJAS.get("real"),
        cfg.forecast_col: COLUMNAS_FIJAS.get("ganador"),
        cfg.best_model_col: COLUMNAS_FIJAS.get("modelo_ganador"),
        cfg.best_score_col: COLUMNAS_FIJAS.get("score"),
        cfg.future_flag_col: COLUMNAS_FIJAS.get("futuro"),
        cfg.cutoff_col: COLUMNAS_FIJAS.get("corte"),
        "run_id": COLUMNAS_FIJAS.get("run"),
    }.items() if v})
    return r


def leer_tabla_destino(conn, cfg: ForecastConfig, meses: int | None = None,
                       filtro: str | None = None) -> pd.DataFrame:
    """Lee la tabla de salida tal cual está, con los nombres de Oracle. Para análisis.

    `meses` limita hacia atrás desde el mes en curso; `filtro` es un predicado
    SQL extra (ej. "SK_CLIENTE IN (900001, 900002)").
    """
    cols = ([c.upper() for c in cfg.category_cols] + [FECHA_DB]
            + [c for c in _renombres(cfg)]
            + ([COLUMNAS_FIJAS["actualizado"]] if COLUMNAS_FIJAS.get("actualizado") else []))
    where = [c for c in (
        f"{FECHA_DB} >= ADD_MONTHS(TRUNC(SYSDATE, 'MM'), -{int(meses)})" if meses else None,
        f"({FILTRO_DESTINO})" if FILTRO_DESTINO else None,
        f"({filtro})" if filtro else None) if c]
    sql = (f"SELECT {', '.join(cols)}\n  FROM {TABLA_SALIDA}"
           + (f"\n WHERE {' AND '.join(where)}" if where else ""))
    df = _fetch_df(conn, sql)
    if df.empty:
        return df
    df[FECHA_DB] = pd.to_datetime(df[FECHA_DB])
    if COLUMNAS_FIJAS.get("corte") in df.columns:
        df[COLUMNAS_FIJAS["corte"]] = pd.to_datetime(df[COLUMNAS_FIJAS["corte"]])
    for c in cfg.category_cols:
        if CATEGORIAS_TIPO.get(c, "str") == "int":
            df[c.upper()] = pd.to_numeric(df[c.upper()]).round().astype("int64")
    return df.sort_values([c.upper() for c in cfg.category_cols] + [FECHA_DB],
                          ignore_index=True)


def a_nombres_motor(df: pd.DataFrame, cfg: ForecastConfig) -> pd.DataFrame:
    """Pasa un frame leído de Oracle a los nombres que usan las funciones del motor."""
    out = df.rename(columns={**_renombres(cfg),
                             **{c.upper(): c for c in cfg.category_cols},
                             FECHA_DB: cfg.date_col})
    return _normalizar_categorias(out, cfg)


def leer_salida_anterior(conn, cfg: ForecastConfig) -> pd.DataFrame | None:
    """Trae la salida de la corrida anterior, ya con los nombres que espera run()."""
    renombres = _renombres(cfg)
    cols = [c.upper() for c in cfg.category_cols] + [FECHA_DB] + list(renombres)
    sql = (f"SELECT {', '.join(cols)}\n  FROM {TABLA_SALIDA}\n"
           f" WHERE {FECHA_DB} >= ADD_MONTHS(TRUNC(SYSDATE, 'MM'), -{MESES_PREVIOS})"
           + (f"\n   AND ({FILTRO_DESTINO})" if FILTRO_DESTINO else ""))
    df = _fetch_df(conn, sql)
    if df.empty:
        return None
    df = df.rename(columns=renombres)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = _normalizar_categorias(df, cfg)
    if cfg.cutoff_col in df.columns:
        df[cfg.cutoff_col] = pd.to_datetime(df[cfg.cutoff_col])
        corte = df[cfg.cutoff_col].max().date()
    else:
        corte = "?"
    log.info("salida anterior: %s filas, corte %s", f"{len(df):,}", corte)
    return df


# ─── escritura ──────────────────────────────────────────────────────────────
def _valores(df: pd.DataFrame, cfg: ForecastConfig) -> list[tuple[str, str, str]]:
    """[(columna Oracle, bind, tipo)] de todo lo que no es clave.

    Un modelo que no produjo columna en esta corrida se saltea, para no pisar
    con NULL lo que ya estaba guardado.
    """
    out: list[tuple[str, str, str]] = []
    if COLUMNAS_FIJAS.get("real"):
        out.append((COLUMNAS_FIJAS["real"], "real", "num"))
    for i, modelo in enumerate(_modelos_esperados(cfg)):
        if _col_motor(cfg, modelo) in df.columns:
            out.append((COLUMNAS_MODELO[modelo], f"m{i}", "num"))
        else:
            log.warning("el modelo '%s' no produjo valores: no se toca la columna %s",
                        modelo, COLUMNAS_MODELO[modelo])
    for clave, tipo in (("ganador", "num"), ("modelo_ganador", "str"), ("score", "num"),
                        ("futuro", "int"), ("corte", "fecha"), ("run", "str")):
        if COLUMNAS_FIJAS.get(clave):
            out.append((COLUMNAS_FIJAS[clave], clave, tipo))
    return out


def _insert_sql(cfg: ForecastConfig, valores: list[tuple[str, str, str]]) -> str:
    cats = [c.upper() for c in cfg.category_cols]
    cols = cats + [FECHA_DB] + [col for col, _, _ in valores]
    binds = [f":k{i}" for i in range(len(cats))] + [":fecha"] + [f":{b}" for _, b, _ in valores]
    if COLUMNAS_FIJAS.get("actualizado"):
        cols.append(COLUMNAS_FIJAS["actualizado"])
        binds.append("SYSDATE")
    return (f"INSERT INTO {TABLA_SALIDA} ({', '.join(cols)})\n"
            f"VALUES ({', '.join(binds)})")


def _delete_sql(desde: pd.Timestamp | None) -> str:
    condiciones = []
    if desde is not None:
        condiciones.append(f"{FECHA_DB} >= :desde")
    if FILTRO_DESTINO:
        condiciones.append(f"({FILTRO_DESTINO})")
    where = f"\n WHERE {' AND '.join(condiciones)}" if condiciones else ""
    return f"DELETE FROM {TABLA_SALIDA}{where}"


def resolver_desde(valor, cfg: ForecastConfig) -> pd.Timestamp | None:
    """Traduce `reescribir_desde` a una fecha. None = borrar todo.

    "auto" -> el corte | "todo"/None -> todo | un entero N -> N períodos antes
    del corte | una fecha -> esa fecha (llevada al inicio del período).
    """
    if valor is None:
        return None
    texto = str(valor).strip().lower()
    if texto in ("todo", "all", "full", ""):
        return None
    if texto == "auto":
        return cfg.resolved_cutoff()
    if texto.lstrip("-").isdigit():
        return shift_period(cfg.resolved_cutoff(), -abs(int(texto)), cfg.freq)
    return floor_to_freq([pd.Timestamp(valor)], cfg.freq).iloc[0]


_TIPO_BIND = {"num": oracledb.DB_TYPE_NUMBER, "int": oracledb.DB_TYPE_NUMBER,
              "str": oracledb.DB_TYPE_VARCHAR, "fecha": oracledb.DB_TYPE_DATE}


def _serie(df: pd.DataFrame, cfg: ForecastConfig, col: str, bind: str, tipo: str) -> list:
    """Columna del DataFrame -> lista de valores listos para bindear."""
    origen = {"real": cfg.actual_col, "ganador": cfg.forecast_col,
              "modelo_ganador": cfg.best_model_col, "score": cfg.best_score_col,
              "futuro": cfg.future_flag_col, "corte": cfg.cutoff_col, "run": "run_id"}
    if bind.startswith("m") and bind[1:].isdigit():
        nombre = next(_col_motor(cfg, m) for m, c in COLUMNAS_MODELO.items() if c == col)
    else:
        nombre = origen[bind]
    s = df[nombre]
    if tipo == "num":
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        return [None if not np.isfinite(x) else float(x) for x in v]
    if tipo == "int":
        v = pd.to_numeric(s, errors="coerce").fillna(0).to_numpy(dtype=int)
        return [int(x) for x in v]
    if tipo == "fecha":
        return list(pd.DatetimeIndex(s).to_pydatetime())
    return [None if pd.isna(x) else str(x) for x in s]


def _filas(df: pd.DataFrame, cfg: ForecastConfig,
           valores: list[tuple[str, str, str]]) -> list[dict]:
    columnas = {}
    for i, c in enumerate(cfg.category_cols):
        if CATEGORIAS_TIPO.get(c, "str") == "int":
            columnas[f"k{i}"] = [int(v) for v in pd.to_numeric(df[c]).to_numpy()]
        else:
            columnas[f"k{i}"] = df[c].astype(str).tolist()
    columnas["fecha"] = list(pd.DatetimeIndex(df[cfg.date_col]).to_pydatetime())
    for col, bind, tipo in valores:
        columnas[bind] = _serie(df, cfg, col, bind, tipo)
    binds = list(columnas)
    return [dict(zip(binds, fila)) for fila in zip(*(columnas[b] for b in binds))]


def guardar(conn, df_out: pd.DataFrame, cfg: ForecastConfig,
            reescribir_desde="auto") -> int:
    """Borra el rango que esta corrida reescribe e inserta las filas nuevas.

    Todo en una transacción: si el INSERT falla, el DELETE se deshace y la tabla
    queda como estaba. Ver `resolver_desde` para las opciones del rango.
    """
    if df_out.empty:
        log.warning("no hay nada para guardar")
        return 0
    desde = resolver_desde(reescribir_desde, cfg)
    escribir = df_out if desde is None else df_out[df_out[cfg.date_col] >= desde]
    if escribir.empty:
        log.warning("no hay filas desde %s: no se escribe nada", desde.date())
        return 0

    # Red de seguridad: no borrar lo que no se va a volver a escribir.
    minimo = pd.Timestamp(escribir[cfg.date_col].min())
    if PROTEGER_HISTORIA and (desde is None or desde < minimo):
        log.info("el borrado se acota a %s: es la fila más vieja que trae esta corrida "
                 "(PROTEGER_HISTORIA)", minimo.date())
        desde = minimo

    valores = _valores(escribir, cfg)
    filas = _filas(escribir, cfg, valores)
    sizes = {f"k{i}": (oracledb.DB_TYPE_NUMBER
                       if CATEGORIAS_TIPO.get(c, "str") == "int" else oracledb.DB_TYPE_VARCHAR)
             for i, c in enumerate(cfg.category_cols)}
    sizes["fecha"] = oracledb.DB_TYPE_DATE
    sizes.update({bind: _TIPO_BIND[tipo] for _, bind, tipo in valores})

    t0 = time.time()
    try:
        with conn.cursor() as cur:
            cur.execute(_delete_sql(desde),
                        {"desde": desde.to_pydatetime()} if desde is not None else {})
            log.info("borradas %s filas de %s (%s)", f"{cur.rowcount:,}", TABLA_SALIDA,
                     f"desde {desde.date()}" if desde is not None else "tabla completa")
        with conn.cursor() as cur:
            cur.setinputsizes(**sizes)
            sql = _insert_sql(cfg, valores)
            for i in range(0, len(filas), BATCH_ROWS):
                cur.executemany(sql, filas[i:i + BATCH_ROWS])
        conn.commit()
    except Exception:
        conn.rollback()
        log.error("falló la escritura: se deshizo el borrado, la tabla quedó como estaba")
        raise
    log.info("insertadas %s filas x %d columnas en %s (%.1fs)", f"{len(filas):,}",
             len(valores) + len(cfg.category_cols) + 1, TABLA_SALIDA, time.time() - t0)
    return len(filas)


# ─── control ────────────────────────────────────────────────────────────────
def resumen(fc, df_out: pd.DataFrame, cfg: ForecastConfig) -> None:
    """Deja en el log lo mínimo para auditar la corrida sin abrir la base."""
    hist = df_out[df_out[cfg.future_flag_col] == 0]
    real = hist[cfg.actual_col].sum()
    err = (hist[cfg.forecast_col] - hist[cfg.actual_col]).abs().sum()
    log.info("run_id=%s corte=%s | %s filas (%s validación, %s futuro) | %s series",
             fc.run_id, cfg.resolved_cutoff().date(), f"{len(df_out):,}", f"{len(hist):,}",
             f"{int((df_out[cfg.future_flag_col] == 1).sum()):,}",
             f"{df_out.groupby(list(cfg.category_cols)).ngroups:,}")
    if real:
        log.info("wMAPE global del modelo elegido: %.2f%%", 100 * err / real)

    # ventana móvil con la que se eligió el modelo
    corte = cfg.resolved_cutoff()
    ventana = cfg.resolved_score_window()
    desde = shift_period(corte, -(ventana - 1), cfg.freq)
    if fc.scores_ is not None and len(fc.scores_):
        pts = fc.scores_["n_points"]
        log.info("selección: ventana móvil %s..%s (%d períodos) | puntos evaluados por serie: "
                 "mín %d, mediana %d, máx %d", desde.date(), corte.date(), ventana,
                 int(pts.min()), int(pts.median()), int(pts.max()))
    else:
        log.info("selección: ventana móvil %s..%s (%d períodos), sin puntos evaluables",
                 desde.date(), corte.date(), ventana)
    ganadores = df_out[df_out[cfg.future_flag_col] == 1].groupby(cfg.best_model_col).size()
    log.info("modelos elegidos: %s", ganadores.sort_values(ascending=False).to_dict())
    if fc.failures_ is not None and len(fc.failures_):
        log.warning("modelos con fallos (series afectadas): %s",
                    fc.failures_["model"].value_counts().to_dict())
