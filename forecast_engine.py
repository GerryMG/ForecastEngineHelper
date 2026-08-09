"""
forecast_engine
===============

Motor de forecast para paneles (multi-serie) agnóstico de la frecuencia temporal.

Idea general
------------
Entrada  : tabla larga  ->  [categorías..., fecha, valor_real] (+ opcionalmente
           una columna por cada modelo con el forecast de corridas anteriores).
Proceso  : 1) saneo y grillado de la serie a la frecuencia elegida (rellenando
              huecos con 0 hasta el último período CERRADO),
           2) backtest de origen móvil ("rolling origin") sobre los últimos
              ``backtest_horizon`` períodos, refiteando cada ``refit_step``,
           3) selección por serie del modelo con menor error,
           4) forecast de ``horizon`` períodos hacia el futuro con todos los
              modelos, marcando el elegido.
Salida   : tabla larga -> [categorías..., fecha, y_real, f_<modelo>...,
           best_model, best_score, yhat, is_future, cutoff].

Corridas incrementales
----------------------
La salida anterior se puede volver a pasar como entrada (``previous``). En ese
caso NO se recalcula el backtest histórico: las filas que eran futuro y ya
tienen real se convierten en filas de validación (el forecast que quedó
guardado fue, por construcción, out-of-sample), se recalcula la selección
sumando el período nuevo y se emiten sólo los nuevos ``horizon`` períodos.

Ejemplo mínimo
--------------
>>> cfg = ForecastConfig(category_cols=["SK_CLIENTE"], date_col="AT_FECHA",
...                      target_col="VENTA", freq="MS", horizon=12,
...                      backtest_horizon=12)
>>> out = PanelForecaster(cfg).run(df)

Dependencias
------------
Obligatorias : pandas, numpy
Opcionales   : statsmodels (ETS/Theta/SARIMA/STL), prophet, joblib (paralelo),
               psutil (control de RAM), pmdarima (auto_arima).
Todo modelo cuya dependencia falte simplemente se desactiva y se informa.
"""

from __future__ import annotations

import logging
import math
import os
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ForecastConfig",
    "PanelForecaster",
    "MODEL_REGISTRY",
    "available_models",
    "evaluate_models",
    "detect_model_cols",
    "score_model",
    "wide_to_long",
    "long_to_wide",
]

LOGGER = logging.getLogger("forecast_engine")
if not LOGGER.handlers:  # configuración mínima para uso en notebook
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# Dependencias opcionales
# --------------------------------------------------------------------------- #
def _has_module(name: str) -> bool:
    """Detecta disponibilidad sin importar (prophet/statsmodels tardan segundos)."""
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # pragma: no cover - depende del entorno
        return False


def _try_import(name: str):
    try:
        mod = __import__(name)
        for part in name.split(".")[1:]:
            mod = getattr(mod, part)
        return mod
    except Exception:  # pragma: no cover
        return None


def _read_first(*paths) -> Optional[str]:
    for p in paths:
        try:
            with open(p) as fh:
                return fh.read().strip()
        except Exception:
            continue
    return None


def container_cpu_limit() -> Optional[float]:
    """Cuota de CPU del cgroup (contenedor). None si no hay límite o no es Linux."""
    v2 = _read_first("/sys/fs/cgroup/cpu.max")
    if v2:
        parts = v2.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                return max(1.0, float(parts[0]) / float(parts[1]))
            except Exception:
                pass
    quota = _read_first("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_first("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        if quota and period and int(quota) > 0:
            return max(1.0, int(quota) / int(period))
    except Exception:
        pass
    return None


def available_cpus() -> int:
    """CPUs realmente usables: afinidad del proceso acotada por el límite del pod."""
    if hasattr(os, "sched_getaffinity"):
        try:
            n = len(os.sched_getaffinity(0))
        except Exception:
            n = os.cpu_count() or 1
    else:
        n = os.cpu_count() or 1
    limit = container_cpu_limit()
    if limit:
        n = min(n, max(1, int(math.floor(limit))))
    return max(1, n)


def container_memory_limit_gb() -> Optional[float]:
    v = _read_first("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
    try:
        if v and v != "max":
            b = int(v)
            if 0 < b < (1 << 62):          # los "sin límite" vienen como un número enorme
                return b / (1024 ** 3)
    except Exception:
        pass
    return None


def available_memory_gb() -> Optional[float]:
    """RAM libre. En un contenedor manda el cgroup, no la del nodo."""
    limit = container_memory_limit_gb()
    if limit is not None:
        used = _read_first("/sys/fs/cgroup/memory.current",
                           "/sys/fs/cgroup/memory/memory.usage_in_bytes")
        try:
            return max(0.1, limit - int(used) / (1024 ** 3))
        except Exception:
            return limit
    if HAS_PSUTIL:
        return _psutil.virtual_memory().available / (1024 ** 3)
    return None


HAS_STATSMODELS = _has_module("statsmodels")
HAS_PROPHET = _has_module("prophet")
HAS_PMDARIMA = _has_module("pmdarima")
_joblib = _try_import("joblib")
HAS_JOBLIB = _joblib is not None
_psutil = _try_import("psutil")
HAS_PSUTIL = _psutil is not None


# --------------------------------------------------------------------------- #
# Utilidades de frecuencia
# --------------------------------------------------------------------------- #
_SEASON_BY_BASE = {
    "M": 12, "MS": 12, "ME": 12, "BM": 12, "BMS": 12, "SM": 24, "SMS": 24,
    "Q": 4, "QS": 4, "QE": 4, "BQ": 4, "BQS": 4,
    "Y": 1, "YS": 1, "YE": 1, "A": 1, "AS": 1,
    "W": 52, "D": 7, "B": 5, "H": 24, "h": 24, "T": 60, "min": 60,
}

_PERIOD_BASE = {
    "MS": "M", "ME": "M", "BM": "M", "BMS": "M", "BME": "M", "SMS": "M", "SM": "M",
    "QS": "Q", "QE": "Q", "BQS": "Q", "BQE": "Q", "BQ": "Q",
    "YS": "Y", "YE": "Y", "AS": "Y", "A": "Y", "BYS": "Y", "BYE": "Y",
    "ME_": "M",
}


def _offset(freq: str):
    return pd.tseries.frequencies.to_offset(freq)


def _period_alias(freq: str) -> str:
    """Alias válido para ``Period`` a partir de una frecuencia de ``date_range``."""
    name = _offset(freq).name  # 'MS', 'W-SUN', 'QS-JAN', 'D', ...
    for candidate in (name, _PERIOD_BASE.get(name.split("-")[0], name.split("-")[0])):
        try:
            pd.period_range("2020-01-01", periods=1, freq=candidate)
            return candidate
        except Exception:
            continue
    # último recurso: mensual
    return "M"


def floor_to_freq(values, freq: str) -> pd.Series:
    """Lleva cada timestamp a la etiqueta canónica de su período en ``freq``.

    Para ``freq='MS'`` toda fecha del mes cae al día 1; para ``'ME'`` al último
    día del mes; para ``'W-SUN'`` al domingo de cierre, etc.
    """
    s = pd.to_datetime(pd.Series(values).values)
    per = pd.PeriodIndex(s, freq=_period_alias(freq))
    ts = per.to_timestamp(how="start").normalize()
    off = _offset(freq)
    snapped = pd.DatetimeIndex([off.rollforward(t) for t in ts]).normalize()
    return pd.Series(snapped)


def shift_period(ts: pd.Timestamp, k: int, freq: str) -> pd.Timestamp:
    """Desplaza ``k`` períodos (puede ser negativo) sobre la grilla de ``freq``."""
    if k == 0:
        return pd.Timestamp(ts)
    per = pd.Period(pd.Timestamp(ts), freq=_period_alias(freq)) + k
    return _offset(freq).rollforward(per.to_timestamp(how="start").normalize())


def period_range(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> pd.DatetimeIndex:
    if pd.Timestamp(end) < pd.Timestamp(start):
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    return pd.date_range(start=start, end=end, freq=freq)


def default_season_length(freq: str) -> int:
    name = _offset(freq).name.split("-")[0]
    return int(_SEASON_BY_BASE.get(name, 1))


def last_closed_period(as_of, freq: str) -> pd.Timestamp:
    """Último período completamente cerrado respecto de ``as_of``.

    Con ``freq='MS'`` y ``as_of='2026-08-09'`` devuelve ``2026-07-01``.
    """
    current = floor_to_freq([pd.Timestamp(as_of)], freq).iloc[0]
    return shift_period(current, -1, freq)


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
@dataclass
class ForecastConfig:
    """Parámetros de la corrida. Todo tiene default razonable para mensual."""

    # --- esquema de la entrada -------------------------------------------- #
    category_cols: Sequence[str] = field(default_factory=list)
    date_col: str = "ds"
    target_col: str = "y"
    #: columnas de la entrada que ya son forecasts (de esta librería o de otro
    #: sistema) y que se quieren evaluar/competir. Vacío = autodetectar por prefijo.
    external_forecast_cols: Sequence[str] = field(default_factory=list)

    # --- grilla temporal --------------------------------------------------- #
    freq: str = "MS"                    #: cualquier alias de pandas: MS, ME, W-SUN, D, QS...
    agg: str = "sum"                    #: cómo colapsar duplicados dentro del período
    fill_value: float = 0.0             #: relleno de períodos faltantes
    start_policy: str = "series"        #: 'series' (desde el 1er dato de cada serie) | 'global'
    as_of: Optional[Any] = None         #: fecha de referencia ("hoy"). None = hoy
    cutoff: Optional[Any] = None        #: último período CERRADO; si se indica manda sobre as_of
    include_open_period: bool = False   #: True = incluir el período en curso como cerrado
    drop_leading_zeros: bool = True     #: recortar ceros previos al primer movimiento real

    # --- horizontes -------------------------------------------------------- #
    horizon: int = 12                   #: n períodos a futuro
    backtest_horizon: int = 12          #: n períodos hacia atrás a validar
    refit_step: int = 1                 #: cada cuántos períodos se re-entrena en el backtest
    score_window: Optional[int] = None  #: períodos usados para elegir modelo (None = backtest_horizon)

    # --- modelos ----------------------------------------------------------- #
    models: Optional[Sequence[str]] = None      #: None = todos los disponibles
    model_params: Mapping[str, dict] = field(default_factory=dict)
    season_length: Optional[int] = None         #: None = inferido de freq (12 en mensual)
    seasonal: bool = True                       #: estacionalidad anual activada por defecto
    min_history: int = 4                        #: mínimo de puntos para modelos "grandes"
    min_seasonal_cycles: float = 2.0            #: ciclos completos exigidos para modelar estacionalidad
    combo_exclude: Sequence[str] = field(default_factory=lambda: ["naive", "drift"])

    # --- selección --------------------------------------------------------- #
    metric: str = "wmape"               #: wmape | mae | rmse | smape | mase
    bias_weight: float = 0.25           #: castigo al sesgo acumulado (0 = sin castigo)
    recency_half_life: Any = "auto"     #: 'auto' (=season_length) | None | nº de períodos
    min_eval_points: int = 3            #: mínimo de puntos evaluados para confiar en un modelo
    min_coverage: float = 0.6           #: fracción del window que el modelo debe cubrir
    fallback_model: str = "snaive"

    # --- recursos ---------------------------------------------------------- #
    n_jobs: int = -1                    #: -1 = todos los cores, -2 = todos menos uno, N = N
    max_workers: Optional[int] = None   #: tope duro de procesos
    memory_fraction: float = 0.75       #: fracción de la RAM libre que se puede usar
    reserve_memory_gb: float = 2.0      #: RAM libre que se deja intacta
    worker_memory_gb: float = 0.9       #: RAM estimada por worker (prophet ~0.7-1 GB)
    batch_size: Any = "auto"            #: batch_size de joblib
    inner_threads: int = 1              #: hilos BLAS por worker (1 evita sobre-suscripción)

    # --- salida ------------------------------------------------------------ #
    model_col_prefix: str = "f_"
    actual_col: str = "y_real"
    forecast_col: str = "yhat"
    best_model_col: str = "best_model"
    best_score_col: str = "best_score"
    future_flag_col: str = "is_future"
    cutoff_col: str = "cutoff"
    non_negative: bool = True
    round_to: Optional[int] = None
    #: filas históricas a devolver. None = todas las que tengan forecast (la
    #: ventana de backtest, más lo que venga acumulado de corridas previas).
    keep_history_periods: Optional[int] = None
    include_full_history: bool = False  #: True = devolver también el histórico sin forecast
    verbose: int = 1

    # --- derivados --------------------------------------------------------- #
    def resolved_season_length(self) -> int:
        if not self.seasonal:
            return 1
        return int(self.season_length or default_season_length(self.freq))

    def resolved_cutoff(self) -> pd.Timestamp:
        if self.cutoff is not None:
            return floor_to_freq([pd.Timestamp(self.cutoff)], self.freq).iloc[0]
        ref = pd.Timestamp(self.as_of) if self.as_of is not None else pd.Timestamp.today().normalize()
        if self.include_open_period:
            return floor_to_freq([ref], self.freq).iloc[0]
        return last_closed_period(ref, self.freq)

    def resolved_score_window(self) -> int:
        return int(self.score_window or self.backtest_horizon)

    def resolved_half_life(self) -> Optional[float]:
        if self.recency_half_life is None:
            return None
        if isinstance(self.recency_half_life, str):
            if self.recency_half_life.lower() == "auto":
                return float(max(2, self.resolved_season_length()))
            return None
        return float(self.recency_half_life)

    def validate(self) -> None:
        if not self.category_cols:
            raise ValueError("category_cols no puede estar vacío (usá una constante si tenés una sola serie)")
        if self.horizon < 1:
            raise ValueError("horizon debe ser >= 1")
        if self.backtest_horizon < 1:
            raise ValueError("backtest_horizon debe ser >= 1")
        if self.refit_step < 1:
            raise ValueError("refit_step debe ser >= 1")
        if self.agg not in ("sum", "mean", "max", "min", "median", "last", "first"):
            raise ValueError(f"agg no soportado: {self.agg}")
        _offset(self.freq)  # levanta si la frecuencia es inválida


# --------------------------------------------------------------------------- #
# Contexto que viaja a los workers
# --------------------------------------------------------------------------- #
@dataclass
class ModelContext:
    freq: str
    season_length: int
    seasonal: bool
    min_history: int
    min_seasonal_cycles: float
    non_negative: bool
    params: Mapping[str, dict]
    random_state: int = 0

    def can_seasonal(self, n: int) -> bool:
        m = self.season_length
        return self.seasonal and m > 1 and n >= int(math.ceil(self.min_seasonal_cycles * m))


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def _rep(value: float, h: int) -> np.ndarray:
    return np.full(h, float(value), dtype=float)


def m_naive(y: np.ndarray, idx, h: int, ctx: ModelContext) -> np.ndarray:
    return _rep(y[-1], h)


def m_snaive(y: np.ndarray, idx, h: int, ctx: ModelContext) -> np.ndarray:
    m = ctx.season_length
    if m <= 1 or len(y) < m:
        return _rep(y[-1], h)
    season = y[-m:]
    return np.array([season[i % m] for i in range(h)], dtype=float)


def m_drift(y: np.ndarray, idx, h: int, ctx: ModelContext) -> np.ndarray:
    n = len(y)
    if n < 2:
        return _rep(y[-1], h)
    slope = (y[-1] - y[0]) / (n - 1)
    return y[-1] + slope * np.arange(1, h + 1, dtype=float)


def m_mean(y: np.ndarray, idx, h: int, ctx: ModelContext) -> np.ndarray:
    win = max(1, min(len(y), max(3, 2 * ctx.season_length)))
    return _rep(float(np.mean(y[-win:])), h)


def m_seasonal_ols(y: np.ndarray, idx, h: int, ctx: ModelContext) -> np.ndarray:
    """Tendencia lineal + dummies estacionales, resuelto por mínimos cuadrados
    con una pizca de regularización (no requiere statsmodels ni sklearn)."""
    n = len(y)
    m = ctx.season_length if ctx.can_seasonal(n) else 1
    t = np.arange(n, dtype=float)
    cols = [np.ones(n), t]
    if m > 1:
        phase = np.arange(n) % m
        for k in range(1, m):
            cols.append((phase == k).astype(float))
    X = np.column_stack(cols)
    lam = 1e-6 * max(1.0, float(np.mean(np.abs(y))))
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    beta = np.linalg.solve(XtX, X.T @ y)
    tf = np.arange(n, n + h, dtype=float)
    fcols = [np.ones(h), tf]
    if m > 1:
        fphase = np.arange(n, n + h) % m
        for k in range(1, m):
            fcols.append((fphase == k).astype(float))
    return np.column_stack(fcols) @ beta


def _croston_core(y: np.ndarray, h: int, alpha: float, variant: str) -> np.ndarray:
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return np.zeros(h)
    if variant == "tsb":
        z = float(y[nz[0]])
        p = 1.0 / max(1.0, float(nz[0] + 1))
        beta = alpha
        for t in range(len(y)):
            if y[t] > 0:
                z += alpha * (y[t] - z)
                p += beta * (1.0 - p)
            else:
                p += beta * (0.0 - p)
        return _rep(z * p, h)
    # Croston clásico / SBA
    z = float(y[nz[0]])
    q = float(nz[0] + 1)
    last = nz[0]
    for t in nz[1:]:
        z += alpha * (y[t] - z)
        q += alpha * ((t - last) - q)
        last = t
    rate = z / max(q, 1e-9)
    if variant == "sba":
        rate *= 1.0 - alpha / 2.0
    return _rep(rate, h)


def _croston_fit(y: np.ndarray, h: int, variant: str) -> np.ndarray:
    best, best_sse = None, np.inf
    for alpha in (0.05, 0.1, 0.2, 0.3):
        # SSE in-sample de un paso: comparamos la tasa estimada con el promedio móvil
        fitted = np.array([_croston_core(y[: t + 1], 1, alpha, variant)[0] for t in range(max(1, len(y) - 24), len(y))])
        actual = y[-len(fitted):]
        sse = float(np.sum((actual - fitted) ** 2))
        if sse < best_sse:
            best_sse, best = sse, alpha
    return _croston_core(y, h, best or 0.1, variant)


def m_croston(y, idx, h, ctx):
    return _croston_fit(np.asarray(y, dtype=float), h, "classic")


def m_croston_sba(y, idx, h, ctx):
    return _croston_fit(np.asarray(y, dtype=float), h, "sba")


def m_tsb(y, idx, h, ctx):
    return _croston_fit(np.asarray(y, dtype=float), h, "tsb")


def _as_series(y: np.ndarray, idx, freq: str) -> "pd.Series":
    s = pd.Series(np.asarray(y, dtype=float), index=pd.DatetimeIndex(idx))
    try:
        s.index.freq = _offset(freq)
    except Exception:
        pass
    return s


def m_ses(y, idx, h, ctx):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    fit = SimpleExpSmoothing(_as_series(y, idx, ctx.freq), initialization_method="estimated").fit()
    return np.asarray(fit.forecast(h), dtype=float)


def m_holt(y, idx, h, ctx):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    fit = ExponentialSmoothing(
        _as_series(y, idx, ctx.freq), trend="add", damped_trend=True,
        seasonal=None, initialization_method="estimated",
    ).fit()
    return np.asarray(fit.forecast(h), dtype=float)


def m_ets(y, idx, h, ctx):
    """Holt-Winters con selección automática de la configuración por AICc."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    s = _as_series(y, idx, ctx.freq)
    n = len(s)
    m = ctx.season_length
    seasonal_ok = ctx.can_seasonal(n)
    positive = bool(np.all(np.asarray(y) > 0))

    candidates: List[dict] = [
        dict(trend=None, seasonal=None),
        dict(trend="add", damped_trend=True, seasonal=None),
        dict(trend="add", damped_trend=False, seasonal=None),
    ]
    if seasonal_ok:
        candidates += [
            dict(trend=None, seasonal="add", seasonal_periods=m),
            dict(trend="add", damped_trend=True, seasonal="add", seasonal_periods=m),
        ]
        if positive:
            candidates.append(dict(trend="add", damped_trend=True, seasonal="mul", seasonal_periods=m))

    best, best_ic = None, np.inf
    for kw in candidates:
        try:
            fit = ExponentialSmoothing(s, initialization_method="estimated", **kw).fit()
            k = len(fit.params_formatted) if hasattr(fit, "params_formatted") else 3
            ic = float(fit.aic) + (2 * k * (k + 1)) / max(1.0, n - k - 1)
            if np.isfinite(ic) and ic < best_ic:
                best_ic, best = ic, fit
        except Exception:
            continue
    if best is None:
        raise RuntimeError("ETS no convergió en ninguna configuración")
    return np.asarray(best.forecast(h), dtype=float)


def m_theta(y, idx, h, ctx):
    from statsmodels.tsa.forecasting.theta import ThetaModel
    n = len(y)
    period = ctx.season_length if ctx.can_seasonal(n) else 1
    model = ThetaModel(_as_series(y, idx, ctx.freq), period=max(1, period),
                       deseasonalize=period > 1, method="auto")
    return np.asarray(model.fit().forecast(h), dtype=float)


def m_stl_ets(y, idx, h, ctx):
    from statsmodels.tsa.forecasting.stl import STLForecast
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    n = len(y)
    m = ctx.season_length
    if not ctx.can_seasonal(n) or m < 2:
        raise RuntimeError("STL requiere al menos 2 ciclos estacionales")
    stlf = STLForecast(
        _as_series(y, idx, ctx.freq),
        ExponentialSmoothing,
        model_kwargs=dict(trend="add", damped_trend=True, seasonal=None,
                          initialization_method="estimated"),
        period=m, robust=True,
    ).fit()
    return np.asarray(stlf.forecast(h), dtype=float)


_SARIMA_GRID = [
    ((1, 1, 1), True), ((0, 1, 1), True), ((1, 0, 0), True),
    ((2, 1, 1), True), ((1, 1, 1), False), ((0, 1, 1), False),
]


def m_sarima(y, idx, h, ctx):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    s = _as_series(y, idx, ctx.freq)
    n = len(s)
    m = ctx.season_length
    seasonal_ok = ctx.can_seasonal(n)
    best, best_ic = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order, use_seasonal in _SARIMA_GRID:
            seasonal_order = (0, 1, 1, m) if (use_seasonal and seasonal_ok) else (0, 0, 0, 0)
            if use_seasonal and not seasonal_ok:
                continue
            try:
                fit = SARIMAX(
                    s, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                    trend=None,
                ).fit(disp=False, maxiter=80)
                k = int(np.sum(np.isfinite(fit.params)))
                ic = float(fit.aic) + (2 * k * (k + 1)) / max(1.0, n - k - 1)
                if np.isfinite(ic) and ic < best_ic:
                    best_ic, best = ic, fit
            except Exception:
                continue
    if best is None:
        raise RuntimeError("SARIMA no convergió")
    return np.asarray(best.forecast(h), dtype=float)


def m_auto_arima(y, idx, h, ctx):
    import pmdarima as pm
    n = len(y)
    m = ctx.season_length if ctx.can_seasonal(n) else 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pm.auto_arima(
            np.asarray(y, dtype=float), seasonal=m > 1, m=max(1, m),
            suppress_warnings=True, error_action="ignore", stepwise=True,
            max_p=3, max_q=3, max_P=2, max_Q=2, information_criterion="aicc",
        )
    return np.asarray(model.predict(n_periods=h), dtype=float)


def m_prophet(y, idx, h, ctx):
    from prophet import Prophet
    # cmdstanpy fija el nivel de su logger al importarse: hay que bajarlo después.
    for _name in ("cmdstanpy", "prophet", "prophet.models"):
        logging.getLogger(_name).setLevel(logging.CRITICAL)
    p = dict(ctx.params.get("prophet", {}))
    n = len(y)
    seasonal_ok = ctx.can_seasonal(n)
    kwargs = dict(
        yearly_seasonality=p.pop("yearly_seasonality", seasonal_ok),
        weekly_seasonality=p.pop("weekly_seasonality", False),
        daily_seasonality=p.pop("daily_seasonality", False),
        seasonality_mode=p.pop("seasonality_mode", "additive"),
        uncertainty_samples=p.pop("uncertainty_samples", 0),  # más rápido y menos RAM
    )
    kwargs.update(p)
    hist = pd.DataFrame({"ds": pd.DatetimeIndex(idx), "y": np.asarray(y, dtype=float)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(**kwargs)
        model.fit(hist)
        future_idx = [shift_period(hist["ds"].iloc[-1], k, ctx.freq) for k in range(1, h + 1)]
        pred = model.predict(pd.DataFrame({"ds": pd.DatetimeIndex(future_idx)}))
    return np.asarray(pred["yhat"].to_numpy(), dtype=float)


@dataclass
class ModelSpec:
    name: str
    fn: Callable[..., np.ndarray]
    requires: Tuple[str, ...] = ()
    min_obs: int = 1
    heavy: bool = False
    description: str = ""


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    s.name: s
    for s in [
        ModelSpec("naive", m_naive, min_obs=1, description="Último valor observado"),
        ModelSpec("snaive", m_snaive, min_obs=1, description="Mismo período del ciclo anterior"),
        ModelSpec("drift", m_drift, min_obs=2, description="Última obs. + pendiente media"),
        ModelSpec("mean", m_mean, min_obs=1, description="Media de los últimos 2 ciclos"),
        ModelSpec("seasonal_ols", m_seasonal_ols, min_obs=4, description="Tendencia + dummies estacionales"),
        ModelSpec("croston", m_croston, min_obs=2, description="Croston clásico (demanda intermitente)"),
        ModelSpec("croston_sba", m_croston_sba, min_obs=2, description="Croston-SBA (corrección de sesgo)"),
        ModelSpec("tsb", m_tsb, min_obs=2, description="Teunter-Syntetos-Babai (intermitente)"),
        ModelSpec("ses", m_ses, requires=("statsmodels",), min_obs=3, description="Suavizado exponencial simple"),
        ModelSpec("holt", m_holt, requires=("statsmodels",), min_obs=4, description="Holt con tendencia amortiguada"),
        ModelSpec("ets", m_ets, requires=("statsmodels",), min_obs=5, description="Holt-Winters auto (AICc)"),
        ModelSpec("theta", m_theta, requires=("statsmodels",), min_obs=5, description="Theta (benchmark M3)"),
        ModelSpec("stl_ets", m_stl_ets, requires=("statsmodels",), min_obs=8, description="STL + ETS"),
        ModelSpec("sarima", m_sarima, requires=("statsmodels",), min_obs=8, heavy=True, description="SARIMA con grilla por AICc"),
        ModelSpec("auto_arima", m_auto_arima, requires=("pmdarima",), min_obs=8, heavy=True, description="auto.arima (pmdarima)"),
        ModelSpec("prophet", m_prophet, requires=("prophet",), min_obs=6, heavy=True, description="Prophet con estacionalidad anual"),
    ]
}

COMBO_MODEL = "combo_median"

_DEP_FLAGS = {"statsmodels": HAS_STATSMODELS, "prophet": HAS_PROPHET, "pmdarima": HAS_PMDARIMA}

#: Modelos que se usan si no se especifica ``models``. auto_arima queda fuera
#: por redundar con sarima; se puede sumar explícitamente.
DEFAULT_MODELS = [
    "snaive", "drift", "seasonal_ols", "croston_sba", "tsb",
    "ses", "holt", "ets", "theta", "stl_ets", "sarima", "prophet",
]


def available_models(candidates: Optional[Sequence[str]] = None) -> List[str]:
    """Modelos utilizables en este entorno (filtra los que no tienen dependencia)."""
    names = list(candidates) if candidates else list(DEFAULT_MODELS)
    out = []
    for name in names:
        if name == COMBO_MODEL:
            out.append(name)
            continue
        spec = MODEL_REGISTRY.get(name)
        if spec is None:
            LOGGER.warning("Modelo desconocido, se ignora: %s", name)
            continue
        missing = [d for d in spec.requires if not _DEP_FLAGS.get(d, False)]
        if missing:
            LOGGER.warning("Modelo '%s' desactivado (falta %s)", name, ", ".join(missing))
            continue
        out.append(name)
    if COMBO_MODEL not in out:
        out.append(COMBO_MODEL)
    return out


# --------------------------------------------------------------------------- #
# Worker: resuelve TODOS los segmentos de UNA serie
# --------------------------------------------------------------------------- #
@dataclass
class SeriesTask:
    key: Tuple
    dates: np.ndarray          # datetime64[ns], grilla completa hasta cutoff
    values: np.ndarray         # float64
    segments: Tuple[Tuple[np.datetime64, int, int], ...]  # (origen, h, offset en pred_dates)
    n_pred: int                # total de puntos predichos
    models: Tuple[str, ...]
    ctx: ModelContext


def _init_worker(inner_threads: int = 1) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(inner_threads))
    for name in ("prophet", "cmdstanpy", "fbprophet"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore")


def run_series_task(task: SeriesTask) -> Dict[str, Any]:
    """Ejecuta todos los modelos sobre todos los segmentos de una serie.

    Devuelve ``{'key', 'preds': {modelo: array(n_pred)}, 'failures': {...}}``.
    """
    _init_worker(1)
    ctx = task.ctx
    dates = pd.DatetimeIndex(task.dates)
    values = np.asarray(task.values, dtype=float)

    preds: Dict[str, np.ndarray] = {
        name: np.full(task.n_pred, np.nan) for name in list(task.models) + [COMBO_MODEL]
    }
    failures: Dict[str, str] = {}

    for origin, h, off in task.segments:
        mask = dates <= pd.Timestamp(origin)
        y_tr = values[mask]
        idx_tr = dates[mask]
        n = len(y_tr)
        if n == 0:
            continue
        seg_preds: Dict[str, np.ndarray] = {}
        for name in task.models:
            spec = MODEL_REGISTRY[name]
            if n < max(spec.min_obs, 1) or (n < ctx.min_history and spec.heavy):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = np.asarray(spec.fn(y_tr, idx_tr, h, ctx), dtype=float).ravel()
                if out.size != h or not np.all(np.isfinite(out)):
                    raise ValueError(f"salida inválida (size={out.size}, finita={np.all(np.isfinite(out))})")
            except Exception as exc:  # el modelo falla -> queda NaN, no rompe la corrida
                failures.setdefault(name, f"{type(exc).__name__}: {exc}")
                continue
            if ctx.non_negative:
                out = np.clip(out, 0.0, None)
            seg_preds[name] = out
            preds[name][off:off + h] = out

        pool = [v for k, v in seg_preds.items() if k not in ctx.params.get("_combo_exclude", ())]
        if len(pool) >= 2:
            preds[COMBO_MODEL][off:off + h] = np.median(np.vstack(pool), axis=0)
        elif len(pool) == 1:
            preds[COMBO_MODEL][off:off + h] = pool[0]

    return {"key": task.key, "preds": preds, "failures": failures}


# --------------------------------------------------------------------------- #
# Métricas y selección
# --------------------------------------------------------------------------- #
def _weights(ages: np.ndarray, half_life: Optional[float]) -> np.ndarray:
    if not half_life:
        return np.ones_like(ages, dtype=float)
    return np.power(0.5, ages.astype(float) / float(half_life))


def score_model(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray,
                metric: str, bias_weight: float, scale: float) -> Dict[str, float]:
    """Error ponderado de un modelo sobre la ventana de evaluación.

    ``score`` = error normalizado + ``bias_weight`` * |sesgo acumulado normalizado|.
    Normalizar por el nivel de la serie hace comparables series de distinta
    magnitud; el término de sesgo penaliza al modelo que "no se pega" a la
    tendencia aunque acierte en promedio absoluto.
    """
    err = y_pred - y_true
    w = weights
    sw = float(np.sum(w)) or 1.0
    denom = float(np.sum(w * np.abs(y_true)))
    level = denom / sw if denom > 0 else max(scale, 1e-9)

    mae = float(np.sum(w * np.abs(err)) / sw)
    rmse = float(math.sqrt(np.sum(w * err ** 2) / sw))
    bias = float(np.sum(w * err) / sw)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape_terms = 2.0 * np.abs(err) / np.clip(np.abs(y_true) + np.abs(y_pred), 1e-9, None)
    smape = float(np.sum(w * smape_terms) / sw)
    wmape = mae / max(level, 1e-9)

    base = {"wmape": wmape, "mae": mae / max(level, 1e-9), "rmse": rmse / max(level, 1e-9),
            "smape": smape, "mase": mae / max(scale, 1e-9)}
    core = base.get(metric)
    if core is None:
        raise ValueError(f"Métrica no soportada: {metric}")
    total = core + bias_weight * abs(bias) / max(level, 1e-9)
    return {"metric": core, "score": float(total), "mae": mae, "rmse": rmse,
            "wmape": wmape, "smape": smape, "bias": bias, "n_points": int(len(y_true))}


def detect_model_cols(df: pd.DataFrame, cfg: "ForecastConfig") -> Dict[str, str]:
    """Mapea columna -> nombre de modelo para las columnas de forecast de un frame."""
    reserved = set(cfg.category_cols) | {
        cfg.date_col, cfg.target_col, cfg.actual_col, cfg.forecast_col,
        cfg.best_model_col, cfg.best_score_col, cfg.future_flag_col, cfg.cutoff_col,
        "run_id",
    }
    out: Dict[str, str] = {}
    for col in df.columns:
        if col in reserved:
            continue
        if col in cfg.external_forecast_cols:
            out[col] = col
        elif cfg.model_col_prefix and str(col).startswith(cfg.model_col_prefix):
            out[col] = str(col)[len(cfg.model_col_prefix):]
    return out


def evaluate_models(history: pd.DataFrame, cfg: "ForecastConfig",
                    model_cols: Optional[Mapping[str, str]] = None,
                    cutoff: Optional[Any] = None,
                    scales: Optional[pd.Series] = None) -> pd.DataFrame:
    """Error de cada modelo, por serie, sobre la ventana móvil de selección.

    Es la función que usa el motor para elegir el ganador; se expone para poder
    reproducir la decisión desde afuera (por ejemplo, sobre la tabla ya escrita)
    y obtener exactamente los mismos números.

    Devuelve una fila por (serie, modelo) con: coverage, metric, score, mae,
    rmse, wmape, smape, bias y n_points.
    """
    cutoff = pd.Timestamp(cutoff) if cutoff is not None else cfg.resolved_cutoff()
    model_cols = dict(model_cols) if model_cols else detect_model_cols(history, cfg)
    window_start = shift_period(cutoff, -(cfg.resolved_score_window() - 1), cfg.freq)
    half_life = cfg.resolved_half_life()
    m = cfg.resolved_season_length()
    cats = list(cfg.category_cols)

    evald = history[(history[cfg.date_col] >= window_start) &
                    (history[cfg.date_col] <= cutoff) &
                    history[cfg.actual_col].notna()]
    rows: List[dict] = []
    for key, g in evald.groupby(cats, observed=True, dropna=False, sort=False):
        key_t = key if isinstance(key, tuple) else (key,)
        g = g.sort_values(cfg.date_col)
        y_true = g[cfg.actual_col].to_numpy(dtype=float)
        ages = np.arange(len(g) - 1, -1, -1, dtype=float)
        w_all = _weights(ages, half_life)
        if scales is not None:
            try:
                scale = float(scales.loc[key_t if len(key_t) > 1 else key_t[0]])
            except Exception:
                scale = _naive_scale(y_true, m)
        else:
            scale = _naive_scale(y_true, m)
        for col, model in model_cols.items():
            if col not in g.columns:
                continue
            pred = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(pred)
            if ok.sum() < max(cfg.min_eval_points, 1) or ok.mean() < cfg.min_coverage:
                continue
            stats = score_model(y_true[ok], pred[ok], w_all[ok], cfg.metric,
                                cfg.bias_weight, scale)
            rows.append({**dict(zip(cats, key_t)), "model": model,
                         "coverage": float(ok.mean()), **stats})
    return pd.DataFrame(rows)


def _naive_scale(y: np.ndarray, m: int) -> float:
    """Escala tipo MASE: error medio del naive estacional in-sample."""
    if len(y) > m and m >= 1:
        d = np.abs(y[m:] - y[:-m])
        if d.size and np.mean(d) > 0:
            return float(np.mean(d))
    lvl = float(np.mean(np.abs(y))) if len(y) else 1.0
    return max(lvl, 1e-9)


# --------------------------------------------------------------------------- #
# Orquestador
# --------------------------------------------------------------------------- #
class PanelForecaster:
    """Ejecuta saneo -> backtest -> selección -> forecast sobre un panel.

    Atributos poblados tras ``run()``:
      ``scores_``       DataFrame de errores por serie y modelo.
      ``selection_``    DataFrame con el modelo elegido por serie.
      ``failures_``     DataFrame con los modelos que fallaron y por qué.
      ``panel_``        DataFrame saneado (grilla completa de reales).
      ``run_id``        identificador de la corrida.
    """

    def __init__(self, config: ForecastConfig):
        config.validate()
        self.cfg = config
        self.models = available_models(config.models)
        self.scores_: Optional[pd.DataFrame] = None
        self.selection_: Optional[pd.DataFrame] = None
        self.failures_: Optional[pd.DataFrame] = None
        self.panel_: Optional[pd.DataFrame] = None
        self.run_id: str = ""
        if config.verbose:
            LOGGER.setLevel(logging.INFO if config.verbose >= 1 else logging.WARNING)

    # -- helpers de nombres ------------------------------------------------- #
    def model_col(self, name: str) -> str:
        return f"{self.cfg.model_col_prefix}{name}"

    def _detect_model_cols(self, df: pd.DataFrame) -> Dict[str, str]:
        """Mapea columna -> nombre de modelo para las columnas de forecast previas."""
        return detect_model_cols(df, self.cfg)

    def _engine_cols(self, model_cols: Mapping[str, str]) -> List[str]:
        """Columnas que produce esta librería (excluye los forecasts externos)."""
        ext = set(self.cfg.external_forecast_cols)
        return [c for c in model_cols if c not in ext]

    # -- 1. saneo ----------------------------------------------------------- #
    def sanitize(self, df: pd.DataFrame, cutoff: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Agrega al período, completa la grilla con ``fill_value`` hasta ``cutoff``
        y descarta lo posterior al último período cerrado."""
        cfg = self.cfg
        cutoff = pd.Timestamp(cutoff) if cutoff is not None else cfg.resolved_cutoff()

        missing = [c for c in list(cfg.category_cols) + [cfg.date_col] if c not in df.columns]
        if missing:
            raise KeyError(f"Faltan columnas en la entrada: {missing}")
        if cfg.target_col not in df.columns:
            raise KeyError(f"Falta la columna de la métrica real: {cfg.target_col}")

        work = df.loc[:, list(cfg.category_cols) + [cfg.date_col, cfg.target_col]].copy()
        work[cfg.date_col] = floor_to_freq(work[cfg.date_col], cfg.freq).values
        work[cfg.target_col] = pd.to_numeric(work[cfg.target_col], errors="coerce")
        work = work.dropna(subset=[cfg.date_col])
        work = work[work[cfg.date_col] <= cutoff]
        work[cfg.target_col] = work[cfg.target_col].fillna(0.0)

        grouped = (work.groupby(list(cfg.category_cols) + [cfg.date_col], observed=True, dropna=False)[cfg.target_col]
                   .agg(cfg.agg).reset_index())

        # grilla completa por serie
        frames = []
        global_start = grouped[cfg.date_col].min()
        for key, g in grouped.groupby(list(cfg.category_cols), observed=True, dropna=False, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            g = g.sort_values(cfg.date_col)
            if cfg.drop_leading_zeros:
                nz = g.index[g[cfg.target_col] != 0]
                start = g.loc[nz[0], cfg.date_col] if len(nz) else g[cfg.date_col].iloc[0]
            else:
                start = g[cfg.date_col].iloc[0]
            if cfg.start_policy == "global":
                start = min(start, global_start)
            idx = period_range(start, cutoff, cfg.freq)
            if len(idx) == 0:
                continue
            s = (g.set_index(cfg.date_col)[cfg.target_col]
                 .reindex(idx, fill_value=cfg.fill_value).astype(float))
            out = pd.DataFrame({cfg.date_col: idx, cfg.target_col: s.to_numpy()})
            for col, val in zip(cfg.category_cols, key):
                out[col] = val
            frames.append(out)

        if not frames:
            return pd.DataFrame(columns=list(cfg.category_cols) + [cfg.date_col, cfg.target_col])
        panel = pd.concat(frames, ignore_index=True)
        return panel[list(cfg.category_cols) + [cfg.date_col, cfg.target_col]].sort_values(
            list(cfg.category_cols) + [cfg.date_col], ignore_index=True
        )

    # -- 2. planificación de recursos --------------------------------------- #
    def plan_workers(self, n_tasks: int) -> int:
        """Cantidad de procesos, acotada por CPU, RAM libre y límites del contenedor.

        Respeta la variable de entorno ``FC_MAX_WORKERS`` (útil para fijarlo desde
        el nodo del pipeline sin tocar el código).
        """
        cfg = self.cfg
        cpu = available_cpus()
        if cfg.n_jobs is None:
            n = cpu
        elif cfg.n_jobs < 0:
            n = max(1, cpu + 1 + cfg.n_jobs)
        elif cfg.n_jobs == 0:
            n = 1
        else:
            n = int(cfg.n_jobs)
        n = min(n, cpu, max(1, n_tasks))
        if cfg.max_workers:
            n = min(n, int(cfg.max_workers))
        env_cap = os.getenv("FC_MAX_WORKERS")
        if env_cap and env_cap.isdigit() and int(env_cap) > 0:
            n = min(n, int(env_cap))

        avail_gb = available_memory_gb()
        if avail_gb is not None and cfg.worker_memory_gb > 0:
            budget = avail_gb * cfg.memory_fraction - cfg.reserve_memory_gb
            by_mem = max(1, int(budget // cfg.worker_memory_gb))
            if by_mem < n:
                LOGGER.info("RAM disponible %.1f GB -> limito a %d workers (pedía %d)",
                            avail_gb, by_mem, n)
            n = max(1, min(n, by_mem))
        if container_cpu_limit() or container_memory_limit_gb():
            LOGGER.info("límites del contenedor: cpu=%s, ram=%s GB",
                        container_cpu_limit(), container_memory_limit_gb())
        return max(1, n)

    # -- 3. armado de segmentos --------------------------------------------- #
    def _plan_segments(self, target_dates: Sequence[pd.Timestamp], cutoff: pd.Timestamp
                       ) -> List[Tuple[pd.Timestamp, int, List[pd.Timestamp]]]:
        """Agrupa las fechas a predecir en bloques contiguos de ``refit_step``.

        Cada bloque devuelve (origen, h, fechas). El origen es el período
        inmediatamente anterior al primer punto del bloque: el entrenamiento
        usa sólo información disponible en ese momento.
        """
        cfg = self.cfg
        dates = sorted(pd.Timestamp(d) for d in set(target_dates))
        segments: List[Tuple[pd.Timestamp, int, List[pd.Timestamp]]] = []
        i = 0
        while i < len(dates):
            block = [dates[i]]
            while len(block) < cfg.refit_step and i + len(block) < len(dates):
                nxt = dates[i + len(block)]
                if nxt == shift_period(block[-1], 1, cfg.freq):
                    block.append(nxt)
                else:
                    break
            origin = shift_period(block[0], -1, cfg.freq)
            segments.append((origin, len(block), block))
            i += len(block)
        return segments

    # -- 4. ejecución -------------------------------------------------------- #
    def _execute(self, panel: pd.DataFrame,
                 backtest_targets: Mapping[Tuple, Sequence[pd.Timestamp]],
                 cutoff: pd.Timestamp) -> pd.DataFrame:
        """Corre backtest + futuro. Devuelve tabla larga con una fila por
        (serie, fecha) y una columna por modelo."""
        cfg = self.cfg
        ctx = ModelContext(
            freq=cfg.freq,
            season_length=cfg.resolved_season_length(),
            seasonal=cfg.seasonal,
            min_history=cfg.min_history,
            min_seasonal_cycles=cfg.min_seasonal_cycles,
            non_negative=cfg.non_negative,
            params={**dict(cfg.model_params), "_combo_exclude": tuple(cfg.combo_exclude)},
        )
        base_models = tuple(m for m in self.models if m != COMBO_MODEL)
        future_dates = [shift_period(cutoff, k, cfg.freq) for k in range(1, cfg.horizon + 1)]

        tasks: List[SeriesTask] = []
        pred_index: List[Tuple[Tuple, List[pd.Timestamp]]] = []
        for key, g in panel.groupby(list(cfg.category_cols), observed=True, dropna=False, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            g = g.sort_values(cfg.date_col)
            dates = pd.DatetimeIndex(g[cfg.date_col])
            values = g[cfg.target_col].to_numpy(dtype=float)

            available = set(dates)
            targets = [d for d in backtest_targets.get(key, ()) if d in available]
            segments_plan = self._plan_segments(targets, cutoff) if targets else []
            flat_dates: List[pd.Timestamp] = []
            segments: List[Tuple[np.datetime64, int, int]] = []
            for origin, h, block in segments_plan:
                if origin < dates[0]:      # sin historia previa al origen -> no evaluable
                    continue
                segments.append((np.datetime64(origin), h, len(flat_dates)))
                flat_dates.extend(block)
            if cfg.horizon > 0:
                segments.append((np.datetime64(cutoff), cfg.horizon, len(flat_dates)))
                flat_dates.extend(future_dates)
            if not segments:
                continue
            tasks.append(SeriesTask(key=key, dates=dates.values, values=values,
                                    segments=tuple(segments), n_pred=len(flat_dates),
                                    models=base_models, ctx=ctx))
            pred_index.append((key, flat_dates))

        if not tasks:
            return pd.DataFrame(columns=list(cfg.category_cols) + [cfg.date_col])

        n_workers = self.plan_workers(len(tasks))
        total_fits = sum(len(t.segments) for t in tasks) * len(base_models)
        LOGGER.info("Series: %d | segmentos totales: %d | modelos: %d | ajustes ~%d | workers: %d",
                    len(tasks), sum(len(t.segments) for t in tasks), len(base_models),
                    total_fits, n_workers)

        if HAS_JOBLIB and n_workers > 1:
            with _joblib.parallel_backend("loky", n_jobs=n_workers,
                                          inner_max_num_threads=cfg.inner_threads):
                results = _joblib.Parallel(batch_size=cfg.batch_size, verbose=0)(
                    _joblib.delayed(run_series_task)(t) for t in tasks
                )
        else:
            if not HAS_JOBLIB:
                LOGGER.warning("joblib no disponible: ejecución secuencial")
            results = [run_series_task(t) for t in tasks]

        # -> tabla larga
        frames = []
        failures: List[dict] = []
        for (key, flat_dates), res in zip(pred_index, results):
            data = {cfg.date_col: pd.DatetimeIndex(flat_dates)}
            for col, val in zip(cfg.category_cols, key):
                data[col] = val
            for model, arr in res["preds"].items():
                data[self.model_col(model)] = arr
            frames.append(pd.DataFrame(data))
            for model, msg in res["failures"].items():
                failures.append({**dict(zip(cfg.category_cols, key)), "model": model, "error": msg})

        self.failures_ = pd.DataFrame(failures)
        if len(self.failures_):
            top = self.failures_["model"].value_counts().to_dict()
            LOGGER.warning("Modelos con fallos (series afectadas): %s", top)
        return pd.concat(frames, ignore_index=True)

    # -- 5. selección -------------------------------------------------------- #
    def _select(self, history: pd.DataFrame, panel: pd.DataFrame, model_cols: Dict[str, str],
                cutoff: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg
        m = cfg.resolved_season_length()
        scales = (panel.groupby(list(cfg.category_cols), observed=True, dropna=False)[cfg.target_col]
                  .apply(lambda s: _naive_scale(s.to_numpy(dtype=float), m)))

        scores = evaluate_models(history, cfg, model_cols, cutoff, scales)
        if scores.empty:
            LOGGER.warning("Sin puntos evaluables: se usa el modelo de fallback '%s'", cfg.fallback_model)
            keys = panel[list(cfg.category_cols)].drop_duplicates().copy()
            keys[cfg.best_model_col] = cfg.fallback_model
            keys[cfg.best_score_col] = np.nan
            return scores, keys

        scores = scores.sort_values(list(cfg.category_cols) + ["score", "model"])
        best = (scores.groupby(list(cfg.category_cols), observed=True, dropna=False, as_index=False)
                .first()[list(cfg.category_cols) + ["model", "score"]]
                .rename(columns={"model": cfg.best_model_col, "score": cfg.best_score_col}))

        all_keys = panel[list(cfg.category_cols)].drop_duplicates()
        best = all_keys.merge(best, on=list(cfg.category_cols), how="left")
        best[cfg.best_model_col] = best[cfg.best_model_col].fillna(cfg.fallback_model)
        return scores, best

    # -- 6. API principal ---------------------------------------------------- #
    def run(self, data: pd.DataFrame, previous: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Corre el pipeline completo.

        Parameters
        ----------
        data : DataFrame
            Fuente de datos con categorías, fecha y métrica real. Puede además
            traer columnas de forecast previas (se detectan por prefijo o por
            ``external_forecast_cols``) y se evalúan como un modelo más.
        previous : DataFrame, opcional
            Salida de una corrida anterior de esta misma librería. Si se pasa,
            el backtest histórico NO se recalcula: sólo se completan los
            períodos que hayan quedado sin predicción.

        Returns
        -------
        DataFrame con categorías, fecha, ``y_real`` (NaN a futuro), una columna
        por modelo, ``best_model``, ``best_score``, ``yhat`` e ``is_future``.
        """
        cfg = self.cfg
        self.run_id = uuid.uuid4().hex[:12]
        cutoff = cfg.resolved_cutoff()
        LOGGER.info("run_id=%s | freq=%s | cutoff=%s | horizon=%d | backtest=%d | refit_step=%d",
                    self.run_id, cfg.freq, cutoff.date(), cfg.horizon, cfg.backtest_horizon, cfg.refit_step)

        # 1) reales saneados
        panel = self.sanitize(data, cutoff=cutoff)
        self.panel_ = panel
        if panel.empty:
            raise ValueError("Tras el saneo no quedaron datos: revisá fechas, cutoff y columnas")
        LOGGER.info("Panel saneado: %d series x %d filas (desde %s)",
                    panel[list(cfg.category_cols)].drop_duplicates().shape[0], len(panel),
                    panel[cfg.date_col].min().date())

        # 2) predicciones ya conocidas (corrida previa y/o columnas externas)
        known = self._collect_known(data, previous, cutoff)
        known_cols = self._detect_model_cols(known) if known is not None else {}

        # 3) qué falta backtestear (por serie)
        eval_start = shift_period(cutoff, -(cfg.backtest_horizon - 1), cfg.freq)
        eval_dates = list(period_range(eval_start, cutoff, cfg.freq))
        pending = self._pending_targets(panel, known, known_cols, eval_dates)

        # 4) correr modelos (backtest pendiente + futuro)
        fresh = self._execute(panel, pending, cutoff)

        # 5) armar la salida
        return self._assemble(panel, fresh, known, known_cols, cutoff)

    def _pending_targets(self, panel: pd.DataFrame, known: Optional[pd.DataFrame],
                         known_cols: Dict[str, str], eval_dates: List[pd.Timestamp]
                         ) -> Dict[Tuple, List[pd.Timestamp]]:
        """Fechas del backtest que hay que calcular, serie por serie.

        Sólo cuentan como resueltas las predicciones hechas por esta librería en
        corridas anteriores (una columna externa no sustituye al backtest). Si en
        la config aparece un modelo que no existe en la salida previa, se
        recalcula toda la ventana para que ese modelo tenga historia comparable.
        """
        cfg = self.cfg
        keys = [tuple(k) for k in panel[list(cfg.category_cols)].drop_duplicates().to_numpy()]
        full = {k: list(eval_dates) for k in keys}
        if known is None or not known_cols:
            return full

        engine_cols = self._engine_cols(known_cols)
        known_models = {known_cols[c] for c in engine_cols}
        missing = [m for m in self.models if m not in known_models]
        if missing:
            LOGGER.info("Modelos sin historia previa (%s): se rehace la ventana completa",
                        ", ".join(missing))
            return full

        resolved = known.loc[known[engine_cols].notna().any(axis=1),
                             list(cfg.category_cols) + [cfg.date_col]]
        covered: Dict[Tuple, set] = {}
        for row in resolved.itertuples(index=False):
            covered.setdefault(tuple(row[:-1]), set()).add(pd.Timestamp(row[-1]))

        pending = {k: [d for d in eval_dates if d not in covered.get(k, ())] for k in keys}
        n_pend = sum(len(v) for v in pending.values())
        n_tot = len(keys) * len(eval_dates)
        LOGGER.info("Backtest: %d de %d (serie, período) ya venían resueltos, faltan %d",
                    n_tot - n_pend, n_tot, n_pend)
        return pending

    # -- helpers de ensamblado ---------------------------------------------- #
    def _collect_known(self, data: pd.DataFrame, previous: Optional[pd.DataFrame],
                       cutoff: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Une las predicciones ya existentes (previous manda sobre data)."""
        cfg = self.cfg
        pieces = []
        for src in (data, previous):
            if src is None or len(src) == 0:
                continue
            cols = self._detect_model_cols(src)
            if not cols:
                continue
            keep = list(cfg.category_cols) + [cfg.date_col] + list(cols)
            piece = src.loc[:, keep].copy()
            piece[cfg.date_col] = floor_to_freq(piece[cfg.date_col], cfg.freq).values
            pieces.append(piece)
        if not pieces:
            return None
        merged = pieces[0]
        for piece in pieces[1:]:
            merged = merged.merge(piece, on=list(cfg.category_cols) + [cfg.date_col],
                                  how="outer", suffixes=("", "__new"))
            for col in [c for c in merged.columns if c.endswith("__new")]:
                base = col[:-5]
                merged[base] = merged[col].combine_first(merged[base]) if base in merged else merged[col]
                merged = merged.drop(columns=[col])
        # Las predicciones propias sólo sirven como validación hasta el cutoff (el
        # futuro se regenera). Las externas se conservan también a futuro, para
        # que un forecast de otro sistema pueda competir en las dos ventanas.
        ext_present = [c for c in merged.columns if c in set(cfg.external_forecast_cols)]
        keep = merged[cfg.date_col] <= cutoff
        if ext_present:
            keep |= merged[ext_present].notna().any(axis=1)
        merged = merged[keep]
        agg_cols = [c for c in merged.columns if c not in list(cfg.category_cols) + [cfg.date_col]]
        merged = (merged.groupby(list(cfg.category_cols) + [cfg.date_col], observed=True,
                                 dropna=False, as_index=False)[agg_cols].last())
        return merged

    def _assemble(self, panel: pd.DataFrame, fresh: pd.DataFrame,
                  known: Optional[pd.DataFrame], known_cols: Dict[str, str],
                  cutoff: pd.Timestamp) -> pd.DataFrame:
        cfg = self.cfg
        keys = list(cfg.category_cols) + [cfg.date_col]

        out = fresh.copy() if len(fresh) else pd.DataFrame(columns=keys)
        if known is not None and len(known):
            out = out.merge(known, on=keys, how="outer", suffixes=("", "__prev"))
            for col in [c for c in out.columns if c.endswith("__prev")]:
                base = col[:-6]
                out[base] = out[base].combine_first(out[col])
                out = out.drop(columns=[col])

        # reales
        actuals = panel.rename(columns={cfg.target_col: cfg.actual_col})
        out = out.merge(actuals, on=keys, how="outer")

        out[cfg.future_flag_col] = (out[cfg.date_col] > cutoff).astype(int)
        out.loc[out[cfg.future_flag_col] == 1, cfg.actual_col] = np.nan
        # nada más allá del horizonte pedido (una columna externa puede traer de más)
        out = out[out[cfg.date_col] <= shift_period(cutoff, cfg.horizon, cfg.freq)]

        # Recorte del histórico: por defecto se devuelven las filas que tienen
        # al menos un forecast (la ventana de backtest de esta corrida más lo
        # acumulado de corridas anteriores) y todo el futuro.
        model_cols = self._detect_model_cols(out)
        engine_cols = self._engine_cols(model_cols)
        if not cfg.include_full_history:
            has_pred = out[engine_cols].notna().any(axis=1) if engine_cols else False
            out = out[has_pred | (out[cfg.future_flag_col] == 1)]
        if cfg.keep_history_periods:
            floor_date = shift_period(cutoff, -(cfg.keep_history_periods - 1), cfg.freq)
            out = out[(out[cfg.date_col] >= floor_date) | (out[cfg.future_flag_col] == 1)]
        out = out.reset_index(drop=True)
        scores, best = self._select(out, panel, model_cols, cutoff)
        self.scores_ = scores
        self.selection_ = best

        out = out.merge(best, on=list(cfg.category_cols), how="left")
        out[cfg.best_model_col] = out[cfg.best_model_col].fillna(cfg.fallback_model)

        # yhat = valor del modelo elegido (con fallback en cascada)
        inv = {v: k for k, v in model_cols.items()}
        chosen = np.full(len(out), np.nan)
        for model, col in inv.items():
            mask = (out[cfg.best_model_col] == model).to_numpy()
            if mask.any() and col in out.columns:
                chosen[mask] = pd.to_numeric(out.loc[mask, col], errors="coerce").to_numpy()
        out[cfg.forecast_col] = chosen
        for fb in (cfg.fallback_model, "snaive", "naive", COMBO_MODEL):
            col = inv.get(fb)
            if col and out[cfg.forecast_col].isna().any():
                out[cfg.forecast_col] = out[cfg.forecast_col].fillna(pd.to_numeric(out[col], errors="coerce"))

        if cfg.non_negative:
            num_cols = list(model_cols) + [cfg.forecast_col]
            out[num_cols] = out[num_cols].clip(lower=0)
        if cfg.round_to is not None:
            num_cols = list(model_cols) + [cfg.forecast_col]
            out[num_cols] = out[num_cols].round(cfg.round_to)

        out[cfg.cutoff_col] = cutoff
        out["run_id"] = self.run_id

        ordered = (list(cfg.category_cols) + [cfg.date_col, cfg.actual_col]
                   + sorted(model_cols) + [cfg.best_model_col, cfg.best_score_col,
                                           cfg.forecast_col, cfg.future_flag_col,
                                           cfg.cutoff_col, "run_id"])
        ordered = [c for c in ordered if c in out.columns]
        out = out.loc[:, ordered].sort_values(keys, ignore_index=True)
        LOGGER.info("Salida: %d filas (%d históricas, %d futuras) | modelos en tabla: %d",
                    len(out), int((out[cfg.future_flag_col] == 0).sum()),
                    int((out[cfg.future_flag_col] == 1).sum()), len(model_cols))
        return out

    # -- reporting ----------------------------------------------------------- #
    def summary(self) -> pd.DataFrame:
        """Ranking global de modelos (cuántas series ganó y error mediano)."""
        if self.scores_ is None or self.scores_.empty:
            return pd.DataFrame()
        cfg = self.cfg
        wins = (self.selection_[cfg.best_model_col].value_counts()
                .rename_axis("model").rename("series_ganadas").reset_index())
        agg = (self.scores_.groupby("model", as_index=False)
               .agg(score_mediano=("score", "median"), wmape_mediano=("wmape", "median"),
                    series_evaluadas=("score", "size")))
        return agg.merge(wins, on="model", how="left").fillna({"series_ganadas": 0}) \
                  .sort_values("score_mediano", ignore_index=True)


# --------------------------------------------------------------------------- #
# Conversión ancho <-> largo (para persistir en base de datos)
# --------------------------------------------------------------------------- #
def wide_to_long(df: pd.DataFrame, cfg: ForecastConfig,
                 model_name_col: str = "model_name",
                 value_col: str = "model_yhat") -> pd.DataFrame:
    """Pasa la salida ancha a formato largo (una fila por modelo).

    Conviene para persistir en una tabla relacional cuyo set de modelos puede
    cambiar sin tener que alterar el DDL.
    """
    id_cols = [c for c in (list(cfg.category_cols) + [cfg.date_col, cfg.actual_col,
                                                      cfg.best_model_col, cfg.best_score_col,
                                                      cfg.forecast_col, cfg.future_flag_col,
                                                      cfg.cutoff_col, "run_id"])
               if c in df.columns]
    model_cols = [c for c in df.columns if c not in id_cols]
    long = df.melt(id_vars=id_cols, value_vars=model_cols,
                   var_name=model_name_col, value_name=value_col)
    prefix = cfg.model_col_prefix
    long[model_name_col] = long[model_name_col].apply(
        lambda c: c[len(prefix):] if prefix and str(c).startswith(prefix) else c)
    return long.dropna(subset=[value_col]).reset_index(drop=True)


def long_to_wide(df: pd.DataFrame, cfg: ForecastConfig,
                 model_name_col: str = "model_name",
                 value_col: str = "model_yhat") -> pd.DataFrame:
    """Inversa de :func:`wide_to_long`; reconstruye la entrada para ``previous``."""
    key_cols = list(cfg.category_cols) + [cfg.date_col]
    id_cols = [c for c in (key_cols + [cfg.actual_col, cfg.best_model_col, cfg.best_score_col,
                                       cfg.forecast_col, cfg.future_flag_col,
                                       cfg.cutoff_col, "run_id"])
               if c in df.columns]
    work = df.copy()
    work[cfg.date_col] = pd.to_datetime(work[cfg.date_col])

    # Se pivotea sólo sobre las claves (nunca nulas). Usar pivot_table sobre todas
    # las columnas id explotaría el producto cartesiano en cuanto haya nulos.
    values = (work.groupby(key_cols + [model_name_col], observed=True, dropna=False)[value_col]
                  .last().unstack(model_name_col))
    values.columns = [f"{cfg.model_col_prefix}{c}" for c in values.columns]
    values = values.reset_index()

    meta = work.loc[:, id_cols].drop_duplicates(subset=key_cols, keep="last")
    wide = meta.merge(values, on=key_cols, how="outer")
    wide.columns.name = None
    return wide.sort_values(key_cols, ignore_index=True)
