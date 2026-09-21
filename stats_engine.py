"""
stats_engine
============

Motor de estadísticas descriptivas por cliente (o por cualquier combinación de
categorías), misma filosofía que ``forecast_engine``.

Entrada  : tabla larga -> [categorías..., FECHA, MT_VENTA, MT_MARGEN]
           a cualquier granularidad; el motor la lleva a grano diario.
Salida   : una fila por combinación de claves con todas las métricas del
           catálogo (ver ``catalogo()``) más BD_HISTORIAL y FECHA_CORTE.

Categorías
----------
La convención de nombres define el comportamiento:
    SK_* / BK_*   claves: definen el grano (GROUP BY)
    BD_*          descripciones: NO agrupan; se toma el valor de la fila más
                  reciente. Si una descripción cambia, el cliente no se parte.

DÓNDE MODIFICAR UNA MÉTRICA
---------------------------
Cada columna de salida está declarada en ``catalogo()`` con su descripción y la
función que la calcula. Esa función está en la sección correspondiente de este
archivo (A. ventas por ventana, B. comportamiento, ...). Cambiar el cálculo de
una columna es cambiar esa función; nada más depende de ella.

Cómo es rápido
--------------
Nada se calcula cliente por cliente. Los datos quedan ordenados por
(cliente, día) y cada métrica es una suma vectorizada por grupo
(``np.bincount``) o una reducción sobre tramos contiguos (``ufunc.reduceat``).
Las métricas que requieren meses con cero (tendencias, estacionalidad) no
rellenan ceros: los meses sin venta no aportan a las sumas, y los términos que
dependen de la cantidad de meses (n, Σx, Σx²) salen de fórmulas cerradas.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["StatsConfig", "StatsEngine", "Fechas", "catalogo", "ajustar_actividad", "prob_activo",
           "historial_a_filas", "MODELOS_ACTIVIDAD", "MESES_ES"]

LOGGER = logging.getLogger("stats_engine")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)

MESES_ES = ["ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO", "JULIO", "AGOSTO",
            "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"]

_EPOCA = pd.Timestamp("1970-01-01")


def sin_residuo(suma, suma_abs, tol: float, piso: float = 0.0):
    """Anula la suma cuando no se distingue del residuo de cancelación.

    Sumar en float64 una venta y su devolución no da cero exacto: cada número
    decimal se guarda redondeado, y al ir acumulando queda un residuo del orden
    de 1e-16 veces lo más grande que pasó por la suma. Con 300.000 USD de venta
    y 300.000 de devolución el neto puede quedar en -4.7e-11 en vez de 0. Como
    número no molesta; como denominador de un porcentaje, explota: un margen de
    -0,01 dividido por -4.7e-11 da 21 millones por ciento.

    Por eso el cero no se prueba contra 0 exacto sino contra la escala real del
    grupo, que es la suma de los valores absolutos: lo que pasó por la suma.

        suma = -4.7e-11, suma_abs = 600.000  ->  razón 7.8e-17  ->  es cero
        suma = -0,01,    suma_abs = 600.000  ->  razón 1.7e-8   ->  es un neto real

    tol = 0 devuelve la suma tal cual.

    `piso` es una escala mínima de referencia (la del panel típico): sirve para el
    residuo que ya llega cancelado desde la fuente, donde no hay nada con qué
    compararlo dentro de la propia suma.
    """
    if not tol:
        return suma
    suma = np.asarray(suma, dtype=float)
    escala = np.maximum(np.asarray(suma_abs, dtype=float), float(piso or 0.0))
    return np.where(np.abs(suma) <= tol * escala, 0.0, suma)


def escala_decimal(arrays, max_decimales: int = 6) -> float:
    """Potencia de 10 que vuelve enteros a todos los importes, o 0 si no la hay.

    El dinero es decimal: 1.234,56 son 123.456 centavos. Los enteros se suman SIN
    ERROR en float64 mientras no pasen de 2^53, así que sumar en esa escala hace que
    una venta y su devolución den cero exacto, sin tolerancias ni umbrales. Es lo
    mismo que hace Oracle con NUMBER, y cuesta lo mismo que sumar en float.

    Se elige la escala más chica que sirva y que no desborde 2^53 con el bruto del
    panel. Si los importes traen más decimales de los buscados, se usa el máximo y
    el resto se redondea (medio millonésimo de dólar, si max_decimales=6). Devuelve
    0 cuando no hay escala usable: ahí entra `sin_residuo` como plan B.
    """
    finitos = [np.asarray(a, dtype=float)[np.isfinite(np.asarray(a, dtype=float))]
               for a in arrays]
    bruto = sum(float(np.abs(a).sum()) for a in finitos) or 1.0
    for d in range(int(max_decimales) + 1):
        e = 10.0 ** d
        if bruto * e >= 2.0 ** 53:      # ya no entra, y con más decimales entra menos
            return 0.0
        if all(np.array_equal(np.round(a, d), a) for a in finitos):
            return e
    e = 10.0 ** int(max_decimales)
    return e if bruto * e < 2.0 ** 53 else 0.0


def escala_tipica(suma_abs) -> float:
    """Escala de referencia del panel: la mediana de lo que movió cada grupo."""
    a = np.asarray(suma_abs, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.median(a)) if a.size else 0.0


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
@dataclass
class StatsConfig:
    """Parámetros del cálculo. Los defaults son las definiciones acordadas."""

    categorias: Sequence[str] = field(default_factory=list)
    col_fecha: str = "FECHA"
    col_venta: str = "MT_VENTA"
    col_margen: str = "MT_MARGEN"

    #: "hoy". Ayer = hoy - 1 es el último día incluido; el mes en curso es el de hoy.
    fecha_ejecucion: Optional[Any] = None

    #: ventanas móviles "hasta ayer", en días
    dias_ventana: Dict[str, int] = field(default_factory=lambda: {
        "R6": 183, "R12": 365, "R24": 730, "R36": 1095})
    #: ventanas a mes cerrado, en meses calendario
    meses_ventana: Dict[str, int] = field(default_factory=lambda: {
        "R6": 6, "R12": 12, "R24": 24, "R36": 36})
    #: variantes de frecuencia, ticket e IDD (además del histórico)
    variantes: Tuple[str, ...] = ("R12", "R24", "R36")

    ddof: int = 1                      #: desvío muestral (como STDEV.S de Excel)
    idd_unidad: str = "gon"            #: gon (0..100) | grados | radianes
    idd_normalizar: bool = False       #: True = pendiente relativa al promedio antes del ángulo
    margen_escala: float = 100.0       #: márgenes en % (100) o fracción (1)
    #: Suma el dinero en su escala decimal (centavos): la venta que se anula con su
    #: devolución da cero EXACTO, sin tolerancias. Es lo que hace Oracle con NUMBER.
    suma_exacta: bool = True
    #: hasta cuántos decimales busca esa escala. Más allá, redondea.
    max_decimales: int = 6
    #: Plan B para cuando no hay escala decimal usable (importes con muchísimos
    #: decimales o un bruto que desborda 2^53). Un denominador se toma como cero
    #: cuando |suma| <= tolerancia_cero * suma de los
    #: valores absolutos. Una venta y su devolución no se anulan exacto en punto flotante:
    #: dejan un residuo minúsculo, y si ese residuo cae en el denominador de un porcentaje
    #: el resultado explota. 1e-9 = un centavo en diez millones. 0 desactiva la limpieza.
    tolerancia_cero: float = 1e-9
    modelo_actividad: str = "pareto"   #: pareto | mbgnbd | bgnbd
    #: el modelo se ajusta sobre una muestra estable de hasta N grupos (None = todos).
    #: Sus 4 parámetros describen a la población: la probabilidad se calcula para todos.
    max_clientes_ajuste: Optional[int] = 20_000
    #: hilos para el ajuste y la probabilidad de actividad (None = todos los CPUs del pod)
    hilos_actividad: Optional[int] = None
    #: categorías que segmentan la actividad: un modelo por cada combinación de valores
    #: (vacío = un modelo para todo el panel). Tienen que estar en `categorias`.
    segmentos_actividad: Sequence[str] = ()
    #: un segmento con menos grupos que esto usa los parámetros del panel completo
    min_clientes_segmento: int = 1_000
    incluir_historial: bool = True     #: armar BD_HISTORIAL
    decimales_historial: int = 2
    verbose: int = 1

    # -- derivados ---------------------------------------------------------- #
    def claves(self) -> List[str]:
        return [c for c in self.categorias if not str(c).upper().startswith("BD_")]

    def descripciones(self) -> List[str]:
        return [c for c in self.categorias if str(c).upper().startswith("BD_")]

    def validate(self) -> None:
        if not self.claves():
            raise ValueError("Hace falta al menos una categoría clave (SK_* o BK_*)")
        if self.idd_unidad not in ("gon", "grados", "radianes"):
            raise ValueError("idd_unidad debe ser gon, grados o radianes")
        if not 0 <= self.max_decimales <= 15:
            raise ValueError("max_decimales debe estar entre 0 y 15")
        if not 0 <= self.tolerancia_cero < 1:
            raise ValueError("tolerancia_cero debe estar entre 0 y 1 (0 = sin limpieza)")
        if self.modelo_actividad not in MODELOS_ACTIVIDAD:
            raise ValueError(f"modelo_actividad debe ser uno de {list(MODELOS_ACTIVIDAD)}")
        fuera = [c for c in self.segmentos_actividad if c not in self.categorias]
        if fuera:
            raise ValueError(f"segmentos_actividad {fuera} tienen que estar en categorias {list(self.categorias)}")
        faltan =[v for v in self.variantes if v not in self.dias_ventana or v not in self.meses_ventana]
        if faltan:
            raise ValueError(f"Variantes sin ventana definida: {faltan}")


# --------------------------------------------------------------------------- #
# Fechas de referencia
# --------------------------------------------------------------------------- #
def _dia(ts) -> int:
    return int((pd.Timestamp(ts).normalize() - _EPOCA).days)


def _mes_idx(ts) -> int:
    ts = pd.Timestamp(ts)
    return (ts.year - 1970) * 12 + ts.month - 1


def _inicio_mes(idx: int) -> pd.Timestamp:
    return pd.Timestamp(year=1970 + idx // 12, month=idx % 12 + 1, day=1)


@dataclass(frozen=True)
class Fechas:
    hoy: pd.Timestamp
    ayer: pd.Timestamp
    m_actual: int          # índice de mes del mes en curso (meses desde 1970-01)
    m_mc: int              # último mes cerrado

    @classmethod
    def desde(cls, fecha_ejecucion) -> "Fechas":
        hoy = (pd.Timestamp(fecha_ejecucion) if fecha_ejecucion is not None
               else pd.Timestamp.today()).normalize()
        m = _mes_idx(hoy)
        return cls(hoy=hoy, ayer=hoy - pd.Timedelta(days=1), m_actual=m, m_mc=m - 1)

    @property
    def d_ayer(self) -> int:
        return _dia(self.ayer)


# --------------------------------------------------------------------------- #
# Contexto: datos ordenados + resultados intermedios compartidos
# --------------------------------------------------------------------------- #
class Contexto:
    """Arrays ordenados por (grupo, día), un registro por grupo-día.

    Todo lo que usan varias métricas se calcula una sola vez (cached_property).
    """

    def __init__(self, cfg: StatsConfig, fechas: Fechas, gid: np.ndarray, dia: np.ndarray,
                 venta: np.ndarray, margen: np.ndarray, n_grupos: int):
        self.cfg = cfg
        self.f = fechas
        self.gid = gid
        self.dia = dia
        self.venta = venta
        self.margen = margen
        self.G = n_grupos
        self.hash_grupo: Optional[np.ndarray] = None    # identifica al grupo entre corridas
        self.segmento: Optional[np.ndarray] = None      # segmento de actividad de cada grupo
        self.escala = 0.0        # escala decimal del dinero (100 = centavos); 0 = sin escala
        self.residuos = 0        # sumas de dinero que quedaron en cero por cancelación

    # -- helpers vectorizados ------------------------------------------------ #
    def suma(self, grupos: np.ndarray, valores: np.ndarray, mascara=None,
             neta: bool = False) -> np.ndarray:
        """Suma por grupo. neta=True es para el dinero, que puede sumar cero de verdad:
        se suma en la escala decimal (exacto) o, si no hay, se limpia el residuo."""
        w = valores if mascara is None else np.where(mascara, valores, 0.0)
        if neta and self.escala:          # centavos: enteros, sin error posible
            return np.bincount(grupos, weights=np.round(w * self.escala),
                               minlength=self.G) / self.escala
        s = np.bincount(grupos, weights=w, minlength=self.G).astype(float)
        if not neta or not self.cfg.tolerancia_cero:
            return s
        escala = np.bincount(grupos, weights=np.abs(w), minlength=self.G).astype(float)
        limpia = sin_residuo(s, escala, self.cfg.tolerancia_cero, escala_tipica(escala))
        self.residuos += int(np.count_nonzero((s != 0) & (limpia == 0)))
        return limpia

    def conteo(self, grupos: np.ndarray, mascara=None) -> np.ndarray:
        w = None if mascara is None else mascara.astype(float)
        return np.bincount(grupos, weights=w, minlength=self.G).astype(float)

    def reducir(self, grupos: np.ndarray, valores: np.ndarray, ufunc) -> np.ndarray:
        """max/min por grupo sobre arrays ordenados por grupo. Grupos sin datos -> NaN."""
        out = np.full(self.G, np.nan)
        if len(grupos) == 0:
            return out
        ini = np.flatnonzero(np.r_[True, grupos[1:] != grupos[:-1]])
        out[grupos[ini]] = ufunc.reduceat(valores.astype(float), ini)
        return out

    def media_std(self, n: np.ndarray, s: np.ndarray, ss: np.ndarray):
        ddof = self.cfg.ddof
        with np.errstate(divide="ignore", invalid="ignore"):
            media = np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)
            var = (ss - s * s / np.where(n > 0, n, 1)) / np.where(n > ddof, n - ddof, 1)
            std = np.where(n > ddof, np.sqrt(np.maximum(var, 0.0)), np.nan)
        return media, std

    # -- estructura por grupo ------------------------------------------------ #
    @cached_property
    def inicios(self) -> np.ndarray:
        return np.flatnonzero(np.r_[True, self.gid[1:] != self.gid[:-1]])

    @cached_property
    def finales(self) -> np.ndarray:
        return np.r_[self.inicios[1:], len(self.gid)] - 1

    @cached_property
    def primera(self) -> np.ndarray:
        return self.dia[self.inicios]

    @cached_property
    def ultima(self) -> np.ndarray:
        return self.dia[self.finales]

    @cached_property
    def dias_compra(self) -> np.ndarray:
        return self.conteo(self.gid)

    @cached_property
    def anio_fila(self) -> np.ndarray:
        return self.dia.astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970

    @cached_property
    def mes_fila(self) -> np.ndarray:
        return self.dia.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)

    @cached_property
    def primer_mes(self) -> np.ndarray:
        return self.mes_fila[self.inicios]

    # -- intervalos entre días de compra ------------------------------------- #
    @cached_property
    def intervalos(self) -> Dict[str, np.ndarray]:
        """Un registro por par de días de compra consecutivos del mismo grupo."""
        mismo = self.gid[1:] == self.gid[:-1]
        return {
            "gid": self.gid[1:][mismo],
            "desde": self.dia[:-1][mismo],
            "hasta": self.dia[1:][mismo],
            "dias": (self.dia[1:] - self.dia[:-1])[mismo].astype(float),
        }

    # -- agregado mensual (disperso: sólo meses con movimiento) --------------- #
    @cached_property
    def mensual(self) -> Dict[str, np.ndarray]:
        m = self.mes_fila
        idx = np.flatnonzero(np.r_[True, (self.gid[1:] != self.gid[:-1]) | (m[1:] != m[:-1])])
        # el mes que se anula con sus devoluciones tiene que valer 0, no 1e-11
        if self.escala:
            venta = np.add.reduceat(np.round(self.venta * self.escala), idx) / self.escala
            margen = np.add.reduceat(np.round(self.margen * self.escala), idx) / self.escala
        else:
            tol = self.cfg.tolerancia_cero
            venta = np.add.reduceat(self.venta, idx)
            margen = np.add.reduceat(self.margen, idx)
            if tol:
                bv = np.add.reduceat(np.abs(self.venta), idx)
                bm = np.add.reduceat(np.abs(self.margen), idx)
                venta = sin_residuo(venta, bv, tol, escala_tipica(bv))
                margen = sin_residuo(margen, bm, tol, escala_tipica(bm))
        return {"gid": self.gid[idx], "mes": m[idx], "venta": venta, "margen": margen}

    # -- estacionalidad (G x 12), compartida por varias columnas ------------- #
    @cached_property
    def estacional(self) -> Tuple[np.ndarray, np.ndarray]:
        """Promedio y desvío de la venta de cada mes calendario, por grupo.

        Se cuentan los meses cerrados desde el primer mes con compra; los meses
        sin venta cuentan como 0. La cantidad de años de cada mes calendario sale
        de una fórmula cerrada, así que no hace falta rellenar los ceros.
        """
        mm = self.mensual
        lo, hi = self.primer_mes, self.f.m_mc
        cal = np.arange(12)
        n = (hi - cal)[None, :] // 12 - (lo[:, None] - 1 - cal[None, :]) // 12
        n = np.where(lo[:, None] > hi, 0, np.maximum(n, 0)).astype(float)

        ok = (mm["mes"] >= lo[mm["gid"]]) & (mm["mes"] <= hi)
        clave = mm["gid"] * 12 + mm["mes"] % 12
        s = np.bincount(clave, weights=np.where(ok, mm["venta"], 0.0),
                        minlength=self.G * 12).reshape(self.G, 12)
        ss = np.bincount(clave, weights=np.where(ok, mm["venta"] ** 2, 0.0),
                         minlength=self.G * 12).reshape(self.G, 12)
        return self.media_std(n, s, ss)

    # -- actividad (un ajuste por segmento, o uno para todo el panel) ---------- #
    @cached_property
    def actividad(self) -> np.ndarray:
        """P(activo) por grupo. Sin segmentos, un modelo para todo el panel. Con segmentos,
        cada uno usa sus propios parámetros; si es chico (min_clientes_segmento) o su
        ajuste quedó en el borde, usa los del panel completo (GLOBAL)."""
        cfg = self.cfg
        modelo = cfg.modelo_actividad
        nombres = _NOMBRES_PARAMS[modelo]
        x = self.dias_compra - 1
        tx = (self.ultima - self.primera).astype(float)
        T = (self.f.d_ayer - self.primera).astype(float)
        self.parametros_actividad: Dict[str, Dict[str, float]] = {}
        self.detalle_actividad: List[Dict[str, Any]] = []
        self.segmento_usado = np.full(self.G, "GLOBAL", dtype=object)
        prob = np.empty(self.G)

        def ajustar(nombre: str, mascara: Optional[np.ndarray], hilos: Optional[int] = cfg.hilos_actividad):
            muestra = self.muestra_ajuste(mascara)
            info: Dict[str, Any] = {}
            t0 = time.time()
            params = ajustar_actividad(x[muestra], tx[muestra], T[muestra], modelo,
                                       hilos=hilos, info=info, etiqueta=nombre)
            LOGGER.info("%s [%s] ajustado sobre %s de %s grupos en %.1fs: %s", MODELOS_ACTIVIDAD[modelo],
                        nombre, f"{int(muestra.sum()):,}",
                        f"{self.G if mascara is None else int(mascara.sum()):,}", time.time() - t0,
                        ", ".join(f"{k}={v:.4f}" for k, v in zip(nombres, params)))
            return params, info

        def globales():
            if "GLOBAL" not in self.parametros_actividad:
                params, _ = ajustar("GLOBAL", None)
                self.parametros_actividad["GLOBAL"] = dict(zip(nombres, params))
            return tuple(self.parametros_actividad["GLOBAL"].values())

        def anotar(segmento, n, usado, motivo, params):
            fila = {"segmento": segmento, "grupos": n, "parametros": usado, "motivo": motivo,
                    **dict(zip(nombres, params))}
            if modelo == "pareto":
                r, alpha, s, beta = params
                fila["compra_cada_dias"] = alpha / r       # 1 / tasa media de compra
                fila["vida_media_dias"] = beta / s          # 1 / tasa media de abandono
            self.detalle_actividad.append(fila)

        if self.segmento is None:
            params = globales()
            prob[:] = prob_activo(params, x, tx, T, modelo, hilos=cfg.hilos_actividad)
            anotar("GLOBAL", self.G, "GLOBAL", "", params)
            return prob

        codigos, etiquetas = pd.factorize(self.segmento, sort=True)
        orden = np.argsort(codigos, kind="stable")
        bordes = np.r_[0, np.cumsum(np.bincount(codigos, minlength=len(etiquetas)))]
        miembros = [orden[bordes[k]:bordes[k + 1]] for k in range(len(etiquetas))]
        grandes = [k for k, idx in enumerate(miembros) if len(idx) >= cfg.min_clientes_segmento]

        # los segmentos se ajustan en paralelo, repartiendo los CPUs entre ellos
        cpus = cfg.hilos_actividad or _cpus_disponibles()
        por_ajuste = max(1, cpus // max(len(grandes), 1))

        def ajustar_segmento(k):
            mascara = np.zeros(self.G, dtype=bool)
            mascara[miembros[k]] = True
            return ajustar(etiquetas[k], mascara, por_ajuste)

        if cpus > 1 and len(grandes) > 1:
            with ThreadPoolExecutor(min(cpus, len(grandes))) as pool:
                ajustes = dict(zip(grandes, pool.map(ajustar_segmento, grandes)))
        else:
            ajustes = {k: ajustar_segmento(k) for k in grandes}

        for k, seg in enumerate(etiquetas):
            idx = miembros[k]
            n = len(idx)
            if k not in ajustes:
                motivo = f"{n:,} grupos, menos que min_clientes_segmento ({cfg.min_clientes_segmento:,})"
            else:
                params, info = ajustes[k]
                motivo =(f"ajuste en el borde ({', '.join(info['en_borde'])}): el segmento casi no "
                          f"muestra abandono, o todos abandonan") if info["en_borde"] else ""
            if motivo:
                LOGGER.info("segmento [%s] usa los parámetros GLOBAL: %s", seg, motivo)
                params, usado = globales(), "GLOBAL"
            else:
                self.parametros_actividad[seg] = dict(zip(nombres, params))
                usado = seg
            prob[idx] = prob_activo(params, x[idx], tx[idx], T[idx], modelo, hilos=cfg.hilos_actividad)
            self.segmento_usado[idx] = usado
            anotar(seg, n, usado, motivo, params)
        return prob

    def muestra_ajuste(self, mascara: Optional[np.ndarray] = None) -> np.ndarray:
        """Grupos que participan del ajuste (dentro de `mascara`, si viene). La pertenencia
        sale de un hash de las claves, no de un sorteo: un mismo cliente queda dentro o
        fuera todos los días, así la probabilidad no oscila sólo porque cambió la muestra."""
        todos = np.ones(self.G, dtype=bool) if mascara is None else mascara
        n = int(todos.sum())
        n_max = self.cfg.max_clientes_ajuste
        if not n_max or n <= n_max or self.hash_grupo is None:
            return todos
        u = (self.hash_grupo >> np.uint64(11)).astype(np.float64) / float(2 ** 53)
        return todos & (u < n_max / n)


# --------------------------------------------------------------------------- #
# Ventanas de fechas
# --------------------------------------------------------------------------- #
def rango_ventana(f: Fechas, cfg: StatsConfig, ventana: str, ap: bool = False) -> Tuple[int, int]:
    """(día inicial, día final) inclusive. Si inicial > final la ventana está vacía.

    YTD       1-ene del año en curso -> ayer
    YTD_MC    1-ene del año en curso -> fin del último mes cerrado
    Rn        últimos N días hasta ayer (N = dias_ventana)
    Rn_MC     últimos N meses cerrados completos (N = meses_ventana)
    ap=True   la misma ventana un año atrás (calendario para YTD y MC, 365 días para Rn)
    """
    anio = f.hoy.year - (1 if ap else 0)
    if ventana == "YTD":
        fin = f.ayer - pd.DateOffset(years=1) if ap else f.ayer
        return _dia(pd.Timestamp(year=anio, month=1, day=1)), _dia(fin)
    if ventana == "YTD_MC":
        fin = _inicio_mes(f.m_actual - (12 if ap else 0)) - pd.Timedelta(days=1)
        return _dia(pd.Timestamp(year=anio, month=1, day=1)), _dia(fin)
    if ventana.endswith("_MC"):
        n = cfg.meses_ventana[ventana[:-3]]
        corr = 12 if ap else 0
        ini = _inicio_mes(f.m_actual - n - corr)
        fin = _inicio_mes(f.m_actual - corr) - pd.Timedelta(days=1)
        return _dia(ini), _dia(fin)
    n = cfg.dias_ventana[ventana]
    fin = f.d_ayer - (365 if ap else 0)
    return fin - n + 1, fin


# =========================================================================== #
# MÉTRICAS
# Cada función recibe el contexto y devuelve un array de largo G.
# =========================================================================== #

# ── A. Venta por ventana ────────────────────────────────────────────────────
def venta_ventana(ctx: Contexto, ventana: str, ap: bool = False) -> np.ndarray:
    ini, fin = rango_ventana(ctx.f, ctx.cfg, ventana, ap)
    return ctx.suma(ctx.gid, ctx.venta, (ctx.dia >= ini) & (ctx.dia <= fin), neta=True)


# ── B. Comportamiento por día de compra ─────────────────────────────────────
# "Día de compra" = día con registro en la fuente, incluidas las devoluciones.
def max_money(ctx: Contexto) -> np.ndarray:
    return ctx.reducir(ctx.gid, ctx.venta, np.maximum)


def min_money(ctx: Contexto) -> np.ndarray:
    return ctx.reducir(ctx.gid, ctx.venta, np.minimum)


def max_days(ctx: Contexto) -> np.ndarray:
    """Máximo de días SIN comprar entre dos compras (consecutivos = 0)."""
    iv = ctx.intervalos
    return ctx.reducir(iv["gid"], iv["dias"] - 1, np.maximum)


def min_days(ctx: Contexto) -> np.ndarray:
    iv = ctx.intervalos
    return ctx.reducir(iv["gid"], iv["dias"] - 1, np.minimum)


def rango_dias(ctx: Contexto) -> np.ndarray:
    return (ctx.ultima - ctx.primera).astype(float)


def dias_sin_compra(ctx: Contexto) -> np.ndarray:
    return (ctx.f.d_ayer - ctx.ultima).astype(float)


def dias_comprados(ctx: Contexto) -> np.ndarray:
    return ctx.dias_compra


def volumen_compra(ctx: Contexto) -> np.ndarray:
    return ctx.suma(ctx.gid, ctx.venta, neta=True)


def _intervalos_en_ventana(ctx: Contexto, ventana: Optional[str]):
    iv = ctx.intervalos
    if ventana is None:
        return iv, None
    ini, fin = rango_ventana(ctx.f, ctx.cfg, ventana)
    return iv, (iv["desde"] >= ini) & (iv["hasta"] <= fin)


def frecuencia_compra(ctx: Contexto, ventana: Optional[str] = None) -> np.ndarray:
    """Cada cuántos días compra: promedio de días entre compras (diario = 1).

    En una ventana cuentan sólo los intervalos con ambos días dentro de ella.
    """
    iv, m = _intervalos_en_ventana(ctx, ventana)
    n = ctx.conteo(iv["gid"], m)
    return ctx.media_std(n, ctx.suma(iv["gid"], iv["dias"], m), n)[0]


def frecuencia_compra_std(ctx: Contexto, ventana: Optional[str] = None) -> np.ndarray:
    iv, m = _intervalos_en_ventana(ctx, ventana)
    n = ctx.conteo(iv["gid"], m)
    return ctx.media_std(n, ctx.suma(iv["gid"], iv["dias"], m),
                         ctx.suma(iv["gid"], iv["dias"] ** 2, m))[1]


def _dias_en_ventana(ctx: Contexto, ventana: Optional[str]):
    if ventana is None:
        return None
    ini, fin = rango_ventana(ctx.f, ctx.cfg, ventana)
    return (ctx.dia >= ini) & (ctx.dia <= fin)


def ticket_promedio(ctx: Contexto, ventana: Optional[str] = None) -> np.ndarray:
    """Venta por día de compra."""
    m = _dias_en_ventana(ctx, ventana)
    return ctx.media_std(ctx.conteo(ctx.gid, m), ctx.suma(ctx.gid, ctx.venta, m, neta=True),
                         ctx.dias_compra)[0]


def ticket_promedio_std(ctx: Contexto, ventana: Optional[str] = None) -> np.ndarray:
    m = _dias_en_ventana(ctx, ventana)
    return ctx.media_std(ctx.conteo(ctx.gid, m), ctx.suma(ctx.gid, ctx.venta, m),
                         ctx.suma(ctx.gid, ctx.venta ** 2, m))[1]


# ── C. Tendencias ───────────────────────────────────────────────────────────
def _angulo(pendiente: np.ndarray, unidad: str) -> np.ndarray:
    a = np.arctan(pendiente)
    return {"gon": a * 200 / np.pi, "grados": a * 180 / np.pi, "radianes": a}[unidad]


def _pendiente_venta(ctx: Contexto, meses: Optional[int], normalizar: bool) -> np.ndarray:
    """Pendiente MCO de la venta mensual contra el número de mes.

    Meses cerrados desde la primera compra (o los últimos `meses`), con ceros.
    x va de 0 a n-1 dentro de la ventana de cada grupo, para no perder precisión.
    """
    mm = ctx.mensual
    hi = ctx.f.m_mc
    lo = ctx.primer_mes if meses is None else np.maximum(ctx.primer_mes, hi - meses + 1)
    n = (hi - lo + 1).astype(float)

    lo_fila = lo[mm["gid"]]
    ok = (mm["mes"] >= lo_fila) & (mm["mes"] <= hi)
    x = (mm["mes"] - lo_fila).astype(float)
    sy = ctx.suma(mm["gid"], mm["venta"], ok, neta=True)
    sxy = ctx.suma(mm["gid"], x * mm["venta"], ok)
    sx = n * (n - 1) / 2
    sxx = (n - 1) * n * (2 * n - 1) / 6
    with np.errstate(divide="ignore", invalid="ignore"):
        den = n * sxx - sx * sx
        p = np.where((n >= 2) & (den > 0), (n * sxy - sx * sy) / den, np.nan)
        if normalizar:      # sy neta: el grupo que vende y devuelve todo no tiene promedio
            media = sy / np.where(n > 0, n, 1)
            p = np.where(media > 0, p / media, np.nan)
    return p


def idd_pendiente(ctx: Contexto) -> np.ndarray:
    """Pendiente cruda, USD por mes, sobre toda la historia."""
    return _pendiente_venta(ctx, None, normalizar=False)


def idd_porcentual(ctx: Contexto, ventana: Optional[str] = None) -> np.ndarray:
    meses = None if ventana is None else ctx.cfg.meses_ventana[ventana]
    return _angulo(_pendiente_venta(ctx, meses, ctx.cfg.idd_normalizar), ctx.cfg.idd_unidad)


def idd_porcentual_margen(ctx: Contexto, ventana: Optional[str] = None) -> np.ndarray:
    """Tendencia del margen % mensual. Sólo meses con venta positiva (el % no
    está definido sin venta). Positivo = el cliente se vuelve más rentable."""
    mm = ctx.mensual
    hi = ctx.f.m_mc
    meses = None if ventana is None else ctx.cfg.meses_ventana[ventana]
    lo = ctx.primer_mes if meses is None else np.maximum(ctx.primer_mes, hi - meses + 1)
    lo_fila = lo[mm["gid"]]
    ok = (mm["mes"] >= lo_fila) & (mm["mes"] <= hi) & (mm["venta"] > 0)   # venta ya limpia
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(ok, ctx.cfg.margen_escala * mm["margen"] / mm["venta"], 0.0)
    x = (mm["mes"] - lo_fila).astype(float)
    g = mm["gid"]
    n = ctx.conteo(g, ok)
    sx, sy = ctx.suma(g, x, ok), ctx.suma(g, y, ok)
    sxx, sxy = ctx.suma(g, x * x, ok), ctx.suma(g, x * y, ok)
    with np.errstate(divide="ignore", invalid="ignore"):
        den = n * sxx - sx * sx
        p = np.where((n >= 2) & (den > 1e-12), (n * sxy - sx * sy) / den, np.nan)
    return _angulo(p, ctx.cfg.idd_unidad)


# ── D. Margen ───────────────────────────────────────────────────────────────
def margen_bruto(ctx: Contexto, ventana: Optional[str] = None) -> np.ndarray:
    """Margen ponderado: suma de margen / suma de venta, en %.

    La venta va neta (`sin_residuo`): el grupo que vendió y devolvió lo mismo tiene
    venta cero, no un residuo de 1e-11, y su margen % queda nulo en vez de explotar.
    """
    m = _dias_en_ventana(ctx, ventana)
    v = ctx.suma(ctx.gid, ctx.venta, m, neta=True)
    mg = ctx.suma(ctx.gid, ctx.margen, m, neta=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v != 0, ctx.cfg.margen_escala * mg / v, np.nan)


# ── E. Años ─────────────────────────────────────────────────────────────────
def anio_inicial(ctx: Contexto) -> np.ndarray:
    return ctx.anio_fila[ctx.inicios].astype(float)


def venta_anio_inicial(ctx: Contexto) -> np.ndarray:
    return ctx.suma(ctx.gid, ctx.venta, ctx.anio_fila == ctx.anio_fila[ctx.inicios][ctx.gid],
                    neta=True)


def anio_final_cerrado(ctx: Contexto) -> np.ndarray:
    m = ctx.anio_fila < ctx.f.hoy.year
    return ctx.reducir(ctx.gid[m], ctx.anio_fila[m], np.maximum)


def venta_anio_final_cerrado(ctx: Contexto) -> np.ndarray:
    fin = anio_final_cerrado(ctx)
    return np.where(np.isnan(fin), np.nan,
                    ctx.suma(ctx.gid, ctx.venta, ctx.anio_fila == fin[ctx.gid], neta=True))


# ── F. Estacionalidad ───────────────────────────────────────────────────────
def promedio_mes(ctx: Contexto, mes: int) -> np.ndarray:
    """mes: 1..12"""
    return ctx.estacional[0][:, mes - 1]


def std_mes(ctx: Contexto, mes: int) -> np.ndarray:
    return ctx.estacional[1][:, mes - 1]


def _extremo_mensual(ctx: Contexto, maximo: bool):
    prom = ctx.estacional[0]
    vacio = np.isnan(prom).all(axis=1)
    relleno = np.where(np.isnan(prom), -np.inf if maximo else np.inf, prom)
    idx = relleno.argmax(axis=1) if maximo else relleno.argmin(axis=1)
    valor = np.where(vacio, np.nan, prom[np.arange(ctx.G), idx])
    nombre = np.where(vacio, None, np.array(MESES_ES, dtype=object)[idx])
    return nombre, valor


def max_mensual_nombre(ctx: Contexto) -> np.ndarray:
    return _extremo_mensual(ctx, True)[0]


def max_mensual(ctx: Contexto) -> np.ndarray:
    return _extremo_mensual(ctx, True)[1]


def min_mensual_nombre(ctx: Contexto) -> np.ndarray:
    return _extremo_mensual(ctx, False)[0]


def min_mensual(ctx: Contexto) -> np.ndarray:
    return _extremo_mensual(ctx, False)[1]


def venta_mes_actual_esperada(ctx: Contexto) -> np.ndarray:
    return promedio_mes(ctx, ctx.f.hoy.month)


def venta_siguiente_mes_esperada(ctx: Contexto) -> np.ndarray:
    return promedio_mes(ctx, ctx.f.hoy.month % 12 + 1)


def venta_cumplida_actual(ctx: Contexto) -> np.ndarray:
    ini = _dia(_inicio_mes(ctx.f.m_actual))
    return ctx.suma(ctx.gid, ctx.venta, (ctx.dia >= ini) & (ctx.dia <= ctx.f.d_ayer), neta=True)


# ── G. Actividad ────────────────────────────────────────────────────────────
MODELOS_ACTIVIDAD = {"bgnbd": "BG/NBD", "mbgnbd": "MBG/NBD", "pareto": "Pareto/NBD"}
_NOMBRES_PARAMS = {"bgnbd": ("r", "alpha", "a", "b"), "mbgnbd": ("r", "alpha", "a", "b"),
                   "pareto": ("r", "alpha", "s", "beta")}
_MIN_FILAS_HILOS = 20_000       # por debajo, repartir en hilos cuesta más de lo que ahorra


def _cpus_disponibles() -> int:
    """CPUs usables: afinidad del proceso acotada por el límite del contenedor.
    En un pod de OpenShift os.cpu_count() devuelve los CPUs del nodo, no los del pod."""
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 1
    try:
        cuota, periodo = open("/sys/fs/cgroup/cpu.max").read().split()
        if cuota != "max":
            n = min(n, max(1, int(float(cuota) / float(periodo))))
    except Exception:
        pass
    try:
        cuota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        periodo = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if cuota > 0:
            n = min(n, max(1, cuota // periodo))
    except Exception:
        pass
    return max(1, n)


def _mapear(fn, arrays, pool, hilos: int) -> np.ndarray:
    """fn(*arrays) evaluada por tramos contiguos en el pool. scipy.special libera el
    GIL, así que la hipergeométrica escala con los núcleos."""
    if pool is None:
        return fn(*arrays)
    bordes = np.linspace(0, len(arrays[0]), hilos + 1).astype(int)
    tramos = [(bordes[i], bordes[i + 1]) for i in range(hilos) if bordes[i + 1] > bordes[i]]
    return np.concatenate(list(pool.map(lambda t: fn(*(a[t[0]:t[1]] for a in arrays)), tramos)))


def prob_activo_col(ctx: Contexto) -> np.ndarray:
    return ctx.actividad


def prob_inactivo_col(ctx: Contexto) -> np.ndarray:
    return 1.0 - ctx.actividad


def segmento_actividad_col(ctx: Contexto) -> np.ndarray:
    ctx.actividad
    return ctx.segmento_usado


def _log_a0_pareto(r, alpha, s, beta, x, tx, T):
    """log A0 de Pareto/NBD (Fader, Hardie & Lee 2005, nota de implementación).

    La hipergeométrica 2F1(a, b; a+1; z) desborda con clientes de muchas compras.
    Se usa la transformación de Euler 2F1(a,b;a+1;z) = (1-z)^(1-b) 2F1(1, a+1-b; a+1; z),
    cuyo segundo factor está acotado por 1/(1-z), y todo se opera en logaritmos.
    """
    from scipy.special import hyp2f1
    if alpha >= beta:
        mayor, b = alpha, s + 1.0
    else:
        mayor, b = beta, r + x
    dif = abs(alpha - beta)
    a = r + s + x

    def log_termino(t):
        z = dif / (mayor + t)
        return ((1.0 - b) * np.log1p(-z) + np.log(hyp2f1(1.0, a + 1.0 - b, a + 1.0, z))
                - a * np.log(mayor + t))

    t1, t2 = log_termino(tx), log_termino(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        return t1 + np.log(-np.expm1(np.minimum(t2 - t1, 0.0)))     # log(e^t1 - e^t2)


def _log_verosimilitud(p, x, tx, T, modelo):
    from scipy.special import gammaln
    if modelo == "pareto":
        r, alpha, s, beta = p
        a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha) + s * np.log(beta)
        a2 = np.logaddexp(-(r + x) * np.log(alpha + T) - s * np.log(beta + T),
                          np.log(s) + _log_a0_pareto(r, alpha, s, beta, x, tx, T) - np.log(r + s + x))
        return a1 + a2
    r, alpha, a, b = p
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a3 = -(r + x) * np.log(alpha + T)
    if modelo == "mbgnbd":
        a2 = gammaln(a + b) + gammaln(b + x + 1) - gammaln(b) - gammaln(a + b + x + 1)
        a4 = np.log(a) - np.log(b + x) - (r + x) * np.log(alpha + tx)
    else:
        a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        a4 = np.where(x > 0, np.log(a) - np.log(np.maximum(b + x - 1, 1e-300))
                      - (r + x) * np.log(alpha + tx), -np.inf)
    return a1 + a2 + np.logaddexp(a3, a4)


def ajustar_actividad(x, tx, T, modelo: str = "pareto", hilos: Optional[int] = None,
                      info: Optional[dict] = None, etiqueta: str = "") -> Tuple[float, float, float, float]:
    """Ajusta el modelo de actividad por máxima verosimilitud sobre los grupos recibidos.
    Si viene `info`, deja ahí `convergio` y `en_borde` (parámetros pegados al límite).

    x  = compras repetidas (días de compra - 1)
    tx = días entre la primera y la última compra
    T  = días entre la primera compra y el fin de la observación

    Devuelve (r, alpha, a, b) para BG/NBD y MBG/NBD, o (r, alpha, s, beta) para
    Pareto/NBD. Los clientes con el mismo (x, tx, T) se agrupan y pesan por su
    cantidad: el costo depende de las combinaciones distintas, no de los clientes.
    """
    from scipy.optimize import minimize
    x, tx, T = (np.asarray(v, float) for v in (x, tx, T))
    escala = max(float(np.max(T)) / 10.0, 1.0)          # condiciona el optimizador
    unicos, peso = np.unique(np.stack([x, tx / escala, T / escala], axis=1), axis=0, return_counts=True)
    ux, utx, uT = unicos.T
    peso = peso.astype(float)

    hilos = hilos or _cpus_disponibles()
    pool = ThreadPoolExecutor(hilos) if hilos > 1 and len(ux) >= _MIN_FILAS_HILOS else None

    def nll(logp):
        q = np.exp(logp)

        def tramo(a, b, c):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):   # errstate es por hilo
                return _log_verosimilitud(q, a, b, c, modelo)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            valor = -np.sum(peso * _mapear(tramo, (ux, utx, uT), pool, hilos)) / peso.sum()
        return valor if np.isfinite(valor) else 1e300

    try:
        res = minimize(nll, np.zeros(4), method="L-BFGS-B", bounds=[(-12.0, 12.0)] * 4)
    finally:
        if pool is not None:
            pool.shutdown()
    donde = f" [{etiqueta}]" if etiqueta else ""
    if not res.success:
        LOGGER.warning("el ajuste de actividad%s no convergió del todo: %s", donde, res.message)
    en_borde = [n for n, b in zip(_NOMBRES_PARAMS[modelo], np.abs(res.x) > 11.5) if b]
    if info is not None:
        info.update(convergio=bool(res.success), en_borde=en_borde)
    if en_borde:
        LOGGER.warning("el ajuste de actividad%s quedó en el borde (%s): los datos casi no muestran "
                       "abandono, o todos abandonan, y las probabilidades se concentran en 0 o 1",
                       donde, ", ".join(en_borde))
    q = np.exp(res.x)
    if modelo == "pareto":
        return float(q[0]), float(q[1] * escala), float(q[2]), float(q[3] * escala)
    return float(q[0]), float(q[1] * escala), float(q[2]), float(q[3])


def _prob_activo(params, x, tx, T, modelo: str) -> np.ndarray:
    from scipy.special import expit
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if modelo == "pareto":
            r, alpha, s, beta = params
            t = (np.log(s) + _log_a0_pareto(r, alpha, s, beta, x, tx, T) - np.log(r + s + x)
                 + (r + x) * np.log(alpha + T) + s * np.log(beta + T))
            return expit(-t)
        r, alpha, a, b = params
        ratio = (r + x) * (np.log(alpha + T) - np.log(alpha + tx))
        if modelo == "mbgnbd":
            return expit(-(np.log(a) - np.log(b + x) + ratio))
        t = np.log(a) - np.log(np.maximum(b + x - 1, 1e-300)) + ratio
        return np.where(x > 0, expit(-t), 1.0)


def prob_activo(params, x, tx, T, modelo: str = "pareto", hilos: Optional[int] = None) -> np.ndarray:
    """P(sigue activo | x, tx, T) según el modelo. La explicación está en el notebook."""
    x, tx, T = (np.asarray(v, float) for v in (x, tx, T))
    hilos = hilos or _cpus_disponibles()
    if modelo != "pareto" or hilos <= 1 or len(x) < _MIN_FILAS_HILOS:
        return _prob_activo(params, x, tx, T, modelo)
    with ThreadPoolExecutor(hilos) as pool:
        return _mapear(lambda a, b, c: _prob_activo(params, a, b, c, modelo), (x, tx, T), pool, hilos)


# ── H. Historial ────────────────────────────────────────────────────────────
def historial_json(ctx: Contexto) -> np.ndarray:
    """JSON por grupo: {"f0": primera compra, "d": días desde f0, "v": venta, "m": margen}.

    Se arma en bloque: cada array se pasa a texto una sola vez, se concatena con
    "|" como separador entre grupos y se corta. Evita un loop de miles de joins.
    """
    if not ctx.cfg.incluir_historial:
        return np.full(ctx.G, None, dtype=object)
    dec = ctx.cfg.decimales_historial
    fin = np.r_[ctx.gid[1:] != ctx.gid[:-1], True]
    sep = np.where(fin, "|", ",")

    def bloque(textos: np.ndarray) -> List[str]:
        return "".join(np.char.add(textos, sep).tolist()).split("|")[:-1]

    d = bloque((ctx.dia - ctx.primera[ctx.gid]).astype(np.int64).astype(str))
    v = bloque(np.round(np.nan_to_num(ctx.venta), dec).astype(str))
    m = bloque(np.round(np.nan_to_num(ctx.margen), dec).astype(str))
    f0 = np.datetime_as_string(ctx.primera.astype("datetime64[D]"))
    return np.array([f'{{"f0":"{a}","d":[{b}],"v":[{c}],"m":[{e}]}}'
                     for a, b, c, e in zip(f0, d, v, m)], dtype=object)


def historial_a_filas(estado: pd.DataFrame, cfg: StatsConfig) -> pd.DataFrame:
    """Inversa de BD_HISTORIAL: una fila por grupo-día con las columnas de entrada.

    `estado` trae las categorías y BD_HISTORIAL. Es lo que usa la corrida
    incremental para no volver a leer toda la fuente.
    """
    cats = list(cfg.categorias)
    textos = estado["BD_HISTORIAL"].tolist()
    n = np.zeros(len(textos), dtype=np.int64)
    f0 = np.zeros(len(textos), dtype="datetime64[D]")
    d: List[int] = []
    v: List[float] = []
    m: List[float] = []
    for i, texto in enumerate(textos):
        h = json.loads(texto)
        n[i] = len(h["d"])
        f0[i] = np.datetime64(h["f0"], "D")
        d.extend(h["d"])
        v.extend(h["v"])
        m.extend(h["m"])
    fila = np.repeat(np.arange(len(textos)), n)
    salida = {c: estado[c].to_numpy()[fila] for c in cats}
    salida[cfg.col_fecha] = (np.repeat(f0, n) + np.asarray(d, dtype="timedelta64[D]")).astype("datetime64[ns]")
    salida[cfg.col_venta] = np.asarray(v, dtype=float)
    salida[cfg.col_margen] = np.asarray(m, dtype=float)
    return pd.DataFrame(salida)


# =========================================================================== #
# CATÁLOGO: qué columnas salen, con qué descripción y qué función las calcula
# =========================================================================== #
@dataclass(frozen=True)
class Metrica:
    nombre: str
    tipo: str
    descripcion: str
    funcion: Callable[..., np.ndarray]
    args: Tuple[Tuple[str, Any], ...] = ()

    def calcular(self, ctx: Contexto) -> np.ndarray:
        return self.funcion(ctx, **dict(self.args))


def catalogo(cfg: Optional[StatsConfig] = None) -> List[Metrica]:
    """Todas las columnas de salida, en el orden de la especificación."""
    cfg = cfg or StatsConfig(categorias=["SK_CLIENTE"])
    u = {"gon": "gradianes (0 plano, 100 creciente, -100 decreciente)",
         "grados": "grados (0 plano, 90 creciente, -90 decreciente)",
         "radianes": "radianes"}[cfg.idd_unidad]
    M: List[Metrica] = []

    def add(nombre, desc, fn, tipo="NUMBER", **kw):
        M.append(Metrica(nombre, tipo, desc, fn, tuple(kw.items())))

    def variantes(base, desc_hist, desc_var, fn):
        add(base, desc_hist, fn)
        for v in cfg.variantes:
            add(f"{base}_{v}", desc_var(v), fn, ventana=v)

    d = cfg.dias_ventana
    # A. Venta por ventana
    ventanas = [
        ("YTD", "del año en curso, desde el 1 de enero hasta ayer"),
        ("YTD_MC", "del año en curso a mes cerrado, desde el 1 de enero hasta el fin del último mes cerrado (en enero es 0)"),
        ("R12", f"de los últimos {d['R12']} días hasta ayer"),
        ("R12_MC", "de los últimos 12 meses cerrados completos"),
        ("R6", f"de los últimos {d['R6']} días hasta ayer"),
        ("R6_MC", "de los últimos 6 meses cerrados completos"),
    ]
    for v, desc in ventanas:
        add(f"MT_VENTA_{v}", f"Venta {desc}. USD.", venta_ventana, ventana=v)
        add(f"MT_VENTA_{v}_AP", f"Año pasado de MT_VENTA_{v}: la misma ventana corrida un año atrás. USD.",
            venta_ventana, ventana=v, ap=True)

    # B. Comportamiento
    add("MT_MAXMONEY", "Máxima venta en un día de compra (suma del día). USD.", max_money)
    add("MT_MINMONEY", "Mínima venta en un día de compra; negativa si hubo devolución neta. USD.", min_money)
    add("MT_MAXDAYS", "Máximo de días sin comprar entre dos días de compra (dos días seguidos = 0). Nulo con un solo día de compra.", max_days)
    add("MT_MINDAYS", "Mínimo de días sin comprar entre dos días de compra (dos días seguidos = 0). Nulo con un solo día de compra.", min_days)
    add("MT_RANGODIAS", "Días entre el primer y el último día de compra.", rango_dias)
    add("MT_DIASNCOMPRA", "Días desde el último día de compra hasta ayer (compró ayer = 0).", dias_sin_compra)

    # C. Tendencia de venta
    variantes("MT_IDDPORCENTUAL",
              f"Pendiente de la recta de venta mensual (meses cerrados desde la primera compra, sin compra = 0), expresada en {u}. Histórico.",
              lambda v: f"Pendiente de la recta de venta mensual de los últimos {cfg.meses_ventana[v]} meses cerrados, expresada en {u}.",
              idd_porcentual)
    add("MT_IDDPENDIENTE", "Pendiente de la recta de venta mensual en USD por mes, toda la historia (meses cerrados, sin compra = 0).", idd_pendiente)

    # D. Margen
    add("MT_MARGENBRUTO", "Margen bruto histórico en %: suma del margen / suma de la venta.", margen_bruto)
    add("MT_MARGENBRUTO_R12", f"Margen bruto en % de los últimos {d['R12']} días: suma del margen / suma de la venta.",
        margen_bruto, ventana="R12")

    # B (cont.)
    add("MT_DIASCOMPRADOS", "Cantidad de días distintos con compra, incluidos días con devolución.", dias_comprados)
    add("MT_VOLUMENCOMPRA", "Venta total de toda la historia. USD.", volumen_compra)

    variantes("MT_FRECUENCIACOMPRA",
              "Cada cuántos días compra en promedio: promedio de días entre días de compra consecutivos (compra diaria = 1). Histórico.",
              lambda v: f"Cada cuántos días compra en promedio, contando sólo intervalos con ambos días dentro de los últimos {d[v]} días.",
              frecuencia_compra)
    variantes("MT_TICKETPROMEDIO",
              "Venta promedio por día de compra (venta / días de compra). Histórico. USD.",
              lambda v: f"Venta promedio por día de compra en los últimos {d[v]} días. USD.",
              ticket_promedio)
    variantes("MT_FRECUENCIACOMPRA_STD",
              "Desvío estándar de los días entre días de compra consecutivos. Histórico.",
              lambda v: f"Desvío estándar de los días entre compras, intervalos dentro de los últimos {d[v]} días.",
              frecuencia_compra_std)
    variantes("MT_TICKETPROMEDIO_STD",
              "Desvío estándar de la venta por día de compra. Histórico. USD.",
              lambda v: f"Desvío estándar de la venta por día de compra en los últimos {d[v]} días. USD.",
              ticket_promedio_std)

    # G. Actividad
    modelo = MODELOS_ACTIVIDAD[cfg.modelo_actividad]
    seg = list(cfg.segmentos_actividad)
    alcance = (f"ajustado por segmento de {', '.join(seg)}; ver BD_SEGMENTO_ACTIVIDAD" if seg
               else "ajustado sobre todo el panel")
    add("MT_PROB_ACTIVO", f"Probabilidad de que el cliente siga activo (modelo {modelo} {alcance}).", prob_activo_col)
    add("BD_SEGMENTO_ACTIVIDAD",
        (f"Parámetros usados para MT_PROB_ACTIVO: el segmento ({' | '.join(seg)}) o GLOBAL si el segmento tiene "
         f"menos de {cfg.min_clientes_segmento:,} grupos o su ajuste no fue confiable." if seg
         else "Parámetros usados para MT_PROB_ACTIVO: GLOBAL (sin segmentación)."),
        segmento_actividad_col, tipo="VARCHAR2(400)")

    # E. Años
    add("MT_ANIO_INICIAL", "Año calendario de la primera compra.", anio_inicial)
    add("MT_VENTA_ANIO_INICIAL", "Venta del año calendario de la primera compra, aunque sean pocos meses. USD.", venta_anio_inicial)
    add("MT_ANIO_FINAL_CERRADO", "Último año calendario cerrado (anterior al año en curso) con compra.", anio_final_cerrado)
    add("MT_VENTA_ANIO_FINAL_CERRADO", "Venta del último año calendario cerrado con compra. USD.", venta_anio_final_cerrado)

    # F. Estacionalidad: extremos
    add("BD_MAXMENSUAL", "Mes calendario con mayor venta mensual promedio (ENERO..DICIEMBRE).", max_mensual_nombre, tipo="VARCHAR2(20)")
    add("MT_MAXMENSUAL", "Venta mensual promedio del mejor mes calendario. USD.", max_mensual)
    add("BD_MINMENSUAL", "Mes calendario con menor venta mensual promedio (ENERO..DICIEMBRE).", min_mensual_nombre, tipo="VARCHAR2(20)")
    add("MT_MINMENSUAL", "Venta mensual promedio del peor mes calendario. USD.", min_mensual)

    # C. Tendencia de margen
    variantes("MT_IDDPORCENTUALMARGEN",
              f"Pendiente de la recta del margen % mensual (sólo meses con venta positiva), expresada en {u}. Positivo = más rentable. Histórico.",
              lambda v: f"Pendiente de la recta del margen % mensual de los últimos {cfg.meses_ventana[v]} meses cerrados, expresada en {u}.",
              idd_porcentual_margen)

    # F. Estacionalidad: promedio y desvío por mes
    for i, mes in enumerate(MESES_ES, start=1):
        add(f"MT_PROM_{mes}", f"Venta promedio de {mes.lower()} en los años de vida del cliente (meses cerrados, sin compra = 0). USD.",
            promedio_mes, mes=i)
    for i, mes in enumerate(MESES_ES, start=1):
        add(f"MT_STD_{mes}", f"Desvío estándar de la venta de {mes.lower()} en los años de vida del cliente. USD.",
            std_mes, mes=i)

    # F. Mes actual
    add("MT_VENTA_MES_ACTUAL", "Venta esperada del mes en curso: promedio histórico del mismo mes calendario. USD.", venta_mes_actual_esperada)
    add("MT_VENTA_SIGUIENTE_MES", "Venta esperada del mes siguiente: promedio histórico de ese mes calendario. USD.", venta_siguiente_mes_esperada)
    add("MT_VENTA_CUMPLIDA_ACTUAL", "Venta real del mes en curso, desde el día 1 hasta ayer. USD.", venta_cumplida_actual)

    add("MT_PROB_INACTIVO", f"Probabilidad de que el cliente esté inactivo: 1 - MT_PROB_ACTIVO ({modelo}).", prob_inactivo_col)

    # H. Historial
    add("BD_HISTORIAL", 'JSON con la historia diaria: f0 = fecha de primera compra, d = días desde f0, v = venta USD, m = margen USD.',
        historial_json, tipo="CLOB")
    return M


# =========================================================================== #
# Orquestador
# =========================================================================== #
class StatsEngine:
    def __init__(self, config: StatsConfig):
        config.validate()
        self.cfg = config
        self.fechas = Fechas.desde(config.fecha_ejecucion)
        self.metricas = catalogo(config)
        self.tiempos_: Dict[str, float] = {}
        self.ctx: Optional[Contexto] = None
        LOGGER.setLevel(logging.INFO if config.verbose else logging.WARNING)

    def catalogo(self) -> pd.DataFrame:
        """Columna, tipo, descripción y función que la calcula."""
        return pd.DataFrame([{"columna": m.nombre, "tipo": m.tipo, "descripcion": m.descripcion,
                              "funcion": m.funcion.__name__,
                              "parametros": ", ".join(f"{k}={v}" for k, v in m.args)}
                             for m in self.metricas])

    def preparar(self, df: pd.DataFrame) -> Tuple[Contexto, pd.DataFrame]:
        """Lleva la entrada a grano (grupo, día), ordenada, y arma las categorías."""
        cfg = self.cfg
        faltan = [c for c in list(cfg.categorias) + [cfg.col_fecha, cfg.col_venta, cfg.col_margen]
                  if c not in df.columns]
        if faltan:
            raise KeyError(f"Faltan columnas en la entrada: {faltan}")

        dia = pd.to_datetime(df[cfg.col_fecha]).to_numpy().astype("datetime64[D]").astype(np.int64)
        dentro = dia <= self.fechas.d_ayer
        if not dentro.all():
            LOGGER.info("se descartan %s filas posteriores a ayer (%s)",
                        f"{int((~dentro).sum()):,}", self.fechas.ayer.date())
        base = df.loc[dentro]
        dia = dia[dentro]
        if base.empty:
            raise ValueError("No quedan filas hasta ayer")

        claves = cfg.claves()
        gid = base.groupby(claves, sort=True, dropna=False).ngroup().to_numpy()
        venta = pd.to_numeric(base[cfg.col_venta], errors="coerce").fillna(0.0).to_numpy(float)
        margen = pd.to_numeric(base[cfg.col_margen], errors="coerce").fillna(0.0).to_numpy(float)

        orden = np.lexsort((dia, gid))
        g, d = gid[orden], dia[orden]
        cambio_grupo = np.r_[True, g[1:] != g[:-1]]
        ultima_fila = np.r_[g[1:] != g[:-1], True]

        # categorías: claves de la primera fila del grupo, descripciones de la más reciente
        cats = base[claves].iloc[orden[cambio_grupo]].reset_index(drop=True)
        for c in cfg.descripciones():
            cats[c] = base[c].iloc[orden[ultima_fila]].to_numpy()

        # un registro por grupo-día
        nuevo = np.flatnonzero(cambio_grupo | np.r_[True, d[1:] != d[:-1]])
        vo, mo = venta[orden], margen[orden]
        # el día que se anula con su devolución tiene que valer 0, no 1e-12
        escala = escala_decimal((vo, mo), cfg.max_decimales) if cfg.suma_exacta else 0.0
        if escala:
            v_dia = np.add.reduceat(np.round(vo * escala), nuevo) / escala
            mg_dia = np.add.reduceat(np.round(mo * escala), nuevo) / escala
            LOGGER.info("dinero en escala de %d decimales: las sumas son exactas",
                        int(round(np.log10(escala))))
        else:
            v_dia, mg_dia = np.add.reduceat(vo, nuevo), np.add.reduceat(mo, nuevo)
            if cfg.tolerancia_cero:
                bv, bm = np.add.reduceat(np.abs(vo), nuevo), np.add.reduceat(np.abs(mo), nuevo)
                v_dia = sin_residuo(v_dia, bv, cfg.tolerancia_cero, escala_tipica(bv))
                mg_dia = sin_residuo(mg_dia, bm, cfg.tolerancia_cero, escala_tipica(bm))
                LOGGER.info("sin escala decimal usable: los ceros se deciden con "
                            "tolerancia_cero=%g", cfg.tolerancia_cero)
        ctx = Contexto(cfg, self.fechas, g[nuevo], d[nuevo], v_dia, mg_dia, int(g.max()) + 1)
        ctx.escala = escala
        ctx.hash_grupo = pd.util.hash_pandas_object(cats[claves], index=False).to_numpy(np.uint64)
        if cfg.segmentos_actividad:
            etiqueta = None
            for c in cfg.segmentos_actividad:
                txt = cats[c].astype(str).where(cats[c].notna(), "(sin dato)")
                etiqueta = txt if etiqueta is None else etiqueta + " | " + txt
            ctx.segmento = etiqueta.to_numpy(dtype=object)
        return ctx, cats

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        t_inicio = time.time()
        f = self.fechas
        LOGGER.info("ejecución %s | ayer %s | mes cerrado %s | métricas %d",
                    f.hoy.date(), f.ayer.date(), _inicio_mes(f.m_mc).strftime("%Y-%m"),
                    len(self.metricas))
        t0 = time.time()
        ctx, cats = self.preparar(df)
        self.ctx = ctx
        self.tiempos_ = {"preparar": time.time() - t0}
        LOGGER.info("%s filas de entrada -> %s grupos x %s días de compra (%.1fs)",
                    f"{len(df):,}", f"{ctx.G:,}", f"{len(ctx.gid):,}", self.tiempos_["preparar"])

        salida: Dict[str, Any] = {c: cats[c].to_numpy() for c in cats.columns}
        for m in self.metricas:
            t0 = time.time()
            salida[m.nombre] = m.calcular(ctx)
            self.tiempos_[m.nombre] = time.time() - t0
        salida["FECHA_CORTE"] = np.full(ctx.G, f.ayer)

        out = pd.DataFrame(salida)
        if ctx.residuos:      # sólo cuando no hubo escala decimal
            LOGGER.info("%s sumas de dinero quedaron en cero (contando todas las columnas): "
                        "venta y devolución se anulan y lo que sobraba era residuo de punto "
                        "flotante. Sus porcentajes salen nulos (tolerancia_cero=%g)",
                        f"{ctx.residuos:,}", self.cfg.tolerancia_cero)
        LOGGER.info("estadísticas: %s filas x %d columnas en %.1fs", f"{len(out):,}",
                    out.shape[1], time.time() - t_inicio)
        return out

    def actividad_segmentos(self) -> pd.DataFrame:
        """Después de run(): parámetros de cada segmento, cuáles se usaron y por qué."""
        return pd.DataFrame(getattr(self.ctx, "detalle_actividad", []))

    def tiempos(self, top: int = 10) -> pd.Series:
        """Qué columnas tardaron más (las compartidas se cargan a la primera que las usa)."""
        return pd.Series(self.tiempos_).sort_values(ascending=False).head(top)
