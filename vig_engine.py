# -*- coding: utf-8 -*-
"""
Motor de vigilancia de métricas.

Toma cualquier serie (una métrica, por categoría, al grano que quieras), le busca
anomalías con una batería de detectores, les pone un nivel de gravedad, las agrupa
en eventos con identidad estable y arma el resumen de notificaciones.

Genérico: cada vigilancia se declara con su propia consulta, sus categorías y su
métrica. El motor no sabe si mira ventas, devoluciones o litros.

Estructura del archivo
  1. Configuración       Vigilancia, VigConfig
  2. Series              armado por grano, historia y merge incremental
  3. Detectores          la batería (hueco, salto, escalón, tendencia, ...)
  4. Gravedad y eventos  nivel, identidad estable, estado y atribución
  5. Orquestador         VigEngine.run()
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("vig")

_EPOCA = pd.Timestamp("1970-01-01")
NIVELES = ("INFO", "ATENCION", "ALERTA", "CRITICO")
GRANOS = ("dia", "semana", "mes")


def _dia(ts) -> int:
    return int((pd.Timestamp(ts).normalize() - _EPOCA).days)


def sin_residuo(suma, suma_abs, tol: float, piso: float = 0.0):
    """Anula la suma cuando no se distingue del residuo de cancelación de float64.

    Sumar un importe y su reverso no da cero exacto: queda un residuo del orden de
    1e-16 veces lo que pasó por la suma. En una serie eso convierte un período de
    valor cero en uno de 1e-10, y si además es el denominador de un ratio, el punto
    se dispara a millones y la vigilancia grita por nada. El cero se prueba contra la
    escala real del período: la suma de los valores absolutos.

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
    un importe y su reverso den cero exacto, sin tolerancias ni umbrales. Es lo mismo
    que hace Oracle con NUMBER, y cuesta lo mismo que sumar en float.

    Se elige la escala más chica que sirva y que no desborde 2^53 con el bruto. Si los
    valores traen más decimales de los buscados, se usa el máximo y el resto se
    redondea. Devuelve 0 cuando no hay escala usable: ahí entra `sin_residuo`.
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


def _mediana(A: np.ndarray, axis: int = 1) -> np.ndarray:
    """Mediana ignorando nulos. Una franja toda nula da nulo y no avisa por consola: en
    una serie con huecos eso es lo normal, no un problema."""
    A = np.asarray(A, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(A, axis=axis)


def escala_tipica(suma_abs) -> float:
    """Escala de referencia del panel: la mediana de lo que movió cada grupo."""
    a = np.asarray(suma_abs, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.median(a)) if a.size else 0.0


# =========================================================================== #
# 1. Configuración
# =========================================================================== #
@dataclass
class Vigilancia:
    """Una métrica vigilada: de dónde sale, cómo se agrupa y qué se le mira."""

    nombre: str                                  #: identifica la vigilancia (va en las tablas)
    sql: str                                     #: consulta con :desde y :hasta
    categorias: Sequence[str]                    #: una serie por combinación de valores
    metrica: str = "MT_VALOR"                    #: columna con el valor
    descripcion: str = ""                        #: para el catálogo y las notificaciones
    unidad: str = "USD"                          #: USD, unidades, galones, %, lo que sea
    agregacion: str = "suma"                     #: suma | promedio | conteo | ratio
    denominador: Optional[str] = None            #: sólo para agregacion="ratio"
    #: denominador mínimo para que el ratio tenga sentido, en la unidad del denominador.
    #: Un período con 0,0000064 de base y 25 arriba da 390.000.000: correcto y sin
    #: sentido, y dispara una alerta crítica. Por debajo, el período queda nulo (hueco).
    min_denominador: float = 0.0
    granos: Sequence[str] = ("dia", "mes")       #: en qué granos se vigila
    detectores: Optional[Sequence[str]] = None   #: None = todos los del config
    #: columna con la que se explica una anomalía (quién la causó). Tiene que venir en el SQL.
    atribucion: Optional[str] = None
    #: por debajo de esta materialidad (en unidad de la métrica) el evento no pasa de INFO
    materialidad_minima: float = 0.0
    #: dirección que importa: "ambas", "baja" (sólo caídas) o "sube" (sólo subas)
    direccion: str = "ambas"
    #: "auto" usa escala logarítmica si la métrica es positiva. El ruido de una venta es
    #: multiplicativo (±20%), no de tantos USD: medirlo en línea recta infla las colas y
    #: llena de falsas alarmas. "lineal" para métricas que pueden ser negativas.
    escala: str = "auto"
    #: Perillas de VigConfig que esta vigilancia pisa, por nombre. Sirve para lo que no
    #: puede ser igual para todos: qué tan estable es la métrica. Un stock casi no se
    #: mueve y un 5% ya es raro; las devoluciones diarias por canal se mueven 40% solas.
    #:     ajustes={"desvio_relativo_minimo": 0.25, "piso_sigma_relativo": 0.10,
    #:              "umbral_z": {"ATENCION": 4, "ALERTA": 6, "CRITICO": 10},
    #:              "nivel_notificacion": "CRITICO"}
    #: Acepta cualquier campo de VigConfig menos `vigilancias`. Lo que no nombres, se
    #: hereda del config global.
    ajustes: Dict[str, Any] = field(default_factory=dict)
    activa: bool = True

    def claves(self) -> List[str]:
        return list(self.categorias)


@dataclass
class VigConfig:
    """Parámetros comunes a todas las vigilancias."""

    vigilancias: Sequence[Vigilancia] = field(default_factory=list)
    col_fecha: str = "FECHA"
    fecha_ejecucion: Optional[Any] = None

    #: cuánta historia se guarda y se usa como referencia, en días
    dias_historia: int = 1095
    #: cuántos períodos hacia atrás se revisan en cada corrida, por grano
    periodos_evaluados: Dict[str, int] = field(default_factory=lambda: {"dia": 30, "semana": 8, "mes": 6})
    #: períodos de referencia para la mediana y el desvío robustos, por grano
    periodos_base: Dict[str, int] = field(default_factory=lambda: {"dia": 91, "semana": 26, "mes": 18})
    #: mínimo de períodos con datos para que un detector opine (abajo de eso, sólo INFO)
    min_periodos: int = 6

    detectores: Sequence[str] = ("hueco", "salto", "escalon", "tendencia", "estacional", "racha",
                                 "congelado", "dia_cerrado")
    #: detectores que NO corren en cierto grano. Comparar contra "el mismo día de la semana"
    #: es ruido: la estacionalidad se mira en semana y mes.
    detectores_excluidos: Dict[str, Sequence[str]] = field(
        default_factory=lambda: {"dia": ("estacional",)})
    #: en grano día, comparar cada día contra los de su mismo día de la semana
    desestacionalizar_dia: bool = True
    #: cuando varios detectores marcan lo mismo, cuál manda (de más específico a menos)
    prioridad_detectores: Sequence[str] = ("hueco", "congelado", "escalon", "salto", "tendencia",
                                           "racha", "dia_cerrado",
                                           "estacional", "nueva")
    unificar_eventos: bool = True   #: un problema = un evento, con los demás como confirmación
    #: períodos seguidos que necesita cada detector para que sea un evento y no una casualidad.
    #: Un escalón de un solo período no es un escalón: con muchas series y muchos días,
    #: siempre aparece alguno por azar.
    duracion_minima: Dict[str, int] = field(default_factory=lambda: {"escalon": 3, "tendencia": 3})
    #: umbrales de desvío robusto (z) que definen el nivel
    umbral_z: Dict[str, float] = field(default_factory=lambda: {"ATENCION": 3.0, "ALERTA": 5.0, "CRITICO": 8.0})
    #: períodos seguidos que suben un nivel la gravedad
    persistencia_sube_nivel: int = 3
    #: peso de la serie comparado con la serie promedio de su vigilancia (1 = promedio).
    #: Menos que esto y no pasa de ATENCION. Es a escala: con 3 series o con 3.000, "chica"
    #: significa lo mismo. Medirlo contra el total haría que con muchas series nada sea grave.
    peso_relativo_minimo: float = 0.2
    #: desde qué nivel marca cada detector. El salto mira un solo período: con ruido de colas
    #: largas, marcar desde ATENCION llena la tabla de eventos que no son nada.
    umbral_marca: Dict[str, str] = field(default_factory=lambda: {
        "salto": "ALERTA", "estacional": "ALERTA", "hueco": "ATENCION",
        "escalon": "ATENCION", "tendencia": "ATENCION", "congelado": "ATENCION",
        "dia_cerrado": "ATENCION"})
    #: Nivel máximo al que puede llegar cada detector. `dia_cerrado` es actividad en un día
    #: que normalmente está cerrado: en ventas es alguien que abrió un domingo (queda
    #: anotado, no se avisa); en consumo de agua o energía es LA fuga. Para esas métricas,
    #: por vigilancia: ajustes={"nivel_maximo": {"dia_cerrado": "CRITICO"}}.
    nivel_maximo: Dict[str, str] = field(default_factory=lambda: {"dia_cerrado": "ATENCION"})
    #: desvío mínimo contra lo esperado para que sea un evento, en fracción.
    #: Sin esto, una serie muy estable convierte una diferencia del 2% en un desvío enorme.
    desvio_relativo_minimo: float = 0.05
    #: piso del ruido: el desvío nunca se considera menor a esta fracción del nivel de la
    #: serie. Los totales mensuales son tan parejos que, sin piso, un 4% da "desvío 17".
    piso_sigma_relativo: float = 0.02
    #: Para ser CRÍTICO, lo observado tiene que apartarse al menos esta fracción de lo
    #: esperado. Una serie muy estable puede dar un desvío estadístico enorme con un 3%
    #: de diferencia: es raro, pero no es grave. Por debajo, el evento queda en ALERTA.
    #: En una métrica muy estable (un stock) bajalo por vigilancia con `ajustes`.
    critico_desvio_relativo_minimo: float = 0.20
    #: Un valor que es esta cantidad de veces lo esperado en un período casi nunca es
    #: negocio: es una carga duplicada, un acumulado anual en un día o un error de
    #: unidades. Se avisa igual, con la etiqueta "posible error de dato". 0 lo apaga.
    factor_sospecha_dato: float = 20.0
    #: En grano día, un día de la semana cuya mediana no llega a esta fracción de la
    #: mediana de la serie es un día CERRADO (el domingo de un B2B): sus valores no se
    #: evalúan, porque un cero ahí es lo normal y no un apagón.
    dia_cerrado_relativo: float = 0.05
    #: desde qué nivel se notifica
    nivel_notificacion: str = "ALERTA"
    #: desde qué nivel se guarda un evento. Lo que queda abajo no se pierde: la serie y su
    #: historia siempre se guardan en la tabla de series. "INFO" guarda absolutamente todo.
    nivel_minimo_evento: str = "ATENCION"
    #: cuántos causantes se guardan al explicar un evento
    top_atribucion: int = 5
    #: Suma los valores de cada período en su escala decimal (centavos): un importe y
    #: su reverso dan cero EXACTO, sin tolerancias. Es lo que hace Oracle con NUMBER.
    suma_exacta: bool = True
    #: hasta cuántos decimales busca esa escala. Más allá, redondea.
    max_decimales: int = 6
    #: Plan B para cuando no hay escala decimal usable: una suma cuenta como cero
    #: cuando no llega a esta fracción de lo que pasó por ella. 0 lo desactiva.
    tolerancia_cero: float = 1e-9
    decimales: int = 4
    verbose: int = 1
    #: caché de los configs efectivos por vigilancia (lo llena `para`)
    _cache_ajustes: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def validate(self) -> None:
        if not self.vigilancias:
            raise ValueError("Hace falta al menos una vigilancia")
        vistos = set()
        for v in self.vigilancias:
            if v.nombre in vistos:
                raise ValueError(f"vigilancia repetida: {v.nombre}")
            vistos.add(v.nombre)
            if not v.categorias:
                raise ValueError(f"{v.nombre}: hace falta al menos una categoría")
            if v.agregacion not in ("suma", "promedio", "conteo", "ratio"):
                raise ValueError(f"{v.nombre}: agregacion debe ser suma, promedio, conteo o ratio")
            if v.agregacion == "ratio" and not v.denominador:
                raise ValueError(f"{v.nombre}: agregacion='ratio' necesita denominador")
            if v.min_denominador < 0:
                raise ValueError(f"{v.nombre}: min_denominador no puede ser negativo")
            propios = {f.name for f in fields(self)} - {"vigilancias", "_cache_ajustes"}
            fuera = [k for k in v.ajustes if k not in propios]
            if fuera:
                raise ValueError(f"{v.nombre}: ajustes desconocidos {fuera}. "
                                 f"Tienen que ser campos de VigConfig: {sorted(propios)}")
            self.para(v).validate_propio(v.nombre)
            if v.direccion not in ("ambas", "baja", "sube"):
                raise ValueError(f"{v.nombre}: direccion debe ser ambas, baja o sube")
            malos = [g for g in v.granos if g not in GRANOS]
            if malos:
                raise ValueError(f"{v.nombre}: granos desconocidos {malos}; hay {GRANOS}")
            desconocidos = [d for d in (v.detectores or self.detectores) if d not in DETECTORES]
            if desconocidos:
                raise ValueError(f"{v.nombre}: detectores desconocidos {desconocidos}; hay {list(DETECTORES)}")
        if not 0 <= self.max_decimales <= 15:
            raise ValueError("max_decimales debe estar entre 0 y 15")
        if not 0 <= self.tolerancia_cero < 1:
            raise ValueError("tolerancia_cero debe estar entre 0 y 1 (0 = sin limpieza)")
        if self.nivel_notificacion not in NIVELES:
            raise ValueError(f"nivel_notificacion debe ser uno de {NIVELES}")

    def para(self, v: Vigilancia) -> "VigConfig":
        """El config efectivo de una vigilancia: el global con sus `ajustes` encima."""
        if not v.ajustes:
            return self
        clave = v.nombre
        if clave not in self._cache_ajustes:
            self._cache_ajustes[clave] = replace(self, **dict(v.ajustes))
        return self._cache_ajustes[clave]

    def validate_propio(self, quien: str = "") -> None:
        """Valida sólo lo que no depende de las vigilancias (para los configs con ajustes)."""
        donde = f"{quien}: " if quien else ""
        if self.nivel_notificacion not in NIVELES:
            raise ValueError(f"{donde}nivel_notificacion debe ser uno de {NIVELES}")
        if self.nivel_minimo_evento not in NIVELES:
            raise ValueError(f"{donde}nivel_minimo_evento debe ser uno de {NIVELES}")
        faltan = [n for n in ("ATENCION", "ALERTA", "CRITICO") if n not in self.umbral_z]
        if faltan:
            raise ValueError(f"{donde}umbral_z le falta {faltan}")
        if not (self.umbral_z["ATENCION"] <= self.umbral_z["ALERTA"] <= self.umbral_z["CRITICO"]):
            raise ValueError(f"{donde}umbral_z tiene que ir de menor a mayor: {self.umbral_z}")
        if self.min_periodos < 2:
            raise ValueError(f"{donde}min_periodos tiene que ser al menos 2")
        if not 0 <= self.critico_desvio_relativo_minimo <= 10:
            raise ValueError(f"{donde}critico_desvio_relativo_minimo va de 0 a 10 (fracción)")
        if self.factor_sospecha_dato < 0:
            raise ValueError(f"{donde}factor_sospecha_dato no puede ser negativo (0 = apagado)")
        if not 0 <= self.dia_cerrado_relativo < 1:
            raise ValueError(f"{donde}dia_cerrado_relativo va de 0 a 1")
        malos = {k: x for k, x in self.nivel_maximo.items() if x not in NIVELES}
        if malos:
            raise ValueError(f"{donde}nivel_maximo tiene niveles desconocidos {malos}; hay {NIVELES}")

    def umbral_de(self, detector: str) -> float:
        return self.umbral_z.get(self.umbral_marca.get(detector, "ATENCION"), 3.0)

    def detectores_de(self, v: Vigilancia, grano: Optional[str] = None) -> List[str]:
        elegidos = list(v.detectores or self.detectores)
        if grano:
            fuera = set(self.detectores_excluidos.get(grano, ()))
            elegidos = [d for d in elegidos if d not in fuera]
        return elegidos


@dataclass(frozen=True)
class Fechas:
    hoy: pd.Timestamp
    ayer: pd.Timestamp

    @classmethod
    def desde(cls, fecha_ejecucion) -> "Fechas":
        hoy = (pd.Timestamp(fecha_ejecucion) if fecha_ejecucion is not None
               else pd.Timestamp.today()).normalize()
        return cls(hoy=hoy, ayer=hoy - pd.Timedelta(days=1))

    @property
    def d_ayer(self) -> int:
        return _dia(self.ayer)


# =========================================================================== #
# 2. Series
# =========================================================================== #
def indice_periodo(fechas: pd.Series, grano: str) -> np.ndarray:
    """Número de período de cada fecha: día, semana (lunes) o mes."""
    f = pd.DatetimeIndex(fechas)
    if grano == "dia":
        return (f.normalize() - _EPOCA).days.to_numpy(np.int64)
    if grano == "semana":
        lunes = f.normalize() - pd.to_timedelta(f.dayofweek, unit="D")
        return ((lunes - _EPOCA).days // 7).to_numpy(np.int64)
    if grano == "mes":
        return ((f.year - 1970) * 12 + f.month - 1).to_numpy(np.int64)
    raise ValueError(f"grano desconocido: {grano}")


def inicio_periodo(idx: np.ndarray, grano: str) -> pd.DatetimeIndex:
    """Primer día de cada período, para mostrar y guardar."""
    idx = np.asarray(idx, dtype=np.int64)
    if grano == "dia":
        return pd.DatetimeIndex(_EPOCA + pd.to_timedelta(idx, unit="D"))
    if grano == "semana":
        return pd.DatetimeIndex(_EPOCA + pd.to_timedelta(idx * 7, unit="D"))
    return pd.DatetimeIndex([pd.Timestamp(year=1970 + i // 12, month=i % 12 + 1, day=1) for i in idx])


def ultimo_periodo_cerrado(ayer: pd.Timestamp, grano: str) -> int:
    """Último período COMPLETO. El mes en curso no se compara contra meses enteros:
    la mitad de un mes siempre parece una caída."""
    idx = int(indice_periodo(pd.Series([ayer]), grano)[0])
    if grano == "dia":
        return idx
    if grano == "semana":
        return idx if ayer.dayofweek == 6 else idx - 1          # cierra el domingo
    return idx if ayer.day == ayer.days_in_month else idx - 1   # cierra a fin de mes


def periodos_por_ano(grano: str) -> int:
    return {"dia": 365, "semana": 52, "mes": 12}[grano]


def armar_series(df: pd.DataFrame, v: Vigilancia, cfg: VigConfig, grano: str) -> pd.DataFrame:
    """Filas de la fuente -> una fila por categoría y período, con el valor agregado."""
    claves = v.claves()
    faltan = [c for c in claves + [cfg.col_fecha, v.metrica] if c not in df.columns]
    if v.agregacion == "ratio" and v.denominador not in df.columns:
        faltan.append(v.denominador)
    if faltan:
        raise KeyError(f"{v.nombre}: faltan columnas en la fuente: {faltan}")

    base = pd.DataFrame({c: df[c].astype(object).where(df[c].notna(), "(sin dato)").astype(str)
                         for c in claves})
    base["periodo"] = indice_periodo(df[cfg.col_fecha], grano)
    base["valor"] = pd.to_numeric(df[v.metrica], errors="coerce").fillna(0.0)
    base["bruto"] = base["valor"].abs()
    if v.agregacion == "ratio":
        base["denominador"] = pd.to_numeric(df[v.denominador], errors="coerce").fillna(0.0)
        base["denominador_bruto"] = base["denominador"].abs()

    tol = cfg.tolerancia_cero
    # escala decimal de la métrica: en centavos los enteros se suman sin error y el
    # período que se anula con sus reversos da cero exacto
    columnas_dinero = [base["valor"].to_numpy(float)]
    if v.agregacion == "ratio":
        columnas_dinero.append(base["denominador"].to_numpy(float))
    e = escala_decimal(columnas_dinero, cfg.max_decimales) if cfg.suma_exacta else 0.0
    if e:
        base["valor"] = np.round(base["valor"].to_numpy(float) * e)
        base["bruto"] = base["valor"].abs()
        if v.agregacion == "ratio":
            base["denominador"] = np.round(base["denominador"].to_numpy(float) * e)
            base["denominador_bruto"] = base["denominador"].abs()
    g = base.groupby(claves + ["periodo"], sort=False)
    if v.agregacion == "conteo":
        out = g.size().reset_index(name="valor")
    elif v.agregacion == "promedio":
        out = g["valor"].mean().reset_index()
        if e:
            out["valor"] = out["valor"].to_numpy(float) / e
    elif v.agregacion == "ratio":
        out = g.agg(numerador=("valor", "sum"), denominador=("denominador", "sum"),
                    num_bruto=("bruto", "sum"),
                    den_bruto=("denominador_bruto", "sum")).reset_index()
        # el denominador que se anula con sus reversos vale 0, no 1e-10: el ratio queda
        # nulo (un hueco de la serie) en vez de dispararse a millones
        bd, bn = out.pop("den_bruto").to_numpy(float), out.pop("num_bruto").to_numpy(float)
        den, num = out["denominador"].to_numpy(float), out["numerador"].to_numpy(float)
        if e:                              # enteros: el que se anula ya vale 0
            den, num = den / e, num / e
        else:
            den = sin_residuo(den, bd, tol, escala_tipica(bd))
            num = sin_residuo(num, bn, tol, escala_tipica(bn))
        out["denominador"], out["numerador"] = den, num
        sirve = den != 0
        if v.min_denominador > 0:      # base despreciable: el ratio no significa nada
            sirve &= np.abs(den) >= v.min_denominador
        with np.errstate(invalid="ignore", divide="ignore"):
            out["valor"] = np.where(sirve, num / np.where(sirve, den, 1.0), np.nan)
    else:
        out = g.agg(valor=("valor", "sum"), bruto=("bruto", "sum")).reset_index()
        bruto = out.pop("bruto").to_numpy(float)
        out["valor"] = (out["valor"].to_numpy(float) / e if e else
                        sin_residuo(out["valor"].to_numpy(float), bruto, tol, escala_tipica(bruto)))
    out["clave"] = _clave_texto(out, claves)
    return out


def _clave_texto(df: pd.DataFrame, claves: Sequence[str]) -> np.ndarray:
    s = df[claves[0]].astype(str)
    for c in list(claves)[1:]:
        s = s.str.cat(df[c].astype(str), sep=" | ")
    return s.to_numpy(dtype=object)


def historial_json(periodos: np.ndarray, valores: np.ndarray, decimales: int = 4) -> str:
    """Historia de una serie: {"p0": primer período, "v": valores, "h": huecos}.

    Los períodos sin dato se guardan como null: no es lo mismo "no vendió" que
    "no sabemos". El motor los rellena con 0 sólo cuando la serie ya existía.
    """
    if not len(periodos):
        return json.dumps({"p0": 0, "v": []})
    p0, pn = int(periodos.min()), int(periodos.max())
    lleno = np.full(pn - p0 + 1, np.nan)
    lleno[periodos.astype(np.int64) - p0] = valores
    v = [None if not np.isfinite(x) else round(float(x), decimales) for x in lleno]
    return json.dumps({"p0": p0, "v": v}, separators=(",", ":"))


def historial_a_serie(texto: str) -> Tuple[np.ndarray, np.ndarray]:
    """Inversa de historial_json: (períodos, valores) con NaN en los huecos."""
    h = json.loads(texto)
    valores = np.array([np.nan if x is None else float(x) for x in h["v"]], dtype=float)
    periodos = np.arange(int(h["p0"]), int(h["p0"]) + len(valores), dtype=np.int64)
    return periodos, valores


def combinar_historia(guardada: Optional[pd.DataFrame], nueva: pd.DataFrame, v: Vigilancia,
                      cfg: VigConfig, grano: str, desde_relectura: int) -> pd.DataFrame:
    """Historia guardada (antes de la relectura) + lo que se acaba de leer.

    Lo releído pisa a lo guardado: si la fuente se corrigió, la corrección entra sola.
    """
    if guardada is None or guardada.empty:
        return nueva
    filas = []
    for _, fila in guardada.iterrows():
        if fila["BD_VIGILANCIA"] != v.nombre or fila["BD_GRANO"] != grano:
            continue
        periodos, valores = historial_a_serie(fila["BD_HISTORIAL"])
        ok = np.isfinite(valores) & (periodos < desde_relectura)
        if not ok.any():
            continue
        filas.append(pd.DataFrame({"clave": fila["BD_CLAVE"], "periodo": periodos[ok], "valor": valores[ok]}))
    if not filas:
        return nueva
    viejo = pd.concat(filas, ignore_index=True)
    nueva = nueva[nueva["periodo"] >= desde_relectura]
    return pd.concat([viejo, nueva[["clave", "periodo", "valor"]]], ignore_index=True)


def dias_cerrados(M: np.ndarray, periodos: np.ndarray, umbral: float = 0.05) -> np.ndarray:
    """Qué celdas son de un día de la semana en que la serie está CERRADA.

    Un día cuya mediana no llega a `umbral` veces el nivel típico de la serie, o que
    nunca trae dato, está cerrado: el domingo de un B2B. Ahí un cero es lo normal, no
    un apagón. Se mide sobre los valores ORIGINALES y en valor absoluto, así sirve en
    escala lineal y logarítmica, y con métricas que pueden ser negativas. El nivel
    típico es la mediana de los valores no nulos: con una serie que abre tres días
    por semana, la mediana de todo daría cero y nada parecería cerrado.
    """
    cerrados = np.zeros(M.shape, dtype=bool)
    if not umbral or M.size == 0:
        return cerrados
    A = np.abs(np.asarray(M, dtype=float))
    base = _mediana(np.where(np.isfinite(A) & (A > 0), A, np.nan))
    hay_base = np.isfinite(base) & (base > 0)
    dow = np.asarray(periodos, dtype=np.int64) % 7
    for d in range(7):
        col = dow == d
        if not col.any():
            continue
        med = _mediana(A[:, col])
        cierra = hay_base & (~np.isfinite(med) | (med <= umbral * np.where(hay_base, base, 0.0)))
        cerrados[:, col] = cierra[:, None]
    return cerrados


def desestacionalizar_semanal(M: np.ndarray, periodos: np.ndarray,
                              aditivo: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Quita el patrón de día de la semana de cada serie.

    Sin esto, "siete días seguidos por encima de lo normal" pasa todas las semanas en
    cualquier negocio que venda distinto los lunes que los sábados. Devuelve la serie
    ajustada y el factor de cada día, para poder informar los valores originales. Los
    días cerrados los resuelve `dias_cerrados`.
    """
    dow = np.asarray(periodos, dtype=np.int64) % 7
    with np.errstate(invalid="ignore"):
        base = _mediana(M)
    ajuste = np.zeros_like(M) if aditivo else np.ones_like(M)
    for d in range(7):
        col = dow == d
        if not col.any():
            continue
        with np.errstate(invalid="ignore"):
            med = _mediana(M[:, col])
        if aditivo:
            a = np.where(np.isfinite(med) & np.isfinite(base), med - base, 0.0)
        else:
            a = np.divide(med, base, out=np.ones_like(base), where=np.isfinite(base) & (base > 0))
            a = np.where(np.isfinite(a) & (a > 0.05), a, 1.0)
        ajuste[:, col] = a[:, None]
    return (M - ajuste, ajuste) if aditivo else (M / ajuste, ajuste)


def matriz_series(series: pd.DataFrame, desde: int, hasta: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(claves, períodos, matriz claves x períodos con NaN donde no hay dato)."""
    claves, codigos = np.unique(series["clave"].to_numpy(dtype=object), return_inverse=True)
    periodos = np.arange(desde, hasta + 1, dtype=np.int64)
    M = np.full((len(claves), len(periodos)), np.nan)
    p = series["periodo"].to_numpy(np.int64)
    dentro = (p >= desde) & (p <= hasta)
    M[codigos[dentro], p[dentro] - desde] = series["valor"].to_numpy(float)[dentro]
    return claves, periodos, M


# =========================================================================== #
# 3. Detectores
# =========================================================================== #
@dataclass
class Deteccion:
    """Lo que devuelve un detector sobre la ventana evaluada (series x períodos)."""
    marca: np.ndarray        #: bool: acá hay algo raro
    z: np.ndarray            #: cuán raro, en desvíos robustos
    esperado: np.ndarray     #: con qué se lo comparó


@dataclass
class Detector:
    nombre: str
    descripcion: str
    funcion: Callable[..., Deteccion]


def _sigma(mad: np.ndarray) -> np.ndarray:
    """MAD -> desvío comparable al estándar. Sin dispersión, un piso para no dividir por cero."""
    s = 1.4826 * mad
    return np.where(s > 0, s, np.nan)


def _base_movil(M: np.ndarray, idx: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mediana, MAD y cantidad de datos de la ventana ANTERIOR a cada período evaluado."""
    n, E = M.shape[0], len(idx)
    mediana = np.full((n, E), np.nan)
    mad = np.full((n, E), np.nan)
    cuenta = np.zeros((n, E))
    for k, t in enumerate(idx):
        ini = max(0, t - w)
        tramo = M[:, ini:t]
        if tramo.shape[1] == 0:
            continue
        with np.errstate(invalid="ignore"):
            med = _mediana(tramo)
            mediana[:, k] = med
            mad[:, k] = _mediana(np.abs(tramo - med[:, None]))
        cuenta[:, k] = np.isfinite(tramo).sum(axis=1)
    return mediana, mad, cuenta


def _z(obs: np.ndarray, esperado: np.ndarray, mad: np.ndarray, cuenta: np.ndarray,
       min_periodos: int, piso_relativo: float = 0.0, log: bool = False) -> np.ndarray:
    """Desvío robusto, con un piso de ruido para no volver enorme una diferencia chica.

    En escala logarítmica el piso es directamente relativo (log1p del porcentaje); en
    escala lineal, una fracción del nivel esperado.
    """
    s = _sigma(mad)
    if log:
        piso = np.full_like(np.nan_to_num(esperado), np.log1p(max(piso_relativo, 1e-6)))
        escala = np.maximum(np.nan_to_num(s), piso)
    else:
        escala = np.where(np.isfinite(s), s, np.abs(esperado) * 0.5)
        escala = np.maximum(np.nan_to_num(escala), np.abs(np.nan_to_num(esperado)) * piso_relativo)
    escala = np.where(escala > 0, escala, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (obs - esperado) / escala
    z = np.where(np.isfinite(z), z, 0.0)
    return np.where(cuenta >= min_periodos, z, 0.0)


def det_hueco(M, idx, cfg: VigConfig, v: "Vigilancia", grano: str, log: bool = False) -> Deteccion:
    """Se apagó: venía con movimiento y ahora es cero (o no vino en la fuente).

    En escala logarítmica el cero no existe: se lo trata como un valor cien veces
    menor que lo normal, que es lo que significa apagarse.
    """
    w = cfg.periodos_base[grano]
    mediana, mad, cuenta = _base_movil(M, idx, w)
    obs = M[:, idx]
    vacio = ~np.isfinite(obs) | ((obs == 0) & (not log))
    tenia = np.isfinite(mediana) & (cuenta >= cfg.min_periodos) & ((mediana != 0) | log)
    apagado = (mediana - np.log(100.0)) if log else np.zeros_like(mediana)
    z = _z(apagado, mediana, mad, cuenta, cfg.min_periodos, cfg.piso_sigma_relativo, log)
    return Deteccion(marca=vacio & tenia, z=z, esperado=mediana)


def det_salto(M, idx, cfg: VigConfig, v: "Vigilancia", grano: str, log: bool = False) -> Deteccion:
    """Pico o caída puntual contra la mediana móvil (Hampel): el típico error de carga."""
    w = cfg.periodos_base[grano]
    mediana, mad, cuenta = _base_movil(M, idx, w)
    obs = np.where(np.isfinite(M[:, idx]), M[:, idx], 0.0)
    z = _z(obs, mediana, mad, cuenta, cfg.min_periodos, cfg.piso_sigma_relativo, log)
    return Deteccion(marca=np.abs(z) >= cfg.umbral_de("salto"), z=z, esperado=mediana)


def det_escalon(M, idx, cfg: VigConfig, v: "Vigilancia", grano: str, log: bool = False) -> Deteccion:
    """Cambió de nivel y se quedó ahí: pérdida real, o cambio de criterio en la fuente."""
    w = cfg.periodos_base[grano]
    k = {"dia": 7, "semana": 3, "mes": 2}[grano]
    mediana, mad, cuenta = _base_movil(M, idx, w)
    n, E = M.shape[0], len(idx)
    reciente = np.full((n, E), np.nan)
    for j, t in enumerate(idx):
        tramo = M[:, max(0, t - k + 1):t + 1]
        with np.errstate(invalid="ignore"):
            reciente[:, j] = _mediana(np.where(np.isfinite(tramo), tramo, 0.0))
    # error estándar de una mediana de k valores: 1.2533 * sigma / sqrt(k)
    z = _z(reciente, mediana, mad, cuenta, cfg.min_periodos, cfg.piso_sigma_relativo, log) * (np.sqrt(k) / 1.2533)
    return Deteccion(marca=np.abs(z) >= cfg.umbral_de("escalon"), z=z, esperado=mediana)


def _pendiente(M: np.ndarray, desde: int, hasta: int) -> np.ndarray:
    """Pendiente por mínimos cuadrados de cada serie en [desde, hasta)."""
    tramo = M[:, desde:hasta]
    if tramo.shape[1] < 3:
        return np.zeros(M.shape[0])
    # sólo con los períodos que tienen dato: un día cerrado o faltante no es un cero,
    # y tratarlo como cero doblaba la recta hacia abajo
    ok = np.isfinite(tramo)
    cuantos = ok.sum(axis=1)
    x = np.broadcast_to(np.arange(tramo.shape[1], dtype=float), tramo.shape)
    xm = np.where(ok, x, 0.0).sum(axis=1) / np.maximum(cuantos, 1)
    ym = np.where(ok, tramo, 0.0).sum(axis=1) / np.maximum(cuantos, 1)
    dx = np.where(ok, x - xm[:, None], 0.0)
    dy = np.where(ok, tramo - ym[:, None], 0.0)
    den = (dx * dx).sum(axis=1)
    return np.where((cuantos >= 3) & (den > 0), (dx * dy).sum(axis=1) / np.where(den > 0, den, 1.0), 0.0)


def _error_pendiente(sigma: np.ndarray, w: int) -> np.ndarray:
    """Error estándar de una pendiente ajustada sobre w puntos con ruido sigma."""
    if w < 3:
        return np.full_like(sigma, np.inf)
    return sigma * np.sqrt(12.0 / (w * (w * w - 1)))


def det_tendencia(M, idx, cfg: VigConfig, v: "Vigilancia", grano: str, log: bool = False) -> Deteccion:
    """La pendiente de la última ventana se dio vuelta o se quebró contra la anterior.

    El ruido se mide con las diferencias entre períodos consecutivos, no con la
    dispersión de la ventana: en una serie que ya venía subiendo, la dispersión es
    grande por la propia tendencia y taparía el quiebre.
    """
    w = {"dia": 28, "semana": 8, "mes": 6}[grano]
    base = cfg.periodos_base[grano]
    mediana, mad, cuenta = _base_movil(M, idx, base)
    n, E = M.shape[0], len(idx)
    z = np.zeros((n, E))
    esperado = np.full((n, E), np.nan)
    for j, t in enumerate(idx):
        tramo = M[:, max(0, t - base):t + 1]
        y = np.where(np.isfinite(tramo), tramo, np.nan)
        with np.errstate(invalid="ignore"):
            dif = np.abs(np.diff(y, axis=1))
            mad_dif = _mediana(dif)
        sigma = 1.4826 * mad_dif / np.sqrt(2.0)
        piso = (np.log1p(cfg.piso_sigma_relativo) if log
                else np.abs(np.nan_to_num(mediana[:, j])) * cfg.piso_sigma_relativo)
        sigma = np.maximum(np.nan_to_num(sigma), piso)
        sigma = np.where(sigma > 0, sigma, np.nan)
        actual = _pendiente(M, max(0, t - w + 1), t + 1)
        ini_previa, fin_previa = max(0, t - 2 * w + 1), max(0, t - w + 1)
        previa = _pendiente(M, ini_previa, fin_previa)
        se = _error_pendiente(sigma, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            zz = (actual - previa) / (se * np.sqrt(2.0))
        z[:, j] = np.where(np.isfinite(zz), zz, 0.0)
        # el esperado que se REPORTA es un nivel, no una pendiente: se ancla en el centro
        # de la ventana anterior y se estira su propia recta hasta este período. Así
        # "observado 28.021 contra 29.960 esperados" se puede leer; una pendiente, no.
        anterior = M[:, ini_previa:fin_previa]
        ancla = _mediana(np.where(np.isfinite(anterior), anterior, np.nan))
        centro = (ini_previa + max(fin_previa - 1, ini_previa)) / 2.0
        esperado[:, j] = ancla + previa * (t - centro)
    z = np.where(cuenta >= max(cfg.min_periodos, w), z, 0.0)
    return Deteccion(marca=np.abs(z) >= cfg.umbral_de("tendencia"), z=z, esperado=esperado)


def det_estacional(M, idx, cfg: VigConfig, v: "Vigilancia", grano: str, log: bool = False) -> Deteccion:
    """Contra el mismo período de otros años (o el mismo día de semana), no contra el promedio."""
    ciclo = periodos_por_ano(grano) if grano != "dia" else 7
    n, E = M.shape[0], len(idx)
    esperado = np.full((n, E), np.nan)
    mad = np.full((n, E), np.nan)
    cuenta = np.zeros((n, E))
    for j, t in enumerate(idx):
        pares = np.arange(t - ciclo, -1, -ciclo)[:12]
        if not len(pares):
            continue
        tramo = M[:, pares]
        with np.errstate(invalid="ignore"):
            med = _mediana(tramo)
            esperado[:, j] = med
            mad[:, j] = _mediana(np.abs(tramo - med[:, None]))
        cuenta[:, j] = np.isfinite(tramo).sum(axis=1)
    obs = np.where(np.isfinite(M[:, idx]), M[:, idx], 0.0)
    minimo = max(3, cfg.min_periodos // 2)
    z = _z(obs, esperado, mad, cuenta, minimo, cfg.piso_sigma_relativo, log)
    return Deteccion(marca=(np.abs(z) >= cfg.umbral_de("estacional")) & (cuenta >= minimo),
                     z=z, esperado=esperado)


def det_racha(M, idx, cfg: VigConfig, v: "Vigilancia", grano: str, log: bool = False) -> Deteccion:
    """Muchos períodos seguidos del mismo lado de lo normal: el aviso temprano, antes del escalón."""
    largo = {"dia": 7, "semana": 4, "mes": 3}[grano]
    w = cfg.periodos_base[grano]
    mediana, mad, cuenta = _base_movil(M, idx, w)
    obs = np.where(np.isfinite(M[:, idx]), M[:, idx], 0.0)
    z = _z(obs, mediana, mad, cuenta, cfg.min_periodos, cfg.piso_sigma_relativo, log)
    n, E = M.shape
    marca = np.zeros((n, len(idx)), dtype=bool)
    for j, t in enumerate(idx):
        ini = max(0, t - largo + 1)
        if t - ini + 1 < largo:
            continue
        tramo = np.where(np.isfinite(M[:, ini:t + 1]), M[:, ini:t + 1], 0.0)
        ref = mediana[:, j][:, None]
        arriba = np.all(tramo > ref, axis=1)
        abajo = np.all(tramo < ref, axis=1)
        marca[:, j] = (arriba | abajo) & (cuenta[:, j] >= cfg.min_periodos)
    return Deteccion(marca=marca, z=z, esperado=mediana)


def det_congelado(M, idx, cfg: VigConfig, v: "Vigilancia", grano: str, log: bool = False,
                  extra: Optional[dict] = None) -> Deteccion:
    """El dato dejó de actualizarse: repite exactamente el mismo valor.

    Es la falla típica de cualquier telemetría (un medidor trabado, un ETL que copia
    el último valor) y ningún otro detector la ve, porque el valor está en su nivel
    normal. Repetir sólo es sospechoso si en ESA serie es raro: se estima de su propia
    historia la probabilidad p de que un período repita el anterior, y se avisa
    cuando la racha es improbable (p elevado a la racha, menos de 1 en 1.000). Un
    contrato fijo repite siempre y nunca salta; un consumo con ruido casi nunca repite.
    El z es la cantidad de períodos iguales: con los umbrales por defecto, 3 es
    ATENCION, 5 ALERTA y 8 CRITICO. Los ceros los ve `hueco`, no éste.
    """
    A = np.asarray(extra["original"], dtype=float)
    C = extra["cerrados"]
    n, T = A.shape
    E = len(idx)
    B = np.where(C, np.nan, A)                       # los días cerrados no cuentan
    finito = np.isfinite(B)
    pos = np.where(finito, np.arange(T)[None, :], -1)
    ultimo = np.maximum.accumulate(pos, axis=1)
    previo_idx = np.concatenate([np.full((n, 1), -1), ultimo[:, :-1]], axis=1)
    previo = np.where(previo_idx >= 0,
                      np.take_along_axis(B, np.maximum(previo_idx, 0), axis=1), np.nan)
    comparable = finito & np.isfinite(previo)
    igual = comparable & (B != 0) & (np.abs(B - previo) <= 1e-9 * np.maximum(np.abs(B), 1.0))

    ini = int(idx[0])
    p = (igual[:, :ini].sum(axis=1) + 1.0) / (comparable[:, :ini].sum(axis=1) + 2.0)
    racha = np.zeros(n)
    rachas = np.zeros((n, T))
    for t in range(T):                               # un día cerrado ni suma ni corta
        racha = np.where(finito[:, t], np.where(igual[:, t], racha + 1.0, 0.0), racha)
        rachas[:, t] = racha
    r = rachas[:, idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        improbable = np.power(p[:, None], r) <= 1e-3
    marca = (r >= 2) & improbable & finito[:, idx]
    z = np.where(marca, r + 1.0, 0.0)                # períodos con el mismo valor
    # no hay un nivel "esperado" contra el cual medir: el valor no es confiable
    return Deteccion(marca=marca, z=z, esperado=np.full((n, E), np.nan))


def det_dia_cerrado(M, idx, cfg: VigConfig, v: "Vigilancia", grano: str, log: bool = False,
                    extra: Optional[dict] = None) -> Deteccion:
    """Hubo actividad en un día que normalmente está cerrado.

    Los días cerrados (ver `dias_cerrados`) no se evalúan con los demás detectores: un
    cero ahí es lo normal. Pero lo contrario sí importa: consumo de agua o energía en un
    día sin operación es una fuga o algo que quedó prendido. Se compara contra lo que
    suele pasar esos días (casi nada), con un piso de ruido relativo al nivel típico
    de la serie. Llega como máximo a `nivel_maximo["dia_cerrado"]` (ATENCION por
    defecto: en ventas, abrir un domingo no es grave).
    """
    A = np.asarray(extra["original"], dtype=float)
    C = extra["cerrados"]
    n, E = A.shape[0], len(idx)
    marca = np.zeros((n, E), dtype=bool)
    z = np.zeros((n, E))
    esperado = np.full((n, E), np.nan)
    if not C.any():
        return Deteccion(marca=marca, z=z, esperado=esperado)
    Aa = np.abs(A)
    tipico = _mediana(np.where(~C & np.isfinite(Aa) & (Aa > 0), Aa, np.nan))
    for j, t in enumerate(idx):
        cerrado_hoy = C[:, t] & np.isfinite(A[:, t])
        if not cerrado_hoy.any():
            continue
        antes = np.where(C[:, :t], A[:, :t], np.nan)
        med = _mediana(antes)
        mad = _mediana(np.abs(antes - med[:, None]))
        escala = np.maximum(1.4826 * np.nan_to_num(mad),
                            cfg.piso_sigma_relativo * np.nan_to_num(tipico))
        escala = np.where(escala > 0, escala, np.nan)
        exceso = A[:, t] - np.nan_to_num(med)
        with np.errstate(invalid="ignore", divide="ignore"):
            zz = exceso / escala
        relevante = exceso >= cfg.desvio_relativo_minimo * np.nan_to_num(tipico)
        ok = cerrado_hoy & np.isfinite(zz) & relevante
        z[:, j] = np.where(ok, zz, 0.0)
        esperado[:, j] = med
        marca[:, j] = ok & (zz >= cfg.umbral_de("dia_cerrado"))
    return Deteccion(marca=marca, z=z, esperado=esperado)


def det_nueva(M, idx, cfg: VigConfig, v: "Vigilancia", grano: str, log: bool = False) -> Deteccion:
    """Categoría que aparece por primera vez. Nunca es grave, pero queda anotada."""
    n, E = M.shape[0], len(idx)
    marca = np.zeros((n, E), dtype=bool)
    obs = M[:, idx]
    for j, t in enumerate(idx):
        antes = np.isfinite(M[:, :t]).any(axis=1)
        marca[:, j] = (~antes) & np.isfinite(obs[:, j])
    # sin historia no hay nada que esperar: nulo, no cero
    return Deteccion(marca=marca, z=np.zeros((n, E)), esperado=np.full((n, E), np.nan))


DETECTORES: Dict[str, Detector] = {d.nombre: d for d in [
    Detector("hueco", "Venía con movimiento y se apagó: cero, o la categoría dejó de venir en la fuente. "
                      "Es el que encuentra las cargas que fallaron.", det_hueco),
    Detector("salto", "Pico o caída puntual contra la mediana móvil, medida con desvío robusto (MAD), "
                      "que no se deja arrastrar por el propio pico.", det_salto),
    Detector("escalon", "El nivel cambió y se quedó ahí: compara las últimas mediciones contra la "
                        "referencia. Pérdida real de negocio o cambio de criterio en la fuente.", det_escalon),
    Detector("tendencia", "La pendiente de la última ventana se dio vuelta o se quebró contra la "
                          "ventana anterior.", det_tendencia),
    Detector("estacional", "Compara contra el mismo período de otros años (o el mismo día de la semana), "
                           "así diciembre no se compara con noviembre.", det_estacional),
    Detector("racha", "Varios períodos seguidos del mismo lado de lo normal: avisa antes de que el "
                      "escalón sea evidente.", det_racha),
    Detector("nueva", "Categoría que aparece por primera vez.", det_nueva),
    Detector("congelado", "El dato dejó de actualizarse: repite exactamente el mismo valor, en una "
                          "serie donde repetir es raro. Medidor trabado o ETL que copia el último valor.",
             det_congelado),
    Detector("dia_cerrado", "Actividad en un día que normalmente está cerrado. En agua o energía es una "
                            "fuga; en ventas, alguien que abrió un domingo (por eso tiene nivel máximo).",
             det_dia_cerrado),
]}


# =========================================================================== #
# 4. Gravedad, eventos y atribución
# =========================================================================== #
def nivel_por_z(z: float, cfg: VigConfig) -> str:
    a = abs(float(z))
    if a >= cfg.umbral_z["CRITICO"]:
        return "CRITICO"
    if a >= cfg.umbral_z["ALERTA"]:
        return "ALERTA"
    if a >= cfg.umbral_z["ATENCION"]:
        return "ATENCION"
    return "INFO"


def _subir(nivel: str, pasos: int = 1) -> str:
    return NIVELES[min(NIVELES.index(nivel) + pasos, len(NIVELES) - 1)]


def _bajar_hasta(nivel: str, tope: str) -> str:
    return nivel if NIVELES.index(nivel) <= NIVELES.index(tope) else tope


#: detectores que ya miran una ventana: su "persistencia" son ventanas que se pisan,
#: no evidencia nueva, así que no sube el nivel.
DETECTORES_VENTANA = ("escalon", "tendencia", "racha")
#: no suben de nivel por persistencia: su z ya mide cuánto dura (o no es un desvío)
SIN_PERSISTENCIA = DETECTORES_VENTANA + ("congelado", "dia_cerrado")
#: su "esperado" no es un nivel normal contra el cual medir cuánto se movió
SIN_DESVIO_RELATIVO = ("nueva", "congelado")
#: miran los valores ORIGINALES (sin ajuste semanal ni logaritmo) y los días cerrados
NECESITAN_ORIGINAL = ("congelado", "dia_cerrado")


def calcular_nivel(z: float, periodos: int, materialidad: float, peso_relativo: float,
                   v: Vigilancia, cfg: VigConfig, detector: str = "",
                   desvio_rel: float = float("inf"), factor: float = 0.0) -> Tuple[str, str]:
    """Gravedad = cuán raro es, cuánto se movió, cuánto lleva así y cuánto pesa.

    Devuelve (nivel, por qué). `desvio_rel` es cuánto se apartó de lo esperado, en
    fracción; `factor`, cuántas veces lo esperado llegó a valer.
    """
    nivel = nivel_por_z(z, cfg)
    razones = [f"desvío {abs(z):.1f}"]
    if (periodos >= cfg.persistencia_sube_nivel and nivel != "INFO"
            and detector not in SIN_PERSISTENCIA):
        nivel = _subir(nivel)
        razones.append(f"{periodos} períodos seguidos")
    if (nivel == "CRITICO" and detector not in SIN_DESVIO_RELATIVO
            and desvio_rel + 1e-9 < cfg.critico_desvio_relativo_minimo):
        # raro no es lo mismo que grave: un 3% en una serie muy pareja da un desvío enorme
        nivel = "ALERTA"
        razones.append(f"se apartó {desvio_rel:.0%} de lo esperado; para crítico hace falta "
                       f"{cfg.critico_desvio_relativo_minimo:.0%}")
    if cfg.factor_sospecha_dato and factor >= cfg.factor_sospecha_dato:
        razones.append(f"posible error de dato: {factor:,.0f} veces lo esperado")
    tope = cfg.nivel_maximo.get(detector)
    if tope in NIVELES and NIVELES.index(nivel) > NIVELES.index(tope):
        nivel = tope
        razones.append(f"{detector} llega como máximo a {tope} (nivel_maximo)")
    if materialidad < v.materialidad_minima:
        nivel = _bajar_hasta(nivel, "INFO")
        razones.append(f"materialidad {materialidad:,.0f} {v.unidad} por debajo del mínimo")
    elif peso_relativo < cfg.peso_relativo_minimo:
        nivel = _bajar_hasta(nivel, "ATENCION")
        razones.append(f"pesa {peso_relativo:.2f} veces la serie promedio")
    return nivel, "; ".join(razones)


def id_evento(vigilancia: str, grano: str, clave: str, detector: str, inicio: str) -> str:
    crudo = "|".join([vigilancia, grano, clave, detector, inicio])
    return hashlib.sha1(crudo.encode("utf-8")).hexdigest()[:20]


def agrupar_eventos(marca: np.ndarray, z: np.ndarray, esperado: np.ndarray, obs: np.ndarray,
                    claves: np.ndarray, periodos: np.ndarray, detector: str,
                    v: Vigilancia, cfg: VigConfig, participacion: np.ndarray,
                    peso: np.ndarray, neutros: Optional[np.ndarray] = None) -> List[dict]:
    """Períodos marcados seguidos de la misma serie = un evento.

    `neutros` son los días cerrados: no inician ni cortan un evento (un apagón de dos
    semanas no se parte en dos por el domingo del medio) y tampoco suman materialidad.
    """
    eventos = []
    for i in range(marca.shape[0]):
        fila = marca[i]
        if not fila.any():
            continue
        neutro = neutros[i] if neutros is not None else np.zeros(len(fila), dtype=bool)
        fin = -1
        for ini in np.flatnonzero(fila):
            if ini <= fin:
                continue                    # ya quedó adentro del evento anterior
            fin = ini
            while fin + 1 < len(fila) and (fila[fin + 1] or neutro[fin + 1]):
                fin += 1
            while fin > ini and neutro[fin]:
                fin -= 1                    # un evento no termina en un día cerrado
            activos = ~neutro[ini:fin + 1]
            tramo = np.arange(ini, fin + 1)[activos]
            zs = z[i, tramo]
            peor = float(zs[np.argmax(np.abs(zs))]) if len(zs) else 0.0
            if detector != "congelado":        # un dato que no se actualiza no tiene dirección
                if v.direccion == "baja" and peor > 0:
                    continue
                if v.direccion == "sube" and peor < 0:
                    continue
            esp = np.nan_to_num(esperado[i, tramo])
            ob = np.where(np.isfinite(obs[i, tramo]), obs[i, tramo], 0.0)
            materialidad = float(np.nansum(np.abs(ob - esp)))
            n_periodos = int(len(tramo))
            if n_periodos < int(cfg.duracion_minima.get(detector, 1)):
                continue
            escala = np.maximum(np.abs(esp), 1e-12)
            desvio_rel = float(np.max(np.abs(ob - esp) / escala)) if len(tramo) else 0.0
            if detector not in SIN_DESVIO_RELATIVO and desvio_rel < cfg.desvio_relativo_minimo:
                continue
            # la sospecha de dato compara contra un nivel normal; un día cerrado espera ~0
            con_base = (np.abs(esp) > 0) & (detector not in ("dia_cerrado", "congelado"))
            factor = (float(np.max(np.abs(ob[con_base]) / np.abs(esp[con_base])))
                      if con_base.any() else 0.0)
            nivel, motivo = calcular_nivel(peor, n_periodos, materialidad, float(peso[i]),
                                           v, cfg, detector, desvio_rel, factor)
            if (detector != "nueva"
                    and NIVELES.index(nivel) < NIVELES.index(cfg.nivel_minimo_evento)):
                continue
            eventos.append({
                "clave": str(claves[i]), "detector": detector, "periodo_inicio": int(periodos[ini]),
                "periodo_fin": int(periodos[fin]), "periodos": n_periodos, "z": round(peor, 4),
                "observado": float(np.nan_to_num(obs[i, fin])), "esperado": float(esperado[i, fin]),
                "materialidad": round(materialidad, 2), "participacion": round(float(participacion[i]), 6),
                "peso_relativo": round(float(peso[i]), 4),
                "nivel": nivel, "motivo_nivel": motivo, "fila": i})
    return eventos


# =========================================================================== #
# 5. Conciliación con lo de ayer, causas y atribución
# =========================================================================== #
ESTADOS = ("NUEVO", "EN_CURSO", "CERRADO")


def unificar(eventos: pd.DataFrame, cfg: VigConfig) -> pd.DataFrame:
    """Un problema = un evento.

    Un apagón lo marcan a la vez el hueco, el salto, el escalón y la racha. Si se
    notifican los cuatro, nadie lee las alertas. Se agrupan los que pisan el mismo
    tramo de la misma serie: manda el detector más específico y los demás quedan
    anotados como confirmación.
    """
    eventos = eventos.copy()
    eventos["principal"] = True
    eventos["confirman"] = ""
    if eventos.empty or not cfg.unificar_eventos:
        return eventos
    prioridad = {d: i for i, d in enumerate(cfg.prioridad_detectores)}
    for _, grupo in eventos.groupby(["vigilancia", "grano", "clave"], sort=False):
        if len(grupo) == 1:
            continue
        bloques: List[dict] = []
        for idx, fila in grupo.sort_values("periodo_inicio").iterrows():
            for b in bloques:
                if int(fila["periodo_inicio"]) <= b["fin"] + 1:
                    b["fin"] = max(b["fin"], int(fila["periodo_fin"]))
                    b["idx"].append(idx)
                    break
            else:
                bloques.append({"fin": int(fila["periodo_fin"]), "idx": [idx]})
        for b in bloques:
            if len(b["idx"]) == 1:
                continue
            orden = sorted(b["idx"], key=lambda i: (prioridad.get(eventos.at[i, "detector"], 99),
                                                    -abs(float(eventos.at[i, "z"]))))
            principal, otros = orden[0], orden[1:]
            eventos.loc[otros, "principal"] = False
            eventos.at[principal, "confirman"] = ", ".join(sorted({eventos.at[i, "detector"] for i in otros}))
            # "en juego" es el del principal: el que cuadra con SUS períodos, observado y
            # esperado. Heredar el más grande de otro detector mezclaba tramos y esperados
            # distintos y daba "1 período, esperado 8k, en juego 3M". Los otros quedan
            # guardados como filas propias (principal = NO) con su propia materialidad.
    return eventos


def conciliar(nuevos: pd.DataFrame, previos: Optional[pd.DataFrame], f: Fechas) -> pd.DataFrame:
    """Le da identidad a cada evento: si ya venía de antes, conserva su id y su inicio.

    Sin esto, el mismo problema se notificaría como nuevo todos los días. Los eventos
    abiertos que ya no aparecen se cierran.
    """
    abiertos: Dict[Tuple[str, str, str, str], dict] = {}
    if previos is not None and len(previos):
        for _, p in previos.iterrows():
            if str(p.get("estado", "")) == "CERRADO":
                continue
            abiertos[(p["vigilancia"], p["grano"], p["clave"], p["detector"])] = p.to_dict()

    filas = []
    usados = set()
    for _, n in nuevos.iterrows():
        llave = (n["vigilancia"], n["grano"], n["clave"], n["detector"])
        previo = abiertos.get(llave)
        fila = n.to_dict()
        continua = previo is not None and int(n["periodo_inicio"]) <= int(previo["periodo_fin"]) + 1
        if continua:
            usados.add(llave)
            fila["id_evento"] = previo["id_evento"]
            fila["periodo_inicio"] = int(previo["periodo_inicio"])
            if previo.get("fecha_inicio") is not None:
                fila["fecha_inicio"] = pd.Timestamp(previo["fecha_inicio"])
            fila["periodos"] = int(n["periodo_fin"]) - int(previo["periodo_inicio"]) + 1
            fila["estado"] = "EN_CURSO"
            fila["nivel_anterior"] = previo.get("nivel", "")
            fila["fecha_deteccion"] = previo.get("fecha_deteccion", f.hoy)
        else:
            fila["id_evento"] = id_evento(n["vigilancia"], n["grano"], n["clave"], n["detector"],
                                          str(n["fecha_inicio"])[:10])
            fila["estado"] = "NUEVO"
            fila["nivel_anterior"] = ""
            fila["fecha_deteccion"] = f.hoy
        fila["fecha_cierre"] = pd.NaT
        filas.append(fila)

    for llave, p in abiertos.items():
        if llave in usados:
            continue
        cerrado = dict(p)
        cerrado["estado"] = "CERRADO"
        cerrado["fecha_cierre"] = f.hoy
        cerrado["nivel_anterior"] = p.get("nivel", "")
        filas.append(cerrado)
    return pd.DataFrame(filas)


def atribuir(crudo: pd.DataFrame, eventos: pd.DataFrame, v: Vigilancia, cfg: VigConfig,
             grano: str) -> Dict[str, str]:
    """Quién causó cada evento: los hijos que más aportaron al cambio.

    Compara el tramo del evento contra el tramo anterior de igual largo, ordena por
    cuánto se movió cada uno y arma la frase.
    """
    if not v.atribucion or v.atribucion not in crudo.columns or eventos.empty:
        return {}
    base = crudo.copy()
    base["periodo"] = indice_periodo(base[cfg.col_fecha], grano)
    base["valor"] = pd.to_numeric(base[v.metrica], errors="coerce").fillna(0.0)
    base["clave"] = _clave_texto(base, v.claves())
    salida: Dict[str, str] = {}
    for _, e in eventos.iterrows():
        ini, fin = int(e["periodo_inicio"]), int(e["periodo_fin"])
        largo = fin - ini + 1
        sub = base[base["clave"] == e["clave"]]
        if sub.empty:
            continue
        ahora = sub[(sub["periodo"] >= ini) & (sub["periodo"] <= fin)].groupby(v.atribucion)["valor"].sum()
        antes = sub[(sub["periodo"] >= ini - largo) & (sub["periodo"] < ini)].groupby(v.atribucion)["valor"].sum()
        delta = ahora.reindex(antes.index.union(ahora.index)).fillna(0) - antes.reindex(
            antes.index.union(ahora.index)).fillna(0)
        if delta.empty:
            continue
        signo = -1.0 if float(delta.sum()) < 0 else 1.0
        aportan = (delta * signo).sort_values(ascending=False)
        aportan = aportan[aportan > 0].head(cfg.top_atribucion)
        if aportan.empty:
            continue
        # si el cambio neto se anula, el reparto no significa nada (sería 4.000.000 %)
        total = float(sin_residuo(abs(delta.sum()), delta.abs().sum(), cfg.tolerancia_cero))
        parte = float(aportan.sum()) / total if total else 0.0
        detalle = ", ".join(f"{k} ({delta[k]:+,.0f} {v.unidad})" for k in aportan.index)
        salida[e["id_evento"]] = (f"{len(aportan)} de {len(delta)} explican el {parte:.0%} del cambio: "
                                  f"{detalle}")[:1000]
    return salida


def buscar_causas(eventos: pd.DataFrame, causas: Optional[pd.DataFrame], f: Fechas) -> pd.DataFrame:
    """Pega la causa que alguien ya escribió y el historial de veces que pasó lo mismo.

    `causas` es la tabla que llenan ustedes: vigilancia, clave (o * para todas),
    detector opcional, desde, hasta, causa, accion, autor.
    """
    eventos = eventos.copy()
    eventos["causa"] = ""
    eventos["historia_causas"] = ""
    if causas is None or causas.empty or eventos.empty:
        return eventos
    c = causas.copy()
    for col in ("desde", "hasta"):
        c[col] = pd.to_datetime(c[col], errors="coerce")
    for i, e in eventos.iterrows():
        mismo = c[(c["vigilancia"] == e["vigilancia"])
                  & ((c["clave"] == e["clave"]) | (c["clave"] == "*"))
                  & ((c["detector"].fillna("") == "") | (c["detector"] == e["detector"]))]
        if mismo.empty:
            continue
        ini, fin = pd.Timestamp(e["fecha_inicio"]), pd.Timestamp(e["fecha_fin"])
        solapa = mismo[(mismo["desde"] <= fin) & (mismo["hasta"].fillna(f.hoy) >= ini)]
        if len(solapa):
            eventos.at[i, "causa"] = str(solapa.iloc[-1]["causa"])[:1000]
        pasadas = mismo[mismo["hasta"].fillna(mismo["desde"]) < ini]
        if len(pasadas):
            ultima = pasadas.sort_values("desde").iloc[-1]
            eventos.at[i, "historia_causas"] = (
                f"ya pasó {len(pasadas)} vez/veces; la última el "
                f"{pd.Timestamp(ultima['desde']).date()}: {ultima['causa']}")[:1000]
    return eventos


# =========================================================================== #
# 6. Orquestador
# =========================================================================== #
@dataclass
class Resultado:
    series: pd.DataFrame
    eventos: pd.DataFrame
    notificaciones: pd.DataFrame
    resumen: pd.DataFrame


class VigEngine:
    def __init__(self, config: VigConfig):
        config.validate()
        self.cfg = config
        self.fechas = Fechas.desde(config.fecha_ejecucion)
        self.tiempos_: Dict[str, float] = {}
        LOGGER.setLevel(logging.INFO if config.verbose else logging.WARNING)

    # -- una vigilancia, un grano ------------------------------------------- #
    def _procesar(self, v: Vigilancia, grano: str, crudo: pd.DataFrame,
                  estado: Optional[pd.DataFrame], desde_relectura: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg, f = self.cfg.para(v), self.fechas      # con los `ajustes` de esta vigilancia
        series = armar_series(crudo, v, cfg, grano)
        series = combinar_historia(estado, series, v, cfg, grano, desde_relectura)
        if series.empty:
            return pd.DataFrame(), pd.DataFrame()

        ultimo = ultimo_periodo_cerrado(f.ayer, grano)
        series = series[series["periodo"] <= ultimo]
        if series.empty:
            return pd.DataFrame(), pd.DataFrame()
        primero = int(indice_periodo(pd.Series([f.ayer - pd.Timedelta(days=cfg.dias_historia)]), grano)[0])
        claves, periodos, M = matriz_series(series, primero, ultimo)
        evaluados = min(int(cfg.periodos_evaluados[grano]), M.shape[1])
        idx = np.arange(M.shape[1] - evaluados, M.shape[1])

        # cuánto pesa cada serie: participación en el total de la ventana evaluada
        totales = np.nansum(np.where(np.isfinite(M), M, 0.0), axis=1)
        suma = float(np.abs(totales).sum())
        participacion = np.abs(totales) / suma if suma else np.zeros_like(totales)
        peso = participacion * max(len(claves), 1)     # 1 = la serie promedio de esta vigilancia

        # escala: logarítmica si la métrica es positiva (el ruido de una venta es relativo)
        finitos = np.isfinite(M)
        positivos = finitos & (M > 0)
        usar_log = (v.escala == "log") or (
            v.escala == "auto" and finitos.any() and positivos.sum() >= 0.95 * finitos.sum())
        if usar_log:
            with np.errstate(divide="ignore", invalid="ignore"):
                X = np.where(M > 0, np.log(M), np.nan)
        else:
            X = M
        ajuste = np.zeros_like(M) if usar_log else np.ones_like(M)
        if grano == "dia" and cfg.desestacionalizar_dia:
            X, ajuste = desestacionalizar_semanal(X, periodos, aditivo=usar_log)
        # días cerrados (el domingo de un B2B): sobre los valores originales, en las dos
        # escalas. No se evalúan, y salen de las medianas y de las rectas.
        cerrados = (dias_cerrados(M, periodos, cfg.dia_cerrado_relativo) if grano == "dia"
                    else np.zeros(M.shape, dtype=bool))
        if cerrados.any():
            X = np.where(cerrados, np.nan, X)

        def a_original(esperado_detectado: np.ndarray) -> np.ndarray:
            """Del espacio en el que miran los detectores, de vuelta a USD (o lo que sea)."""
            e = esperado_detectado + ajuste[:, idx] if usar_log else esperado_detectado * ajuste[:, idx]
            with np.errstate(over="ignore"):
                return np.exp(e) if usar_log else e

        M_det = X
        eventos: List[dict] = []
        esperado_ultimo = np.full(len(claves), np.nan)
        z_ultimo = np.zeros(len(claves))
        neutros = cerrados[:, idx]          # días cerrados: ni marcan ni cortan un evento
        extra = {"original": M, "cerrados": cerrados}
        for nombre in cfg.detectores_de(v, grano):
            if nombre in NECESITAN_ORIGINAL:
                d = DETECTORES[nombre].funcion(M_det, idx, cfg, v, grano, usar_log, extra=extra)
                esperado = d.esperado        # ya viene en unidades originales
            else:
                d = DETECTORES[nombre].funcion(M_det, idx, cfg, v, grano, usar_log)
                esperado = a_original(d.esperado)
            en_cerrados = nombre == "dia_cerrado"      # éste justamente mira los días cerrados
            if neutros.any() and not en_cerrados:
                d = Deteccion(marca=d.marca & ~neutros, z=np.where(neutros, 0.0, d.z),
                              esperado=d.esperado)
            eventos += [dict(e, vigilancia=v.nombre, grano=grano)
                        for e in agrupar_eventos(d.marca, d.z, esperado, M[:, idx], claves,
                                                 periodos[idx], nombre, v, cfg, participacion, peso,
                                                 None if en_cerrados else neutros)]
            if nombre == "salto":
                esperado_ultimo = esperado[:, -1]
                z_ultimo = d.z[:, -1]

        # una fila por serie, siempre, con su historia completa
        historial = (pd.DataFrame({"clave": series["clave"], "periodo": series["periodo"],
                                   "valor": series["valor"]})
                     .sort_values(["clave", "periodo"]).groupby("clave"))
        textos = {k: historial_json(g["periodo"].to_numpy(np.int64), g["valor"].to_numpy(float),
                                    cfg.decimales) for k, g in historial}
        LOGGER.debug("[%s/%s] escala %s", v.nombre, grano, "logarítmica" if usar_log else "lineal")
        con_datos = np.isfinite(M).sum(axis=1)
        filas_serie = pd.DataFrame({
            "vigilancia": v.nombre, "grano": grano, "clave": claves,
            "categorias": " | ".join(v.claves()), "unidad": v.unidad,
            "ultimo": np.nan_to_num(M[:, -1]), "esperado": esperado_ultimo, "z": z_ultimo,
            "nivel": [nivel_por_z(x, cfg) for x in z_ultimo],
            "periodos_con_datos": con_datos, "participacion": participacion,
            "fecha_ultimo_periodo": inicio_periodo([periodos[-1]] * len(claves), grano),
            "historial": [textos.get(k, historial_json(np.zeros(0), np.zeros(0))) for k in claves],
        })
        if not eventos:
            return filas_serie, pd.DataFrame()
        ev = pd.DataFrame(eventos)
        ev["fecha_inicio"] = inicio_periodo(ev["periodo_inicio"].to_numpy(), grano)
        ev["fecha_fin"] = inicio_periodo(ev["periodo_fin"].to_numpy(), grano)
        ev["unidad"] = v.unidad
        ev["categorias"] = " | ".join(v.claves())
        return filas_serie, ev

    # -- todo ---------------------------------------------------------------- #
    def run(self, datos: Dict[str, pd.DataFrame], estado: Optional[pd.DataFrame] = None,
            eventos_previos: Optional[pd.DataFrame] = None,
            causas: Optional[pd.DataFrame] = None,
            desde_relectura: Optional[pd.Timestamp] = None) -> Resultado:
        cfg, f = self.cfg, self.fechas
        t_inicio = time.time()
        series_todas, eventos_todos = [], []
        for v in cfg.vigilancias:
            if not v.activa:
                continue
            crudo = datos.get(v.nombre)
            if crudo is None or crudo.empty:
                LOGGER.warning("[%s] la fuente no devolvió filas: se saltea", v.nombre)
                continue
            for grano in v.granos:
                t0 = time.time()
                corte = (int(indice_periodo(pd.Series([desde_relectura]), grano)[0])
                         if desde_relectura is not None else -10 ** 9)
                s, e = self._procesar(v, grano, crudo, estado, corte)
                if len(s):
                    series_todas.append(s)
                if len(e):
                    eventos_todos.append((v, grano, crudo, e))
                self.tiempos_[f"{v.nombre}:{grano}"] = time.time() - t0
                LOGGER.info("[%s/%s] %s series, %s eventos (%.1fs)", v.nombre, grano,
                            f"{len(s):,}", f"{len(e):,}", time.time() - t0)

        series = pd.concat(series_todas, ignore_index=True) if series_todas else pd.DataFrame()
        if not eventos_todos:
            vacio = pd.DataFrame()
            self.tiempos_["total"] = time.time() - t_inicio
            return Resultado(series, vacio, vacio, self._resumen(series, vacio))

        crudos = unificar(pd.concat([e for _, _, _, e in eventos_todos], ignore_index=True), cfg)
        eventos = conciliar(crudos, eventos_previos, f)
        eventos = buscar_causas(eventos, causas, f)

        # atribución sólo de lo que importa: explicar cuesta y no se explica lo trivial
        eventos["atribucion"] = ""
        for v, grano, crudo, _ in eventos_todos:
            sel = eventos[(eventos["vigilancia"] == v.nombre) & (eventos["grano"] == grano)
                          & eventos["nivel"].isin(("ALERTA", "CRITICO"))
                          & (eventos["estado"] != "CERRADO") & eventos["principal"].fillna(True)]
            if not len(sel):
                continue
            textos = atribuir(crudo, sel, v, cfg.para(v), grano)
            if textos:
                eventos["atribucion"] = np.where(eventos["id_evento"].isin(textos),
                                                 eventos["id_evento"].map(textos).fillna(""),
                                                 eventos["atribucion"])
        notificaciones = self._notificar(eventos)
        self.tiempos_["total"] = time.time() - t_inicio
        LOGGER.info("%s series, %s eventos (%s nuevos, %s en curso, %s cerrados), %s notificaciones en %.1fs",
                    f"{len(series):,}", f"{len(eventos):,}",
                    int((eventos['estado'] == 'NUEVO').sum()), int((eventos['estado'] == 'EN_CURSO').sum()),
                    int((eventos['estado'] == 'CERRADO').sum()), f"{len(notificaciones):,}",
                    time.time() - t_inicio)
        return Resultado(series, eventos, notificaciones, self._resumen(series, eventos))

    def _notificar(self, eventos: pd.DataFrame) -> pd.DataFrame:
        """Se notifica lo nuevo y lo que empeoró; lo que sigue igual no vuelve a molestar."""
        cfg = self.cfg
        if eventos.empty:
            return pd.DataFrame()
        # el piso puede ser propio de cada vigilancia (una métrica ruidosa avisa sólo lo grave)
        piso_de = {v.nombre: NIVELES.index(cfg.para(v).nivel_notificacion) for v in cfg.vigilancias}
        piso_fila = eventos["vigilancia"].map(piso_de).fillna(NIVELES.index(cfg.nivel_notificacion))
        grave = eventos["nivel"].map(NIVELES.index) >= piso_fila
        anterior = eventos["nivel_anterior"].map(lambda n: NIVELES.index(n) if n in NIVELES else -1)
        actual = eventos["nivel"].map(lambda n: NIVELES.index(n))
        nuevo = eventos["estado"] == "NUEVO"
        empeoro = (eventos["estado"] == "EN_CURSO") & (actual > anterior)
        principal = eventos["principal"].fillna(True) if "principal" in eventos else True
        sel = eventos[grave & (nuevo | empeoro) & principal].copy()
        sel["motivo_notificacion"] = np.where(sel["estado"] == "NUEVO", "evento nuevo",
                                              "el evento empeoró de nivel")
        return sel.sort_values(["nivel", "materialidad"], ascending=[True, False]).reset_index(drop=True)

    def _resumen(self, series: pd.DataFrame, eventos: pd.DataFrame) -> pd.DataFrame:
        """Cuántos eventos por vigilancia, grano y nivel, y cuánta plata hay atrás."""
        if eventos is None or eventos.empty:
            return pd.DataFrame(columns=["vigilancia", "grano", "nivel", "estado", "eventos",
                                         "materialidad", "series_vigiladas"])
        conteo = (eventos.groupby(["vigilancia", "grano", "nivel", "estado"], dropna=False)
                  .agg(eventos=("id_evento", "count"), materialidad=("materialidad", "sum"))
                  .reset_index())
        if len(series):
            vigiladas = series.groupby(["vigilancia", "grano"]).size().rename("series_vigiladas").reset_index()
            conteo = conteo.merge(vigiladas, on=["vigilancia", "grano"], how="left")
        else:
            conteo["series_vigiladas"] = 0
        return conteo.sort_values(["vigilancia", "grano", "nivel"]).reset_index(drop=True)

    def tiempos(self, top: int = 10) -> pd.Series:
        return pd.Series(self.tiempos_).sort_values(ascending=False).head(top)


# =========================================================================== #
# 7. Calibración: elegir los umbrales con los datos propios
# =========================================================================== #
def inyectar_fallas(df: pd.DataFrame, v: Vigilancia, cfg: VigConfig, n: int = 20,
                    tipos: Sequence[str] = ("apagon", "escalon", "pico"),
                    dias: int = 7, semilla: int = 0) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Mete fallas conocidas en series REALES, para medir cuántas detecta cada umbral.

    apagon  -> la serie se va a cero los últimos `dias`
    escalon -> cae a la mitad los últimos 3 x `dias`
    pico    -> el último día se multiplica por 5

    Devuelve los datos con las fallas y qué falla le tocó a cada serie.
    """
    rng = np.random.default_rng(semilla)
    f = Fechas.desde(cfg.fecha_ejecucion)
    base = df.copy()
    base["_clave"] = _clave_texto(base, v.claves())
    claves = base["_clave"].dropna().unique()
    if not len(claves):
        return df, {}
    elegidas = rng.choice(claves, size=min(int(n), len(claves)), replace=False)
    fecha = pd.to_datetime(base[cfg.col_fecha])
    marcadas: Dict[str, str] = {}
    for i, clave in enumerate(elegidas):
        tipo = tipos[i % len(tipos)]
        es = base["_clave"] == clave
        if tipo == "apagon":
            corte = f.ayer - pd.Timedelta(days=dias - 1)
            base.loc[es & (fecha >= corte), v.metrica] = 0.0
        elif tipo == "escalon":
            corte = f.ayer - pd.Timedelta(days=3 * dias - 1)
            base.loc[es & (fecha >= corte), v.metrica] *= 0.5
        else:
            base.loc[es & (fecha >= f.ayer), v.metrica] *= 5.0
        marcadas[str(clave)] = tipo
    return base.drop(columns="_clave"), marcadas


def calibrar(datos: Dict[str, pd.DataFrame], cfg_base: VigConfig,
             rejilla: Optional[Dict[str, Sequence[Any]]] = None,
             objetivo_alertas_dia: Optional[float] = None,
             n_fallas: int = 20, semilla: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Prueba umbrales sobre TUS datos y devuelve, para cada uno, cuántas alertas tira
    por día y cuántas fallas inyectadas detecta.

    La rejilla por defecto mueve el umbral de ALERTA y el desvío mínimo relativo:

        calibrar(datos, cfg, objetivo_alertas_dia=3)

    El umbral recomendado es el que más fallas detecta sin pasarse del objetivo.
    """
    from dataclasses import replace
    from itertools import product

    rejilla = dict(rejilla or {"alerta": [4.0, 5.0, 6.0, 8.0],
                               "desvio_relativo_minimo": [0.05, 0.10, 0.20]})
    claves = list(rejilla)
    combos = [dict(zip(claves, valores)) for valores in product(*(rejilla[k] for k in claves))]

    # las mismas fallas para todas las configuraciones, así se comparan entre sí
    datos_falla: Dict[str, pd.DataFrame] = {}
    verdad: Dict[str, Dict[str, str]] = {}
    for v in cfg_base.vigilancias:
        df = datos.get(v.nombre)
        if df is None or df.empty:
            continue
        datos_falla[v.nombre], verdad[v.nombre] = inyectar_fallas(df, v, cfg_base, n=n_fallas,
                                                                  semilla=semilla)
    total_fallas = sum(len(x) for x in verdad.values())

    granos = {g for v in cfg_base.vigilancias for g in v.granos}
    divisor = cfg_base.periodos_evaluados.get("dia", 30) if "dia" in granos else 1

    filas = []
    for combo in combos:
        alerta = float(combo.get("alerta", cfg_base.umbral_z["ALERTA"]))
        umbral = {"ATENCION": round(alerta * 0.6, 2), "ALERTA": alerta, "CRITICO": round(alerta * 1.6, 2)}
        extra = {k: v for k, v in combo.items() if k != "alerta"}
        cfg = replace(cfg_base, umbral_z=umbral, verbose=0, **extra)

        t0 = time.time()
        limpio = VigEngine(cfg).run(datos)
        alertas = len(limpio.notificaciones)

        detectadas = 0
        if total_fallas:
            con_falla = VigEngine(cfg).run(datos_falla)
            avisadas = set(zip(con_falla.notificaciones.get("vigilancia", []),
                               con_falla.notificaciones.get("clave", []))) if len(con_falla.notificaciones) else set()
            detectadas = sum(1 for vig, marcadas in verdad.items()
                             for clave in marcadas if (vig, clave) in avisadas)

        fila = {**umbral, **extra, "eventos": int(len(limpio.eventos)),
                "alertas": int(alertas), "alertas_por_dia": round(alertas / max(divisor, 1), 2),
                "fallas_detectadas": int(detectadas), "fallas_inyectadas": int(total_fallas),
                "deteccion": round(detectadas / total_fallas, 3) if total_fallas else None,
                "segundos": round(time.time() - t0, 1)}
        filas.append(fila)
        if verbose:
            LOGGER.info("umbral ALERTA %.1f, desvío mínimo %s -> %.2f alertas/día, detecta %s de %s",
                        alerta, extra.get("desvio_relativo_minimo", "-"), fila["alertas_por_dia"],
                        detectadas, total_fallas)

    out = pd.DataFrame(filas)
    if out.empty:
        return out
    out = out.sort_values(["deteccion", "alertas_por_dia"], ascending=[False, True]).reset_index(drop=True)
    out["recomendado"] = False
    if objetivo_alertas_dia is not None:
        caben = out[out["alertas_por_dia"] <= float(objetivo_alertas_dia)]
        if len(caben):
            out.loc[caben.index[0], "recomendado"] = True
        else:
            out.loc[out["alertas_por_dia"].idxmin(), "recomendado"] = True
            LOGGER.warning("ninguna configuración baja de %s alertas por día: se marca la más silenciosa. "
                           "Subí el umbral o el desvío mínimo, o vigilá menos series.",
                           objetivo_alertas_dia)
    else:
        out.loc[0, "recomendado"] = True
    return out


def bloque_config_vig(fila: pd.Series) -> str:
    """El texto para pegar en vig_oracle.py con los umbrales elegidos."""
    def limpio(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return round(float(x), 6)
        return x

    lineas = [f'UMBRAL_Z = {{"ATENCION": {limpio(fila["ATENCION"])}, "ALERTA": {limpio(fila["ALERTA"])}, '
              f'"CRITICO": {limpio(fila["CRITICO"])}}}']
    for k in ("desvio_relativo_minimo", "piso_sigma_relativo", "peso_relativo_minimo",
              "min_periodos", "nivel_notificacion", "nivel_minimo_evento"):
        if k in fila.index:
            lineas.append(f"{k.upper()} = {limpio(fila[k])!r}")
    lineas += ["", f"# calibrado con datos propios: {fila['alertas_por_dia']} alertas por día, "
                   f"detecta {fila['fallas_detectadas']} de {fila['fallas_inyectadas']} fallas inyectadas",
               f"# eventos guardados por corrida: {fila['eventos']}"]
    return "\n".join(lineas)


def catalogo_detectores() -> pd.DataFrame:
    return pd.DataFrame([{"detector": d.nombre, "descripcion": d.descripcion}
                         for d in DETECTORES.values()])


__all__ = ["Vigilancia", "VigConfig", "VigEngine", "Resultado", "Fechas", "DETECTORES", "NIVELES",
           "unificar", "conciliar",
           "GRANOS", "ESTADOS", "catalogo_detectores", "historial_json", "historial_a_serie",
           "indice_periodo", "inicio_periodo", "nivel_por_z", "ultimo_periodo_cerrado",
           "desestacionalizar_semanal", "dias_cerrados", "calibrar", "inyectar_fallas", "bloque_config_vig"]
