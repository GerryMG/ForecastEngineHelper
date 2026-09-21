# -*- coding: utf-8 -*-
"""
Motor de recomendación dentro del segmento.

La idea: dentro de un grupo comparable, una entidad debería estar comprando lo que
compran sus pares. Lo que no compra, lo que dejó de comprar y lo que compra poco
son las tres recomendaciones.

Todo es genérico. "Entidad" e "ítem" son dos conjuntos cualesquiera de categorías
(cliente/producto, cliente/servicio, sucursal/submarca, vendedor/familia): se
declaran en RecConfig y el motor no sabe de qué se trata.

Estructura del archivo
  1. Configuración          RecConfig
  2. Panel y matrices       Panel, Matriz, Bloque
  3. Algoritmos             la batería de puntuadores
  4. Tipos y potencial      CRUZADA / REPOSICION / BRECHA y el USD en juego
  5. Backtest               qué algoritmo acierta más en cada segmento
  6. Orquestador            RecEngine.run()
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

LOGGER = logging.getLogger("rec")

_EPOCA = pd.Timestamp("1970-01-01")
TIPOS = ("CRUZADA", "REPOSICION", "BRECHA")


def _dia(ts) -> int:
    return int((pd.Timestamp(ts).normalize() - _EPOCA).days)


def sin_residuo(suma, suma_abs, tol: float, piso: float = 0.0):
    """Anula la suma cuando no se distingue del residuo de cancelación de float64.

    Sumar una venta y su devolución no da cero exacto: queda un residuo del orden de
    1e-16 veces lo que pasó por la suma. Como número no molesta; como denominador de
    un porcentaje (el margen del ítem, la participación de la entidad) explota. Por eso
    el cero se prueba contra la escala real, que es la suma de los valores absolutos.

    `piso` es una escala mínima de referencia (la del panel típico): sirve para el
    residuo que ya llega cancelado desde la fuente, donde no hay nada con qué
    compararlo dentro de la propia suma.
    """
    if not tol:
        return suma
    suma = np.asarray(suma, dtype=float)
    escala = np.maximum(np.asarray(suma_abs, dtype=float), float(piso or 0.0))
    return np.where(np.abs(suma) <= tol * escala, 0.0, suma)


def escala_tipica(suma_abs) -> float:
    """Escala de referencia del panel: la mediana de lo que movió cada grupo."""
    a = np.asarray(suma_abs, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.median(a)) if a.size else 0.0


# =========================================================================== #
# 1. Configuración
# =========================================================================== #
@dataclass
class RecConfig:
    """Qué se recomienda, a quién, con quién se compara y con qué reglas."""

    #: a quién se le recomienda. SK_/BK_ agrupan, BD_ describen (valor más reciente).
    entidad: Sequence[str] = ("SK_CLIENTE", "BD_CLIENTE")
    #: qué se recomienda: el nivel al que se arma la matriz (submarca, familia, servicio...).
    item: Sequence[str] = ("BK_ITEM", "BD_ITEM")
    #: jerarquía de segmentación, del más GRUESO al más FINO (ej. ["BD_SEGMENTO", "BD_SUBSEGMENTO"]).
    #: Se usa el nivel más fino que tenga suficientes entidades; si ninguno alcanza, GLOBAL.
    segmentos: Sequence[str] = ()

    col_fecha: str = "FECHA"
    col_valor: str = "MT_VENTA"        #: importe (USD)
    col_margen: str = "MT_MARGEN"      #: margen bruto (USD)
    fecha_ejecucion: Optional[Any] = None

    # -- tipo de documento --------------------------------------------------- #
    #: columna con el tipo de documento (factura, nota de crédito, devolución...).
    col_tipo_documento: Optional[str] = None
    #: sólo estos tipos cuentan como compra. Vacío = todos los que no estén excluidos.
    tipos_venta: Sequence[str] = ()
    #: estos restan. Si en la fuente ya vienen en negativo, dejá `devolucion_ya_negativa=True`.
    tipos_devolucion: Sequence[str] = ()
    devolucion_ya_negativa: bool = True
    #: estos se ignoran por completo (fletes, ajustes, documentos internos).
    tipos_excluidos: Sequence[str] = ()
    #: si después de restar devoluciones el par queda en cero o negativo, no cuenta como compra.
    excluir_netos_no_positivos: bool = True

    # -- cómo se mide la afinidad ------------------------------------------- #
    #: "canasta" mira lo que se compra JUNTO (mismo cliente, mismo día: si el cliente compra
    #: una vez por día, el día es el ticket). "repertorio" mira todo lo que compra en la
    #: ventana, junto o no. Si tenés número de documento, pasalo en `col_documento`.
    afinidad: str = "repertorio"
    col_documento: Optional[str] = None

    # -- tamaño de la entidad ------------------------------------------------ #
    #: agrega un nivel de segmentación por tamaño (cuantiles de compra en la ventana).
    #: Un cliente grande compra de todo y arrastra la afinidad; uno mediano parece no
    #: comprar lo que en realidad no le corresponde. [] = no segmentar por tamaño.
    cortes_tamano: Sequence[float] = ()
    etiquetas_tamano: Sequence[str] = ("CHICO", "MEDIANO", "GRANDE", "TOP")

    # -- ventanas ----------------------------------------------------------- #
    #: ventana que arma la matriz de compras. 0 = TODA la historia que traiga la fuente.
    dias_afinidad: int = 365
    dias_backtest: int = 90            #: tramo final reservado para medir aciertos

    # -- recortes ----------------------------------------------------------- #
    min_entidades_segmento: int = 200  #: menos que esto y sube al nivel de arriba
    min_adopciones_backtest: int = 30  #: menos que esto y el segmento usa el ganador del panel
    min_soporte: int = 5               #: entidades del segmento que compran el ítem
    min_penetracion: float = 0.02      #: fracción del segmento que lo compra
    max_items_reco: int = 10           #: recomendaciones por entidad

    # -- batería ------------------------------------------------------------ #
    algoritmos: Sequence[str] = ("popularidad", "coseno_item", "coseno_entidad",
                                 "svd", "kmeans_valor", "reglas")
    #: backtest = mide y elige el mejor por segmento; rrf = fusiona todos;
    #: ponderado = usa `pesos`; o el nombre de un algoritmo para forzarlo.
    seleccion: str = "backtest"
    metrica_seleccion: str = "precision"    #: precision | usd | recall
    pesos: Dict[str, float] = field(default_factory=dict)
    k_vecinos: int = 50                #: vecinos en coseno_entidad
    k_factores: int = 32               #: factores latentes en svd
    k_clusters: int = 8                #: grupos de kmeans_valor
    semilla: int = 0

    # -- tipos de recomendación --------------------------------------------- #
    incluir_tipos: Sequence[str] = TIPOS
    factor_reposicion: float = 1.5     #: silencio > factor x intervalo típico del par
    #: días de compra que necesita un par para hablar de "su ritmo". Con 2 compras hay un
    #: solo hueco y el ritmo es una casualidad, no un ritmo: el mínimo razonable es 3.
    min_compras_reposicion: int = 3
    #: qué tan irregular puede ser: desvío / promedio de los intervalos. None = no filtrar.
    #: Un cliente que compró en enero, en marzo y en diciembre no está "atrasado".
    max_cv_intervalo: Optional[float] = 1.0
    brecha_ratio: float = 0.5          #: compra menos de esta fracción de lo que compran sus pares

    # -- topes del USD potencial --------------------------------------------- #
    escalar_potencial: bool = True     #: ajusta el USD potencial por tamaño de la entidad
    tope_escala: float = 3.0           #: tope de ese ajuste
    #: horizonte de la estimación: "USD esperados en los próximos N días". Es la unidad
    #: común de los tres tipos, para que el ranking compare lo mismo.
    horizonte_dias: int = 90
    #: cuánta evidencia de los pares se le presta al cliente que tiene poca propia.
    #: k = 3 significa "sus datos valen tanto como los pares cuando tiene 3 intervalos".
    #: 0 = no prestar nada (sólo su historia).
    peso_prior_pares: float = 3.0
    #: multiplicar el valor por la probabilidad de que la compra ocurra:
    #: recompra (reposición, por la distribución de intervalos del segmento) y
    #: adopción (cruzada, por la tasa que midió el backtest en ese segmento).
    usar_probabilidad: bool = True
    #: vida media para pesar la afinidad por recencia: lo de hace `n` días pesa la mitad.
    #: 0 = todo pesa igual. Sirve cuando el mix de compra cambia con el tiempo.
    vida_media_afinidad_dias: int = 0
    #: casos mínimos para creerle a la curva de recuperación de un ítem; con menos, se usa
    #: la del panel entero.
    min_casos_recuperacion: int = 30
    #: por qué se ordena. "esperado" = USD por probabilidad (asigna bien el esfuerzo del
    #: vendedor); "bruto" = el tamaño de la oportunidad sin descontar la probabilidad
    #: (deja arriba a los clientes muy atrasados, que son campañas de recuperación).
    ordenar_por: str = "esperado"
    #: piso de la probabilidad. Con 0, un ítem que nadie recupera nunca vale 0 y cae al
    #: fondo; con 0,05 se le deja una chance mínima y sigue compitiendo.
    piso_prob: float = 0.0
    #: la reposición no puede valer más que esta fracción de lo que la entidad compró de
    #: ESE ítem en la ventana. 0 = sin tope.
    tope_potencial_por_historico: float = 1.0
    #: ninguna recomendación puede valer más que esta fracción de la compra total de la
    #: entidad en la ventana. 0 = sin tope.
    tope_potencial_relativo: float = 1.0
    #: días de compra mínimos de la ENTIDAD para recomendarle algo. Con una o dos compras
    #: en el año no hay con qué sostener una recomendación.
    min_dias_compra_entidad: int = 3

    #: Cuándo una suma de dinero es cero. Una venta y su devolución no se anulan exacto
    #: en punto flotante: queda un residuo de 1e-10 que, como denominador, explota. Una
    #: suma cuenta como cero cuando no llega a esta fracción de lo que pasó por ella.
    tolerancia_cero: float = 1e-9

    filas_bloque: int = 2048           #: entidades por bloque (memoria acotada)
    decimales: int = 4
    verbose: int = 1

    # -- derivados ---------------------------------------------------------- #
    @staticmethod
    def _separar(columnas: Sequence[str]) -> Tuple[List[str], List[str]]:
        """(claves, descripciones). Por convención SK_/BK_ agrupan y BD_ describen.

        Si no hay ninguna SK_/BK_, se agrupa por todas: "recomendar a nivel BD_SUBMARCA"
        quiere decir que esa columna ES la clave, no una descripción de otra cosa.
        """
        cols = [str(c) for c in columnas]
        claves = [c for c in cols if not c.upper().startswith("BD_")]
        if not claves:
            return cols, []
        return claves, [c for c in cols if c.upper().startswith("BD_")]

    def claves_entidad(self) -> List[str]:
        return self._separar(self.entidad)[0]

    def desc_entidad(self) -> List[str]:
        return self._separar(self.entidad)[1]

    def claves_item(self) -> List[str]:
        return self._separar(self.item)[0]

    def desc_item(self) -> List[str]:
        return self._separar(self.item)[1]

    def columnas(self) -> List[str]:
        cols = list(self.entidad) + list(self.item) + list(self.segmentos) + [
            self.col_fecha, self.col_valor, self.col_margen]
        if self.col_tipo_documento:
            cols.append(self.col_tipo_documento)
        if self.col_documento:
            cols.append(self.col_documento)
        return cols

    def validate(self) -> None:
        if not list(self.entidad):
            raise ValueError("entidad no puede estar vacía: es a quién se le recomienda")
        if not list(self.item):
            raise ValueError("item no puede estar vacío: es qué se recomienda")
        repetidas = set(self.claves_entidad()) & set(self.claves_item())
        if repetidas:
            raise ValueError(f"las claves {sorted(repetidas)} están en entidad y en item")
        desconocidos = [a for a in self.algoritmos if a not in ALGORITMOS]
        if desconocidos:
            raise ValueError(f"algoritmos desconocidos {desconocidos}; hay {list(ALGORITMOS)}")
        if not self.algoritmos:
            raise ValueError("hace falta al menos un algoritmo")
        if self.seleccion not in ("backtest", "rrf", "ponderado") and self.seleccion not in self.algoritmos:
            raise ValueError("seleccion debe ser backtest, rrf, ponderado o el nombre de un algoritmo activo")
        if self.ordenar_por not in ("esperado", "bruto"):
            raise ValueError("ordenar_por debe ser esperado o bruto")
        if not 0 <= self.tolerancia_cero < 1:
            raise ValueError("tolerancia_cero debe estar entre 0 y 1 (0 = sin limpieza)")
        if self.metrica_seleccion not in ("precision", "usd", "recall"):
            raise ValueError("metrica_seleccion debe ser precision, usd o recall")
        malos = [t for t in self.incluir_tipos if t not in TIPOS]
        if malos:
            raise ValueError(f"incluir_tipos {malos} no está en {TIPOS}")
        if self.afinidad not in ("repertorio", "canasta"):
            raise ValueError("afinidad debe ser repertorio o canasta")
        if self.cortes_tamano:
            cortes = list(self.cortes_tamano)
            if sorted(cortes) != cortes or not all(0 < c < 1 for c in cortes):
                raise ValueError("cortes_tamano son cuantiles crecientes entre 0 y 1")
            if len(self.etiquetas_tamano) < len(cortes) + 1:
                raise ValueError(f"hacen falta {len(cortes) + 1} etiquetas_tamano")


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
# 2. Panel y matrices
# =========================================================================== #
def _codificar(df: pd.DataFrame, columnas: Sequence[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Código entero por combinación de columnas + tabla de valores únicos."""
    columnas = list(columnas)
    if len(columnas) == 1:
        codigos, valores = pd.factorize(df[columnas[0]], sort=True)
        return codigos.astype(np.int64), pd.DataFrame({columnas[0]: valores})
    idx = pd.MultiIndex.from_frame(df[columnas])
    codigos, valores = pd.factorize(idx, sort=True)
    return codigos.astype(np.int64), valores.to_frame(index=False)[columnas]


def _ultimo_valor(df: pd.DataFrame, codigos: np.ndarray, n: int,
                  columnas: Sequence[str], orden_dia: np.ndarray) -> pd.DataFrame:
    """Valor más reciente de cada columna descriptiva, por código."""
    out = pd.DataFrame(index=pd.RangeIndex(n))
    if not len(columnas):
        return out
    orden = np.lexsort((orden_dia, codigos))
    c = codigos[orden]
    ultima = np.r_[c[1:] != c[:-1], True]
    destino = c[ultima]
    for col in columnas:
        valores = df[col].to_numpy()[orden][ultima]
        serie = pd.Series(valores, index=destino)
        out[col] = serie.reindex(pd.RangeIndex(n)).to_numpy()
    return out


class Matriz:
    """Entidades x ítems dentro de una ventana, ya agregadas."""

    def __init__(self, n_ent: int, n_item: int, tab: pd.DataFrame, tolerancia_cero: float = 0.0):
        self.n_ent, self.n_item, self.tab = n_ent, n_item, tab
        self.tol = float(tolerancia_cero)
        e = tab["e"].to_numpy(np.int64)
        i = tab["i"].to_numpy(np.int64)
        forma = (n_ent, n_item)

        def csr(valores):
            return sp.csr_matrix((np.asarray(valores, float), (e, i)), shape=forma)

        self.R = csr(np.ones(len(tab)))                     # compró: 1 / 0
        self.V = csr(tab["usd"].to_numpy(float))            # USD en la ventana
        self.M = csr(tab["margen"].to_numpy(float))         # margen USD
        self.D = csr(tab["dias"].to_numpy(float))           # días de compra
        self.U = csr(tab["ultimo"].to_numpy(float))         # último día (época)
        self.P = csr(tab["primero"].to_numpy(float))        # primer día (época)
        bruto_ent = np.asarray(abs(self.V).sum(1)).ravel()
        self.venta_entidad = sin_residuo(np.asarray(self.V.sum(1)).ravel(), bruto_ent,
                                         self.tol, escala_tipica(bruto_ent))
        self.items_entidad = np.asarray(self.R.sum(1)).ravel()
        self.dias_entidad = np.zeros(n_ent)      # días de compra de la entidad, lo llena Panel
        self.huecos: Tuple[np.ndarray, np.ndarray, np.ndarray] = ()   # los llena Panel
        self.censuras: Tuple[np.ndarray, np.ndarray, np.ndarray] = ()

    #: razones silencio/intervalo en las que se mide la curva
    REJILLA_ATRASO = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0)

    def curva_recuperacion(self, horizonte: float, min_casos: int = 30
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """P(vuelve a comprar el ítem dentro del horizonte | lleva X veces su intervalo sin comprar).

        Se estima con la historia y nada más: para cada nivel de atraso se cuenta cuántos
        casos llegaron a ese atraso (huecos observados más silencios todavía abiertos) y
        cuántos de esos volvieron a comprar dentro del horizonte. No supone ninguna
        distribución.

        Devuelve (rejilla de atrasos, curva por ítem, curva global).
        """
        clave = (round(float(horizonte), 3), int(min_casos))
        if getattr(self, "_curva_cache", (None,))[0] == clave:
            return self._curva_cache[1]
        if not hasattr(self, "huecos"):
            vacio = np.full((self.n_item, len(self.REJILLA_ATRASO)), np.nan)
            return np.array(self.REJILLA_ATRASO), vacio, np.full(len(self.REJILLA_ATRASO), np.nan)
        it_h, gap, iv_h = self.huecos
        it_c, sil, iv_c = self.censuras
        rejilla = np.array(self.REJILLA_ATRASO, dtype=float)
        por_item = np.full((self.n_item, len(rejilla)), np.nan)
        global_ = np.full(len(rejilla), np.nan)
        for k, a in enumerate(rejilla):
            umbral_h = a * iv_h
            umbral_c = a * iv_c
            en_riesgo_h = gap >= umbral_h
            # un silencio abierto sólo cuenta como "no volvió" si ya observamos el horizonte
            # completo después de haberse atrasado; si no, todavía no sabemos y se excluye
            en_riesgo_c = sil >= umbral_c + horizonte
            volvio = en_riesgo_h & (gap <= umbral_h + horizonte)
            riesgo_item = (np.bincount(it_h[en_riesgo_h], minlength=self.n_item)
                           + np.bincount(it_c[en_riesgo_c], minlength=self.n_item)).astype(float)
            volvio_item = np.bincount(it_h[volvio], minlength=self.n_item).astype(float)
            suficiente = riesgo_item >= min_casos
            por_item[suficiente, k] = volvio_item[suficiente] / riesgo_item[suficiente]
            riesgo_total = float(en_riesgo_h.sum() + en_riesgo_c.sum())
            if riesgo_total >= min_casos:
                global_[k] = float(volvio.sum()) / riesgo_total
        self._curva_cache = (clave, (rejilla, por_item, global_))
        return self._curva_cache[1]

    def __repr__(self) -> str:
        densidad = self.R.nnz / max(self.n_ent * self.n_item, 1)
        return (f"Matriz({self.n_ent:,} entidades x {self.n_item:,} ítems, "
                f"{self.R.nnz:,} pares, densidad {densidad:.2%})")


class Panel:
    """Eventos codificados + catálogos de entidades, ítems y segmentos."""

    def __init__(self, cfg: RecConfig, fechas: Fechas, ent: np.ndarray, item: np.ndarray,
                 dia: np.ndarray, usd: np.ndarray, margen: np.ndarray,
                 entidades: pd.DataFrame, items: pd.DataFrame, segmentos: pd.DataFrame):
        self.cfg, self.f = cfg, fechas
        self.ent, self.item, self.dia, self.usd, self.margen = ent, item, dia, usd, margen
        self.entidades, self.items, self.segmentos = entidades, items, segmentos
        self.n_ent, self.n_item = len(entidades), len(items)
        self.doc: Optional[np.ndarray] = None       # documento de cada fila, si la fuente lo trae

    def matriz(self, desde: int, hasta: int) -> Matriz:
        """Agrega los eventos de [desde, hasta] a una fila por entidad-ítem."""
        m = (self.dia >= desde) & (self.dia <= hasta)
        base = pd.DataFrame({"e": self.ent[m], "i": self.item[m], "d": self.dia[m],
                             "v": self.usd[m], "g": self.margen[m],
                             "va": np.abs(self.usd[m]), "ga": np.abs(self.margen[m])})
        por_dia = base.groupby(["e", "i", "d"], sort=False, as_index=False).agg(
            v=("v", "sum"), g=("g", "sum"), va=("va", "sum"), ga=("ga", "sum"))
        tab = por_dia.groupby(["e", "i"], sort=False, as_index=False).agg(
            usd=("v", "sum"), margen=("g", "sum"), dias=("d", "size"),
            primero=("d", "min"), ultimo=("d", "max"),
            usd_bruto=("va", "sum"), margen_bruto=("ga", "sum"))
        # el par que se compró y se devolvió entero vale 0, no 1e-10 (ver `sin_residuo`)
        tol = self.cfg.tolerancia_cero
        bu, bm = tab.pop("usd_bruto").to_numpy(float), tab.pop("margen_bruto").to_numpy(float)
        tab["usd"] = sin_residuo(tab["usd"].to_numpy(float), bu, tol, escala_tipica(bu))
        tab["margen"] = sin_residuo(tab["margen"].to_numpy(float), bm, tol, escala_tipica(bm))

        # ritmo real de cada par: promedio y desvío de los huecos entre compras. Con esto se
        # distingue "compra cada 20 días" de "compró dos veces y justo pasaron 20 días".
        por_dia = por_dia.sort_values(["e", "i", "d"])
        por_dia["hueco"] = por_dia.groupby(["e", "i"], sort=False)["d"].diff()
        huecos = por_dia.groupby(["e", "i"], sort=False, as_index=False)["hueco"].agg(
            intervalo_medio="mean", intervalo_desvio="std", n_intervalos="count")
        tab = tab.merge(huecos, on=["e", "i"], how="left")

        # cada hueco observado es un caso de "se atrasó y volvió"; cada silencio final que
        # sigue abierto es un caso de "se atrasó y todavía no volvió". Con los dos se estima
        # después, sin suponer nada, cuánta gente vuelve.
        con_hueco = por_dia[por_dia["hueco"].notna()][["e", "i", "hueco"]]
        con_hueco = con_hueco.merge(tab[["e", "i", "intervalo_medio"]], on=["e", "i"], how="left")
        censura = tab[["i", "ultimo", "intervalo_medio"]].copy()
        censura["silencio"] = float(hasta) - censura["ultimo"]

        if self.cfg.excluir_netos_no_positivos:
            # comprado y devuelto entero no es una compra
            tab = tab[tab["usd"] > 0].reset_index(drop=True)   # 1e-10 ya es 0, no pasa
        m = Matriz(self.n_ent, self.n_item, tab, self.cfg.tolerancia_cero)
        dias_ent = por_dia.drop_duplicates(["e", "d"]).groupby("e").size()
        m.dias_entidad = dias_ent.reindex(range(self.n_ent), fill_value=0).to_numpy(float)
        ok_h = con_hueco["intervalo_medio"].notna().to_numpy()
        m.huecos = (con_hueco["i"].to_numpy(np.int32)[ok_h],
                    con_hueco["hueco"].to_numpy(np.float32)[ok_h],
                    con_hueco["intervalo_medio"].to_numpy(np.float32)[ok_h])
        ok_c = censura["intervalo_medio"].notna().to_numpy()
        m.censuras = (censura["i"].to_numpy(np.int32)[ok_c],
                      censura["silencio"].to_numpy(np.float32)[ok_c],
                      censura["intervalo_medio"].to_numpy(np.float32)[ok_c])
        return m

    def pares(self, desde: int, hasta: int) -> sp.csr_matrix:
        """Pares entidad-ítem con compra en la ventana, como matriz binaria."""
        m = (self.dia >= desde) & (self.dia <= hasta)
        return sp.csr_matrix((np.ones(int(m.sum())), (self.ent[m], self.item[m])),
                             shape=(self.n_ent, self.n_item))


    def canastas(self, desde: int, hasta: int) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """Matriz canasta x ítem y a qué entidad pertenece cada canasta.

        La canasta es el documento si la fuente lo trae; si no, el día: para un cliente
        que compra una vez por día, el día ES el ticket.
        """
        m = (self.dia >= desde) & (self.dia <= hasta)
        ent, item = self.ent[m], self.item[m]
        if self.doc is not None:
            marca = self.doc[m]
            clave = pd.factorize(pd.Series(ent).astype(str) + "|" + pd.Series(marca).astype(str))[0]
        else:
            ancho = int(self.dia.max()) + 1
            clave = ent * ancho + self.dia[m]
        _, fila = np.unique(clave, return_inverse=True)
        n = int(fila.max()) + 1 if len(fila) else 0
        B = sp.csr_matrix((np.ones(len(fila)), (fila, item)), shape=(n, self.n_item))
        B.data[:] = 1.0
        entidad_canasta = np.zeros(n, dtype=np.int64)
        entidad_canasta[fila] = ent
        dia_canasta = np.zeros(n, dtype=np.int64)
        dia_canasta[fila] = self.dia[m]
        return B, entidad_canasta, dia_canasta


def preparar(df: pd.DataFrame, cfg: RecConfig, fechas: Fechas) -> Panel:
    """Valida la entrada, codifica entidades e ítems y arma el panel."""
    faltan = [c for c in cfg.columnas() if c not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en la entrada: {faltan}")
    dia = pd.to_datetime(df[cfg.col_fecha]).to_numpy().astype("datetime64[D]").astype(np.int64)
    dentro = dia <= fechas.d_ayer
    if not dentro.all():
        LOGGER.info("se descartan %s filas posteriores a ayer", f"{int((~dentro).sum()):,}")
    df = df.loc[dentro].reset_index(drop=True)
    dia = dia[dentro]
    if cfg.col_tipo_documento:
        tipo = df[cfg.col_tipo_documento].astype(str)
        permitidos = set(cfg.tipos_venta) | set(cfg.tipos_devolucion)
        sirve = ~tipo.isin(list(cfg.tipos_excluidos))
        if permitidos:
            sirve &= tipo.isin(list(permitidos))
        if not sirve.all():
            LOGGER.info("se descartan %s filas por tipo de documento (%s)",
                        f"{int((~sirve).sum()):,}", sorted(set(tipo[~sirve]))[:8])
        df = df.loc[sirve.to_numpy()].reset_index(drop=True)
        dia = dia[sirve.to_numpy()]
    if df.empty:
        raise ValueError("No quedan filas hasta ayer")

    for que, columnas in (("entidad", cfg.claves_entidad()), ("item", cfg.claves_item())):
        vacias = [c for c in columnas if df[c].isna().all()]
        if vacias:
            raise ValueError(f"las columnas {vacias} de {que} vienen todas nulas: revisá el SQL")
    ent, entidades = _codificar(df, cfg.claves_entidad())
    item, items = _codificar(df, cfg.claves_item())
    entidades = pd.concat([entidades, _ultimo_valor(df, ent, len(entidades), cfg.desc_entidad(), dia)], axis=1)
    items = pd.concat([items, _ultimo_valor(df, item, len(items), cfg.desc_item(), dia)], axis=1)
    segmentos = _ultimo_valor(df, ent, len(entidades), cfg.segmentos, dia)
    for c in cfg.segmentos:
        col = segmentos[c]
        segmentos[c] = col.astype(object).where(col.notna(), "(sin dato)").astype(str)

    usd = pd.to_numeric(df[cfg.col_valor], errors="coerce").fillna(0.0).to_numpy(float)
    margen = pd.to_numeric(df[cfg.col_margen], errors="coerce").fillna(0.0).to_numpy(float)
    if cfg.col_tipo_documento:
        tipo = df[cfg.col_tipo_documento].astype(str).to_numpy()
        devuelve = np.isin(tipo, list(cfg.tipos_devolucion))
        if devuelve.any() and not cfg.devolucion_ya_negativa:
            usd = np.where(devuelve, -np.abs(usd), usd)
            margen = np.where(devuelve, -np.abs(margen), margen)
        LOGGER.info("tipos de documento: %s filas de venta, %s de devolución", 
                    f"{int((~devuelve).sum()):,}", f"{int(devuelve.sum()):,}")
    LOGGER.info("panel: %s filas -> %s entidades x %s ítems", f"{len(df):,}",
                f"{len(entidades):,}", f"{len(items):,}")
    panel = Panel(cfg, fechas, ent, item, dia, usd, margen, entidades, items, segmentos)
    if cfg.col_documento and cfg.col_documento in df.columns:
        panel.doc = df[cfg.col_documento].to_numpy()
    return panel


def asignar_segmentos(panel: Panel, matriz: Matriz) -> pd.DataFrame:
    """Nivel de segmentación más fino con datos suficientes, por entidad.

    `cfg.segmentos` es una jerarquía del nivel más grueso al más fino. Se prueba primero
    la combinación completa (SEGMENTO | SUBSEGMENTO | ...) y, si a ese grupo le faltan
    entidades, se va soltando el último nivel hasta llegar a GLOBAL.

    Se cuentan las entidades con compras en la ventana: un segmento de 300 entidades de
    las cuales compraron 5 no sirve para comparar contra nadie.
    """
    cfg = panel.cfg
    n = panel.n_ent
    activa = matriz.items_entidad > 0
    etiqueta = np.full(n, "GLOBAL", dtype=object)
    nivel = np.full(n, "GLOBAL", dtype=object)
    asignada = np.zeros(n, dtype=bool)

    for corte in range(len(cfg.segmentos), 0, -1):
        columnas = list(cfg.segmentos)[:corte]
        combinada = panel.segmentos[columnas[0]].astype(str)
        for col in columnas[1:]:
            combinada = combinada.str.cat(panel.segmentos[col].astype(str), sep=" | ")
        codigos, etiquetas = pd.factorize(combinada)
        cuenta = np.bincount(codigos[activa], minlength=len(etiquetas))
        nuevas = (~asignada) & (cuenta[codigos] >= cfg.min_entidades_segmento)
        if nuevas.any():
            etiqueta[nuevas] = combinada.to_numpy(dtype=object)[nuevas]
            nivel[nuevas] = " | ".join(columnas)
            asignada |= nuevas
        if asignada.all():
            break
    return pd.DataFrame({"segmento": etiqueta, "nivel_segmento": nivel})


def agregar_tamano(panel: Panel, matriz: Matriz) -> List[str]:
    """Suma BD_TAMANO a la jerarquía de segmentos: en qué cuantil de compra cae la entidad.

    Sirve para no comparar a un cliente mediano contra uno que compra veinte veces más:
    lo que el grande compra "de todo" no es una oportunidad para el mediano.
    """
    cfg = panel.cfg
    if not cfg.cortes_tamano:
        return list(cfg.segmentos)
    total = matriz.venta_entidad
    positivos = total[total > 0]
    if not len(positivos):
        return list(cfg.segmentos)
    cortes = np.quantile(positivos, list(cfg.cortes_tamano))
    etiquetas = np.array(list(cfg.etiquetas_tamano)[:len(cortes) + 1], dtype=object)
    idx = np.clip(np.searchsorted(cortes, total, side="right"), 0, len(etiquetas) - 1)
    panel.segmentos = panel.segmentos.copy()
    panel.segmentos["BD_TAMANO"] = etiquetas[idx]
    return list(cfg.segmentos) + ["BD_TAMANO"]


class Bloque:
    """Un segmento ya recortado: matrices, soporte, penetración y candidatos."""

    def __init__(self, cfg: RecConfig, matriz: Matriz, filas: np.ndarray, etiqueta: str, nivel: str,
                 canastas: Optional[Tuple[sp.csr_matrix, np.ndarray, np.ndarray]] = None,
                 d_ayer: Optional[int] = None):
        self.cfg, self.etiqueta, self.nivel, self.filas = cfg, etiqueta, nivel, filas
        self.R = matriz.R[filas]
        self.V = matriz.V[filas]
        self.M = matriz.M[filas]
        self.U = matriz.U[filas]
        self.D = matriz.D[filas]
        self.P = matriz.P[filas]
        self.n = len(filas)
        self.n_item = matriz.n_item
        self._pares: Optional[pd.DataFrame] = None            # caché de los pares del bloque
        self.dias_ventana = float(cfg.dias_afinidad or 365)   # el motor la ajusta a la real
        self.p_adopcion = 1.0                                 # el backtest la ajusta

        # matriz con la que se mide la afinidad: canastas (lo que se compra junto) o
        # el repertorio de cada entidad (todo lo que compra en la ventana)
        vida = float(cfg.vida_media_afinidad_dias or 0)

        def peso(dias_del_dato: np.ndarray) -> np.ndarray:
            """Lo viejo pesa menos: a `vida` días de antigüedad, la mitad."""
            if not vida or d_ayer is None:
                return np.ones(len(dias_del_dato))
            edad = np.maximum(float(d_ayer) - np.asarray(dias_del_dato, float), 0.0)
            return np.power(0.5, edad / vida)

        self.A = (self.R > 0).astype(float)
        self.n_afinidad = self.n
        if canastas is not None:
            B, entidad_canasta, dia_canasta = canastas
            pertenece = np.zeros(matriz.n_ent, dtype=bool)
            pertenece[filas] = True
            sel = np.flatnonzero(pertenece[entidad_canasta])
            if len(sel):
                self.A = B[sel]
                self.n_afinidad = len(sel)
                if vida:
                    self.A = (sp.diags(peso(dia_canasta[sel])) @ self.A).tocsr()
        elif vida:
            ultimo = self.U.tocoo()
            self.A = sp.csr_matrix((peso(ultimo.data), (ultimo.row, ultimo.col)),
                                   shape=self.R.shape)

        self.soporte = np.asarray((self.R > 0).sum(0)).ravel().astype(float)
        self.penetracion = self.soporte / max(self.n, 1)
        self.candidato = (self.soporte >= cfg.min_soporte) & (self.penetracion >= cfg.min_penetracion)

        tol = cfg.tolerancia_cero
        bruto_v = np.asarray(abs(self.V).sum(0)).ravel()
        bruto_m = np.asarray(abs(self.M).sum(0)).ravel()
        usd_item = sin_residuo(np.asarray(self.V.sum(0)).ravel(), bruto_v, tol, escala_tipica(bruto_v))
        margen_item = sin_residuo(np.asarray(self.M.sum(0)).ravel(), bruto_m, tol, escala_tipica(bruto_m))
        seguro = np.maximum(self.soporte, 1.0)
        self.usd_medio_comprador = np.where(self.soporte > 0, usd_item / seguro, 0.0)
        # el ítem cuya venta se anula con sus devoluciones no tiene margen %, no uno gigante
        self.margen_pct_item = np.where(usd_item > 0, margen_item / np.where(usd_item > 0, usd_item, 1.0), 0.0)

        # ritmo y ticket del ítem EN ESTE SEGMENTO: es la evidencia que se le presta a
        # quien tiene poca historia propia
        self.iv_item = np.full(self.n_item, np.nan)
        self.cv_item = np.full(self.n_item, np.nan)
        self.ticket_item = np.zeros(self.n_item)
        pares = matriz.tab
        propios = np.isin(pares["e"].to_numpy(np.int64), filas)
        if propios.any():
            sub = pares.loc[propios]
            i_sub = sub["i"].to_numpy(np.int64)
            dias_sub = sub["dias"].to_numpy(float)
            usd_sub = sub["usd"].to_numpy(float)
            total_dias = np.bincount(i_sub, weights=dias_sub, minlength=self.n_item)
            total_usd = np.bincount(i_sub, weights=usd_sub, minlength=self.n_item)
            self.ticket_item = np.where(total_dias > 0, total_usd / np.maximum(total_dias, 1.0), 0.0)
            iv = pd.to_numeric(sub["intervalo_medio"], errors="coerce").to_numpy(float)
            de = pd.to_numeric(sub["intervalo_desvio"], errors="coerce").to_numpy(float)
            con_ritmo = np.isfinite(iv) & (iv > 0)
            if con_ritmo.any():
                tabla = pd.DataFrame({"i": i_sub[con_ritmo], "iv": iv[con_ritmo],
                                      "cv": np.where(iv[con_ritmo] > 0, de[con_ritmo] / iv[con_ritmo], np.nan)})
                agr = tabla.groupby("i").agg(iv=("iv", "median"), cv=("cv", "median"))
                self.iv_item[agr.index.to_numpy()] = agr["iv"].to_numpy()
                self.cv_item[agr.index.to_numpy()] = agr["cv"].to_numpy()

        bruto_ent = np.asarray(abs(self.V).sum(1)).ravel()
        self.venta_entidad = sin_residuo(np.asarray(self.V.sum(1)).ravel(), bruto_ent,
                                         tol, escala_tipica(bruto_ent))
        self.dias_entidad = matriz.dias_entidad[filas]
        valido = self.venta_entidad > 0
        positivas = self.venta_entidad[valido]
        self.venta_media = float(positivas.mean()) if len(positivas) else 0.0
        # participación de cada ítem en la compra de la entidad. La entidad cuya compra neta
        # es cero (compró y devolvió todo) no tiene participaciones: su fila queda en cero y
        # tampoco cuenta en el promedio del ítem, para no ensuciar la brecha de los demás.
        escala = sp.diags(np.where(valido, 1.0 / np.where(valido, self.venta_entidad, 1.0), 0.0))
        self.share = (escala @ self.V).tocsr()
        soporte_share = (np.asarray((self.R[valido] > 0).sum(0)).ravel().astype(float)
                         if valido.any() else np.zeros(self.n_item))
        self.share_medio_comprador = np.where(
            soporte_share > 0,
            np.asarray(self.share.sum(0)).ravel() / np.maximum(soporte_share, 1.0), 0.0)

    def __repr__(self) -> str:
        return (f"Bloque({self.etiqueta!r} nivel={self.nivel} {self.n:,} entidades, "
                f"{int(self.candidato.sum()):,} ítems candidatos)")


# =========================================================================== #
# 3. Algoritmos: la batería
# =========================================================================== #
def _argmax_por_fila(valores: np.ndarray, indptr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Máximo y posición del máximo dentro de cada fila de una CSR expandida.

    `valores` tiene una fila por elemento no nulo; `indptr` marca dónde empieza cada
    fila. Devuelve (filas con datos, máximo, posición del máximo dentro de `valores`).
    """
    largos = np.diff(indptr)
    llenas = np.flatnonzero(largos > 0)
    if not len(llenas):
        return llenas, np.zeros((0,) + valores.shape[1:]), np.zeros((0,) + valores.shape[1:], dtype=np.int64)
    inicios = indptr[llenas]
    maximo = np.maximum.reduceat(valores, inicios, axis=0)
    repetido = np.repeat(maximo, largos[llenas], axis=0)
    posicion = np.arange(len(valores))
    if valores.ndim == 2:
        posicion = posicion[:, None]
    marca = np.where(valores >= repetido, posicion, -1)
    return llenas, maximo, np.maximum.reduceat(marca, inicios, axis=0)


class Algoritmo:
    """Puntúa, para cada entidad del segmento, los ítems candidatos."""

    nombre = "base"
    descripcion = ""

    def __init__(self, cfg: RecConfig):
        self.cfg = cfg
        self.b: Optional[Bloque] = None
        self.cand = np.zeros(0, dtype=np.int64)

    def ajustar(self, b: Bloque) -> None:
        self.b = b
        self.cand = np.flatnonzero(b.candidato)

    def puntuar(self, filas: np.ndarray) -> np.ndarray:
        """(len(filas) x len(self.cand)). Más alto = más recomendable."""
        raise NotImplementedError

    def limite_filas(self) -> int:
        return self.cfg.filas_bloque

    def explicar(self, filas: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Un texto por par (entidad, ítem) que diga por qué salió recomendado."""
        pen = self.b.penetracion[items]
        return np.array([f"lo compran {p:.0%} de los pares del segmento" for p in pen], dtype=object)


def _explicar_por_afinidad(alg: "Algoritmo", matriz: sp.csr_matrix, filas: np.ndarray,
                           items: np.ndarray, etiqueta: str) -> np.ndarray:
    """El ítem que ya compra la entidad y más empuja a la recomendación."""
    b = alg.b
    nombres = b.nombres_item
    salida = np.empty(len(filas), dtype=object)
    salida[:] = ""
    trozo = 50_000
    for ini in range(0, len(filas), trozo):
        f = filas[ini:ini + trozo]
        it = items[ini:ini + trozo]
        sub = (b.R[f] > 0).tocsr()
        if sub.nnz == 0:
            continue
        columnas = np.repeat(it, np.diff(sub.indptr))
        valores = np.asarray(matriz[sub.indices, columnas]).ravel()[:, None]
        llenas, maximo, posicion = _argmax_por_fila(valores, sub.indptr)
        if not len(llenas):
            continue
        propio = sub.indices[posicion.ravel()]
        cuantos = np.diff(sub.indptr)[llenas]
        salida[ini + llenas] = [
            (f"{etiqueta} {nombres[p]}" + (f" y {c - 1} ítem(s) más" if c > 1 else ""))
            if m > 0 else "sin afinidad medible con lo que ya compra"
            for p, c, m in zip(propio, cuantos, maximo.ravel())]
    vacias = np.array([s == "" for s in salida])
    if vacias.any():
        pen = b.penetracion[items[vacias]]
        salida[vacias] = [f"lo compran {p:.0%} de los pares del segmento" for p in pen]
    return salida


class Popularidad(Algoritmo):
    nombre = "popularidad"
    descripcion = ("Penetración del ítem en el segmento. Es la línea de base: ofrecer lo que más "
                   "compran los pares, sin personalizar.")

    def puntuar(self, filas: np.ndarray) -> np.ndarray:
        return np.repeat(self.b.penetracion[self.cand][None, :], len(filas), axis=0)


class CosenoItem(Algoritmo):
    nombre = "coseno_item"
    descripcion = ("Afinidad ítem-ítem por coseno sobre quién compra qué. El puntaje suma la "
                   "afinidad entre el ítem candidato y lo que la entidad ya compra.")

    def ajustar(self, b: Bloque) -> None:
        super().ajustar(b)
        R = b.A.tocsc()
        norma = np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel())
        Rn = (R @ sp.diags(1.0 / np.where(norma > 0, norma, 1.0))).tocsr()
        S = (Rn.T @ Rn).tocsr()
        S.setdiag(0.0)
        S.eliminate_zeros()
        self.S = S
        self.Sc = S[:, self.cand].tocsc()

    def puntuar(self, filas: np.ndarray) -> np.ndarray:
        Rf = (self.b.R[filas] > 0).astype(float)
        return np.asarray((Rf @ self.Sc).todense())

    def explicar(self, filas: np.ndarray, items: np.ndarray) -> np.ndarray:
        return _explicar_por_afinidad(self, self.S, filas, items, "afín a lo que ya compra:")


class CosenoEntidad(Algoritmo):
    nombre = "coseno_entidad"
    descripcion = ("Vecinos parecidos: coseno entre entidades por lo que compran. El puntaje pesa "
                   "a los k pares más parecidos que sí compran el ítem.")

    def ajustar(self, b: Bloque) -> None:
        super().ajustar(b)
        R = (b.R > 0).astype(float).tocsr()
        norma = np.sqrt(np.asarray(R.multiply(R).sum(1)).ravel())
        self.Rn = (sp.diags(1.0 / np.where(norma > 0, norma, 1.0)) @ R).tocsr()
        self.Rc = R[:, self.cand].tocsr()

    def _vecinos(self, filas: np.ndarray) -> sp.csr_matrix:
        """k pares más parecidos de cada fila, con su parecido como peso."""
        sim = (self.Rn[filas] @ self.Rn.T).tocsr()
        k = max(int(self.cfg.k_vecinos), 1)
        datos, indices, ptr = [], [], [0]
        for r in range(len(filas)):
            ini, fin = sim.indptr[r], sim.indptr[r + 1]
            d, j = sim.data[ini:fin], sim.indices[ini:fin]
            propio = j != filas[r]
            d, j = d[propio], j[propio]
            if len(d) > k:
                sel = np.argpartition(d, -k)[-k:]
                d, j = d[sel], j[sel]
            datos.append(d)
            indices.append(j)
            ptr.append(ptr[-1] + len(d))
        datos = np.concatenate(datos) if datos else np.zeros(0)
        indices = np.concatenate(indices) if indices else np.zeros(0, dtype=int)
        return sp.csr_matrix((datos, indices, np.array(ptr)), shape=(len(filas), self.b.n))

    def puntuar(self, filas: np.ndarray) -> np.ndarray:
        return np.asarray((self._vecinos(filas) @ self.Rc).todense())

    def explicar(self, filas: np.ndarray, items: np.ndarray) -> np.ndarray:
        salida = np.empty(len(filas), dtype=object)
        R = (self.b.R > 0).astype(float).tocsc()
        trozo = 20_000
        for ini in range(0, len(filas), trozo):
            f, it = filas[ini:ini + trozo], items[ini:ini + trozo]
            W = (self._vecinos(f) > 0).astype(float).tocsr()
            cuenta = np.asarray(W.multiply(R[:, it].T).sum(1)).ravel()
            salida[ini:ini + trozo] = [f"{int(c)} de sus pares más parecidos lo compran" for c in cuenta]
        return salida


class Svd(Algoritmo):
    nombre = "svd"
    descripcion = ("Factores latentes (SVD truncada) sobre la matriz de compras: patrones de consumo "
                   "que no se ven mirando ítem por ítem.")

    def ajustar(self, b: Bloque) -> None:
        super().ajustar(b)
        from sklearn.decomposition import TruncatedSVD
        R = (b.R > 0).astype(float).tocsr()
        k = int(min(self.cfg.k_factores, min(R.shape) - 1)) if min(R.shape) > 2 else 0
        if k < 2:
            self.U = None
            return
        svd = TruncatedSVD(n_components=k, random_state=self.cfg.semilla)
        self.U = svd.fit_transform(R)
        self.Vt = svd.components_[:, self.cand]

    def puntuar(self, filas: np.ndarray) -> np.ndarray:
        if self.U is None:
            return np.repeat(self.b.penetracion[self.cand][None, :], len(filas), axis=0)
        return self.U[filas] @ self.Vt


class KmeansValor(Algoritmo):
    nombre = "kmeans_valor"
    descripcion = ("k-means sobre cómo reparte cada entidad su dinero entre los ítems (participación "
                   "en USD, no unidades). El puntaje es la penetración del ítem dentro del grupo.")

    def ajustar(self, b: Bloque) -> None:
        super().ajustar(b)
        from sklearn.cluster import KMeans
        k = int(max(1, min(self.cfg.k_clusters, b.n)))
        R = (b.R > 0).astype(float).tocsr()[:, self.cand]
        if k < 2 or b.n < 4:
            self.etiquetas = np.zeros(b.n, dtype=int)
            self.pen = b.penetracion[self.cand][None, :]
            self.tam = np.array([float(b.n)])
            return
        km = KMeans(n_clusters=k, random_state=self.cfg.semilla, n_init=10)
        self.etiquetas = km.fit_predict(b.share)
        self.tam = np.bincount(self.etiquetas, minlength=k).astype(float)
        sumas = np.zeros((k, len(self.cand)))
        for c in range(k):
            filas = np.flatnonzero(self.etiquetas == c)
            if len(filas):
                sumas[c] = np.asarray(R[filas].sum(0)).ravel()
        self.pen = sumas / np.maximum(self.tam[:, None], 1.0)

    def puntuar(self, filas: np.ndarray) -> np.ndarray:
        if self.pen.shape[0] == 1:
            return np.repeat(self.pen, len(filas), axis=0)
        return self.pen[self.etiquetas[filas]]

    def explicar(self, filas: np.ndarray, items: np.ndarray) -> np.ndarray:
        if self.pen.shape[0] == 1:
            return super().explicar(filas, items)
        posicion = -np.ones(self.b.n_item, dtype=np.int64)
        posicion[self.cand] = np.arange(len(self.cand))
        grupo = self.etiquetas[filas]
        col = posicion[items]
        pen = np.where(col >= 0, self.pen[grupo, np.maximum(col, 0)], 0.0)
        return np.array([f"en su grupo de gasto (k-means #{g}, {int(self.tam[g])} pares) lo compran {p:.0%}"
                         for g, p in zip(grupo, pen)], dtype=object)


class Reglas(Algoritmo):
    nombre = "reglas"
    descripcion = ("Reglas de asociación: confianza P(compra el candidato | compra lo que ya tiene), "
                   "exigiendo lift mayor a 1. El puntaje es la mejor regla que le aplica.")

    def ajustar(self, b: Bloque) -> None:
        super().ajustar(b)
        R = b.A.tocsr()
        soporte = np.asarray((R > 0).sum(0)).ravel().astype(float)
        penetracion = soporte / max(b.n_afinidad, 1)
        co = (R.T @ R).tocsr()
        co.setdiag(0.0)
        co.eliminate_zeros()
        conf = (sp.diags(1.0 / np.maximum(soporte, 1.0)) @ co).tocsr()
        pen = np.where(penetracion > 0, penetracion, 1.0)
        lift = (conf @ sp.diags(1.0 / pen)).tocsr()
        conf.data = np.where(lift.data > 1.0, conf.data, 0.0)
        conf.eliminate_zeros()
        self.C = conf
        self.Cc = conf[:, self.cand].tocsr()

    def limite_filas(self) -> int:
        items_medios = max(float(np.mean(np.diff(self.b.R.indptr))), 1.0)
        tope = int(4_000_000 / max(len(self.cand), 1) / items_medios)
        return int(max(64, min(self.cfg.filas_bloque, tope)))

    def puntuar(self, filas: np.ndarray) -> np.ndarray:
        sub = (self.b.R[filas] > 0).tocsr()
        out = np.zeros((len(filas), len(self.cand)))
        if sub.nnz == 0:
            return out
        valores = np.asarray(self.Cc[sub.indices].todense())
        llenas, maximo, _ = _argmax_por_fila(valores, sub.indptr)
        out[llenas] = maximo
        return out

    def explicar(self, filas: np.ndarray, items: np.ndarray) -> np.ndarray:
        return _explicar_por_afinidad(self, self.C, filas, items, "regla: como compra")


ALGORITMOS: Dict[str, type] = {a.nombre: a for a in
                               (Popularidad, CosenoItem, CosenoEntidad, Svd, KmeansValor, Reglas)}


# =========================================================================== #
# 4. Tipos de recomendación y USD en juego
# =========================================================================== #
def construir_algoritmo(nombre: str, cfg: RecConfig) -> Algoritmo:
    if nombre in ("rrf", "ponderado"):
        return Fusion(cfg, modo=nombre)
    return ALGORITMOS[nombre](cfg)


class Fusion(Algoritmo):
    """Combina toda la batería: por posición (rrf) o por puntaje normalizado (ponderado)."""

    nombre = "fusion"

    def __init__(self, cfg: RecConfig, modo: str = "rrf"):
        super().__init__(cfg)
        self.modo = modo
        self.descripcion = ("Fusión de la batería por posición en cada lista (Reciprocal Rank Fusion)."
                            if modo == "rrf" else
                            "Fusión de la batería por puntaje normalizado, con los pesos configurados.")
        self.algos = [ALGORITMOS[n](cfg) for n in cfg.algoritmos]

    def ajustar(self, b: Bloque) -> None:
        super().ajustar(b)
        for a in self.algos:
            a.ajustar(b)

    def limite_filas(self) -> int:
        return min([a.limite_filas() for a in self.algos] + [self.cfg.filas_bloque])

    def puntuar(self, filas: np.ndarray) -> np.ndarray:
        total = np.zeros((len(filas), len(self.cand)))
        for a in self.algos:
            peso = float(self.cfg.pesos.get(a.nombre, 1.0))
            P = a.puntuar(filas)
            if self.modo == "rrf":
                posicion = np.argsort(np.argsort(-P, axis=1), axis=1)
                total += peso / (60.0 + posicion + 1.0)
            else:
                maximo = P.max(axis=1, keepdims=True)
                total += peso * P / np.where(maximo > 0, maximo, 1.0)
        return total

    def explicar(self, filas: np.ndarray, items: np.ndarray) -> np.ndarray:
        for a in self.algos:
            if isinstance(a, (CosenoItem, Reglas)):
                return a.explicar(filas, items)
        return self.algos[0].explicar(filas, items)


def _pares_bloque(b: Bloque, matriz: Matriz) -> pd.DataFrame:
    """Los pares entidad-ítem del bloque, con la fila local. Se calcula una sola vez."""
    if getattr(b, "_pares", None) is not None:
        return b._pares
    mapa = np.full(matriz.n_ent, -1, dtype=np.int64)
    mapa[b.filas] = np.arange(b.n)
    f = mapa[matriz.tab["e"].to_numpy(np.int64)]
    sel = f >= 0
    out = matriz.tab.loc[sel].copy()
    out["f"] = f[sel]
    b._pares = out
    return out


def _escala_tamano(b: Bloque, filas: np.ndarray, cfg: RecConfig) -> np.ndarray:
    """Ajusta el potencial al tamaño de la entidad: un cliente chico no compra como uno grande."""
    if not cfg.escalar_potencial or b.venta_media <= 0:
        return np.ones(len(filas))
    razon = b.venta_entidad[filas] / b.venta_media
    razon = np.where(razon > 0, razon, 1.0)
    return np.clip(razon, 1.0 / cfg.tope_escala, cfg.tope_escala)


def _mezclar(propio: np.ndarray, n_propio: np.ndarray, pares: np.ndarray, k: float) -> np.ndarray:
    """Estimación encogida hacia los pares: (n*propio + k*pares) / (n + k).

    Con una sola compra, `n` es chico y manda la evidencia del segmento. Con veinte,
    manda la del cliente. Es lo que evita inventar un ritmo con un solo intervalo y, al
    mismo tiempo, no desperdiciar la historia de quien sí la tiene.
    """
    propio = np.asarray(propio, float)
    pares = np.asarray(pares, float)
    n = np.maximum(np.asarray(n_propio, float), 0.0)
    hay_propio = np.isfinite(propio) & (propio > 0)
    hay_pares = np.isfinite(pares) & (pares > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mezcla = (np.where(hay_propio, propio, 0.0) * n + np.where(hay_pares, pares, 0.0) * k) / np.maximum(
            np.where(hay_propio, n, 0.0) + np.where(hay_pares, k, 0.0), 1e-12)
    mezcla = np.where(hay_propio | hay_pares, mezcla, np.nan)
    return np.where(hay_propio & ~hay_pares, propio, np.where(hay_pares & ~hay_propio, pares, mezcla))


def _prob_recompra(silencio: np.ndarray, intervalo: np.ndarray, item: np.ndarray,
                   matriz: Matriz, cfg: RecConfig) -> np.ndarray:
    """P(vuelve a comprar el ítem dentro del horizonte), leída de la curva de recuperación.

    No es "¿este silencio es normal?" —eso ya lo dice el puntaje— sino "de los que
    llegaron a este nivel de atraso, ¿cuántos volvieron?". Sale de la historia: primero la
    curva del ítem, y si el ítem no tiene casos suficientes, la del panel.
    """
    horizonte = float(cfg.horizonte_dias) if cfg.horizonte_dias else 90.0
    rejilla, por_item, global_ = matriz.curva_recuperacion(horizonte, cfg.min_casos_recuperacion)
    with np.errstate(invalid="ignore", divide="ignore"):
        atraso = np.where(intervalo > 0, silencio / intervalo, np.nan)
    atraso = np.clip(np.nan_to_num(atraso, nan=rejilla[0]), rejilla[0], rejilla[-1])
    # curva por ítem, rellenando con la del panel donde el ítem no tiene casos suficientes
    curva = np.where(np.isfinite(por_item), por_item, global_[None, :]) if len(por_item) else None
    if curva is None:
        valida = np.isfinite(global_)
        p = (np.interp(atraso, rejilla[valida], global_[valida]) if valida.any()
             else np.zeros(len(atraso)))
        return np.clip(p, 0.0, 1.0)
    # interpolación lineal vectorizada entre los dos puntos de rejilla que rodean al atraso
    k = np.clip(np.searchsorted(rejilla, atraso, side="left"), 1, len(rejilla) - 1)
    x0, x1 = rejilla[k - 1], rejilla[k]
    y0, y1 = curva[item, k - 1], curva[item, k]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(x1 > x0, (atraso - x0) / (x1 - x0), 0.0)
        p = y0 + w * (y1 - y0)
    p = np.where(np.isfinite(p), p, np.where(np.isfinite(y0), y0, np.where(np.isfinite(y1), y1, 0.0)))
    return np.clip(np.where(np.isfinite(p), p, 0.0), 0.0, 1.0)


def recomendar_cruzadas(b: Bloque, alg: Algoritmo, cfg: RecConfig, top: Optional[int] = None) -> pd.DataFrame:
    """Ítems que la entidad NO compra y sus pares sí. Devuelve fila local, ítem y puntaje."""
    if not len(alg.cand) or b.n == 0:
        return pd.DataFrame(columns=["f", "i", "puntaje"])
    top = int(top or cfg.max_items_reco)
    paso = max(1, min(cfg.filas_bloque, alg.limite_filas()))
    trozos = []
    for ini in range(0, b.n, paso):
        filas = np.arange(ini, min(ini + paso, b.n))
        P = alg.puntuar(filas)
        tiene = np.asarray((b.R[filas] > 0)[:, alg.cand].todense())
        P = np.where(tiene, -np.inf, P)
        k = min(top, P.shape[1])
        idx = np.argpartition(-P, k - 1, axis=1)[:, :k] if P.shape[1] > k else np.tile(np.arange(P.shape[1]), (len(filas), 1))
        val = np.take_along_axis(P, idx, axis=1)
        orden = np.argsort(-val, axis=1)
        idx = np.take_along_axis(idx, orden, axis=1)
        val = np.take_along_axis(val, orden, axis=1)
        ok = np.isfinite(val) & (val > 0)
        if not ok.any():
            continue
        trozos.append(pd.DataFrame({
            "f": np.repeat(filas, idx.shape[1])[ok.ravel()],
            "i": alg.cand[idx.ravel()[ok.ravel()]],
            "puntaje": val.ravel()[ok.ravel()]}))
    if not trozos:
        return pd.DataFrame(columns=["f", "i", "puntaje"])
    return pd.concat(trozos, ignore_index=True)


def recomendar_reposicion(b: Bloque, matriz: Matriz, cfg: RecConfig, d_ayer: int) -> pd.DataFrame:
    """Lo que compraba con cierto ritmo y hace rato no compra.

    El ritmo y el ticket se estiman mezclando la evidencia del cliente con la del ítem en
    su segmento (`peso_prior_pares`): con una compra manda el segmento, con veinte manda
    él. El valor es lo que se espera en los próximos `horizonte_dias`, multiplicado por la
    probabilidad de que la compra ocurra.
    """
    columnas = ["f", "i", "puntaje", "usd_potencial", "usd_si_compra", "prob", "dias_sin_comprar",
                "dias_compra_item", "intervalo_tipico", "intervalo_esperado", "compras_esperadas",
                "motivo"]
    p = _pares_bloque(b, matriz)
    if p.empty:
        return pd.DataFrame(columns=columnas)
    fila = p["f"].to_numpy(np.int64)
    item = p["i"].to_numpy(np.int64)
    dias = p["dias"].to_numpy(float)
    usd = p["usd"].to_numpy(float)
    silencio = d_ayer - p["ultimo"].to_numpy(float)
    iv_propio = pd.to_numeric(p["intervalo_medio"], errors="coerce").to_numpy(float)
    de_propio = pd.to_numeric(p["intervalo_desvio"], errors="coerce").to_numpy(float)
    n_int = pd.to_numeric(p["n_intervalos"], errors="coerce").fillna(0).to_numpy(float)

    # ritmo y ticket: lo propio mezclado con lo del ítem en el segmento
    escala = _escala_tamano(b, fila, cfg)
    iv_est = _mezclar(iv_propio, n_int, b.iv_item[item], cfg.peso_prior_pares)
    ticket_propio = np.where(dias > 0, usd / np.maximum(dias, 1.0), np.nan)
    ticket_est = _mezclar(ticket_propio, dias, b.ticket_item[item] * escala, cfg.peso_prior_pares)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv_propio = np.where(iv_propio > 0, de_propio / iv_propio, np.nan)
    cv_est = _mezclar(cv_propio, np.maximum(n_int - 1, 0), b.cv_item[item], cfg.peso_prior_pares)

    ok = ((dias >= cfg.min_compras_reposicion) & np.isfinite(iv_est) & (iv_est > 0)
          & np.isfinite(ticket_est) & (ticket_est > 0)
          & (silencio > cfg.factor_reposicion * iv_est))
    if cfg.max_cv_intervalo is not None:      # la regularidad sólo se le exige a quien tiene ritmo propio
        medible = n_int >= 2
        ok &= ~medible | ~np.isfinite(cv_propio) | (cv_propio <= float(cfg.max_cv_intervalo))
    if not ok.any():
        return pd.DataFrame(columns=columnas)

    iv, sil, compras = iv_est[ok], silencio[ok], dias[ok]
    ticket, comprado = ticket_est[ok], usd[ok]
    horizonte = float(cfg.horizonte_dias) if cfg.horizonte_dias else float(np.max(sil))
    esperadas = np.minimum(horizonte / iv, horizonte)          # a lo sumo una compra por día
    si_compra = ticket * esperadas
    if cfg.tope_potencial_por_historico:                       # no más de N veces su propio ritmo
        propio_en_horizonte = comprado / max(b.dias_ventana, 1.0) * horizonte
        techo = propio_en_horizonte * float(cfg.tope_potencial_por_historico)
        si_compra = np.minimum(si_compra, np.where(techo > 0, techo, np.inf))
    prob = (_prob_recompra(sil, iv, item[ok], matriz, cfg) if cfg.usar_probabilidad
            else np.ones(len(iv)))
    return pd.DataFrame({
        "f": fila[ok], "i": item[ok],
        "puntaje": sil / iv,
        "usd_potencial": si_compra * prob,
        "usd_si_compra": si_compra,
        "prob": prob,
        "dias_sin_comprar": sil,
        "dias_compra_item": compras,
        "intervalo_tipico": np.where(np.isfinite(iv_propio[ok]), iv_propio[ok], np.nan),
        "intervalo_esperado": iv,
        "compras_esperadas": esperadas,
        "motivo": [_motivo_reposicion(c, ip, iv_, s, pr)
                   for c, ip, iv_, s, pr in zip(compras, iv_propio[ok], iv, sil, prob)]})


def _motivo_reposicion(compras, iv_propio, iv_est, silencio, prob) -> str:
    """El texto dice con qué evidencia se armó: la propia, la de los pares, o las dos."""
    if np.isfinite(iv_propio) and compras >= 3:
        base = f"compró {compras:.0f} veces, cada {iv_propio:.0f} días en promedio"
    elif np.isfinite(iv_propio):
        base = (f"compró {compras:.0f} veces (cada {iv_propio:.0f} días); sus pares lo compran "
                f"cada {iv_est:.0f}")
    else:
        base = f"compró {compras:.0f} vez; sus pares lo compran cada {iv_est:.0f} días"
    return f"{base}, lleva {silencio:.0f} sin comprar (probabilidad de recompra {prob:.0%})"


def recomendar_brecha(b: Bloque, matriz: Matriz, cfg: RecConfig) -> pd.DataFrame:
    """Lo que compra, pero mucho menos de lo que le dedican sus pares."""
    columnas = ["f", "i", "puntaje", "usd_potencial", "usd_si_compra", "prob", "motivo"]
    p = _pares_bloque(b, matriz)
    if p.empty:
        return pd.DataFrame(columns=columnas)
    f = p["f"].to_numpy(np.int64)
    i = p["i"].to_numpy(np.int64)
    venta_ent = b.venta_entidad[f]
    share = np.where(venta_ent > 0, p["usd"].to_numpy(float) / np.where(venta_ent > 0, venta_ent, 1.0), 0.0)
    medio = b.share_medio_comprador[i]
    ok = b.candidato[i] & (venta_ent > 0) & (medio > 0) & (share < cfg.brecha_ratio * medio)
    if not ok.any():
        return pd.DataFrame(columns=columnas)
    sh, me, ve = share[ok], medio[ok], venta_ent[ok]
    horizonte = float(cfg.horizonte_dias) if cfg.horizonte_dias else b.dias_ventana
    falta = (me - sh) * ve * horizonte / max(b.dias_ventana, 1.0)
    return pd.DataFrame({
        "f": f[ok], "i": i[ok],
        "puntaje": 1.0 - sh / me,
        "usd_potencial": falta,
        "usd_si_compra": falta,
        "prob": np.ones(len(falta)),
        "motivo": [f"sus pares le dedican {m:.1%} de su compra y esta entidad {s:.1%}"
                   for m, s in zip(me, sh)]})


def armar_recomendaciones(b: Bloque, matriz: Matriz, alg: Algoritmo, cfg: RecConfig,
                          d_ayer: int) -> pd.DataFrame:
    """Los tres tipos juntos, todos medidos en USD esperados en el mismo horizonte."""
    horizonte = float(cfg.horizonte_dias) if cfg.horizonte_dias else b.dias_ventana
    partes = []
    if "CRUZADA" in cfg.incluir_tipos:
        cru = recomendar_cruzadas(b, alg, cfg)
        if len(cru):
            f = cru["f"].to_numpy(np.int64)
            i = cru["i"].to_numpy(np.int64)
            escala = _escala_tamano(b, f, cfg)
            # lo que gastaría en el horizonte si lo adoptara: el ticket del ítem en el
            # segmento por las compras que hace un par en ese lapso
            iv_pares = b.iv_item[i]
            con_ritmo = np.isfinite(iv_pares) & (iv_pares > 0)
            esperadas = np.where(con_ritmo,
                                 np.minimum(horizonte / np.where(con_ritmo, iv_pares, 1.0), horizonte),
                                 1.0)
            si_compra = np.where(con_ritmo,
                                 b.ticket_item[i] * escala * esperadas,
                                 b.usd_medio_comprador[i] * escala * horizonte / max(b.dias_ventana, 1.0))
            prob = np.full(len(f), float(b.p_adopcion) if cfg.usar_probabilidad else 1.0)
            cru["usd_si_compra"] = si_compra
            cru["prob"] = prob
            cru["usd_potencial"] = si_compra * prob
            cru["dias_sin_comprar"] = np.nan
            cru["dias_compra_item"] = 0.0
            cru["intervalo_tipico"] = np.nan
            cru["intervalo_esperado"] = iv_pares
            cru["compras_esperadas"] = esperadas
            cru["motivo"] = alg.explicar(f, i)
            cru["tipo"] = "CRUZADA"
            cru["algoritmo"] = alg.nombre
            partes.append(cru)
    if "REPOSICION" in cfg.incluir_tipos:
        rep = recomendar_reposicion(b, matriz, cfg, d_ayer)
        if len(rep):
            rep["tipo"] = "REPOSICION"
            rep["algoritmo"] = "regla_reposicion"
            partes.append(rep)
    if "BRECHA" in cfg.incluir_tipos:
        bre = recomendar_brecha(b, matriz, cfg)
        if len(bre):
            evidencia = _pares_bloque(b, matriz)[["f", "i", "dias", "intervalo_medio"]]
            bre = bre.merge(evidencia, on=["f", "i"], how="left")
            bre["tipo"] = "BRECHA"
            bre["algoritmo"] = "regla_brecha"
            bre["dias_sin_comprar"] = np.nan
            bre["dias_compra_item"] = bre.pop("dias").fillna(0).astype(float)
            bre["intervalo_tipico"] = pd.to_numeric(bre.pop("intervalo_medio"), errors="coerce")
            bre["intervalo_esperado"] = b.iv_item[bre["i"].to_numpy(np.int64)]
            bre["compras_esperadas"] = np.nan
            partes.append(bre)
    if not partes:
        return pd.DataFrame()
    out = pd.concat(partes, ignore_index=True)
    filas_local = out["f"].to_numpy(np.int64)

    # ninguna recomendación puede valer más que lo que la entidad compra en ese lapso
    if cfg.tope_potencial_relativo:
        techo = (b.venta_entidad[filas_local] * float(cfg.tope_potencial_relativo)
                 * horizonte / max(b.dias_ventana, 1.0))
        techo = np.where(techo > 0, techo, np.inf)
        out["usd_si_compra"] = np.minimum(out["usd_si_compra"].to_numpy(float), techo)
        out["usd_potencial"] = np.minimum(out["usd_potencial"].to_numpy(float), techo)

    # y a una entidad con una o dos compras en el año no se le recomienda nada
    out["dias_compra_entidad"] = b.dias_entidad[filas_local]
    if cfg.min_dias_compra_entidad:
        out = out[out["dias_compra_entidad"] >= float(cfg.min_dias_compra_entidad)]
        if out.empty:
            return pd.DataFrame()
    if cfg.piso_prob:
        piso = float(cfg.piso_prob)
        subio = out["prob"].to_numpy(float) < piso
        if subio.any():
            out.loc[subio, "prob"] = piso
            out.loc[subio, "usd_potencial"] = out.loc[subio, "usd_si_compra"] * piso
    out["margen_potencial"] = out["usd_potencial"] * b.margen_pct_item[out["i"].to_numpy()]
    orden = "usd_potencial" if cfg.ordenar_por == "esperado" else "usd_si_compra"
    out = out.sort_values(["f", orden, "puntaje"], ascending=[True, False, False])
    # un mismo cliente-ítem puede caer en dos tipos (dejó de comprarlo Y compra menos que
    # sus pares): queda una sola fila, la del tipo que mejor lo explica
    out = out.drop_duplicates(["f", "i"], keep="first")
    out["ranking"] = out.groupby("f").cumcount() + 1
    out = out[out["ranking"] <= cfg.max_items_reco].reset_index(drop=True)
    i = out["i"].to_numpy(np.int64)
    out["entidad"] = b.filas[out["f"].to_numpy(np.int64)]
    out["penetracion"] = b.penetracion[i]
    out["soporte"] = b.soporte[i]
    out["usd_medio_par"] = b.usd_medio_comprador[i]
    out["usd_entidad"] = b.venta_entidad[out["f"].to_numpy(np.int64)]
    out["segmento"] = b.etiqueta
    out["nivel_segmento"] = b.nivel
    return out


# =========================================================================== #
# 5. Backtest: qué algoritmo acierta más en cada segmento
# =========================================================================== #
def backtest(panel: Panel, cfg: RecConfig) -> pd.DataFrame:
    """Entrena con datos hasta hace `dias_backtest` y mide qué se compró después.

    Sólo evalúa recomendaciones CRUZADAS: son las únicas verificables (el ítem no
    estaba y apareció). Devuelve una fila por segmento y algoritmo.
    """
    f = panel.f
    corte = f.d_ayer - cfg.dias_backtest
    inicio = corte - cfg.dias_afinidad + 1 if cfg.dias_afinidad else int(panel.dia.min())
    entrena = panel.matriz(inicio, corte)
    evalua = panel.matriz(corte + 1, f.d_ayer)
    verdad = (evalua.R > 0)
    nuevos = verdad.astype(int) - (entrena.R > 0).astype(int)
    nuevos.data = np.where(nuevos.data > 0, 1, 0)
    nuevos.eliminate_zeros()
    from dataclasses import replace
    panel.cfg = replace(cfg, segmentos=agregar_tamano(panel, entrena))
    seg = asignar_segmentos(panel, entrena)
    nombres = etiquetas_item(panel)
    canastas = panel.canastas(inicio, corte) if cfg.afinidad == "canasta" else None

    filas = []
    codigos, etiquetas = pd.factorize(seg["segmento"])
    for k, etiqueta in enumerate(etiquetas):
        idx = np.flatnonzero(codigos == k)
        b = Bloque(cfg, entrena, idx, str(etiqueta), str(seg["nivel_segmento"].iloc[idx[0]]),
                   canastas=canastas, d_ayer=corte)
        b.dias_ventana = float(corte - inicio + 1)
        b.nombres_item = nombres
        adopciones = int(nuevos[idx].sum())
        entidades_adoptan = int((np.asarray(nuevos[idx].sum(1)).ravel() > 0).sum())
        for nombre in cfg.algoritmos:
            t0 = time.time()
            alg = construir_algoritmo(nombre, cfg)
            alg.ajustar(b)
            reco = recomendar_cruzadas(b, alg, cfg)
            aciertos = usd = 0
            if len(reco):
                glob = b.filas[reco["f"].to_numpy(np.int64)]
                it = reco["i"].to_numpy(np.int64)
                acierta = np.asarray(nuevos[glob, it]).ravel() > 0
                aciertos = int(acierta.sum())
                usd = float(np.asarray(evalua.V[glob, it]).ravel()[acierta].sum())
            filas.append({
                "segmento": str(etiqueta), "nivel_segmento": b.nivel, "entidades": b.n,
                "items_candidatos": int(b.candidato.sum()), "algoritmo": nombre,
                "recomendados": int(len(reco)), "aciertos": aciertos,
                "precision": aciertos / len(reco) if len(reco) else 0.0,
                "adopciones": adopciones, "entidades_que_adoptan": entidades_adoptan,
                "recall": aciertos / adopciones if adopciones else 0.0,
                "usd_acertado": round(usd, 2), "segundos": round(time.time() - t0, 2)})
    d = pd.DataFrame(filas)
    if d.empty:
        return d
    metrica = {"precision": "precision", "usd": "usd_acertado", "recall": "recall"}[cfg.metrica_seleccion]

    # ganador del panel entero: se usa donde el segmento no tiene con qué medirse
    juntos = d.groupby("algoritmo").agg(aciertos=("aciertos", "sum"), recomendados=("recomendados", "sum"),
                                        usd=("usd_acertado", "sum"))
    juntos["precision"] = juntos["aciertos"] / juntos["recomendados"].replace(0, np.nan)
    ganador_global = juntos.sort_values(["precision", "usd"], ascending=False).index[0]

    d = d.sort_values(["segmento", metrica, "usd_acertado"], ascending=[True, False, False])
    mejor_del_segmento = ~d.duplicated("segmento")
    medible = d["adopciones"] >= cfg.min_adopciones_backtest
    d["elegido"] = np.where(medible, mejor_del_segmento, d["algoritmo"] == ganador_global)
    d["motivo_seleccion"] = np.where(
        medible, "el que más acertó en este segmento",
        f"menos de {cfg.min_adopciones_backtest} adopciones para medir: usa el ganador del panel "
        f"({ganador_global})")
    d = d.sort_values(["segmento", "algoritmo"]).reset_index(drop=True)
    d.attrs["ganador_global"] = ganador_global
    LOGGER.info("backtest: ganador del panel %s (precisión %.4f sobre %s recomendaciones)", ganador_global,
                float(juntos.loc[ganador_global, "precision"] or 0),
                f"{int(juntos.loc[ganador_global, 'recomendados']):,}")
    return d


def etiquetas_item(panel: Panel) -> np.ndarray:
    """Nombre legible de cada ítem: su descripción BD_ si hay, si no la clave."""
    cfg = panel.cfg
    columnas = cfg.desc_item() or cfg.claves_item()
    return panel.items[columnas[0]].astype(str).to_numpy(dtype=object)


# =========================================================================== #
# 6. Orquestador
# =========================================================================== #
def catalogo(cfg: RecConfig) -> List[Tuple[str, str, str]]:
    """(columna, tipo Oracle, descripción) de la salida, en orden."""
    cols: List[Tuple[str, str, str]] = []
    for c in cfg.entidad:
        tipo = "VARCHAR2(200)" if str(c).upper().startswith("BD_") else "NUMBER"
        cols.append((c.upper(), tipo, f"Entidad a la que se recomienda ({c})."))
    for c in cfg.item:
        tipo = "VARCHAR2(200)" if str(c).upper().startswith("BD_") else "NUMBER"
        cols.append((c.upper(), tipo, f"Ítem recomendado ({c})."))
    cols += [
        ("MT_RANKING", "NUMBER", "Orden de la recomendación dentro de la entidad: 1 es la de mayor USD en juego."),
        ("BD_TIPO", "VARCHAR2(20)", "CRUZADA (no lo compra y sus pares sí), REPOSICION (lo compraba y se atrasó) "
                                    "o BRECHA (lo compra mucho menos que sus pares)."),
        ("MT_USD_POTENCIAL", "NUMBER", "USD esperados en el horizonte configurado (por defecto 90 días). "
                                       "Es MT_USD_SI_COMPRA por MT_PROB, y es la columna por la que se ordena: "
                                       "los tres tipos quedan en la misma unidad y se pueden comparar."),
        ("MT_USD_SI_COMPRA", "NUMBER", "USD del horizonte si la compra efectivamente ocurre, sin multiplicar por "
                                       "la probabilidad. Sirve para ver el tamaño de la oportunidad."),
        ("MT_PROB", "NUMBER", "Probabilidad de que ocurra. REPOSICION: que vuelva a comprar el ítem, según la "
                              "distribución de intervalos del segmento. CRUZADA: tasa de adopción que midió el "
                              "backtest en ese segmento. BRECHA: 1, porque ya lo compra."),
        ("MT_INTERVALO_ESPERADO", "NUMBER", "Días entre compras estimados para este cliente y este ítem, mezclando "
                                            "su propia historia con la de sus pares del segmento: con una compra "
                                            "manda el segmento, con veinte manda él."),
        ("MT_COMPRAS_ESPERADAS", "NUMBER", "Compras esperadas en el horizonte, según ese intervalo."),
        ("MT_MARGEN_POTENCIAL", "NUMBER", "MT_USD_POTENCIAL por el margen porcentual del ítem en el segmento. USD."),
        ("MT_PUNTAJE", "NUMBER", "Puntaje del algoritmo o de la regla. Comparable dentro del mismo tipo y segmento."),
        ("MT_PENETRACION_SEGMENTO", "NUMBER", "Fracción de las entidades del segmento que compran el ítem."),
        ("MT_SOPORTE_SEGMENTO", "NUMBER", "Cantidad de entidades del segmento que compran el ítem."),
        ("MT_USD_MEDIO_PAR", "NUMBER", "USD que gasta en el ítem una entidad del segmento que sí lo compra. USD."),
        ("MT_USD_ENTIDAD", "NUMBER", "Compra total de la entidad en la ventana de afinidad. USD."),
        ("MT_DIAS_SIN_COMPRAR", "NUMBER", "Sólo REPOSICION: días desde la última compra del ítem."),
        ("MT_DIAS_COMPRA_ITEM", "NUMBER", "Días en que la entidad compró este ítem en la ventana. Es la "
                                          "evidencia detrás de REPOSICION y BRECHA: con 2 compras el ritmo "
                                          "es una casualidad, no un ritmo."),
        ("MT_INTERVALO_TIPICO", "NUMBER", "Días promedio entre compras del ítem por parte de la entidad."),
        ("MT_DIAS_COMPRA_ENTIDAD", "NUMBER", "Días en que la entidad compró algo en la ventana. Mide cuánta "
                                             "historia sostiene la recomendación."),
        ("BD_SEGMENTO", "VARCHAR2(400)", "Segmento contra el que se comparó a la entidad."),
        ("BD_NIVEL_SEGMENTO", "VARCHAR2(100)", "Nivel de segmentación usado, o GLOBAL si ninguno tenía suficientes entidades."),
        ("BD_ALGORITMO", "VARCHAR2(40)", "Algoritmo que produjo la recomendación, o la regla (regla_reposicion / regla_brecha)."),
        ("BD_MOTIVO", "VARCHAR2(400)", "Por qué se recomienda, en palabras."),
        ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo (el día anterior a la ejecución)."),
    ]
    return cols


class RecEngine:
    def __init__(self, config: RecConfig):
        config.validate()
        self.cfg = config
        self.fechas = Fechas.desde(config.fecha_ejecucion)
        self.diagnostico = pd.DataFrame()
        self.tiempos_: Dict[str, float] = {}
        LOGGER.setLevel(logging.INFO if config.verbose else logging.WARNING)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg, f = self.cfg, self.fechas
        t_inicio = time.time()
        panel = preparar(df, cfg, f)
        desde = (f.d_ayer - cfg.dias_afinidad + 1 if cfg.dias_afinidad
                 else int(panel.dia.min()))
        dias_ventana = float(f.d_ayer - desde + 1)
        t0 = time.time()
        matriz = panel.matriz(desde, f.d_ayer)
        self.tiempos_["matriz"] = time.time() - t0
        LOGGER.info("ventana de afinidad %s a %s: %s", (_EPOCA + pd.Timedelta(days=desde)).date(),
                    f.ayer.date(), matriz)

        from dataclasses import replace
        panel.cfg = replace(cfg, segmentos=agregar_tamano(panel, matriz))
        seg = asignar_segmentos(panel, matriz)
        nombres = etiquetas_item(panel)
        canastas = panel.canastas(desde, f.d_ayer) if cfg.afinidad == "canasta" else None
        p_adopcion = {}
        if len(self.diagnostico):
            base = float(cfg.dias_backtest or 90)
            for _, g in self.diagnostico[self.diagnostico["elegido"]].iterrows():
                tasa = float(g["precision"]) * (float(cfg.horizonte_dias or base) / base)
                p_adopcion[str(g["segmento"])] = float(min(max(tasa, 0.0), 1.0))
        if canastas is not None:
            LOGGER.info("afinidad por canasta: %s canastas (%.1f ítems por canasta)",
                        f"{canastas[0].shape[0]:,}", canastas[0].nnz / max(canastas[0].shape[0], 1))
        self.panel, self.matriz, self.segmentos = panel, matriz, seg

        elegido_por_segmento: Dict[str, str] = {}
        if cfg.seleccion == "backtest":
            t0 = time.time()
            self.diagnostico = backtest(panel, cfg)
            self.tiempos_["backtest"] = time.time() - t0
            if len(self.diagnostico):
                ganadores = self.diagnostico[self.diagnostico["elegido"]]
                elegido_por_segmento = dict(zip(ganadores["segmento"], ganadores["algoritmo"]))
                mejor_global = self.diagnostico.attrs.get("ganador_global", cfg.algoritmos[0])
            else:
                mejor_global = cfg.algoritmos[0]
            LOGGER.info("backtest: %s segmentos evaluados en %.1fs",
                        f"{len(elegido_por_segmento):,}", self.tiempos_.get("backtest", 0.0))
        else:
            mejor_global = cfg.seleccion

        salidas = []
        codigos, etiquetas = pd.factorize(seg["segmento"])
        t0 = time.time()
        for k, etiqueta in enumerate(etiquetas):
            idx = np.flatnonzero(codigos == k)
            nivel = str(seg["nivel_segmento"].iloc[idx[0]])
            b = Bloque(cfg, matriz, idx, str(etiqueta), nivel, canastas=canastas, d_ayer=f.d_ayer)
            b.dias_ventana = dias_ventana
            b.p_adopcion = p_adopcion.get(str(etiqueta), 1.0)
            b.nombres_item = nombres
            nombre_alg = elegido_por_segmento.get(str(etiqueta), mejor_global)
            alg = construir_algoritmo(nombre_alg, cfg)
            alg.ajustar(b)
            parte = armar_recomendaciones(b, matriz, alg, cfg, f.d_ayer)
            if len(parte):
                salidas.append(parte)
            LOGGER.info("segmento [%s] nivel %s: %s entidades, %s ítems candidatos, algoritmo %s -> %s recomendaciones",
                        etiqueta, nivel, f"{b.n:,}", f"{int(b.candidato.sum()):,}", nombre_alg,
                        f"{len(parte):,}")
        self.tiempos_["recomendar"] = time.time() - t0

        if not salidas:
            LOGGER.warning("no salió ninguna recomendación")
            return pd.DataFrame(columns=[c for c, _, _ in catalogo(cfg)])
        rec = pd.concat(salidas, ignore_index=True)
        out = self._ensamblar(rec, panel)
        LOGGER.info("%s recomendaciones para %s entidades en %.1fs", f"{len(out):,}",
                    f"{out[cfg.claves_entidad()[0]].nunique():,}", time.time() - t_inicio)
        self.tiempos_["total"] = time.time() - t_inicio
        return out

    def _ensamblar(self, rec: pd.DataFrame, panel: Panel) -> pd.DataFrame:
        cfg = self.cfg
        ent = rec["entidad"].to_numpy(np.int64)
        item = rec["i"].to_numpy(np.int64)
        salida: Dict[str, Any] = {}
        for c in cfg.entidad:
            salida[c.upper()] = panel.entidades[c].to_numpy()[ent]
        for c in cfg.item:
            salida[c.upper()] = panel.items[c].to_numpy()[item]
        dec = cfg.decimales
        salida.update({
            "MT_RANKING": rec["ranking"].to_numpy(int),
            "BD_TIPO": rec["tipo"].to_numpy(dtype=object),
            "MT_USD_POTENCIAL": np.round(rec["usd_potencial"].to_numpy(float), 2),
            "MT_USD_SI_COMPRA": np.round(rec["usd_si_compra"].to_numpy(float), 2),
            "MT_PROB": np.round(rec["prob"].to_numpy(float), 4),
            "MT_INTERVALO_ESPERADO": np.round(rec["intervalo_esperado"].to_numpy(float), 1),
            "MT_COMPRAS_ESPERADAS": np.round(rec["compras_esperadas"].to_numpy(float), 2),
            "MT_MARGEN_POTENCIAL": np.round(rec["margen_potencial"].to_numpy(float), 2),
            "MT_PUNTAJE": np.round(rec["puntaje"].to_numpy(float), dec),
            "MT_PENETRACION_SEGMENTO": np.round(rec["penetracion"].to_numpy(float), dec),
            "MT_SOPORTE_SEGMENTO": rec["soporte"].to_numpy(float),
            "MT_USD_MEDIO_PAR": np.round(rec["usd_medio_par"].to_numpy(float), 2),
            "MT_USD_ENTIDAD": np.round(rec["usd_entidad"].to_numpy(float), 2),
            "MT_DIAS_SIN_COMPRAR": rec["dias_sin_comprar"].to_numpy(float),
            "MT_DIAS_COMPRA_ITEM": rec["dias_compra_item"].to_numpy(float),
            "MT_INTERVALO_TIPICO": np.round(rec["intervalo_tipico"].to_numpy(float), 1),
            "MT_DIAS_COMPRA_ENTIDAD": rec["dias_compra_entidad"].to_numpy(float),
            "BD_SEGMENTO": rec["segmento"].to_numpy(dtype=object),
            "BD_NIVEL_SEGMENTO": rec["nivel_segmento"].to_numpy(dtype=object),
            "BD_ALGORITMO": rec["algoritmo"].to_numpy(dtype=object),
            "BD_MOTIVO": rec["motivo"].to_numpy(dtype=object),
            "FECHA_CORTE": np.full(len(rec), self.fechas.ayer),
        })
        out = pd.DataFrame(salida)
        claves = [c.upper() for c in cfg.claves_entidad()]
        return out.sort_values(claves + ["MT_RANKING"]).reset_index(drop=True)

    def tiempos(self, top: int = 10) -> pd.Series:
        return pd.Series(self.tiempos_).sort_values(ascending=False).head(top)


# =========================================================================== #
# 7. Calibración: probar configuraciones con los datos propios
# =========================================================================== #
def explorar(df: pd.DataFrame, cfg_base: RecConfig,
             niveles_item: Sequence[Sequence[str]] = (),
             rejilla: Optional[Dict[str, Sequence[Any]]] = None,
             verbose: bool = True) -> pd.DataFrame:
    """Mide, con el backtest, qué configuración acierta más sobre TUS datos.

    Prueba cada nivel de ítem (submarca, familia, marca...) y cada combinación de la
    rejilla, y devuelve una fila por configuración con lo que acertó. No escribe nada.

        explorar(df, cfg, niveles_item=[["BK_SUBMARCA", "BD_SUBMARCA"], ["BK_FAMILIA", "BD_FAMILIA"]],
                 rejilla={"min_penetracion": [0.01, 0.05], "k_vecinos": [25, 50]})
    """
    from dataclasses import replace
    from itertools import product

    if isinstance(niveles_item, str) or (niveles_item and isinstance(niveles_item[0], str)):
        raise ValueError("niveles_item es una lista DE LISTAS: [[\"BK_SUBMARCA\", \"BD_SUBMARCA\"], "
                         "[\"BK_FAMILIA\", \"BD_FAMILIA\"]]")
    rejilla = dict(rejilla or {})
    desconocidas = [k for k in rejilla if k not in cfg_base.__dataclass_fields__]
    if desconocidas:
        raise ValueError(f"la rejilla tiene llaves que no son parámetros: {desconocidas}. "
                         f"Se pueden barrer, por ejemplo: afinidad, min_soporte, min_penetracion, "
                         f"min_entidades_segmento, k_vecinos, k_factores, k_clusters, dias_afinidad")
    claves = list(rejilla)
    combos = [dict(zip(claves, valores)) for valores in product(*(rejilla[k] for k in claves))] or [{}]
    niveles = [list(n) for n in (niveles_item or [list(cfg_base.item)])]
    filas = []
    for nivel in niveles:
        faltan = [c for c in nivel if c not in df.columns]
        if faltan:
            raise KeyError(f"el nivel {nivel} pide columnas que no están en la fuente: {faltan}")
        cfg_nivel = replace(cfg_base, item=nivel, seleccion="backtest", verbose=0)
        cfg_nivel.validate()
        f = Fechas.desde(cfg_base.fecha_ejecucion)
        t0 = time.time()
        panel = preparar(df, cfg_nivel, f)
        prep = time.time() - t0
        densidad = None
        for combo in combos:
            cfg = replace(cfg_nivel, **combo)
            panel.cfg = cfg
            t0 = time.time()
            d = backtest(panel, cfg)
            segundos = time.time() - t0
            if densidad is None:
                m = panel.matriz(f.d_ayer - cfg.dias_afinidad + 1, f.d_ayer)
                densidad = m.R.nnz / max(m.n_ent * m.n_item, 1)
                items_medios = m.R.nnz / max(m.n_ent, 1)
            if d.empty:
                continue
            rec, ac = float(d["recomendados"].sum()), float(d["aciertos"].sum())
            adop = float(d["adopciones"].sum())
            pop = d[d["algoritmo"] == "popularidad"]
            pop_prec = (float(pop["aciertos"].sum()) / float(pop["recomendados"].sum())
                        if float(pop["recomendados"].sum()) else 0.0)
            ganadores = d[d["elegido"]]
            elegido = ganadores.groupby("algoritmo").size().sort_values(ascending=False)
            fila = {"nivel_item": " + ".join(nivel), "items": panel.n_item, "entidades": panel.n_ent,
                    "densidad": round(float(densidad), 5), "items_por_entidad": round(items_medios, 2),
                    **combo,
                    "segmentos": int(d["segmento"].nunique()),
                    "recomendados": int(rec), "aciertos": int(ac),
                    "precision": round(ac / rec, 5) if rec else 0.0,
                    "recall": round(ac / adop, 5) if adop else 0.0,
                    "usd_acertado": round(float(d["usd_acertado"].sum()), 2),
                    "precision_popularidad": round(pop_prec, 5),
                    "mejora_vs_popularidad": round((ac / rec) / pop_prec, 2) if (rec and pop_prec) else None,
                    "algoritmos_elegidos": dict(elegido),
                    "segundos": round(segundos + prep, 1)}
            filas.append(fila)
            if verbose:
                LOGGER.info("%s %s -> precisión %.4f (popularidad %.4f), %s USD acertados en %.0fs",
                            " + ".join(nivel), combo or "", fila["precision"], pop_prec,
                            f"{fila['usd_acertado']:,.0f}", fila["segundos"])
    out = pd.DataFrame(filas)
    if out.empty:
        return out
    return out.sort_values(["precision", "usd_acertado"], ascending=False).reset_index(drop=True)


def bloque_config(fila: pd.Series, cfg_base: RecConfig) -> str:
    """El texto para pegar en rec_oracle.py con la configuración ganadora."""
    def limpio(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return round(float(x), 6)
        return x

    nivel = [str(c) for c in str(fila["nivel_item"]).split(" + ")]
    lineas = [f"ITEM = {nivel!r}", ""]
    for k in ("min_soporte", "min_penetracion", "min_entidades_segmento", "max_items_reco",
              "k_vecinos", "k_factores", "k_clusters", "dias_afinidad", "dias_backtest"):
        if k in fila.index:
            lineas.append(f"{k.upper()} = {limpio(fila[k])!r}")
    lineas += ["", f"# backtest sobre datos propios: precisión {fila['precision']:.4f} contra "
                   f"{fila['precision_popularidad']:.4f} de popularidad",
               f"# algoritmos que ganaron por segmento: "
               f"{ {str(k): int(v) for k, v in dict(fila['algoritmos_elegidos']).items()} }",
               f"# {fila['items']} ítems, {fila['entidades']} entidades, "
               f"{fila['items_por_entidad']} ítems por entidad (densidad {fila['densidad']:.3%})"]
    return "\n".join(lineas)


__all__ = ["RecConfig", "RecEngine", "Fechas", "Panel", "Bloque", "Matriz", "ALGORITMOS",
           "catalogo", "backtest", "preparar", "asignar_segmentos", "construir_algoritmo", "TIPOS",
           "explorar", "bloque_config", "agregar_tamano"]
