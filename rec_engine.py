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


def escala_decimal(arrays, max_decimales: int = 6) -> float:
    """Potencia de 10 que vuelve enteros a todos los importes, o 0 si no la hay.

    El dinero es decimal: 1.234,56 son 123.456 centavos. Los enteros se suman SIN
    ERROR en float64 mientras no pasen de 2^53, así que sumar en esa escala hace que
    una venta y su devolución den cero exacto, sin tolerancias ni umbrales. Es lo
    mismo que hace Oracle con NUMBER, y cuesta lo mismo que sumar en float.

    Se elige la escala más chica que sirva y que no desborde 2^53 con el bruto. Si los
    importes traen más decimales de los buscados, se usa el máximo y el resto se
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
                                 "svd", "kmeans_valor", "reglas", "ease", "secuencia", "tendencia")
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
    #: con quién se compara una BRECHA: los N compradores del ítem en el segmento de tamaño
    #: más parecido. Se listan TODOS en la evidencia, así que el número se puede recalcular.
    pares_comparables: int = 30
    #: con menos pares comparables que esto no hay con qué comparar y no se recomienda.
    min_pares_comparables: int = 5
    #: cómo se resume lo que le dedican esos pares: "mediana" (la mitad le dedica más, la mitad
    #: menos; no la mueven los extremos), "agregado" (USD del ítem / USD total de todos ellos)
    #: o "media" (promedio simple de los porcentajes: la inflan los clientes chicos).
    estadistico_pares: str = "mediana"

    # -- algoritmos nuevos ----------------------------------------------------- #
    #: regularización de EASE, relativa al soporte medio de los ítems. Más alta = más
    #: parecido a la popularidad; más baja = más personal y más ruidoso.
    lambda_ease: float = 0.5
    #: EASE invierte una matriz ítem x ítem: con más ítems activos que esto se queda con los
    #: de más soporte (el resto no participa del modelo).
    max_items_ease: int = 4000
    #: "tendencia": qué tan reciente tiene que ser la primera compra para contar como adopción.
    dias_tendencia: int = 90

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

    #: Base mínima, en la moneda de col_valor, para que un porcentaje tenga sentido.
    #: Un ítem con 0,0000064 de venta y -25,04 de margen daría -388.304.245 %: el número
    #: es correcto y no significa nada. Por debajo, el porcentaje vale 0. 0 lo desactiva.
    min_base_porcentaje: float = 1.0
    #: Suma el dinero en su escala decimal (centavos): lo que se compra y se devuelve
    #: entero da cero EXACTO, sin tolerancias. Es lo que hace Oracle con NUMBER.
    suma_exacta: bool = True
    #: hasta cuántos decimales busca esa escala. Más allá, redondea.
    max_decimales: int = 6
    #: Plan B para cuando no hay escala decimal usable. Una suma cuenta como cero
    #: cuando no llega a esta fracción de lo que pasó por ella.
    tolerancia_cero: float = 1e-9

    # -- tablas para comprobar --------------------------------------------------- #
    #: valores de cada ítem en cada segmento (soporte, ticket, ritmo, participación...).
    guardar_referencia: bool = True
    #: la matriz de compras entidad x ítem de la ventana: con ella cualquier número del
    #: segmento se recalcula en SQL. Es la tabla más grande (una fila por par comprado).
    guardar_matriz: bool = True
    #: la curva de recuperación con sus conteos (de dónde sale la probabilidad de recompra).
    guardar_curva: bool = True
    #: los vecinos de cada entidad (coseno_entidad) y su grupo (kmeans_valor).
    guardar_vecindario: bool = True

    # -- evidencia ------------------------------------------------------------ #
    #: arma, para cada recomendación, con quién se la comparó: los pares, el grupo, los
    #: ítems y las reglas que la sostienen, y el cálculo del USD en juego. No cambia nada
    #: de lo que se recomienda: sólo lo deja escrito para poder verificarlo.
    guardar_evidencia: bool = True
    #: pares (o ítems) que se listan por recomendación y por clase de evidencia. Los
    #: totales (cuántos hay, cuántos compran) van siempre completos.
    max_evidencias: int = 3
    #: evidencia sólo para las recomendaciones con ranking hasta este. 0 = todas.
    evidencia_hasta_ranking: int = 0

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

    def cfg_dias_tendencia_invalido(self) -> bool:
        """Sólo importa si se usa "tendencia": tiene que quedar historia antes del tramo reciente."""
        if "tendencia" not in self.algoritmos:
            return False
        if self.dias_tendencia < 1:
            return True
        return bool(self.dias_afinidad) and self.dias_tendencia >= self.dias_afinidad

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
        if self.min_base_porcentaje < 0:
            raise ValueError("min_base_porcentaje no puede ser negativo (0 = sin mínimo)")
        if not 0 <= self.max_decimales <= 15:
            raise ValueError("max_decimales debe estar entre 0 y 15")
        if not 0 <= self.tolerancia_cero < 1:
            raise ValueError("tolerancia_cero debe estar entre 0 y 1 (0 = sin limpieza)")
        if self.estadistico_pares not in ("mediana", "agregado", "media"):
            raise ValueError("estadistico_pares debe ser mediana, agregado o media")
        if self.pares_comparables < 1 or self.min_pares_comparables < 1:
            raise ValueError("pares_comparables y min_pares_comparables deben ser >= 1")
        if self.min_pares_comparables > self.pares_comparables:
            raise ValueError("min_pares_comparables no puede superar a pares_comparables")
        if self.lambda_ease <= 0:
            raise ValueError("lambda_ease debe ser > 0")
        if self.cfg_dias_tendencia_invalido():
            raise ValueError("dias_tendencia debe ser >= 1 y menor que dias_afinidad")
        if self.max_evidencias < 1:
            raise ValueError("max_evidencias debe ser >= 1 (para no guardar evidencia: guardar_evidencia=False)")
        if self.evidencia_hasta_ranking < 0:
            raise ValueError("evidencia_hasta_ranking no puede ser negativo (0 = todas)")
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
    # por grupos y no factorizando un MultiIndex: en pandas 3 eso pierde los nombres de las
    # columnas y una clave de dos columnas (empresa + cliente) no corría
    grupos = df.groupby(columnas, sort=True, dropna=False)
    codigos = grupos.ngroup().to_numpy(np.int64)
    valores = grupos.size().reset_index()[columnas]
    return codigos, valores


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


def _sumar(X: sp.spmatrix, eje: int, escala: float, tol: float) -> np.ndarray:
    """Suma una matriz de dinero por filas (eje=1) o por columnas (eje=0).

    Con escala decimal se suman enteros y el que se anula da cero exacto; sin ella,
    se limpia el residuo contra lo que pasó por la suma.
    """
    if escala:
        Y = X.copy()
        Y.data = np.round(Y.data * escala)
        return np.asarray(Y.sum(eje)).ravel() / escala
    bruto = np.asarray(abs(X).sum(eje)).ravel()
    return sin_residuo(np.asarray(X.sum(eje)).ravel(), bruto, tol, escala_tipica(bruto))


class Matriz:
    """Entidades x ítems dentro de una ventana, ya agregadas."""

    def __init__(self, n_ent: int, n_item: int, tab: pd.DataFrame, tolerancia_cero: float = 0.0,
                 escala: float = 0.0):
        self.n_ent, self.n_item, self.tab = n_ent, n_item, tab
        self.tol, self.escala = float(tolerancia_cero), float(escala)
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
        self.venta_entidad = _sumar(self.V, 1, self.escala, self.tol)
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
        # los conteos quedan guardados: son la evidencia de cada probabilidad de recompra
        casos = np.zeros((self.n_item, len(rejilla)))
        volvieron = np.zeros((self.n_item, len(rejilla)))
        for k, a in enumerate(rejilla):
            # justo en el umbral cuenta (con una tolerancia que absorbe el redondeo de a x intervalo)
            umbral_h = a * iv_h - 1e-9
            umbral_c = a * iv_c - 1e-9
            en_riesgo_h = gap >= umbral_h
            # un silencio abierto sólo cuenta como "no volvió" si ya observamos el horizonte
            # completo después de haberse atrasado; si no, todavía no sabemos y se excluye
            en_riesgo_c = sil >= umbral_c + horizonte
            volvio = en_riesgo_h & (gap <= umbral_h + horizonte + 2e-9)
            riesgo_item = (np.bincount(it_h[en_riesgo_h], minlength=self.n_item)
                           + np.bincount(it_c[en_riesgo_c], minlength=self.n_item)).astype(float)
            volvio_item = np.bincount(it_h[volvio], minlength=self.n_item).astype(float)
            casos[:, k], volvieron[:, k] = riesgo_item, volvio_item
            suficiente = riesgo_item >= min_casos
            por_item[suficiente, k] = volvio_item[suficiente] / riesgo_item[suficiente]
            riesgo_total = float(en_riesgo_h.sum() + en_riesgo_c.sum())
            if riesgo_total >= min_casos:
                global_[k] = float(volvio.sum()) / riesgo_total
        self._curva_cache = (clave, (rejilla, por_item, global_))
        self.curva_conteos = (casos, volvieron)
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
        # escala decimal del dinero (100 = centavos). Con ella las sumas son exactas.
        self.escala = escala_decimal((usd, margen), cfg.max_decimales) if cfg.suma_exacta else 0.0
        if self.escala:
            LOGGER.info("dinero en escala de %d decimales: las sumas son exactas",
                        int(round(np.log10(self.escala))))
        elif cfg.tolerancia_cero:
            LOGGER.info("sin escala decimal usable: los ceros se deciden con tolerancia_cero=%g",
                        cfg.tolerancia_cero)

    def matriz(self, desde: int, hasta: int) -> Matriz:
        """Agrega los eventos de [desde, hasta] a una fila por entidad-ítem."""
        m = (self.dia >= desde) & (self.dia <= hasta)
        # El par que se compró y se devolvió entero tiene que valer 0, no 1e-10. En escala
        # decimal se suman enteros y sale solo; sin escala hay que limpiar el residuo, y
        # para eso se arrastra también el bruto (la suma de los valores absolutos).
        e = self.escala
        base = pd.DataFrame({"e": self.ent[m], "i": self.item[m], "d": self.dia[m],
                             "v": np.round(self.usd[m] * e) if e else self.usd[m],
                             "g": np.round(self.margen[m] * e) if e else self.margen[m]})
        agr_dia = {"v": ("v", "sum"), "g": ("g", "sum")}
        agr_par = {"usd": ("v", "sum"), "margen": ("g", "sum"), "dias": ("d", "size"),
                   "primero": ("d", "min"), "ultimo": ("d", "max")}
        if not e:
            base["va"], base["ga"] = base["v"].abs(), base["g"].abs()
            agr_dia |= {"va": ("va", "sum"), "ga": ("ga", "sum")}
            agr_par |= {"usd_bruto": ("va", "sum"), "margen_bruto": ("ga", "sum")}

        por_dia = base.groupby(["e", "i", "d"], sort=False, as_index=False).agg(**agr_dia)
        tab = por_dia.groupby(["e", "i"], sort=False, as_index=False).agg(**agr_par)
        if e:
            tab["usd"] = tab["usd"].to_numpy(float) / e
            tab["margen"] = tab["margen"].to_numpy(float) / e
        else:
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
        m = Matriz(self.n_ent, self.n_item, tab, self.cfg.tolerancia_cero, self.escala)
        dias_ent = por_dia.drop_duplicates(["e", "d"]).groupby("e").size()
        m.dias_entidad = dias_ent.reindex(range(self.n_ent), fill_value=0).to_numpy(float)
        ok_h = con_hueco["intervalo_medio"].notna().to_numpy()
        # float64: en float32 un silencio de justo 1,5 veces el intervalo quedaba de un lado o
        # del otro según el redondeo, y los conteos de la curva no se podían rehacer
        m.huecos = (con_hueco["i"].to_numpy(np.int32)[ok_h],
                    con_hueco["hueco"].to_numpy(np.float64)[ok_h],
                    con_hueco["intervalo_medio"].to_numpy(np.float64)[ok_h])
        ok_c = censura["intervalo_medio"].notna().to_numpy()
        m.censuras = (censura["i"].to_numpy(np.int32)[ok_c],
                      censura["silencio"].to_numpy(np.float64)[ok_c],
                      censura["intervalo_medio"].to_numpy(np.float64)[ok_c])
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
        self.d_ayer = d_ayer                                  # último día de la ventana (época)
        self.p_adopcion = 1.0                                 # el backtest la ajusta
        self.origen_prob = ""                                 # de dónde salió, para la evidencia

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

        tol, esc = cfg.tolerancia_cero, matriz.escala
        usd_item = _sumar(self.V, 0, esc, tol)
        margen_item = _sumar(self.M, 0, esc, tol)
        seguro = np.maximum(self.soporte, 1.0)
        self.usd_medio_comprador = np.where(self.soporte > 0, usd_item / seguro, 0.0)
        # el ítem sin venta suficiente no tiene margen %: una base de 0,0000064 daría
        # un porcentaje de cientos de millones, correcto y sin ningún sentido
        base = max(cfg.min_base_porcentaje, 0.0)
        con_base = usd_item > 0
        if base:
            con_base &= usd_item >= base
        self.margen_pct_item = np.where(con_base, margen_item / np.where(con_base, usd_item, 1.0), 0.0)

        # ritmo y ticket del ítem EN ESTE SEGMENTO: es la evidencia que se le presta a
        # quien tiene poca historia propia
        self.iv_item = np.full(self.n_item, np.nan)
        self.cv_item = np.full(self.n_item, np.nan)
        self.n_ritmo_item = np.zeros(self.n_item)       # compradores con ritmo: de quiénes sale iv_item
        self.dias_item = np.zeros(self.n_item)          # días de compra sumados: el divisor del ticket
        self.usd_item = np.zeros(self.n_item)
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
            self.dias_item, self.usd_item = total_dias, total_usd
            self.ticket_item = np.where(total_dias > 0, total_usd / np.maximum(total_dias, 1.0), 0.0)
            iv = pd.to_numeric(sub["intervalo_medio"], errors="coerce").to_numpy(float)
            de = pd.to_numeric(sub["intervalo_desvio"], errors="coerce").to_numpy(float)
            con_ritmo = np.isfinite(iv) & (iv > 0)
            if con_ritmo.any():
                tabla = pd.DataFrame({"i": i_sub[con_ritmo], "iv": iv[con_ritmo],
                                      "cv": np.where(iv[con_ritmo] > 0, de[con_ritmo] / iv[con_ritmo], np.nan)})
                agr = tabla.groupby("i").agg(iv=("iv", "median"), cv=("cv", "median"), n=("iv", "size"))
                self.iv_item[agr.index.to_numpy()] = agr["iv"].to_numpy()
                self.cv_item[agr.index.to_numpy()] = agr["cv"].to_numpy()
                self.n_ritmo_item[agr.index.to_numpy()] = agr["n"].to_numpy()

        self.venta_entidad = _sumar(self.V, 1, esc, tol)
        self.dias_entidad = matriz.dias_entidad[filas]
        # la entidad sin compra suficiente tampoco reparte participaciones
        valido = self.venta_entidad > 0
        if cfg.min_base_porcentaje > 0:
            valido &= self.venta_entidad >= cfg.min_base_porcentaje
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
            self.V = None
            return
        svd = TruncatedSVD(n_components=k, random_state=self.cfg.semilla)
        self.U = svd.fit_transform(R)
        self.V = svd.components_                   # todos los ítems: lo usa la evidencia
        self.Vt = svd.components_[:, self.cand]

    def puntuar(self, filas: np.ndarray) -> np.ndarray:
        if self.U is None:
            return np.repeat(self.b.penetracion[self.cand][None, :], len(filas), axis=0)
        return self.U[filas] @ self.Vt

    def explicar(self, filas: np.ndarray, items: np.ndarray) -> np.ndarray:
        """El puntaje es la suma, sobre lo que ya compra, del aporte latente de cada ítem
        (U = R·Vt): se nombra el que más empuja, como en los demás modelos ítem a ítem."""
        if self.U is None:
            return super().explicar(filas, items)
        V = self.V

        class Aporte:
            def __getitem__(self, clave):
                j, i = clave
                return np.einsum("kn,kn->n", V[:, np.asarray(j)], V[:, np.asarray(i)])
        return _explicar_por_afinidad(self, Aporte(), filas, items, "patrón de consumo (SVD) con lo que ya compra:")


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
        # el lift de cada regla, calculado sobre los MISMOS datos que la confianza. Antes se
        # armaba como otra matriz y se filtraba posición por posición, pero las dos no
        # guardaban las columnas en el mismo orden: se descartaban reglas buenas y quedaban
        # reglas con lift menor a 1.
        lift = conf.data / pen[conf.indices]
        conf.data = np.where(lift > 1.0, conf.data, 0.0)
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


class _Indexada:
    """Una matriz densa sobre un subconjunto de ítems que se lee con índices de TODOS los
    ítems, como una dispersa: lo que no está en el subconjunto vale 0."""

    def __init__(self, M: np.ndarray, posicion: np.ndarray):
        self.M, self.posicion = M, posicion

    def __getitem__(self, clave):
        j, i = clave
        pj, pi = self.posicion[np.asarray(j)], self.posicion[np.asarray(i)]
        out = np.zeros(np.broadcast(pj, pi).shape)
        ok = (pj >= 0) & (pi >= 0)
        out[ok] = self.M[pj[ok], pi[ok]]
        return out


class Ease(Algoritmo):
    nombre = "ease"
    descripcion = ("EASE (Steck, 2019): modelo lineal ítem a ítem resuelto de forma cerrada. Aprende cuánto "
                   "empuja cada ítem que ya compra hacia cada candidato, descontando lo que ya explican los "
                   "demás. Suele igualar o superar a modelos mucho más complejos, y el puntaje es la suma "
                   "exacta de los aportes de cada ítem propio.")

    def ajustar(self, b: Bloque) -> None:
        super().ajustar(b)
        X = (b.A > 0).astype(np.float64).tocsc()
        soporte = np.asarray(X.sum(0)).ravel()
        activos = np.flatnonzero(soporte > 0)
        if len(activos) > self.cfg.max_items_ease:
            activos = np.sort(activos[np.argsort(-soporte[activos], kind="stable")[:self.cfg.max_items_ease]])
        self.posicion = -np.ones(b.n_item, dtype=np.int64)
        self.posicion[activos] = np.arange(len(activos))
        self.activos = activos
        if len(activos) < 2:
            self.B = np.zeros((len(activos), len(activos)))
        else:
            G = (X[:, activos].T @ X[:, activos]).toarray()
            lam = float(self.cfg.lambda_ease) * float(np.mean(np.diag(G)))
            G[np.diag_indices_from(G)] += lam
            P = np.linalg.inv(G)
            self.B = P / (-np.diag(P))[None, :]
            np.fill_diagonal(self.B, 0.0)
        col = self.posicion[self.cand]
        self.Bc = np.zeros((len(activos), len(self.cand)))
        ok = col >= 0
        self.Bc[:, ok] = self.B[:, col[ok]]
        self.pesos = _Indexada(self.B, self.posicion)

    def puntuar(self, filas: np.ndarray) -> np.ndarray:
        Rf = (self.b.R[filas] > 0).astype(np.float64).tocsc()[:, self.activos].tocsr()
        return np.asarray(Rf @ self.Bc)

    def explicar(self, filas: np.ndarray, items: np.ndarray) -> np.ndarray:
        return _explicar_por_afinidad(self, self.pesos, filas, items, "por lo que ya compra (EASE):")


class Secuencia(Algoritmo):
    nombre = "secuencia"
    descripcion = ("Reglas de secuencia: de las entidades que compraron A, qué fracción compró B DESPUÉS (las "
                   "que ya tenían B antes no cuentan). Recomienda lo que suele venir a continuación de lo que ya "
                   "compra, exigiendo lift mayor a 1. Cada regla se comprueba con las fechas de primera compra.")

    def ajustar(self, b: Bloque) -> None:
        super().ajustar(b)
        R = (b.R > 0).astype(np.float64).tocsr()
        Pr = b.P.tocsr()
        Pr.sort_indices()
        R.sort_indices()
        n_item = b.n_item
        soporte = np.asarray(R.sum(0)).ravel()
        # cuántas entidades compraron a ANTES que c (primera compra de cada uno en la ventana)
        antes = sp.csr_matrix((n_item, n_item))
        filas_a, filas_c, acumulados = [], [], 0

        def volcar():
            nonlocal antes, filas_a, filas_c, acumulados
            if filas_a:
                a, c = np.concatenate(filas_a), np.concatenate(filas_c)
                antes = antes + sp.csr_matrix((np.ones(len(a)), (a, c)), shape=(n_item, n_item))
            filas_a, filas_c, acumulados = [], [], 0

        for r in range(b.n):                      # por entidad: sus ítems ordenados por fecha
            a, z = Pr.indptr[r], Pr.indptr[r + 1]
            if z - a < 2:
                continue
            it, dia = Pr.indices[a:z], Pr.data[a:z]
            ii, cc = np.meshgrid(np.arange(z - a), np.arange(z - a), indexing="ij")
            m = dia[ii] < dia[cc]
            filas_a.append(it[ii[m]])
            filas_c.append(it[cc[m]])
            acumulados += int(m.sum())
            if acumulados > 5_000_000:            # memoria acotada
                volcar()
        volcar()
        antes = antes.tocsr()
        antes.sum_duplicates()
        antes.sort_indices()
        filas = np.repeat(np.arange(n_item), np.diff(antes.indptr))
        ambos = np.asarray((R.T @ R).tocsr()[filas, antes.indices]).ravel()
        # elegibles: compraron a y NO tenían c de antes (o del mismo día)
        elegibles = soporte[filas] - (ambos - antes.data)
        conf = np.where(elegibles > 0, antes.data / np.maximum(elegibles, 1.0), 0.0)
        pen = soporte / max(b.n, 1)
        lift = conf / np.where(pen[antes.indices] > 0, pen[antes.indices], 1.0)
        sirve = (lift > 1.0) & (antes.data >= max(int(self.cfg.min_soporte), 1))
        # copias de `antes` con otros valores: misma estructura y mismos tipos de índice (armarlas
        # a mano mezclaba int32 e int64 y scipy no lo acepta), y eliminate_zeros, que compacta EN
        # EL LUGAR, no toca a `antes`
        self.C = antes.copy()
        self.C.data = np.where(sirve, conf, 0.0).astype(np.float64)
        self.C.eliminate_zeros()
        self.antes = antes
        self.elegibles = antes.copy()
        self.elegibles.data = np.asarray(elegibles, dtype=np.float64)
        self.Cc = self.C[:, self.cand].tocsr()

    def limite_filas(self) -> int:
        return Reglas.limite_filas(self)

    def puntuar(self, filas: np.ndarray) -> np.ndarray:
        return Reglas.puntuar(self, filas)

    def explicar(self, filas: np.ndarray, items: np.ndarray) -> np.ndarray:
        return _explicar_por_afinidad(self, self.C, filas, items, "secuencia: después de comprar")


class Tendencia(Algoritmo):
    nombre = "tendencia"
    descripcion = ("Adopción reciente en el segmento: de las entidades que no compraban el ítem, qué fracción "
                   "empezó a comprarlo en los últimos dias_tendencia días. Recomienda lo que el segmento está "
                   "empezando a comprar. No personaliza: es la popularidad de lo nuevo.")

    def ajustar(self, b: Bloque) -> None:
        super().ajustar(b)
        self.corte = (float(b.d_ayer) - float(self.cfg.dias_tendencia)) if b.d_ayer is not None else np.inf
        P = b.P.tocsc()
        columna = np.repeat(np.arange(b.n_item), np.diff(P.indptr))
        # primera compra dentro del tramo reciente = no lo había comprado antes en la ventana
        nuevos = np.bincount(columna[P.data > self.corte], minlength=b.n_item).astype(float)
        previos = b.soporte - nuevos
        self.nuevos = nuevos
        self.elegibles = np.maximum(float(b.n) - previos, 0.0)
        self.tasa = np.where(self.elegibles > 0, nuevos / np.maximum(self.elegibles, 1.0), 0.0)

    def puntuar(self, filas: np.ndarray) -> np.ndarray:
        return np.repeat(self.tasa[self.cand][None, :], len(filas), axis=0)

    def explicar(self, filas: np.ndarray, items: np.ndarray) -> np.ndarray:
        d = int(self.cfg.dias_tendencia)
        antes = max(int(round(self.b.dias_ventana - d)), 0)
        return np.array([f"de las {e:.0f} entidades del segmento que no lo compraron en los {antes} días anteriores, "
                         f"{nv:.0f} empezaron a comprarlo en los últimos {d} ({t:.0%})"
                         for e, nv, t in zip(self.elegibles[items], self.nuevos[items], self.tasa[items])],
                        dtype=object)


ALGORITMOS: Dict[str, type] = {a.nombre: a for a in
                               (Popularidad, CosenoItem, CosenoEntidad, Svd, KmeansValor, Reglas,
                                Ease, Secuencia, Tendencia)}


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

    def _aportes(self, filas: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Lo que aporta cada algoritmo al puntaje fusionado de cada (fila, ítem)."""
        posicion = -np.ones(self.b.n_item, dtype=np.int64)
        posicion[self.cand] = np.arange(len(self.cand))
        col = posicion[items]
        aportes = np.zeros((len(filas), len(self.algos)))
        unicas, donde = np.unique(filas, return_inverse=True)
        paso = max(1, min(self.cfg.filas_bloque, self.limite_filas()))
        for ini in range(0, len(unicas), paso):
            f = unicas[ini:ini + paso]
            sel = (donde >= ini) & (donde < ini + paso) & (col >= 0)
            r = donde[sel] - ini
            for k, a in enumerate(self.algos):
                peso = float(self.cfg.pesos.get(a.nombre, 1.0))
                P = a.puntuar(f)
                if self.modo == "rrf":
                    posicion_p = np.argsort(np.argsort(-P, axis=1), axis=1)
                    C = peso / (60.0 + posicion_p + 1.0)
                else:
                    maximo = P.max(axis=1, keepdims=True)
                    C = peso * P / np.where(maximo > 0, maximo, 1.0)
                aportes[sel, k] = C[r, col[sel]]
        return aportes

    def explicar(self, filas: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Dice qué algoritmo de la batería aportó más a esta recomendación, y su motivo.
        Antes daba siempre el de coseno_item, aunque no fuera el que la puso arriba."""
        aportes = self._aportes(filas, items)
        mejor = np.argmax(aportes, axis=1)
        salida = np.empty(len(filas), dtype=object)
        total = aportes.sum(axis=1)
        for k, a in enumerate(self.algos):
            sel = np.flatnonzero(mejor == k)
            if not len(sel):
                continue
            textos = a.explicar(filas[sel], items[sel])
            frac = aportes[sel, k] / np.where(total[sel] > 0, total[sel], 1.0)
            salida[sel] = [f"fusión {self.modo}; aporta más {a.nombre} ({p:.0%}): {t}"
                           for p, t in zip(frac, textos)]
        return salida


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
        "motivo": [_motivo_reposicion(c, ip, iv_, ivp, npar, s, pr)
                   for c, ip, iv_, ivp, npar, s, pr in zip(compras, iv_propio[ok], iv, b.iv_item[item[ok]],
                                                           b.n_ritmo_item[item[ok]], sil, prob)]})


def _motivo_reposicion(compras, iv_propio, iv_est, iv_pares, n_pares, silencio, prob) -> str:
    """El texto dice con qué evidencia se armó: la propia, la de los pares, o las dos, y
    nombra cada número por lo que es (el propio, el del segmento, o la mezcla)."""
    pares = (f"en su segmento lo compran cada {iv_pares:.0f} días (mediana de {n_pares:.0f} compradores)"
             if np.isfinite(iv_pares) else "en su segmento nadie lo compra con ritmo medible")
    if np.isfinite(iv_propio) and compras >= 3:
        base = f"compró {compras:.0f} veces, cada {iv_propio:.0f} días en promedio"
    elif np.isfinite(iv_propio):
        base = (f"compró {compras:.0f} veces (cada {iv_propio:.0f} días); {pares}: se estima cada "
                f"{iv_est:.0f}")
    else:
        base = f"compró {compras:.0f} vez; {pares}"
    return f"{base}, lleva {silencio:.0f} sin comprar (probabilidad de recompra {prob:.0%})"


_NOMBRE_ESTADISTICO = {"mediana": "mediana", "agregado": "en conjunto", "media": "promedio simple"}


def _candidatos_brecha(b: Bloque, matriz: Matriz, cfg: RecConfig) -> Dict[int, np.ndarray]:
    """Por ítem, las entidades del bloque que lo compran y tienen compra suficiente para
    repartir participaciones: entre ellas se buscan los pares comparables."""
    if getattr(b, "_cand_brecha", None) is None:
        p = _pares_bloque(b, matriz)
        valido = b.venta_entidad > 0
        if cfg.min_base_porcentaje > 0:
            valido &= b.venta_entidad >= cfg.min_base_porcentaje
        sub = p[valido[p["f"].to_numpy(np.int64)]]
        b._cand_brecha = {int(k): g.to_numpy(np.int64) for k, g in sub.groupby("i")["f"]}
    return b._cand_brecha


def pares_comparables(b: Bloque, matriz: Matriz, cfg: RecConfig, f: np.ndarray, i: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Los `pares_comparables` compradores del ítem, en el segmento, de tamaño más parecido
    a cada entidad (sin ella misma). Es UNA sola regla: la usa la BRECHA para comparar y la
    evidencia para listarlos, así que los listados son exactamente los comparados."""
    cand = _candidatos_brecha(b, matriz, cfg)
    pos, par, _ = _cercanos(i, f, lambda k: cand.get(int(k), ()), _tamano_log(b),
                            int(cfg.pares_comparables))
    return pos, par


def _resumen_pares(b: Bloque, cfg: RecConfig, pos: np.ndarray, par: np.ndarray, items: np.ndarray,
                   n_filas: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lo que le dedican al ítem los pares comparables de cada fila: (referencia, cuántos,
    cuartil 1, cuartil 3), según `estadistico_pares`."""
    ref = np.full(n_filas, np.nan)
    q1 = np.full(n_filas, np.nan)
    q3 = np.full(n_filas, np.nan)
    cuantos = np.bincount(pos, minlength=n_filas).astype(float)
    if not len(pos):
        return ref, cuantos, q1, q3
    it = items[pos]
    usd = np.asarray(b.V[par, it]).ravel()
    total = b.venta_entidad[par]
    share = usd / np.where(total > 0, total, 1.0)
    t = pd.DataFrame({"pos": pos, "share": share, "usd": usd, "total": total})
    g = t.groupby("pos")
    idx = g.size().index.to_numpy()
    if cfg.estadistico_pares == "mediana":
        ref[idx] = g["share"].median().to_numpy()
    elif cfg.estadistico_pares == "agregado":
        s = g[["usd", "total"]].sum()
        ref[idx] = (s["usd"] / s["total"].where(s["total"] > 0)).to_numpy()
    else:
        ref[idx] = g["share"].mean().to_numpy()
    q1[idx] = g["share"].quantile(0.25).to_numpy()
    q3[idx] = g["share"].quantile(0.75).to_numpy()
    return ref, cuantos, q1, q3


def recomendar_brecha(b: Bloque, matriz: Matriz, cfg: RecConfig) -> pd.DataFrame:
    """Lo que compra, pero mucho menos de lo que le dedican sus pares comparables.

    Los pares son los `pares_comparables` compradores del ítem en el segmento de tamaño más
    parecido, y lo que le dedican se resume con `estadistico_pares` (la mediana, por
    defecto). Antes se comparaba contra el promedio simple de TODOS los compradores del
    segmento: no se sabía contra quiénes, y los clientes chicos (a los que un ítem les pesa
    mucho) lo inflaban.
    """
    columnas = ["f", "i", "puntaje", "usd_potencial", "usd_si_compra", "prob", "motivo"]
    p = _pares_bloque(b, matriz)
    if p.empty:
        return pd.DataFrame(columns=columnas)
    f = p["f"].to_numpy(np.int64)
    i = p["i"].to_numpy(np.int64)
    venta_ent = b.venta_entidad[f]
    share = np.where(venta_ent > 0, p["usd"].to_numpy(float) / np.where(venta_ent > 0, venta_ent, 1.0), 0.0)
    posible = np.flatnonzero(b.candidato[i] & (venta_ent > 0))
    if not len(posible):
        return pd.DataFrame(columns=columnas)
    ref = np.full(len(f), np.nan)
    cuantos = np.zeros(len(f))
    q1 = np.full(len(f), np.nan)
    q3 = np.full(len(f), np.nan)
    for ini in range(0, len(posible), 50_000):           # por tramos: memoria acotada
        sel = posible[ini:ini + 50_000]
        pos, par = pares_comparables(b, matriz, cfg, f[sel], i[sel])
        r, c, a, z = _resumen_pares(b, cfg, pos, par, i[sel], len(sel))
        ref[sel], cuantos[sel], q1[sel], q3[sel] = r, c, a, z
    ok = ((cuantos >= cfg.min_pares_comparables) & np.isfinite(ref) & (ref > 0)
          & (share < cfg.brecha_ratio * np.nan_to_num(ref)))
    if not ok.any():
        return pd.DataFrame(columns=columnas)
    sh, me, ve = share[ok], ref[ok], venta_ent[ok]
    horizonte = float(cfg.horizonte_dias) if cfg.horizonte_dias else b.dias_ventana
    falta = (me - sh) * ve * horizonte / max(b.dias_ventana, 1.0)
    nombre = _NOMBRE_ESTADISTICO[cfg.estadistico_pares]
    return pd.DataFrame({
        "f": f[ok], "i": i[ok],
        "puntaje": 1.0 - sh / me,
        "usd_potencial": falta,
        "usd_si_compra": falta,
        "prob": np.ones(len(falta)),
        "pares_ref": cuantos[ok], "ref_valor": me, "ref_q1": q1[ok], "ref_q3": q3[ok],
        "motivo": [f"sus {n:.0f} pares de tamaño parecido que lo compran le dedican {_pct1(m)} de su compra "
                   f"({nombre}) y esta entidad {_pct1(s)}" for n, m, s in zip(cuantos[ok], me, sh)]})


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
# 4b. Evidencia: con quién se comparó cada recomendación
# =========================================================================== #
#: clases de evidencia, en el orden en que se listan
EVIDENCIAS = ("CALCULO", "COMPRADOR_SEGMENTO", "VECINO", "MIEMBRO_GRUPO", "ITEM_AFIN", "REGLA",
              "FACTOR_LATENTE", "ITEM_EASE", "SECUENCIA", "CO_COMPRADOR", "ADOPTANTE", "ADOPTANTE_RECIENTE",
              "PAR_RITMO", "PAR_PARTICIPACION")

_COLUMNAS_EVIDENCIA = ["pos", "evidencia", "fuente", "par", "ref", "similitud", "contribucion", "lift",
                       "en_comun", "grupo", "tamano_grupo", "compran_grupo", "detalle"]


def _num(x: float) -> str:
    """Un número legible: sin decimales si es grande, con los que hagan falta si es chico."""
    if x is None or not np.isfinite(x):
        return "sin dato"
    a = abs(float(x))
    return f"{x:,.0f}" if a >= 100 else (f"{x:,.2f}" if a >= 1 else f"{x:.3g}")


def _pct(p: float) -> str:
    """Un porcentaje que no se redondea a 0% cuando es chico."""
    if p is None or not np.isfinite(p):
        return "sin dato"
    return f"{p:.0%}" if abs(p) >= 0.1 else (f"{p:.1%}" if abs(p) >= 0.01 else f"{p:.2%}")


def _vacia() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object if c in ("evidencia", "fuente", "grupo", "detalle")
                                      else float) for c in _COLUMNAS_EVIDENCIA})


def _evidencia_df(pos, evidencia: str, fuente: str, **cols) -> pd.DataFrame:
    """Filas de evidencia con todas las columnas; lo que no se da queda vacío."""
    n = len(pos)
    out = {"pos": np.asarray(pos, np.int64), "evidencia": np.full(n, evidencia, dtype=object),
           "fuente": np.full(n, fuente, dtype=object)}
    for c in _COLUMNAS_EVIDENCIA[3:]:
        v = cols.get(c)
        if v is None:
            out[c] = np.full(n, None if c in ("grupo", "detalle") else np.nan,
                             dtype=object if c in ("grupo", "detalle") else float)
        elif np.isscalar(v) or isinstance(v, str):
            out[c] = np.full(n, v, dtype=object if isinstance(v, str) else float)
        else:
            out[c] = np.asarray(v)
    return pd.DataFrame(out)


def _tamano_log(b: Bloque) -> np.ndarray:
    """Tamaño de cada entidad del bloque, en log: es la distancia con que se eligen los pares."""
    return np.log(np.maximum(b.venta_entidad, 1e-9))


def _compradores(b: Bloque) -> sp.csc_matrix:
    """Quién compra cada ítem (columna) dentro del bloque."""
    if getattr(b, "_compradores", None) is None:
        b._compradores = (b.R > 0).astype(np.int8).tocsc()
        b._compradores.sort_indices()
    return b._compradores


def _cercanos(claves: np.ndarray, f_fila: np.ndarray, candidatos, tam: np.ndarray,
              n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Los `n` candidatos de tamaño más parecido a cada fila, sin contarse a sí misma.

    `claves` agrupa las filas que comparten candidatos (el ítem, el par de ítems, el grupo);
    `candidatos(clave)` devuelve las entidades locales que podrían ser su par. Se ordenan
    por tamaño y a cada fila se le mira una ventana de 2n+2 alrededor del suyo: los n más
    cercanos están siempre adentro. Devuelve (fila, par, cuántos pares tenía cada fila, sin
    contarse a sí misma).
    """
    if not len(claves):
        vacio = np.zeros(0, dtype=np.int64)
        return vacio, vacio, vacio
    codigos, unicos = pd.factorize(pd.Series(claves))
    orden = np.argsort(codigos, kind="stable")
    cortes = np.flatnonzero(np.diff(codigos[orden])) + 1
    filas_out, pares_out = [], []
    total = np.zeros(len(claves), dtype=np.int64)
    ancho = 2 * n + 2
    for grupo in np.split(orden, cortes):
        cand = np.asarray(candidatos(unicos[codigos[grupo[0]]]), dtype=np.int64)
        if not len(cand):
            continue
        total[grupo] = len(cand) - np.isin(f_fila[grupo], cand)
        ls = tam[cand]
        o = np.argsort(ls, kind="stable")
        cand, ls = cand[o], ls[o]
        lf = tam[f_fila[grupo]]
        if len(cand) <= ancho:
            idx = np.tile(np.arange(len(cand)), (len(grupo), 1))
        else:
            ini = np.clip(np.searchsorted(ls, lf) - (n + 1), 0, len(cand) - ancho)
            idx = ini[:, None] + np.arange(ancho)
        pc = cand[idx]
        dist = np.abs(ls[idx] - lf[:, None])
        dist[pc == f_fila[grupo][:, None]] = np.inf
        sel = np.argsort(dist, axis=1, kind="stable")[:, :n]
        pc = np.take_along_axis(pc, sel, axis=1)
        ok = np.isfinite(np.take_along_axis(dist, sel, axis=1))
        filas_out.append(np.repeat(grupo, sel.shape[1])[ok.ravel()])
        pares_out.append(pc.ravel()[ok.ravel()])
    if not filas_out:
        vacio = np.zeros(0, dtype=np.int64)
        return vacio, vacio, total
    return np.concatenate(filas_out), np.concatenate(pares_out), total


def _primeros(pos: np.ndarray, valor: np.ndarray, n: int) -> np.ndarray:
    """Máscara de los `n` de mayor valor dentro de cada `pos`."""
    if not len(pos):
        return np.zeros(0, dtype=bool)
    orden = np.lexsort((-valor, pos))
    rango = np.empty(len(pos), dtype=np.int64)
    p = pos[orden]
    inicio = np.r_[True, p[1:] != p[:-1]]
    corrido = np.arange(len(p)) - np.maximum.accumulate(np.where(inicio, np.arange(len(p)), 0))
    rango[orden] = corrido
    return rango < n


def _por_compradores(b: Bloque, filas: np.ndarray, items: np.ndarray, n: int, evidencia: str,
                     fuente: str, grupo_de=None, nombre_grupo=None, tamano_grupo=None) -> pd.DataFrame:
    """Pares que compran el ítem, los de tamaño más parecido. `grupo_de` (opcional) restringe
    a las entidades del mismo grupo que la fila (el cluster de k-means)."""
    C = _compradores(b)
    tam = _tamano_log(b)
    if grupo_de is None:
        claves = items.astype(np.int64)

        def candidatos(i):
            return C.indices[C.indptr[i]:C.indptr[i + 1]]
    else:
        g = grupo_de[filas]
        claves = g.astype(np.int64) * b.n_item + items

        def candidatos(clave):
            gg, i = divmod(int(clave), b.n_item)
            compran = C.indices[C.indptr[i]:C.indptr[i + 1]]
            return compran[grupo_de[compran] == gg]
    pos, par, total = _cercanos(claves, filas, candidatos, tam, n)
    grupo = (np.full(len(filas), b.etiqueta, dtype=object) if nombre_grupo is None else nombre_grupo)
    tamano = np.full(len(filas), float(b.n)) if tamano_grupo is None else tamano_grupo
    return _evidencia_df(pos, evidencia, fuente, par=par, grupo=np.asarray(grupo, dtype=object)[pos],
                         tamano_grupo=np.asarray(tamano, float)[pos],
                         compran_grupo=total[pos].astype(float))


def _por_item_propio(b: Bloque, filas: np.ndarray, items: np.ndarray, n: int, evidencia: str,
                     fuente: str, valor, A: sp.csr_matrix, solo_positivos: bool = True
                     ) -> Tuple[pd.DataFrame, np.ndarray]:
    """Los ítems que la entidad YA compra y más empujan al recomendado.

    `valor(j, i)` es lo que aporta el ítem propio j al candidato i (afinidad, confianza o
    aporte latente). `A` es la matriz con que se cuentan los que compran los dos. Devuelve
    las filas y, por fila, el ítem propio que más aporta (-1 si ninguno), para listar
    después a quienes compran los dos.
    """
    # quiénes tienen cada par de ítems: una sola multiplicación por matriz, cacheada por bloque
    cache = b.__dict__.setdefault("_co_ocurrencia", {})
    if id(A) not in cache:
        Ab = (A > 0).astype(np.float64).tocsr()
        cache[id(A)] = ((Ab.T @ Ab).tocsr(), np.asarray(Ab.sum(0)).ravel(), A)
    co, soporte_j, _ = cache[id(A)]
    partes, principal = [], np.full(len(filas), -1, dtype=np.int64)
    trozo = 20_000
    for ini in range(0, len(filas), trozo):
        f, it = filas[ini:ini + trozo], items[ini:ini + trozo]
        sub = (b.R[f] > 0).tocsr()
        if sub.nnz == 0:
            continue
        pos = np.repeat(np.arange(len(f)), np.diff(sub.indptr))
        j, i = sub.indices.astype(np.int64), it[pos]
        v = np.asarray(valor(j, i), dtype=float).ravel()
        ok = v > 0 if solo_positivos else v != 0
        # el aporte es la fracción de lo que empuja A FAVOR (en svd hay ítems que restan)
        total = np.bincount(pos[ok], weights=np.abs(v[ok]), minlength=len(f))
        pos, j, i, v = pos[ok], j[ok], i[ok], v[ok]
        top = _primeros(pos, v, n)
        pos, j, i, v = pos[top], j[top], i[top], v[top]
        if not len(pos):
            continue
        mejor = _primeros(pos, v, 1)
        principal[ini + pos[mejor]] = j[mejor]
        en_comun = np.asarray(co[j, i]).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            contrib = np.where(total[pos] != 0, v / total[pos], np.nan)
        partes.append(_evidencia_df(ini + pos, evidencia, fuente, ref=j, similitud=v,
                                    contribucion=contrib, en_comun=en_comun,
                                    tamano_grupo=soporte_j[j]))
    return (pd.concat(partes, ignore_index=True) if partes else _vacia()), principal


def _co_compradores(b: Bloque, filas: np.ndarray, items: np.ndarray, principal: np.ndarray,
                    n: int, fuente: str, en_orden: bool = False) -> pd.DataFrame:
    """Entidades que compran a la vez el ítem propio principal y el recomendado: son los
    pares que hacen verificable una afinidad entre ítems. Con `en_orden`, sólo las que
    compraron primero el principal y DESPUÉS el recomendado (la secuencia)."""
    con = np.flatnonzero(principal >= 0)
    if not len(con):
        return _vacia()
    C = _compradores(b)
    claves = principal[con] * b.n_item + items[con]

    Pc = b.P.tocsc() if en_orden else None

    def candidatos(clave):
        j, i = divmod(int(clave), b.n_item)
        ambos = np.intersect1d(C.indices[C.indptr[j]:C.indptr[j + 1]],
                               C.indices[C.indptr[i]:C.indptr[i + 1]], assume_unique=True)
        if en_orden and len(ambos):
            dj = np.asarray(Pc[ambos, j].todense()).ravel()
            di = np.asarray(Pc[ambos, i].todense()).ravel()
            ambos = ambos[dj < di]
        return ambos
    pos, par, total = _cercanos(claves, filas[con], candidatos, _tamano_log(b), n)
    return _evidencia_df(con[pos], "ADOPTANTE" if en_orden else "CO_COMPRADOR", fuente, par=par,
                         ref=principal[con][pos], compran_grupo=total[pos].astype(float))


def _evidencia_algoritmo(alg: Algoritmo, b: Bloque, filas: np.ndarray, items: np.ndarray,
                         n: int) -> pd.DataFrame:
    """Con quién comparó el algoritmo para recomendar cada (fila, ítem)."""
    if isinstance(alg, Fusion):
        return pd.concat([_evidencia_algoritmo(a, b, filas, items, n) for a in alg.algos],
                         ignore_index=True)
    nombre = alg.nombre
    if isinstance(alg, CosenoEntidad):
        partes = []
        Rb = (b.R > 0).tocsr()
        for ini in range(0, len(filas), 20_000):
            f, it = filas[ini:ini + 20_000], items[ini:ini + 20_000]
            W = alg._vecinos(f).tocoo()
            r, j, s = W.row.astype(np.int64), W.col.astype(np.int64), W.data.astype(float)
            vecinos = np.bincount(r, minlength=len(f)).astype(float)
            compra = np.asarray(Rb[j, it[r]]).ravel() > 0
            r, j, s = r[compra], j[compra], s[compra]
            compran = np.bincount(r, minlength=len(f)).astype(float)
            total = np.bincount(r, weights=s, minlength=len(f))
            top = _primeros(r, s, n)
            r, j, s = r[top], j[top], s[top]
            partes.append(_evidencia_df(ini + r, "VECINO", nombre, par=j, similitud=s,
                                        contribucion=s / np.where(total[r] > 0, total[r], 1.0),
                                        grupo=f"sus {int(alg.cfg.k_vecinos)} pares más parecidos",
                                        tamano_grupo=vecinos[r], compran_grupo=compran[r]))
        return pd.concat(partes, ignore_index=True) if partes else _vacia()
    if isinstance(alg, KmeansValor):
        if alg.pen.shape[0] == 1:
            return _por_compradores(b, filas, items, n, "COMPRADOR_SEGMENTO", nombre)
        g = alg.etiquetas[filas]
        nombres = np.array([f"k-means #{x}" for x in range(len(alg.tam))], dtype=object)
        return _por_compradores(b, filas, items, n, "MIEMBRO_GRUPO", nombre, grupo_de=alg.etiquetas,
                                nombre_grupo=nombres[g], tamano_grupo=alg.tam[g])
    if isinstance(alg, CosenoItem):
        ev, principal = _por_item_propio(b, filas, items, n, "ITEM_AFIN", nombre,
                                         lambda j, i: alg.S[j, i], b.A)
        return pd.concat([ev, _co_compradores(b, filas, items, principal, n, nombre)], ignore_index=True)
    if isinstance(alg, Reglas):
        soporte = np.asarray((b.A > 0).sum(0)).ravel().astype(float)
        pen = soporte / max(b.n_afinidad, 1)
        ev, principal = _por_item_propio(b, filas, items, n, "REGLA", nombre,
                                         lambda j, i: alg.C[j, i], b.A)
        if len(ev):
            i_ev = items[ev["pos"].to_numpy(np.int64)]
            ev["lift"] = ev["similitud"].to_numpy(float) / np.where(pen[i_ev] > 0, pen[i_ev], 1.0)
            ev["contribucion"] = np.nan      # el puntaje es la MEJOR regla, no una suma
        return pd.concat([ev, _co_compradores(b, filas, items, principal, n, nombre)], ignore_index=True)
    if isinstance(alg, Svd) and alg.U is not None:
        V = alg.V

        def aporte(j, i):
            out = np.empty(len(j))
            for k in range(0, len(j), 200_000):
                out[k:k + 200_000] = np.einsum("kn,kn->n", V[:, j[k:k + 200_000]], V[:, i[k:k + 200_000]])
            return out
        ev, principal = _por_item_propio(b, filas, items, n, "FACTOR_LATENTE", nombre, aporte, b.R)
        return pd.concat([ev, _co_compradores(b, filas, items, principal, n, nombre)], ignore_index=True)
    if isinstance(alg, Ease):
        ev, principal = _por_item_propio(b, filas, items, n, "ITEM_EASE", nombre,
                                         lambda j, i: alg.pesos[j, i], b.A)
        return pd.concat([ev, _co_compradores(b, filas, items, principal, n, nombre)], ignore_index=True)
    if isinstance(alg, Secuencia):
        ev, principal = _por_item_propio(b, filas, items, n, "SECUENCIA", nombre,
                                         lambda j, i: alg.C[j, i], b.R)
        if len(ev):
            j = ev["ref"].to_numpy(np.int64)
            i_ev = items[ev["pos"].to_numpy(np.int64)]
            pen = b.soporte / max(b.n, 1)
            ev["lift"] = ev["similitud"].to_numpy(float) / np.where(pen[i_ev] > 0, pen[i_ev], 1.0)
            ev["en_comun"] = np.asarray(alg.antes[j, i_ev]).ravel()          # compraron j y DESPUÉS i
            ev["tamano_grupo"] = np.asarray(alg.elegibles[j, i_ev]).ravel()  # compraron j sin tener i antes
            ev["contribucion"] = np.nan                                       # el puntaje es la mejor
        return pd.concat([ev, _co_compradores(b, filas, items, principal, n, nombre, en_orden=True)],
                         ignore_index=True)
    if isinstance(alg, Tendencia):
        C = _compradores(b)
        Pc = b.P.tocsc()

        def recientes(i):
            compran = C.indices[C.indptr[i]:C.indptr[i + 1]]
            dias = np.asarray(Pc[compran, i].todense()).ravel()
            return compran[dias > alg.corte]
        pos, par, total = _cercanos(items.astype(np.int64), filas, recientes, _tamano_log(b), n)
        return _evidencia_df(pos, "ADOPTANTE_RECIENTE", nombre, par=par,
                             grupo=f"los que no lo compraban hasta hace {int(alg.cfg.dias_tendencia)} días",
                             tamano_grupo=alg.elegibles[items[pos]], compran_grupo=alg.nuevos[items[pos]])
    # popularidad (y svd sin factores, que cae a popularidad): los que lo compran en el segmento
    return _por_compradores(b, filas, items, n, "COMPRADOR_SEGMENTO", nombre)


def _pct1(p: float) -> str:
    """Una participación: con un decimal, o dos si es muy chica (0,06% no es 0,1%)."""
    if p is None or not np.isfinite(p):
        return "sin dato"
    return f"{p:.1%}" if abs(p) >= 0.01 else f"{p:.2%}"


def _formula_mezcla(propio: float, n: float, pares: float, k: float, final: float, que: str) -> str:
    """Cómo se mezcló lo propio con lo del segmento, con los números, para poder rehacerlo."""
    hay_p = np.isfinite(propio) and propio > 0 and n > 0
    hay_s = np.isfinite(pares) and pares > 0 and k > 0
    if hay_p and hay_s:
        return (f"({n:.0f} x {_num(propio)} + {k:g} x {_num(pares)}) / ({n:.0f} + {k:g}) = {_num(final)}: "
                f"lo suyo pesa {n:.0f} ({que}) y lo del segmento {k:g}")
    if hay_p:
        return f"{_num(final)}, el suyo (el segmento no tiene dato)"
    return f"{_num(final)}, el del segmento (no tiene dato propio)"


def _explicar_prob_recompra(sil: float, iv: float, item: int, prob: float, matriz: Matriz,
                            cfg: RecConfig) -> str:
    """De dónde sale la probabilidad de recompra: los casos de la curva, con los conteos que
    quedan en la tabla de la curva de recuperación."""
    if not cfg.usar_probabilidad:
        return "usar_probabilidad apagado: no se descuenta probabilidad"
    h = float(cfg.horizonte_dias) if cfg.horizonte_dias else 90.0
    rejilla, por_item, global_ = matriz.curva_recuperacion(h, cfg.min_casos_recuperacion)
    casos, volvieron = getattr(matriz, "curva_conteos", (None, None))
    atraso_real = sil / iv if iv > 0 else np.nan
    atraso = float(np.clip(np.nan_to_num(atraso_real, nan=rejilla[0]), rejilla[0], rejilla[-1]))
    k = int(np.clip(np.searchsorted(rejilla, atraso, side="left"), 1, len(rejilla) - 1))
    partes = []
    for j in (k - 1, k):
        if np.isfinite(por_item[item, j]):
            partes.append(f"a {rejilla[j]:g} veces, {volvieron[item, j]:.0f} de {casos[item, j]:.0f} casos de "
                          f"este ítem volvieron ({por_item[item, j]:.0%})")
        elif np.isfinite(global_[j]) and casos is not None:
            partes.append(f"a {rejilla[j]:g} veces, el ítem tiene menos de {cfg.min_casos_recuperacion} casos y "
                          f"se usa todo el panel: {volvieron[:, j].sum():.0f} de {casos[:, j].sum():.0f} volvieron "
                          f"({global_[j]:.0%})")
        else:
            partes.append(f"a {rejilla[j]:g} veces no hay casos suficientes")
    texto = (f"Lleva {atraso_real:.1f} veces su intervalo sin comprar. De los que llegaron a ese atraso en el "
             f"pasado, cuántos volvieron a comprar dentro de {h:.0f} días: " + "; ".join(partes)
             + f". Interpolando entre {rejilla[k - 1]:g} y {rejilla[k]:g} veces: {_pct(prob)}")
    if cfg.piso_prob and prob <= float(cfg.piso_prob) + 1e-9:
        texto += f" (con el piso PISO_PROB = {float(cfg.piso_prob):g})"
    return texto


def _filas_calculo(b: Bloque, matriz: Matriz, rec: pd.DataFrame, cfg: RecConfig) -> pd.DataFrame:
    """Una fila por recomendación con el cálculo del USD en juego, número por número.

    Cada número dice de quiénes sale: los compradores del segmento (todos sus valores están
    en la tabla de referencia), los pares comparables (listados en la evidencia), o la
    curva de recuperación (sus conteos están en su tabla).
    """
    f = rec["f"].to_numpy(np.int64)
    i = rec["i"].to_numpy(np.int64)
    tipo = rec["tipo"].to_numpy(dtype=object)
    h = float(cfg.horizonte_dias) if cfg.horizonte_dias else b.dias_ventana
    ventana = max(b.dias_ventana, 1.0)
    escala = _escala_tamano(b, f, cfg)
    # los mismos números que van a la tabla, redondeados igual, para que el texto los repita
    si_compra = np.round(rec["usd_si_compra"].to_numpy(float), 2)
    prob = np.round(rec["prob"].to_numpy(float), 6)
    pot = np.round(rec["usd_potencial"].to_numpy(float), 2)

    def esperados(k) -> str:
        return f"{_num(si_compra[k])} x {prob[k]:.6f} = {_num(pot[k])} esperados."
    esperadas = pd.to_numeric(rec["compras_esperadas"], errors="coerce").to_numpy(float)
    iv = pd.to_numeric(rec["intervalo_esperado"], errors="coerce").to_numpy(float)

    def col(nombre):
        return (pd.to_numeric(rec[nombre], errors="coerce").to_numpy(float) if nombre in rec
                else np.full(len(rec), np.nan))
    pares_ref, ref_valor, ref_q1, ref_q3 = col("pares_ref"), col("ref_valor"), col("ref_q1"), col("ref_q3")
    origen = (getattr(b, "origen_prob", "") if cfg.usar_probabilidad
              else "usar_probabilidad apagado: no se descuenta")
    p = _pares_bloque(b, matriz).set_index(["f", "i"])
    propio = p.reindex(pd.MultiIndex.from_arrays([f, i]))
    dias_p = propio["dias"].to_numpy(float)
    usd_p = propio["usd"].to_numpy(float)
    iv_p = pd.to_numeric(propio["intervalo_medio"], errors="coerce").to_numpy(float)
    n_int = pd.to_numeric(propio["n_intervalos"], errors="coerce").fillna(0).to_numpy(float)

    def tope(crudo, final, techo_hist=np.inf) -> str:
        if final >= np.round(crudo, 2) - 0.005:          # sin tope (a centavos)
            return ""
        if np.isfinite(techo_hist) and abs(final - np.round(techo_hist, 2)) <= 0.005:
            veces = float(cfg.tope_potencial_por_historico)
            return (f" Tope: no más de {veces:g} {'vez' if veces == 1 else 'veces'} lo que él mismo compra de "
                    f"este ítem en {h:.0f} días, {_num(final)}.")
        return (f" Tope: no más de {float(cfg.tope_potencial_relativo):.0%} de su compra total en "
                f"{h:.0f} días, {_num(final)}.")

    def tamano(k) -> str:
        if not cfg.escalar_potencial or b.venta_media <= 0:
            return "Sin ajuste por tamaño"
        razon = b.venta_entidad[f[k]] / b.venta_media
        texto = (f"Su compra en la ventana ({_num(b.venta_entidad[f[k]])}) es {razon:.2f} veces la de una entidad "
                 f"media del segmento ({_num(b.venta_media)})")
        if abs(razon - escala[k]) > 1e-9:
            texto += (f"; el ajuste va de x{1 / cfg.tope_escala:.2f} a x{cfg.tope_escala:g} (TOPE_ESCALA), "
                      f"así que queda en x{escala[k]:.2f}")
        else:
            texto += f": x{escala[k]:.2f}"
        return texto

    textos = np.empty(len(f), dtype=object)
    for k in range(len(f)):
        fk, ik = f[k], i[k]
        ticket_seg = (f"{_num(b.ticket_item[ik])} por compra (USD {_num(b.usd_item[ik])} / {b.dias_item[ik]:,.0f} "
                      f"días de compra de sus {int(b.soporte[ik]):,} compradores en el segmento)")
        ritmo_seg = (f"cada {b.iv_item[ik]:.0f} días (mediana de {int(b.n_ritmo_item[ik]):,} compradores con ritmo)"
                     if np.isfinite(b.iv_item[ik]) else "sin ritmo medible en el segmento")
        if tipo[k] == "CRUZADA":
            base = (f"No lo compra. En su segmento ({b.etiqueta}) lo compran {int(b.soporte[ik]):,} de "
                    f"{b.n:,} entidades ({b.penetracion[ik]:.1%}).")
            if np.isfinite(b.iv_item[ik]) and b.iv_item[ik] > 0:
                crudo = b.ticket_item[ik] * escala[k] * esperadas[k]
                calculo = (f" Ticket del ítem: {ticket_seg}. Lo compran {ritmo_seg}: en {h:.0f} días son "
                           f"{esperadas[k]:.1f} compras, {_num(b.ticket_item[ik] * esperadas[k])}. {tamano(k)}: "
                           f"{_num(np.round(crudo, 2))}.")
            else:
                crudo = b.usd_medio_comprador[ik] * escala[k] * h / ventana
                calculo = (f" Un comprador del segmento gasta en promedio {_num(b.usd_medio_comprador[ik])} en "
                           f"{ventana:.0f} días (USD {_num(b.usd_item[ik])} / {int(b.soporte[ik]):,} compradores); "
                           f"en {h:.0f} días, {_num(b.usd_medio_comprador[ik] * h / ventana)}. {tamano(k)}: "
                           f"{_num(np.round(crudo, 2))}.")
            texto = base + calculo + tope(crudo, si_compra[k])
            texto += (f" Probabilidad de adopción {_pct(prob[k])} ({origen or 'sin backtest: no se descuenta'}): "
                      + esperados(k))
        elif tipo[k] == "REPOSICION":
            ticket_propio = usd_p[k] / dias_p[k] if dias_p[k] > 0 else np.nan
            ticket_pares = b.ticket_item[ik] * escala[k]
            ticket = float(_mezclar(np.array([ticket_propio]), np.array([dias_p[k]]),
                                    np.array([ticket_pares]), cfg.peso_prior_pares)[0])
            crudo = ticket * esperadas[k]
            techo_hist = (usd_p[k] / ventana * h * float(cfg.tope_potencial_por_historico)
                          if cfg.tope_potencial_por_historico and usd_p[k] > 0 else np.inf)
            sil = float(rec["dias_sin_comprar"].iloc[k])
            ritmo = f", cada {iv_p[k]:.0f} días en promedio" if np.isfinite(iv_p[k]) else ""
            texto = (f"Lo compró {dias_p[k]:.0f} veces{ritmo}; lleva {sil:.0f} días sin comprar. En su segmento "
                     f"lo compran {ritmo_seg}. Intervalo estimado: "
                     + _formula_mezcla(iv_p[k], n_int[k], b.iv_item[ik], cfg.peso_prior_pares, iv[k],
                                       "sus intervalos")
                     + f". Ticket: el suyo {_num(ticket_propio)} (USD {_num(usd_p[k])} / {dias_p[k]:.0f} compras); el del "
                       f"ítem en el segmento {ticket_seg}, ajustado a su tamaño x{escala[k]:.2f} = {_num(ticket_pares)}. "
                       f"Ticket estimado: "
                     + _formula_mezcla(ticket_propio, dias_p[k], ticket_pares, cfg.peso_prior_pares, ticket,
                                       "sus compras")
                     + f". En {h:.0f} días: {esperadas[k]:.1f} compras, {_num(np.round(crudo, 2))}."
                     + tope(crudo, si_compra[k], techo_hist) + " "
                     + _explicar_prob_recompra(sil, iv[k], ik, prob[k], matriz, cfg)
                     + ": " + esperados(k))
        else:
            ve = b.venta_entidad[fk]
            sh = usd_p[k] / ve if ve > 0 else 0.0
            me = ref_valor[k]
            crudo = (me - sh) * ve * h / ventana
            nombre = _NOMBRE_ESTADISTICO[cfg.estadistico_pares]
            texto = (f"Lo compra, pero le dedica {_pct1(sh)} de su compra ({_num(usd_p[k])} de {_num(ve)}). Sus "
                     f"{pares_ref[k]:.0f} pares de tamaño parecido que lo compran (todos en la tabla de pares comparables) le "
                     f"dedican {_pct1(me)} ({nombre}; la mitad de ellos entre {_pct1(ref_q1[k])} y {_pct1(ref_q3[k])}). "
                     f"Para llegar a eso en {h:.0f} días le faltan ({_pct1(me)} - {_pct1(sh)}) x {_num(ve)} x "
                     f"{h:.0f}/{ventana:.0f} = {_num(np.round(crudo, 2))}." + tope(crudo, si_compra[k])
                     + " Ya lo compra, probabilidad 1: " + esperados(k))
        textos[k] = texto
    return _evidencia_df(np.arange(len(f)), "CALCULO", "",
                         grupo=np.full(len(f), b.etiqueta, dtype=object),
                         tamano_grupo=np.full(len(f), float(b.n)), compran_grupo=b.soporte[i],
                         detalle=textos)


def _detalle_pares(ev: pd.DataFrame, rec_nombres: np.ndarray, ref_nombres: np.ndarray, unidad: str) -> np.ndarray:
    """El texto de cada fila de evidencia que no es CALCULO."""
    salida = np.empty(len(ev), dtype=object)
    for k, r in enumerate(ev.itertuples(index=False)):
        rec = rec_nombres[k]
        ref = ref_nombres[k] if ref_nombres[k] is not None and not (isinstance(ref_nombres[k], float)
                                                                  and np.isnan(ref_nombres[k])) else ""
        compra = (f"compra {rec}: {_num(r.usd_par_item)} en {r.dias_par_item:.0f} compras"
                  if np.isfinite(r.usd_par_item) else f"compra {rec}")
        desde = (f", la primera el {pd.Timestamp(r.fecha_item_par).date()}"
                 if r.fecha_item_par is not None and pd.notna(r.fecha_item_par) else "")
        e = r.evidencia
        if e == "COMPRADOR_SEGMENTO":
            t = (f"Par del segmento de tamaño parecido (compra total {_num(r.usd_par_total)}); {compra}. "
                 f"Lo compran {r.compran_grupo:.0f} de {r.tamano_grupo:.0f} entidades del segmento.")
        elif e == "VECINO":
            t = (f"Uno de sus pares más parecidos por lo que compran (coseno {r.similitud:.2f}); {compra}. "
                 f"De sus {r.tamano_grupo:.0f} pares más parecidos, {r.compran_grupo:.0f} lo compran (se listan los "
                 f"de más peso; los {r.tamano_grupo:.0f} están en la tabla de vecindario). Éste aporta "
                 f"{r.contribucion:.0%} del puntaje.")
        elif e == "MIEMBRO_GRUPO":
            t = (f"Está en su mismo grupo de gasto ({r.grupo}, {r.tamano_grupo:.0f} entidades, "
                 f"{r.compran_grupo:.0f} lo compran; el grupo de cada entidad está en la tabla de vecindario); "
                 f"{compra}.")
        elif e == "ITEM_AFIN":
            t = (f"Ya compra {ref}. De las {r.tamano_grupo:.0f} {unidad} con {ref}, {r.en_comun:.0f} también "
                 f"tienen {rec} (coseno {r.similitud:.2f}). Aporta {r.contribucion:.0%} del puntaje.")
        elif e == "REGLA":
            t = (f"Regla {ref} -> {rec}: de las {r.tamano_grupo:.0f} {unidad} con {ref}, {r.en_comun:.0f} "
                 f"tienen {rec} (confianza {r.similitud:.0%}, lift {r.lift:.2f}).")
        elif e == "FACTOR_LATENTE":
            t = (f"Ya compra {ref}; en el patrón de consumo del segmento (SVD) {ref} y {rec} van juntos: "
                 f"{r.en_comun:.0f} de las {r.tamano_grupo:.0f} entidades con {ref} compran los dos. "
                 f"Aporta {r.contribucion:.0%} de lo que empuja a favor.")
        elif e == "ITEM_EASE":
            t = (f"Ya compra {ref}; en el modelo EASE, {ref} empuja hacia {rec} con peso {r.similitud:.3f}, "
                 f"descontando lo que explican los demás ítems ({r.en_comun:.0f} de las {r.tamano_grupo:.0f} "
                 f"{unidad} con {ref} tienen {rec}). Aporta {r.contribucion:.0%} de lo que empuja a favor.")
        elif e == "SECUENCIA":
            t = (f"Secuencia {ref} -> {rec}: de las {r.tamano_grupo:.0f} entidades que compraron {ref} sin tener "
                 f"{rec} de antes, {r.en_comun:.0f} compraron {rec} DESPUÉS ({r.similitud:.0%}, lift {r.lift:.2f}).")
        elif e == "CO_COMPRADOR":
            t = (f"Compra {ref}, como esta entidad, y además {compra}. {r.compran_grupo:.0f} entidades del "
                 f"segmento compran los dos.")
        elif e == "ADOPTANTE":
            ref_desde = (f" el {pd.Timestamp(r.fecha_ref_par).date()}"
                         if r.fecha_ref_par is not None and pd.notna(r.fecha_ref_par) else "")
            t = (f"Compró {ref}{ref_desde} y después {rec}{desde.replace(', la primera', '')}: {compra}. "
                 f"{r.compran_grupo:.0f} entidades del segmento hicieron esa secuencia.")
        elif e == "ADOPTANTE_RECIENTE":
            t = (f"Empezó a comprar {rec} hace poco{desde}: {compra}. De las {r.tamano_grupo:.0f} entidades del "
                 f"segmento que no lo compraban, {r.compran_grupo:.0f} empezaron en el mismo tramo.")
        elif e == "PAR_RITMO":
            t = (f"Ejemplo, no el total: uno de los {r.compran_grupo:.0f} compradores del segmento con ritmo, el de "
                 f"tamaño más parecido. Compra {rec} cada {r.intervalo_par:.0f} días, {_num(r.ticket_par)} por compra "
                 f"({r.dias_par_item:.0f} compras). El ritmo y el ticket del segmento salen de todos ellos (tabla de "
                 f"referencia).")
        elif e == "PAR_PARTICIPACION":
            t = (f"Uno de los {r.compran_grupo:.0f} pares comparables, de los de tamaño más parecido (la lista completa, "
                 f"con la que se calcula la mediana, está en la tabla de pares comparables): le dedica "
                 f"{_pct1(r.participacion_par)} de su compra a {rec} ({_num(r.usd_par_item)} de {_num(r.usd_par_total)}).")
        else:
            t = ""
        salida[k] = t
    return salida


def completar_detalle(ev: pd.DataFrame, cfg: RecConfig) -> pd.DataFrame:
    """Escribe BD_DETALLE en las filas de evidencia que no lo tienen (todas menos CALCULO).

    El texto de cada par sale entero de sus columnas, así que no se guarda en memoria para
    millones de filas: se arma cuando se escribe, de a lotes, o para las filas que se miran.
    """
    if ev.empty or "BD_DETALLE" not in ev:
        return ev
    falta = ev["BD_DETALLE"].isna().to_numpy()
    if not falta.any():
        return ev
    sub = ev.loc[falta]
    nombre = (cfg.desc_item() or cfg.claves_item())[0].upper()
    campos = pd.DataFrame({
        "evidencia": sub["BD_EVIDENCIA"].astype(object).to_numpy(),
        "usd_par_item": sub["MT_USD_PAR_ITEM"].to_numpy(float), "dias_par_item": sub["MT_DIAS_PAR_ITEM"].to_numpy(float),
        "fecha_item_par": sub["FECHA_PRIMERA_ITEM_PAR"].to_numpy(), "fecha_ref_par": sub["FECHA_PRIMERA_REF_PAR"].to_numpy(),
        "usd_par_total": sub["MT_USD_PAR_TOTAL"].to_numpy(float), "compran_grupo": sub["MT_COMPRAN_EN_GRUPO"].to_numpy(float),
        "tamano_grupo": sub["MT_TAMANO_GRUPO"].to_numpy(float), "similitud": sub["MT_SIMILITUD"].to_numpy(float),
        "contribucion": sub["MT_CONTRIBUCION"].to_numpy(float), "lift": sub["MT_LIFT"].to_numpy(float),
        "en_comun": sub["MT_EN_COMUN"].to_numpy(float), "grupo": sub["BD_GRUPO"].astype(object).to_numpy(),
        "intervalo_par": sub["MT_INTERVALO_PAR"].to_numpy(float), "ticket_par": sub["MT_TICKET_PAR"].to_numpy(float),
        "participacion_par": sub["MT_PARTICIPACION_PAR"].to_numpy(float)})
    unidad = "canastas" if cfg.afinidad == "canasta" else "entidades"
    textos = _detalle_pares(campos, sub[nombre].astype(object).to_numpy(),
                            sub["REF_" + nombre].astype(object).to_numpy(), unidad)
    ev = ev.copy()
    detalle = np.array(ev["BD_DETALLE"].astype(object).to_numpy(), dtype=object)   # copia escribible
    detalle[falta] = textos
    ev["BD_DETALLE"] = detalle
    return ev


def evidencias_bloque(b: Bloque, matriz: Matriz, alg: Algoritmo, rec: pd.DataFrame,
                      cfg: RecConfig) -> pd.DataFrame:
    """La evidencia de todas las recomendaciones de un segmento.

    Para cada recomendación, una fila CALCULO con el USD en juego desarmado, y después la
    evidencia de su tipo: CRUZADA, con quién comparó el algoritmo que la eligió; REPOSICION,
    los pares que compran el ítem con ritmo; BRECHA, los pares y lo que le dedican. No
    cambia ninguna recomendación: las lee y las explica.
    """
    if rec.empty:
        return pd.DataFrame()
    if cfg.evidencia_hasta_ranking:
        rec = rec[rec["ranking"] <= int(cfg.evidencia_hasta_ranking)]
        if rec.empty:
            return pd.DataFrame()
    rec = rec.reset_index(drop=True)
    n = int(cfg.max_evidencias)
    f = rec["f"].to_numpy(np.int64)
    i = rec["i"].to_numpy(np.int64)
    tipo = rec["tipo"].to_numpy(dtype=object)
    partes = [_filas_calculo(b, matriz, rec, cfg)]

    cru = np.flatnonzero(tipo == "CRUZADA")
    if len(cru):
        ev = _evidencia_algoritmo(alg, b, f[cru], i[cru], n)
        ev["pos"] = cru[ev["pos"].to_numpy(np.int64)]
        partes.append(ev)

    p = _pares_bloque(b, matriz)
    rep = np.flatnonzero(tipo == "REPOSICION")
    if len(rep):
        con_ritmo = p[pd.to_numeric(p["intervalo_medio"], errors="coerce").notna()]
        por_item = {k: g.to_numpy(np.int64) for k, g in con_ritmo.groupby("i")["f"]}
        pos, par, total = _cercanos(i[rep], f[rep], lambda k: por_item.get(int(k), ()), _tamano_log(b), n)
        partes.append(_evidencia_df(rep[pos], "PAR_RITMO", "regla_reposicion", par=par,
                                    grupo=b.etiqueta, tamano_grupo=b.soporte[i[rep][pos]],
                                    compran_grupo=total[pos].astype(float)))
    bre = np.flatnonzero(tipo == "BRECHA")
    b.comparables = None
    if len(bre):
        # TODOS los pares con los que se comparó van a su propia tabla (angosta: son 30 por
        # brecha); acá quedan los más parecidos como ejemplo
        pos, par = pares_comparables(b, matriz, cfg, f[bre], i[bre])
        cuantos = np.bincount(pos, minlength=len(bre)).astype(float)
        rango = np.arange(len(pos)) - np.searchsorted(pos, pos)      # vienen del más parecido al menos
        it = i[bre][pos]
        usd_par = np.asarray(b.V[par, it]).ravel()
        total_par = b.venta_entidad[par]
        b.comparables = pd.DataFrame({
            "entidad": b.filas[f[bre][pos]], "i": it, "par_entidad": b.filas[par], "orden": rango + 1,
            "usd_par": usd_par, "total_par": total_par,
            "share_par": usd_par / np.where(total_par > 0, total_par, 1.0),
            "ranking": rec["ranking"].to_numpy()[bre][pos]})
        ej = rango < n
        partes.append(_evidencia_df(bre[pos][ej], "PAR_PARTICIPACION", "regla_brecha", par=par[ej],
                                    grupo=f"sus {int(cfg.pares_comparables)} pares comparables",
                                    tamano_grupo=b.soporte[it[ej]], compran_grupo=cuantos[pos][ej]))

    ev = pd.concat([x for x in partes if len(x)], ignore_index=True)
    pos = ev["pos"].to_numpy(np.int64)
    item_rec = i[pos]
    # lo que hizo cada par con el ítem recomendado, para poder verificarlo contra la fuente
    par = ev["par"].to_numpy(float)
    con_par = np.isfinite(par)
    pl = np.where(con_par, par, 0).astype(np.int64)
    usd = np.full(len(ev), np.nan)
    dias = np.full(len(ev), np.nan)
    share = np.full(len(ev), np.nan)
    total_par = np.full(len(ev), np.nan)
    intervalo = np.full(len(ev), np.nan)
    if con_par.any():
        usd[con_par] = np.asarray(b.V[pl[con_par], item_rec[con_par]]).ravel()
        dias[con_par] = np.asarray(b.D[pl[con_par], item_rec[con_par]]).ravel()
        share[con_par] = np.asarray(b.share[pl[con_par], item_rec[con_par]]).ravel()
        total_par[con_par] = b.venta_entidad[pl[con_par]]
        iv_par = pd.to_numeric(p.set_index(["f", "i"])["intervalo_medio"], errors="coerce")
        intervalo[con_par] = iv_par.reindex(pd.MultiIndex.from_arrays(
            [pl[con_par], item_rec[con_par]])).to_numpy(float)
        sin_compra = dias == 0              # el par no compró el ítem (no debería pasar)
        usd[sin_compra] = dias[sin_compra] = share[sin_compra] = np.nan
    ev["usd_par_item"] = usd
    ev["dias_par_item"] = dias
    ev["ticket_par"] = np.where(dias > 0, usd / np.where(dias > 0, dias, 1.0), np.nan)
    ev["intervalo_par"] = intervalo
    ev["participacion_par"] = share
    ev["usd_par_total"] = total_par
    # cuándo compró el par por primera vez (en la ventana) el ítem recomendado y el de referencia
    fecha_item = np.full(len(ev), np.nan)
    fecha_ref = np.full(len(ev), np.nan)
    if con_par.any():
        fecha_item[con_par] = np.asarray(b.P[pl[con_par], item_rec[con_par]]).ravel()
        ref = ev["ref"].to_numpy(float)
        con_ref = con_par & np.isfinite(ref)
        if con_ref.any():
            fecha_ref[con_ref] = np.asarray(b.P[pl[con_ref], ref[con_ref].astype(np.int64)]).ravel()

    def a_fecha(d):
        d = np.where(d > 0, d, np.nan)
        return pd.to_datetime(d, unit="D", origin=_EPOCA).to_numpy(dtype=object)
    ev["fecha_item_par"] = a_fecha(fecha_item)
    ev["fecha_ref_par"] = a_fecha(fecha_ref)
    # en la fila CALCULO, las mismas columnas llevan la referencia del segmento
    calc = (ev["evidencia"] == "CALCULO").to_numpy()
    ic = item_rec[calc]
    ev.loc[calc, "usd_par_item"] = b.usd_medio_comprador[ic]
    ev.loc[calc, "ticket_par"] = b.ticket_item[ic]
    ev.loc[calc, "intervalo_par"] = b.iv_item[ic]
    ev.loc[calc, "participacion_par"] = b.share_medio_comprador[ic]
    ev.loc[calc, "fuente"] = rec["algoritmo"].to_numpy(dtype=object)[pos[calc]]
    # el texto de cada par se arma después, desde sus columnas (completar_detalle): son millones
    # una sola numeración por recomendación: CALCULO primero, después en el orden de EVIDENCIAS
    clase = pd.Categorical(ev["evidencia"], categories=list(EVIDENCIAS), ordered=True).codes
    ev["_clase"] = clase
    ev["_orden_original"] = np.arange(len(ev))
    # dentro de cada clase, la de más peso primero (afinidad, coseno, confianza); los pares sin
    # peso ya vienen ordenados por cercanía de tamaño
    ev["_peso"] = -pd.to_numeric(ev["similitud"], errors="coerce").fillna(0.0)
    # en un empate de peso, primero el ítem de índice mayor: es el que nombra el motivo
    ev["_desempate"] = -pd.to_numeric(ev["ref"], errors="coerce").fillna(0.0)
    ev = ev.sort_values(["pos", "_clase", "_peso", "_desempate", "_orden_original"]).reset_index(drop=True)
    ev["orden"] = ev.groupby("pos").cumcount()
    pos = ev["pos"].to_numpy(np.int64)
    ev["entidad"] = b.filas[f[pos]]
    ev["i"] = i[pos]
    ev["ranking"] = rec["ranking"].to_numpy()[pos]
    ev["tipo"] = tipo[pos]
    ev["algoritmo"] = rec["algoritmo"].to_numpy(dtype=object)[pos]
    par = ev["par"].to_numpy(float)
    ev["par_entidad"] = np.where(np.isfinite(par), b.filas[np.where(np.isfinite(par), par, 0).astype(np.int64)], -1)
    ev["ref_item"] = np.where(np.isfinite(ev["ref"].to_numpy(float)), ev["ref"].to_numpy(float), -1).astype(np.int64)
    return ev.drop(columns=["_clase", "_orden_original", "_peso", "_desempate", "pos", "par", "ref"])


def referencia_bloque(b: Bloque, cfg: RecConfig) -> pd.DataFrame:
    """Los valores de cada ítem en el segmento que usan los cálculos, con sus componentes
    (sumas y conteos), para poder rehacerlos desde la matriz de compras."""
    i = np.flatnonzero(b.soporte > 0)
    if not len(i):
        return pd.DataFrame()
    return pd.DataFrame({
        "segmento": b.etiqueta, "nivel_segmento": b.nivel, "i": i,
        "entidades": float(b.n), "compradores": b.soporte[i], "penetracion": b.penetracion[i],
        "candidato": b.candidato[i], "usd_item": b.usd_item[i], "dias_item": b.dias_item[i],
        "ticket": b.ticket_item[i], "usd_medio_comprador": b.usd_medio_comprador[i],
        "con_ritmo": b.n_ritmo_item[i], "intervalo_mediana": b.iv_item[i], "cv_mediana": b.cv_item[i],
        "participacion_media": b.share_medio_comprador[i], "margen_pct": b.margen_pct_item[i],
        "venta_media_entidad": b.venta_media, "prob_adopcion": b.p_adopcion,
        "origen_prob": getattr(b, "origen_prob", "") or "", "dias_ventana": b.dias_ventana})


def vecindario_bloque(b: Bloque, alg: Algoritmo, filas: np.ndarray) -> pd.DataFrame:
    """Con quién se agrupó a cada entidad: sus k vecinos (coseno_entidad) y su grupo de
    k-means (kmeans_valor). Con esto se rehacen los "N de sus pares más parecidos" y los
    "en su grupo lo compran X%" de los motivos."""
    algos = alg.algos if isinstance(alg, Fusion) else [alg]
    partes = []
    for a in algos:
        if isinstance(a, CosenoEntidad) and len(filas):
            for ini in range(0, len(filas), 20_000):
                f = filas[ini:ini + 20_000]
                W = a._vecinos(f).tocsr()
                W.sort_indices()
                r = np.repeat(np.arange(len(f)), np.diff(W.indptr))
                orden = np.lexsort((-W.data, r))
                r, j, s = r[orden], W.indices[orden], W.data[orden]
                rango = np.arange(len(r)) - np.searchsorted(r, r)
                partes.append(pd.DataFrame({"f": f[r], "algoritmo": a.nombre, "par": j, "similitud": s,
                                            "orden": rango + 1, "grupo": f"sus {int(a.cfg.k_vecinos)} pares más parecidos",
                                            "tamano": np.bincount(r, minlength=len(f))[r].astype(float)}))
        elif isinstance(a, KmeansValor) and a.pen.shape[0] > 1:
            todas = np.arange(b.n)
            partes.append(pd.DataFrame({"f": todas, "algoritmo": a.nombre, "par": -1, "similitud": np.nan,
                                        "orden": 0, "grupo": [f"k-means #{g}" for g in a.etiquetas],
                                        "tamano": a.tam[a.etiquetas]}))
    if not partes:
        return pd.DataFrame()
    out = pd.concat(partes, ignore_index=True)
    out["segmento"] = b.etiqueta
    out["entidad"] = b.filas[out["f"].to_numpy(np.int64)]
    par = out["par"].to_numpy(np.int64)
    out["par_entidad"] = np.where(par >= 0, b.filas[np.maximum(par, 0)], -1)
    return out


def tabla_curva(matriz: Matriz, cfg: RecConfig) -> pd.DataFrame:
    """La curva de recuperación con sus conteos: por ítem y para todo el panel."""
    h = float(cfg.horizonte_dias) if cfg.horizonte_dias else 90.0
    rejilla, por_item, global_ = matriz.curva_recuperacion(h, cfg.min_casos_recuperacion)
    if not hasattr(matriz, "curva_conteos"):
        return pd.DataFrame()
    casos, volvieron = matriz.curva_conteos
    filas = []
    con_casos = np.flatnonzero(casos.sum(1) > 0)
    for k, a in enumerate(rejilla):
        filas.append(pd.DataFrame({"i": con_casos, "atraso": a, "casos": casos[con_casos, k],
                                   "volvieron": volvieron[con_casos, k], "prob": por_item[con_casos, k],
                                   "alcance": "ITEM"}))
        filas.append(pd.DataFrame({"i": [-1], "atraso": [a], "casos": [casos[:, k].sum()],
                                   "volvieron": [volvieron[:, k].sum()], "prob": [global_[k]],
                                   "alcance": ["PANEL"]}))
    out = pd.concat(filas, ignore_index=True)
    out["horizonte"] = h
    out["min_casos"] = float(cfg.min_casos_recuperacion)
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


def catalogo_evidencia(cfg: RecConfig) -> List[Tuple[str, str, str]]:
    """(columna, tipo Oracle, descripción) de la tabla de evidencia, en orden."""
    cols: List[Tuple[str, str, str]] = []
    for c in cfg.entidad:
        tipo = "VARCHAR2(200)" if str(c).upper().startswith("BD_") else "NUMBER"
        cols.append((c.upper(), tipo, f"Entidad a la que se recomienda ({c}). Junto con el ítem, une con la "
                                      f"tabla de recomendaciones."))
    for c in cfg.item:
        tipo = "VARCHAR2(200)" if str(c).upper().startswith("BD_") else "NUMBER"
        cols.append((c.upper(), tipo, f"Ítem recomendado ({c})."))
    cols += [
        ("MT_RANKING", "NUMBER", "Ranking de la recomendación que se explica (el mismo de la tabla de recomendaciones)."),
        ("BD_TIPO", "VARCHAR2(20)", "Tipo de la recomendación que se explica: CRUZADA, REPOSICION o BRECHA."),
        ("BD_ALGORITMO", "VARCHAR2(40)", "Algoritmo o regla que produjo la recomendación."),
        ("MT_ORDEN", "NUMBER", "Orden de la fila dentro de la recomendación. 0 es la fila CALCULO; después, la "
                               "evidencia de mayor peso primero."),
        ("BD_EVIDENCIA", "VARCHAR2(30)",
         "Qué es esta fila. CALCULO: el USD en juego desarmado número por número. COMPRADOR_SEGMENTO: un par del "
         "segmento que compra el ítem (popularidad). VECINO: uno de los pares más parecidos por lo que compran, "
         "que compra el ítem (coseno_entidad). MIEMBRO_GRUPO: una entidad del mismo grupo de k-means que compra "
         "el ítem (kmeans_valor). ITEM_AFIN: un ítem que la entidad ya compra y se compra con el recomendado "
         "(coseno_item). REGLA: una regla de asociación que le aplica (reglas). FACTOR_LATENTE: un ítem que ya "
         "compra y va con el recomendado en el patrón de consumo (svd). CO_COMPRADOR: una entidad que compra a "
         "la vez el ítem de referencia y el recomendado. PAR_RITMO: un par que compra el ítem con ritmo "
         "(reposición): es un ejemplo, el ritmo del segmento sale de todos (tabla de referencia). "
         "PAR_PARTICIPACION: uno de los pares comparables de una BRECHA; se listan TODOS los comparados. "
         "ITEM_EASE: un ítem que ya compra y su peso en el modelo EASE (ease). SECUENCIA: una regla 'compró A y "
         "después B' que le aplica (secuencia). ADOPTANTE: una entidad que compró la referencia y DESPUÉS el "
         "recomendado. ADOPTANTE_RECIENTE: una entidad que empezó a comprar el ítem en los últimos "
         "dias_tendencia días (tendencia)."),
        ("BD_ALGORITMO_EVIDENCIA", "VARCHAR2(40)", "Algoritmo que aporta esta evidencia. Igual a BD_ALGORITMO, "
                                                   "salvo en una fusión (rrf / ponderado), donde cada algoritmo "
                                                   "de la batería deja la suya."),
    ]
    for c in cfg.entidad:
        tipo = "VARCHAR2(200)" if str(c).upper().startswith("BD_") else "NUMBER"
        cols.append((f"PAR_{c.upper()}", tipo, f"El par con el que se comparó ({c}). Vacío en las filas que "
                                               f"no hablan de un par (CALCULO e ítems)."))
    for c in cfg.item:
        tipo = "VARCHAR2(200)" if str(c).upper().startswith("BD_") else "NUMBER"
        cols.append((f"REF_{c.upper()}", tipo, f"Ítem de referencia ({c}): el que la entidad ya compra y empuja "
                                               f"la recomendación (ITEM_AFIN, REGLA, FACTOR_LATENTE) o el que el "
                                               f"CO_COMPRADOR compra junto con el recomendado."))
    cols += [
        ("MT_SIMILITUD", "NUMBER", "VECINO: coseno entre la entidad y el par. ITEM_AFIN: coseno entre los dos "
                                   "ítems. REGLA: confianza, P(compra el recomendado | compra la referencia). "
                                   "FACTOR_LATENTE: aporte del ítem de referencia al puntaje SVD."),
        ("MT_CONTRIBUCION", "NUMBER", "Fracción del puntaje que explica esta fila. VECINO e ITEM_AFIN: el "
                                      "puntaje es la suma de esos aportes. FACTOR_LATENTE: fracción de lo que "
                                      "empuja a favor (en svd hay ítems que restan y no se listan)."),
        ("MT_LIFT", "NUMBER", "REGLA y SECUENCIA: cuántas veces más probable es comprar el recomendado "
                              "teniendo la referencia que en el segmento en general."),
        ("MT_EN_COMUN", "NUMBER", "ITEM_AFIN, REGLA, FACTOR_LATENTE, ITEM_EASE: entidades (o canastas, si la "
                                  "afinidad es por canasta) que tienen la referencia y el recomendado. SECUENCIA: "
                                  "entidades que compraron la referencia y DESPUÉS el recomendado."),
        ("MT_USD_PAR_ITEM", "NUMBER", "USD que el par compró del ítem recomendado en la ventana. En CALCULO: lo "
                                      "que gasta en la ventana un comprador medio del segmento."),
        ("MT_DIAS_PAR_ITEM", "NUMBER", "Días en que el par compró el ítem recomendado en la ventana."),
        ("MT_TICKET_PAR", "NUMBER", "USD por compra del par en el ítem. En CALCULO: el ticket del ítem en el "
                                    "segmento, el que se usa para el USD en juego."),
        ("MT_INTERVALO_PAR", "NUMBER", "Días promedio entre compras del par en el ítem. En CALCULO: la mediana "
                                       "del segmento, la que se le presta a quien tiene poca historia."),
        ("MT_PARTICIPACION_PAR", "NUMBER", "Fracción de su compra que el par le dedica al ítem. En CALCULO: la "
                                           "media de los compradores del segmento (la base de BRECHA)."),
        ("MT_USD_PAR_TOTAL", "NUMBER", "Compra total del par en la ventana: los pares se eligen de tamaño parecido."),
        ("FECHA_PRIMERA_ITEM_PAR", "DATE", "Primera compra del ítem recomendado por el par dentro de la ventana."),
        ("FECHA_PRIMERA_REF_PAR", "DATE", "Primera compra del ítem de referencia por el par dentro de la ventana "
                                          "(en ADOPTANTE: es anterior a la del recomendado)."),
        ("BD_GRUPO", "VARCHAR2(400)", "Contra qué grupo se comparó: el segmento, el grupo de k-means o los k "
                                      "pares más parecidos."),
        ("MT_TAMANO_GRUPO", "NUMBER", "Entidades de ese grupo (en ITEM_AFIN, REGLA y FACTOR_LATENTE: las que "
                                      "tienen el ítem de referencia; en PAR_RITMO y PAR_PARTICIPACION: las que "
                                      "compran el ítem)."),
        ("MT_COMPRAN_EN_GRUPO", "NUMBER", "De ese grupo, cuántas compran el ítem recomendado (en CO_COMPRADOR: "
                                          "cuántas compran los dos; en PAR_RITMO, cuántas con ritmo medible). "
                                          "Es el total, aunque se listen sólo algunas."),
        ("BD_DETALLE", "VARCHAR2(4000)", "La fila en palabras, con los números para verificarla. En CALCULO "
                                         "lleva cada fórmula con sus valores, por eso es largo."),
        ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo."),
    ]
    return cols


def _cols_categoria(cfg: RecConfig, columnas: Sequence[str], prefijo: str, que: str) -> List[Tuple[str, str, str]]:
    return [(f"{prefijo}{c.upper()}", "VARCHAR2(200)" if str(c).upper().startswith("BD_") else "NUMBER",
             f"{que} ({c}).") for c in columnas]


def catalogo_referencia(cfg: RecConfig) -> List[Tuple[str, str, str]]:
    """Tabla de referencia: un ítem en un segmento, con todo lo que los cálculos usan de él."""
    return ([("BD_SEGMENTO", "VARCHAR2(400)", "Segmento."),
             ("BD_NIVEL_SEGMENTO", "VARCHAR2(100)", "Nivel de segmentación usado, o GLOBAL.")]
            + _cols_categoria(cfg, cfg.item, "", "Ítem")
            + [("MT_ENTIDADES_SEGMENTO", "NUMBER", "Entidades del segmento con compras en la ventana."),
               ("MT_COMPRADORES", "NUMBER", "Entidades del segmento que compran el ítem (soporte)."),
               ("MT_PENETRACION", "NUMBER", "MT_COMPRADORES / MT_ENTIDADES_SEGMENTO."),
               ("BD_CANDIDATO", "VARCHAR2(5)", "SI si pasa MIN_SOPORTE y MIN_PENETRACION: sólo entonces se recomienda."),
               ("MT_USD_ITEM", "NUMBER", "USD del ítem sumando a todos sus compradores del segmento."),
               ("MT_DIAS_COMPRA_ITEM", "NUMBER", "Días de compra del ítem sumando a todos sus compradores."),
               ("MT_TICKET", "NUMBER", "MT_USD_ITEM / MT_DIAS_COMPRA_ITEM: el ticket del ítem en el segmento, el "
                                       "que usan CRUZADA y REPOSICION."),
               ("MT_USD_MEDIO_COMPRADOR", "NUMBER", "MT_USD_ITEM / MT_COMPRADORES."),
               ("MT_COMPRADORES_CON_RITMO", "NUMBER", "Compradores con al menos 2 días de compra (con intervalo)."),
               ("MT_INTERVALO_MEDIANA", "NUMBER", "Mediana del intervalo medio de esos compradores: el ritmo del "
                                                  "segmento que se le presta a quien tiene poca historia."),
               ("MT_CV_MEDIANA", "NUMBER", "Mediana de la irregularidad (desvío / promedio de los intervalos)."),
               ("MT_PARTICIPACION_MEDIA", "NUMBER", "Promedio simple de la participación del ítem en la compra de sus "
                                                    "compradores. Es informativa: la BRECHA ya no la usa, compara "
                                                    "contra sus pares comparables."),
               ("MT_MARGEN_PCT", "NUMBER", "Margen / venta del ítem en el segmento (0 si la venta no llega a "
                                           "MIN_BASE_PORCENTAJE)."),
               ("MT_VENTA_MEDIA_ENTIDAD", "NUMBER", "Compra media de una entidad del segmento: la base del ajuste por "
                                                    "tamaño."),
               ("MT_PROB_ADOPCION", "NUMBER", "Probabilidad de adopción de una CRUZADA en el segmento."),
               ("BD_ORIGEN_PROB", "VARCHAR2(400)", "De dónde sale esa probabilidad (backtest del segmento o del panel)."),
               ("MT_DIAS_VENTANA", "NUMBER", "Días de la ventana de afinidad."),
               ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo.")])


def catalogo_matriz(cfg: RecConfig) -> List[Tuple[str, str, str]]:
    """La matriz de compras: una fila por entidad e ítem comprado en la ventana."""
    return (_cols_categoria(cfg, cfg.entidad, "", "Entidad") + _cols_categoria(cfg, cfg.item, "", "Ítem")
            + [("BD_SEGMENTO", "VARCHAR2(400)", "Segmento de la entidad."),
               ("MT_USD", "NUMBER", "USD netos del ítem en la ventana (ventas menos devoluciones)."),
               ("MT_MARGEN", "NUMBER", "Margen USD del ítem en la ventana."),
               ("MT_DIAS_COMPRA", "NUMBER", "Días distintos en que compró el ítem."),
               ("FECHA_PRIMERA", "DATE", "Primera compra del ítem dentro de la ventana."),
               ("FECHA_ULTIMA", "DATE", "Última compra del ítem."),
               ("MT_INTERVALO_MEDIO", "NUMBER", "Días promedio entre compras (vacío con un solo día de compra)."),
               ("MT_INTERVALO_DESVIO", "NUMBER", "Desvío de esos intervalos."),
               ("MT_USD_ENTIDAD", "NUMBER", "Compra total de la entidad en la ventana."),
               ("MT_PARTICIPACION", "NUMBER", "MT_USD / MT_USD_ENTIDAD: lo que le dedica al ítem."),
               ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo.")])


def catalogo_curva(cfg: RecConfig) -> List[Tuple[str, str, str]]:
    """La curva de recuperación con sus conteos."""
    return (_cols_categoria(cfg, cfg.item, "", "Ítem (vacío en las filas del panel entero)")
            + [("BD_ALCANCE", "VARCHAR2(10)", "ITEM: casos de ese ítem. PANEL: todos los ítems juntos (se usa cuando "
                                              "el ítem no llega a MT_MIN_CASOS)."),
               ("MT_ATRASO", "NUMBER", "Veces su intervalo que llevaba sin comprar."),
               ("MT_CASOS", "NUMBER", "Casos que llegaron a ese atraso (huecos pasados y silencios ya observados)."),
               ("MT_VOLVIERON", "NUMBER", "De esos, cuántos volvieron a comprar dentro del horizonte."),
               ("MT_PROB", "NUMBER", "MT_VOLVIERON / MT_CASOS, si hay casos suficientes."),
               ("MT_HORIZONTE_DIAS", "NUMBER", "El horizonte de la recompra."),
               ("MT_MIN_CASOS", "NUMBER", "Casos mínimos para creerle a la curva del ítem."),
               ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo.")])


def catalogo_pares_comparables(cfg: RecConfig) -> List[Tuple[str, str, str]]:
    """Todos los pares con los que se comparó cada BRECHA."""
    return (_cols_categoria(cfg, cfg.entidad, "", "Entidad de la BRECHA")
            + _cols_categoria(cfg, cfg.item, "", "Ítem de la BRECHA")
            + _cols_categoria(cfg, cfg.entidad, "PAR_", "Par comparable")
            + [("MT_RANKING", "NUMBER", "Ranking de la BRECHA en la tabla de recomendaciones."),
               ("MT_ORDEN", "NUMBER", "1 = el par de tamaño más parecido."),
               ("MT_USD_PAR_ITEM", "NUMBER", "USD que el par compró del ítem en la ventana."),
               ("MT_USD_PAR_TOTAL", "NUMBER", "Compra total del par en la ventana."),
               ("MT_PARTICIPACION_PAR", "NUMBER", "MT_USD_PAR_ITEM / MT_USD_PAR_TOTAL. La mediana (o lo que diga "
                                                  "ESTADISTICO_PARES) de estas filas es el número del motivo."),
               ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo.")])


def catalogo_vecindario(cfg: RecConfig) -> List[Tuple[str, str, str]]:
    """Con quién se agrupó a cada entidad."""
    return ([("BD_SEGMENTO", "VARCHAR2(400)", "Segmento."),
             ("BD_ALGORITMO", "VARCHAR2(40)", "coseno_entidad (sus vecinos) o kmeans_valor (su grupo).")]
            + _cols_categoria(cfg, cfg.entidad, "", "Entidad")
            + _cols_categoria(cfg, cfg.entidad, "PAR_", "Vecino (sólo coseno_entidad)")
            + [("BD_GRUPO", "VARCHAR2(200)", "El grupo: 'k-means #3' o 'sus 50 pares más parecidos'."),
               ("MT_ORDEN", "NUMBER", "Lugar del vecino, 1 el más parecido (0 en kmeans_valor)."),
               ("MT_SIMILITUD", "NUMBER", "Coseno entre la entidad y el vecino, por lo que compran."),
               ("MT_TAMANO_GRUPO", "NUMBER", "Entidades del grupo, o vecinos de la entidad."),
               ("FECHA_CORTE", "DATE", "Último día incluido en el cálculo.")])


class RecEngine:
    def __init__(self, config: RecConfig):
        config.validate()
        self.cfg = config
        self.fechas = Fechas.desde(config.fecha_ejecucion)
        self.diagnostico = pd.DataFrame()
        self.evidencia = pd.DataFrame()
        self.referencia = self.matriz_compras = self.curva = self.vecindario = pd.DataFrame()
        self.pares_comparables = pd.DataFrame()
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
        # La probabilidad de adopción de una CRUZADA es lo que acertó el backtest. Se lee
        # DESPUÉS de correrlo: antes se leía del diagnóstico de una corrida anterior, que en
        # una corrida nueva está vacío, y todas las cruzadas salían con probabilidad 100%.
        p_adopcion, p_global = self._probabilidad_adopcion()

        salidas, evidencias, referencias, vecindarios, comparables = [], [], [], [], []
        codigos, etiquetas = pd.factorize(seg["segmento"])
        t0 = time.time()
        for k, etiqueta in enumerate(etiquetas):
            idx = np.flatnonzero(codigos == k)
            nivel = str(seg["nivel_segmento"].iloc[idx[0]])
            b = Bloque(cfg, matriz, idx, str(etiqueta), nivel, canastas=canastas, d_ayer=f.d_ayer)
            b.dias_ventana = dias_ventana
            if str(etiqueta) in p_adopcion:
                b.p_adopcion, b.origen_prob = p_adopcion[str(etiqueta)]
            elif p_global is not None:
                b.p_adopcion, b.origen_prob = p_global
            b.nombres_item = nombres
            nombre_alg = elegido_por_segmento.get(str(etiqueta), mejor_global)
            alg = construir_algoritmo(nombre_alg, cfg)
            alg.ajustar(b)
            parte = armar_recomendaciones(b, matriz, alg, cfg, f.d_ayer)
            if len(parte):
                salidas.append(parte)
                if cfg.guardar_evidencia:
                    ev_b = evidencias_bloque(b, matriz, alg, parte, cfg)
                    if len(ev_b):
                        evidencias.append(self._ensamblar_evidencia(ev_b, panel))
                    if getattr(b, "comparables", None) is not None and len(b.comparables):
                        comparables.append(b.comparables)
            if cfg.guardar_referencia:
                referencias.append(referencia_bloque(b, cfg))
            if cfg.guardar_vecindario:
                con_reco = np.unique(parte["f"].to_numpy(np.int64)) if len(parte) else np.zeros(0, np.int64)
                vecindarios.append(vecindario_bloque(b, alg, con_reco))
            LOGGER.info("segmento [%s] nivel %s: %s entidades, %s ítems candidatos, algoritmo %s -> %s recomendaciones",
                        etiqueta, nivel, f"{b.n:,}", f"{int(b.candidato.sum()):,}", nombre_alg,
                        f"{len(parte):,}")
        self.tiempos_["recomendar"] = time.time() - t0

        self.evidencia = pd.DataFrame(columns=[c for c, _, _ in catalogo_evidencia(cfg)])
        self._tablas_soporte(panel, matriz, seg, referencias, vecindarios, comparables)
        if not salidas:
            LOGGER.warning("no salió ninguna recomendación")
            return pd.DataFrame(columns=[c for c, _, _ in catalogo(cfg)])
        rec = pd.concat(salidas, ignore_index=True)
        out = self._ensamblar(rec, panel)
        evidencias = [e for e in evidencias if len(e)]
        if evidencias:
            self.evidencia = pd.concat(evidencias, ignore_index=True)
            del evidencias
            for c in self.evidencia.columns:       # textos repetidos: categorías, no millones de objetos
                if self.evidencia[c].dtype == object and c != "BD_DETALLE":
                    self.evidencia[c] = self.evidencia[c].astype("category")
            LOGGER.info("evidencia: %s filas para %s recomendaciones (%s)", f"{len(self.evidencia):,}",
                        f"{self.evidencia.loc[self.evidencia['BD_EVIDENCIA'] == 'CALCULO'].shape[0]:,}",
                        self.evidencia["BD_EVIDENCIA"].value_counts().to_dict())
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
            "MT_PROB": np.round(rec["prob"].to_numpy(float), 6),
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

    def _tablas_soporte(self, panel: Panel, matriz: Matriz, seg: pd.DataFrame,
                        referencias: List[pd.DataFrame], vecindarios: List[pd.DataFrame],
                        comparables: List[pd.DataFrame]) -> None:
        """Las tablas con las que se comprueba cualquier número de un motivo o de un cálculo."""
        cfg, ayer = self.cfg, self.fechas.ayer
        ent_cols = {c.upper(): panel.entidades[c].to_numpy() for c in cfg.entidad}
        item_cols = {c.upper(): panel.items[c].to_numpy() for c in cfg.item}

        def claves(ent=None, item=None, par=None):
            out = {}
            if ent is not None:
                out.update({c: v[ent] for c, v in ent_cols.items()})
            if item is not None:
                out.update({c: (v[item] if (item >= 0).all() else
                                np.where(item >= 0, v.astype(object)[np.maximum(item, 0)], None))
                            for c, v in item_cols.items()})
            if par is not None:
                out.update({"PAR_" + c: (v[par] if (par >= 0).all() else
                                         np.where(par >= 0, v.astype(object)[np.maximum(par, 0)], None))
                            for c, v in ent_cols.items()})
            return out

        def vacia(catalogo_fn):
            return pd.DataFrame(columns=[c for c, _, _ in catalogo_fn(cfg)])

        def fecha(d):
            return pd.to_datetime(np.asarray(d, float), unit="D", origin=_EPOCA).to_numpy()

        self.referencia, self.matriz_compras = vacia(catalogo_referencia), vacia(catalogo_matriz)
        self.curva, self.vecindario = vacia(catalogo_curva), vacia(catalogo_vecindario)
        self.pares_comparables = vacia(catalogo_pares_comparables)
        if comparables:
            c = pd.concat(comparables, ignore_index=True)
            self.pares_comparables = pd.DataFrame({
                **claves(ent=c["entidad"].to_numpy(np.int64), item=c["i"].to_numpy(np.int64),
                         par=c["par_entidad"].to_numpy(np.int64)),
                "MT_RANKING": c["ranking"].to_numpy(int),
                "MT_ORDEN": c["orden"].to_numpy(int),
                "MT_USD_PAR_ITEM": np.round(c["usd_par"].to_numpy(float), 2),
                "MT_USD_PAR_TOTAL": np.round(c["total_par"].to_numpy(float), 2),
                "MT_PARTICIPACION_PAR": c["share_par"].to_numpy(float),     # sin redondear: es la base de la mediana
                "FECHA_CORTE": np.full(len(c), ayer)})
        dec = cfg.decimales
        refs = [r for r in referencias if len(r)]
        if refs:
            r = pd.concat(refs, ignore_index=True)
            self.referencia = pd.DataFrame({
                "BD_SEGMENTO": r["segmento"].to_numpy(dtype=object),
                "BD_NIVEL_SEGMENTO": r["nivel_segmento"].to_numpy(dtype=object),
                **claves(item=r["i"].to_numpy(np.int64)),
                "MT_ENTIDADES_SEGMENTO": r["entidades"].to_numpy(float),
                "MT_COMPRADORES": r["compradores"].to_numpy(float),
                "MT_PENETRACION": np.round(r["penetracion"].to_numpy(float), dec),
                "BD_CANDIDATO": np.where(r["candidato"].to_numpy(bool), "SI", "NO"),
                "MT_USD_ITEM": np.round(r["usd_item"].to_numpy(float), 2),
                "MT_DIAS_COMPRA_ITEM": r["dias_item"].to_numpy(float),
                "MT_TICKET": np.round(r["ticket"].to_numpy(float), 2),
                "MT_USD_MEDIO_COMPRADOR": np.round(r["usd_medio_comprador"].to_numpy(float), 2),
                "MT_COMPRADORES_CON_RITMO": r["con_ritmo"].to_numpy(float),
                "MT_INTERVALO_MEDIANA": np.round(r["intervalo_mediana"].to_numpy(float), 2),
                "MT_CV_MEDIANA": np.round(r["cv_mediana"].to_numpy(float), dec),
                "MT_PARTICIPACION_MEDIA": np.round(r["participacion_media"].to_numpy(float), dec),
                "MT_MARGEN_PCT": np.round(r["margen_pct"].to_numpy(float), dec),
                "MT_VENTA_MEDIA_ENTIDAD": np.round(r["venta_media_entidad"].to_numpy(float), 2),
                "MT_PROB_ADOPCION": np.round(r["prob_adopcion"].to_numpy(float), 6),
                "BD_ORIGEN_PROB": r["origen_prob"].to_numpy(dtype=object),
                "MT_DIAS_VENTANA": r["dias_ventana"].to_numpy(float),
                "FECHA_CORTE": np.full(len(r), ayer)})
        if cfg.guardar_matriz and len(matriz.tab):
            t = matriz.tab
            e = t["e"].to_numpy(np.int64)
            venta = matriz.venta_entidad[e]
            usd = t["usd"].to_numpy(float)
            self.matriz_compras = pd.DataFrame({
                **claves(ent=e, item=t["i"].to_numpy(np.int64)),
                "BD_SEGMENTO": seg["segmento"].to_numpy(dtype=object)[e],
                "MT_USD": np.round(usd, 2),
                "MT_MARGEN": np.round(t["margen"].to_numpy(float), 2),
                "MT_DIAS_COMPRA": t["dias"].to_numpy(float),
                "FECHA_PRIMERA": fecha(t["primero"]), "FECHA_ULTIMA": fecha(t["ultimo"]),
                "MT_INTERVALO_MEDIO": np.round(pd.to_numeric(t["intervalo_medio"], errors="coerce").to_numpy(float), 4),
                "MT_INTERVALO_DESVIO": np.round(pd.to_numeric(t["intervalo_desvio"], errors="coerce").to_numpy(float), 4),
                "MT_USD_ENTIDAD": np.round(venta, 2),
                "MT_PARTICIPACION": np.where(venta > 0, usd / np.where(venta > 0, venta, 1.0), np.nan),
                "FECHA_CORTE": np.full(len(t), ayer)})
        if cfg.guardar_curva:
            c = tabla_curva(matriz, cfg)
            if len(c):
                self.curva = pd.DataFrame({
                    **claves(item=c["i"].to_numpy(np.int64)),
                    "BD_ALCANCE": c["alcance"].to_numpy(dtype=object),
                    "MT_ATRASO": c["atraso"].to_numpy(float),
                    "MT_CASOS": c["casos"].to_numpy(float),
                    "MT_VOLVIERON": c["volvieron"].to_numpy(float),
                    "MT_PROB": np.round(c["prob"].to_numpy(float), 4),
                    "MT_HORIZONTE_DIAS": c["horizonte"].to_numpy(float),
                    "MT_MIN_CASOS": c["min_casos"].to_numpy(float),
                    "FECHA_CORTE": np.full(len(c), ayer)})
        vec = [v for v in vecindarios if len(v)]
        if vec:
            v = pd.concat(vec, ignore_index=True)
            self.vecindario = pd.DataFrame({
                "BD_SEGMENTO": v["segmento"].to_numpy(dtype=object),
                "BD_ALGORITMO": v["algoritmo"].to_numpy(dtype=object),
                **claves(ent=v["entidad"].to_numpy(np.int64), par=v["par_entidad"].to_numpy(np.int64)),
                "BD_GRUPO": v["grupo"].to_numpy(dtype=object),
                "MT_ORDEN": v["orden"].to_numpy(float),
                "MT_SIMILITUD": np.round(v["similitud"].to_numpy(float), dec),
                "MT_TAMANO_GRUPO": v["tamano"].to_numpy(float),
                "FECHA_CORTE": np.full(len(v), ayer)})
        LOGGER.info("tablas para comprobar: referencia %s, matriz %s, curva %s, vecindario %s, pares "
                    "comparables %s filas", f"{len(self.referencia):,}", f"{len(self.matriz_compras):,}",
                    f"{len(self.curva):,}", f"{len(self.vecindario):,}", f"{len(self.pares_comparables):,}")

    def _probabilidad_adopcion(self) -> Tuple[Dict[str, Tuple[float, str]], Optional[Tuple[float, str]]]:
        """P(adopta) de una CRUZADA en el horizonte, por segmento, leída del backtest.

        Es la precisión del algoritmo elegido (qué fracción de lo recomendado se compró en
        `dias_backtest` días), llevada al horizonte. Un segmento sin adopciones suficientes
        para medirse usa la del panel entero, igual que para elegir el algoritmo.
        """
        cfg, d = self.cfg, self.diagnostico
        if not len(d) or not cfg.usar_probabilidad:
            return {}, None
        base = float(cfg.dias_backtest or 90)
        factor = float(cfg.horizonte_dias or base) / base

        def tasa(precision: float) -> float:
            return float(min(max(precision * factor, 0.0), 1.0))

        ganador = d.attrs.get("ganador_global")
        del_panel = d[d["algoritmo"] == ganador]
        recomendados = float(del_panel["recomendados"].sum())
        p_panel = float(del_panel["aciertos"].sum()) / recomendados if recomendados else 0.0
        p_global = (tasa(p_panel), f"lo que acertó {ganador} en el backtest del panel entero, "
                                   f"{_pct(p_panel)} en {base:.0f} días")
        por_segmento = {}
        for _, g in d[d["elegido"]].iterrows():
            if g["adopciones"] >= cfg.min_adopciones_backtest:
                por_segmento[str(g["segmento"])] = (
                    tasa(float(g["precision"])),
                    f"lo que acertó {g['algoritmo']} en el backtest del segmento, {_pct(float(g['precision']))} "
                    f"en {base:.0f} días")
            else:
                por_segmento[str(g["segmento"])] = p_global
        return por_segmento, p_global

    def _ensamblar_evidencia(self, ev: pd.DataFrame, panel: Panel) -> pd.DataFrame:
        cfg = self.cfg
        ent = ev["entidad"].to_numpy(np.int64)
        item = ev["i"].to_numpy(np.int64)
        par = ev["par_entidad"].to_numpy(np.int64)
        ref = ev["ref_item"].to_numpy(np.int64)
        salida: Dict[str, Any] = {}
        for c in cfg.entidad:
            salida[c.upper()] = panel.entidades[c].to_numpy()[ent]
        for c in cfg.item:
            salida[c.upper()] = panel.items[c].to_numpy()[item]
        salida.update({
            "MT_RANKING": ev["ranking"].to_numpy(int),
            "BD_TIPO": ev["tipo"].to_numpy(dtype=object),
            "BD_ALGORITMO": ev["algoritmo"].to_numpy(dtype=object),
            "MT_ORDEN": ev["orden"].to_numpy(int),
            "BD_EVIDENCIA": ev["evidencia"].to_numpy(dtype=object),
            "BD_ALGORITMO_EVIDENCIA": ev["fuente"].to_numpy(dtype=object),
        })
        for c in cfg.entidad:
            v = panel.entidades[c].to_numpy(dtype=object)[np.maximum(par, 0)]
            salida[f"PAR_{c.upper()}"] = np.where(par >= 0, v, None)
        for c in cfg.item:
            v = panel.items[c].to_numpy(dtype=object)[np.maximum(ref, 0)]
            salida[f"REF_{c.upper()}"] = np.where(ref >= 0, v, None)
        dec = cfg.decimales
        salida.update({
            "MT_SIMILITUD": np.round(ev["similitud"].to_numpy(float), dec),
            "MT_CONTRIBUCION": np.round(ev["contribucion"].to_numpy(float), dec),
            "MT_LIFT": np.round(ev["lift"].to_numpy(float), dec),
            "MT_EN_COMUN": ev["en_comun"].to_numpy(float),
            "MT_USD_PAR_ITEM": np.round(ev["usd_par_item"].to_numpy(float), 2),
            "MT_DIAS_PAR_ITEM": ev["dias_par_item"].to_numpy(float),
            "MT_TICKET_PAR": np.round(ev["ticket_par"].to_numpy(float), 2),
            "MT_INTERVALO_PAR": np.round(ev["intervalo_par"].to_numpy(float), 1),
            "MT_PARTICIPACION_PAR": ev["participacion_par"].to_numpy(float),
            "MT_USD_PAR_TOTAL": np.round(ev["usd_par_total"].to_numpy(float), 2),
            "FECHA_PRIMERA_ITEM_PAR": pd.to_datetime(ev["fecha_item_par"]).to_numpy(),
            "FECHA_PRIMERA_REF_PAR": pd.to_datetime(ev["fecha_ref_par"]).to_numpy(),
            "BD_GRUPO": ev["grupo"].to_numpy(dtype=object),
            "MT_TAMANO_GRUPO": ev["tamano_grupo"].to_numpy(float),
            "MT_COMPRAN_EN_GRUPO": ev["compran_grupo"].to_numpy(float),
            "BD_DETALLE": ev["detalle"].to_numpy(dtype=object),
            "FECHA_CORTE": np.full(len(ev), self.fechas.ayer),
        })
        out = pd.DataFrame(salida)
        claves = [c.upper() for c in cfg.claves_entidad()]
        return out          # ya viene por entidad, ranking y orden dentro de cada segmento

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
            fila = {"nivel_item": " + ".join(nivel), "rejilla": ",".join(combo),
                    "items": panel.n_item, "entidades": panel.n_ent,
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


def parametros_barridos(fila: pd.Series) -> List[str]:
    """Los parámetros que se barrieron en la rejilla de esa fila de explorar(). Sólo esos: la
    tabla también tiene columnas de resultado (`segmentos`, `items`...) que se llaman igual que
    un parámetro y no lo son."""
    if "rejilla" in fila.index and isinstance(fila["rejilla"], str):
        return [k for k in fila["rejilla"].split(",") if k]
    return [k for k in fila.index if k in RecConfig.__dataclass_fields__
            and k not in ("item", "segmentos")]


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
    # TODO lo que se barrió en la rejilla, no una lista fija: antes AFINIDAD (que el notebook
    # barre por defecto) no salía, y había que adivinar cuál había ganado
    for k in parametros_barridos(fila):
        v = limpio(fila[k])
        lineas.append(f"{k.upper()} = {list(v) if isinstance(v, tuple) else v!r}")
    lineas += ["", f"# backtest sobre datos propios: precisión {fila['precision']:.4f} contra "
                   f"{fila['precision_popularidad']:.4f} de popularidad",
               f"# algoritmos que ganaron por segmento: "
               f"{ {str(k): int(v) for k, v in dict(fila['algoritmos_elegidos']).items()} }",
               f"# {fila['items']} ítems, {fila['entidades']} entidades, "
               f"{fila['items_por_entidad']} ítems por entidad (densidad {fila['densidad']:.3%})"]
    return "\n".join(lineas)


__all__ = ["RecConfig", "RecEngine", "Fechas", "Panel", "Bloque", "Matriz", "ALGORITMOS",
           "catalogo", "catalogo_evidencia", "catalogo_referencia", "catalogo_matriz", "catalogo_curva",
           "catalogo_vecindario", "catalogo_pares_comparables", "completar_detalle", "backtest", "preparar", "asignar_segmentos",
           "construir_algoritmo", "TIPOS", "EVIDENCIAS", "explorar", "bloque_config", "parametros_barridos",
           "agregar_tamano"]
