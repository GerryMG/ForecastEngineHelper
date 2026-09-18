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
    dias_afinidad: int = 365           #: ventana que arma la matriz de compras
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
    min_compras_reposicion: int = 2    #: días de compra mínimos para hablar de intervalo
    brecha_ratio: float = 0.5          #: compra menos de esta fracción de lo que compran sus pares
    escalar_potencial: bool = True     #: ajusta el USD potencial por tamaño de la entidad
    tope_escala: float = 3.0           #: tope de ese ajuste

    filas_bloque: int = 2048           #: entidades por bloque (memoria acotada)
    decimales: int = 4
    verbose: int = 1

    # -- derivados ---------------------------------------------------------- #
    def claves_entidad(self) -> List[str]:
        return [c for c in self.entidad if not str(c).upper().startswith("BD_")]

    def desc_entidad(self) -> List[str]:
        return [c for c in self.entidad if str(c).upper().startswith("BD_")]

    def claves_item(self) -> List[str]:
        return [c for c in self.item if not str(c).upper().startswith("BD_")]

    def desc_item(self) -> List[str]:
        return [c for c in self.item if str(c).upper().startswith("BD_")]

    def columnas(self) -> List[str]:
        cols = list(self.entidad) + list(self.item) + list(self.segmentos) + [
            self.col_fecha, self.col_valor, self.col_margen]
        if self.col_tipo_documento:
            cols.append(self.col_tipo_documento)
        if self.col_documento:
            cols.append(self.col_documento)
        return cols

    def validate(self) -> None:
        if not self.claves_entidad():
            raise ValueError("entidad necesita al menos una categoría clave (SK_/BK_)")
        if not self.claves_item():
            raise ValueError("item necesita al menos una categoría clave (SK_/BK_)")
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

    def __init__(self, n_ent: int, n_item: int, tab: pd.DataFrame):
        self.n_ent, self.n_item, self.tab = n_ent, n_item, tab
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
        self.venta_entidad = np.asarray(self.V.sum(1)).ravel()
        self.items_entidad = np.asarray(self.R.sum(1)).ravel()

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
                             "v": self.usd[m], "g": self.margen[m]})
        por_dia = base.groupby(["e", "i", "d"], sort=False, as_index=False).agg(
            v=("v", "sum"), g=("g", "sum"))
        tab = por_dia.groupby(["e", "i"], sort=False, as_index=False).agg(
            usd=("v", "sum"), margen=("g", "sum"), dias=("d", "size"),
            primero=("d", "min"), ultimo=("d", "max"))
        if self.cfg.excluir_netos_no_positivos:
            # comprado y devuelto entero no es una compra
            tab = tab[tab["usd"] > 0].reset_index(drop=True)
        return Matriz(self.n_ent, self.n_item, tab)

    def pares(self, desde: int, hasta: int) -> sp.csr_matrix:
        """Pares entidad-ítem con compra en la ventana, como matriz binaria."""
        m = (self.dia >= desde) & (self.dia <= hasta)
        return sp.csr_matrix((np.ones(int(m.sum())), (self.ent[m], self.item[m])),
                             shape=(self.n_ent, self.n_item))


    def canastas(self, desde: int, hasta: int) -> Tuple[sp.csr_matrix, np.ndarray]:
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
        return B, entidad_canasta


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
                 canastas: Optional[Tuple[sp.csr_matrix, np.ndarray]] = None):
        self.cfg, self.etiqueta, self.nivel, self.filas = cfg, etiqueta, nivel, filas
        self.R = matriz.R[filas]
        self.V = matriz.V[filas]
        self.M = matriz.M[filas]
        self.U = matriz.U[filas]
        self.D = matriz.D[filas]
        self.P = matriz.P[filas]
        self.n = len(filas)
        self.n_item = matriz.n_item

        # matriz con la que se mide la afinidad: canastas (lo que se compra junto) o
        # el repertorio de cada entidad (todo lo que compra en la ventana)
        self.A = (self.R > 0).astype(float)
        self.n_afinidad = self.n
        if canastas is not None:
            B, entidad_canasta = canastas
            pertenece = np.zeros(matriz.n_ent, dtype=bool)
            pertenece[filas] = True
            sel = np.flatnonzero(pertenece[entidad_canasta])
            if len(sel):
                self.A = B[sel]
                self.n_afinidad = len(sel)

        self.soporte = np.asarray((self.R > 0).sum(0)).ravel().astype(float)
        self.penetracion = self.soporte / max(self.n, 1)
        self.candidato = (self.soporte >= cfg.min_soporte) & (self.penetracion >= cfg.min_penetracion)

        usd_item = np.asarray(self.V.sum(0)).ravel()
        margen_item = np.asarray(self.M.sum(0)).ravel()
        seguro = np.maximum(self.soporte, 1.0)
        self.usd_medio_comprador = np.where(self.soporte > 0, usd_item / seguro, 0.0)
        self.margen_pct_item = np.where(usd_item > 0, margen_item / np.where(usd_item > 0, usd_item, 1.0), 0.0)

        self.venta_entidad = np.asarray(self.V.sum(1)).ravel()
        positivas = self.venta_entidad[self.venta_entidad > 0]
        self.venta_media = float(positivas.mean()) if len(positivas) else 0.0
        escala = sp.diags(1.0 / np.where(self.venta_entidad > 0, self.venta_entidad, 1.0))
        self.share = (escala @ self.V).tocsr()
        self.share_medio_comprador = np.where(self.soporte > 0,
                                              np.asarray(self.share.sum(0)).ravel() / seguro, 0.0)

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
    """Los pares entidad-ítem del bloque, con la fila local."""
    mapa = np.full(matriz.n_ent, -1, dtype=np.int64)
    mapa[b.filas] = np.arange(b.n)
    f = mapa[matriz.tab["e"].to_numpy(np.int64)]
    sel = f >= 0
    out = matriz.tab.loc[sel].copy()
    out["f"] = f[sel]
    return out


def _escala_tamano(b: Bloque, filas: np.ndarray, cfg: RecConfig) -> np.ndarray:
    """Ajusta el potencial al tamaño de la entidad: un cliente chico no compra como uno grande."""
    if not cfg.escalar_potencial or b.venta_media <= 0:
        return np.ones(len(filas))
    razon = b.venta_entidad[filas] / b.venta_media
    razon = np.where(razon > 0, razon, 1.0)
    return np.clip(razon, 1.0 / cfg.tope_escala, cfg.tope_escala)


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
    """Lo que compraba con cierto ritmo y hace rato no compra."""
    p = _pares_bloque(b, matriz)
    if p.empty:
        return pd.DataFrame(columns=["f", "i", "puntaje", "usd_potencial", "dias_sin_comprar", "motivo"])
    dias = p["dias"].to_numpy(float)
    primero = p["primero"].to_numpy(float)
    ultimo = p["ultimo"].to_numpy(float)
    usd = p["usd"].to_numpy(float)
    silencio = d_ayer - ultimo
    intervalo = np.where(dias >= 2, (ultimo - primero) / np.maximum(dias - 1, 1), np.inf)
    ok = ((dias >= cfg.min_compras_reposicion) & np.isfinite(intervalo) & (intervalo > 0)
          & (silencio > cfg.factor_reposicion * intervalo))
    if not ok.any():
        return pd.DataFrame(columns=["f", "i", "puntaje", "usd_potencial", "dias_sin_comprar", "motivo"])
    ritmo = usd[ok] / np.maximum(ultimo[ok] - primero[ok] + 1.0, 1.0)
    iv, sil = intervalo[ok], silencio[ok]
    return pd.DataFrame({
        "f": p["f"].to_numpy()[ok],
        "i": p["i"].to_numpy()[ok],
        "puntaje": sil / iv,
        "usd_potencial": ritmo * sil,
        "dias_sin_comprar": sil,
        "motivo": [f"compraba cada {v:.0f} días y lleva {s:.0f} sin comprar" for v, s in zip(iv, sil)]})


def recomendar_brecha(b: Bloque, matriz: Matriz, cfg: RecConfig) -> pd.DataFrame:
    """Lo que compra, pero mucho menos de lo que le dedican sus pares."""
    p = _pares_bloque(b, matriz)
    if p.empty:
        return pd.DataFrame(columns=["f", "i", "puntaje", "usd_potencial", "motivo"])
    f = p["f"].to_numpy(np.int64)
    i = p["i"].to_numpy(np.int64)
    venta_ent = b.venta_entidad[f]
    share = np.where(venta_ent > 0, p["usd"].to_numpy(float) / np.where(venta_ent > 0, venta_ent, 1.0), 0.0)
    medio = b.share_medio_comprador[i]
    ok = b.candidato[i] & (venta_ent > 0) & (medio > 0) & (share < cfg.brecha_ratio * medio)
    if not ok.any():
        return pd.DataFrame(columns=["f", "i", "puntaje", "usd_potencial", "motivo"])
    sh, me, ve = share[ok], medio[ok], venta_ent[ok]
    return pd.DataFrame({
        "f": f[ok], "i": i[ok],
        "puntaje": 1.0 - sh / me,
        "usd_potencial": (me - sh) * ve,
        "motivo": [f"sus pares le dedican {m:.1%} de su compra y esta entidad {s:.1%}"
                   for m, s in zip(me, sh)]})


def armar_recomendaciones(b: Bloque, matriz: Matriz, alg: Algoritmo, cfg: RecConfig,
                          d_ayer: int) -> pd.DataFrame:
    """Los tres tipos juntos, ordenados por USD en juego y recortados a max_items_reco."""
    partes = []
    if "CRUZADA" in cfg.incluir_tipos:
        cru = recomendar_cruzadas(b, alg, cfg)
        if len(cru):
            escala = _escala_tamano(b, cru["f"].to_numpy(), cfg)
            cru["usd_potencial"] = b.usd_medio_comprador[cru["i"].to_numpy()] * escala
            cru["dias_sin_comprar"] = np.nan
            cru["motivo"] = alg.explicar(cru["f"].to_numpy(), cru["i"].to_numpy())
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
            bre["tipo"] = "BRECHA"
            bre["algoritmo"] = "regla_brecha"
            bre["dias_sin_comprar"] = np.nan
            partes.append(bre)
    if not partes:
        return pd.DataFrame()
    out = pd.concat(partes, ignore_index=True)
    out["margen_potencial"] = out["usd_potencial"] * b.margen_pct_item[out["i"].to_numpy()]
    out = out.sort_values(["f", "usd_potencial", "puntaje"], ascending=[True, False, False])
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
    entrena = panel.matriz(corte - cfg.dias_afinidad + 1, corte)
    evalua = panel.matriz(corte + 1, f.d_ayer)
    verdad = (evalua.R > 0)
    nuevos = verdad.astype(int) - (entrena.R > 0).astype(int)
    nuevos.data = np.where(nuevos.data > 0, 1, 0)
    nuevos.eliminate_zeros()
    from dataclasses import replace
    panel.cfg = replace(cfg, segmentos=agregar_tamano(panel, entrena))
    seg = asignar_segmentos(panel, entrena)
    nombres = etiquetas_item(panel)
    canastas = (panel.canastas(corte - cfg.dias_afinidad + 1, corte)
                if cfg.afinidad == "canasta" else None)

    filas = []
    codigos, etiquetas = pd.factorize(seg["segmento"])
    for k, etiqueta in enumerate(etiquetas):
        idx = np.flatnonzero(codigos == k)
        b = Bloque(cfg, entrena, idx, str(etiqueta), str(seg["nivel_segmento"].iloc[idx[0]]),
                   canastas=canastas)
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
        ("MT_USD_POTENCIAL", "NUMBER", "USD anuales estimados en juego. CRUZADA: lo que gasta un par medio, "
                                       "ajustado por tamaño. REPOSICION: lo que dejó de comprar desde que se atrasó. "
                                       "BRECHA: lo que le faltaría para igualar a sus pares."),
        ("MT_MARGEN_POTENCIAL", "NUMBER", "MT_USD_POTENCIAL por el margen porcentual del ítem en el segmento. USD."),
        ("MT_PUNTAJE", "NUMBER", "Puntaje del algoritmo o de la regla. Comparable dentro del mismo tipo y segmento."),
        ("MT_PENETRACION_SEGMENTO", "NUMBER", "Fracción de las entidades del segmento que compran el ítem."),
        ("MT_SOPORTE_SEGMENTO", "NUMBER", "Cantidad de entidades del segmento que compran el ítem."),
        ("MT_USD_MEDIO_PAR", "NUMBER", "USD que gasta en el ítem una entidad del segmento que sí lo compra. USD."),
        ("MT_USD_ENTIDAD", "NUMBER", "Compra total de la entidad en la ventana de afinidad. USD."),
        ("MT_DIAS_SIN_COMPRAR", "NUMBER", "Sólo REPOSICION: días desde la última compra del ítem."),
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
        desde = f.d_ayer - cfg.dias_afinidad + 1
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

        salidas = []
        codigos, etiquetas = pd.factorize(seg["segmento"])
        t0 = time.time()
        for k, etiqueta in enumerate(etiquetas):
            idx = np.flatnonzero(codigos == k)
            nivel = str(seg["nivel_segmento"].iloc[idx[0]])
            b = Bloque(cfg, matriz, idx, str(etiqueta), nivel, canastas=canastas)
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
            "MT_MARGEN_POTENCIAL": np.round(rec["margen_potencial"].to_numpy(float), 2),
            "MT_PUNTAJE": np.round(rec["puntaje"].to_numpy(float), dec),
            "MT_PENETRACION_SEGMENTO": np.round(rec["penetracion"].to_numpy(float), dec),
            "MT_SOPORTE_SEGMENTO": rec["soporte"].to_numpy(float),
            "MT_USD_MEDIO_PAR": np.round(rec["usd_medio_par"].to_numpy(float), 2),
            "MT_USD_ENTIDAD": np.round(rec["usd_entidad"].to_numpy(float), 2),
            "MT_DIAS_SIN_COMPRAR": rec["dias_sin_comprar"].to_numpy(float),
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

    rejilla = dict(rejilla or {})
    claves = list(rejilla)
    combos = [dict(zip(claves, valores)) for valores in product(*(rejilla[k] for k in claves))] or [{}]
    niveles = [list(n) for n in (niveles_item or [list(cfg_base.item)])]
    filas = []
    for nivel in niveles:
        cfg_nivel = replace(cfg_base, item=nivel, seleccion="backtest", verbose=0)
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
