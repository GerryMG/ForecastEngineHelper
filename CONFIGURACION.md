# Referencia de configuración

Todo se configura en los archivos `*_oracle.py`. Los `*_engine.py` no se tocan: son el cálculo.

| Análisis | Configuración | Cálculo | Notebooks |
|---|---|---|---|
| Pronóstico | `fc_oracle.py` | `forecast_engine.py` | `run_inicial`, `run_mensual`, `analisis_forecast` |
| Estadísticas | `st_oracle.py` | `stats_engine.py` | `run_estadisticas` |
| Recomendación | `rec_oracle.py` | `rec_engine.py` | `run_recomendaciones`, `calibrar_recomendaciones` |
| Vigilancia | `vig_oracle.py` | `vig_engine.py` | `run_vigilancia`, `calibrar_vigilancia` |

Convención de nombres en todos: `SK_`/`BK_` agrupan (claves), `BD_` describen (se arrastra el valor más
reciente), `MT_` son métricas.

Variables de entorno comunes: `ORA_USER`, `ORA_PASSWORD`, `ORA_DSN` para el origen y
`ORA_DEST_USER`, `ORA_DEST_PASSWORD`, `ORA_DEST_DSN` para el destino.

---

## `st_oracle.py` — Estadísticas por cliente

| Perilla | Qué hace | Cuándo tocarla |
|---|---|---|
| `CATEGORIAS` | claves y descripciones que definen cada fila | siempre |
| `SQL_FUENTE` | la consulta; **tiene que** filtrar por `:desde` y `:hasta` | siempre |
| `TABLA_DESTINO`, `MODO_CARGA` | dónde escribe y cómo (`delete` o `truncate`) | siempre |
| `SEGMENTOS_ACTIVIDAD` | un modelo de actividad por cada valor de estas categorías | si querés P(activo) por canal, región, etc. |
| `MIN_CLIENTES_SEGMENTO` | segmento más chico que esto usa los parámetros del panel | si te quedan muchos segmentos en GLOBAL |
| `MODO_CORRIDA`, `RELEER_MESES` | incremental o completa, y cuánto se relee | si corregís datos viejos seguido |
| `MAX_DIAS_SIN_DATOS` | atraso tolerado de la fuente antes de cortar | si tu ETL carga con más atraso |
| `FECHA_INICIO_FUENTE` | desde dónde lee la corrida completa | rara vez |
| `MIN_BASE_PORCENTAJE` | venta mínima para calcular un margen % (ver abajo) | si tenés ventas microscópicas |
| `SUMA_EXACTA`, `MAX_DECIMALES`, `TOLERANCIA_CERO` | cómo se suma el dinero (ver abajo) | casi nunca |

Variables de entorno: `ST_MODO`, `ST_RELEER_MESES`, `ST_RELEER_DESDE`, `ST_FECHA_EJECUCION`, `ST_DRY_RUN`.

---

## `rec_oracle.py` — Recomendación

### Qué se recomienda

| Perilla | Qué hace |
|---|---|
| `ENTIDAD` | a quién se le recomienda (cliente, sucursal, vendedor) |
| `ITEM` | qué se le recomienda, y a qué nivel (SKU, submarca, familia, servicio) |
| `SEGMENTOS` | con quién se compara: jerarquía del **más grueso al más fino** |
| `SQL_FUENTE` | la consulta, con `:desde` y `:hasta` |
| `TABLA_DESTINO`, `TABLA_DIAGNOSTICO`, `GUARDAR_DIAGNOSTICO` | dónde escribe |
| `TABLA_EVIDENCIA`, `GUARDAR_EVIDENCIA` | con quién se comparó cada recomendación (ver abajo) |
| `TABLA_REFERENCIA`, `TABLA_MATRIZ`, `TABLA_CURVA`, `TABLA_VECINDARIO` y sus `GUARDAR_...` | las tablas para comprobar cualquier número (ver abajo) |

### Cómo se mide la afinidad

| Perilla | Qué hace | Default |
|---|---|---|
| `AFINIDAD` | `"canasta"` = lo que se compra el mismo día (o mismo documento); `"repertorio"` = todo lo del período | `canasta` |
| `COL_DOCUMENTO` | si la fuente trae el número de documento, la canasta pasa a ser el documento | `None` |
| `CORTES_TAMANO` | cuantiles que arman el nivel `BD_TAMANO`; `[]` = no segmentar por tamaño | `[0.5, 0.8, 0.95]` |
| `ETIQUETAS_TAMANO` | los nombres de esos tramos | CHICO…TOP |

### Tipo de documento

| Perilla | Qué hace |
|---|---|
| `COL_TIPO_DOCUMENTO` | columna con el tipo. `None` = no se usa (si ya viene todo neto y filtrado) |
| `TIPOS_VENTA` | sólo estos cuentan como compra. `[]` = todos los no excluidos |
| `TIPOS_DEVOLUCION` | estos restan |
| `DEVOLUCION_YA_NEGATIVA` | `True` si en la fuente ya vienen en negativo |
| `TIPOS_EXCLUIDOS` | se ignoran (fletes, ajustes) |
| `EXCLUIR_NETOS_NO_POSITIVOS` | comprado y devuelto entero no cuenta como compra |

### Recortes y batería

| Perilla | Qué hace | Default |
|---|---|---|
| `DIAS_AFINIDAD` | ventana con la que se arma la matriz. **0 = toda la historia** | 365 |
| `FECHA_INICIO_FUENTE` | desde dónde leer cuando `DIAS_AFINIDAD = 0` | 1900-01-01 |
| `DIAS_BACKTEST` | tramo final reservado para medir aciertos | 90 |
| `MIN_ENTIDADES_SEGMENTO` | menos que esto y el segmento sube de nivel | 200 |
| `MIN_SOPORTE`, `MIN_PENETRACION` | cuántas entidades del segmento tienen que comprar el ítem | 5 / 2% |
| `MAX_ITEMS_RECO` | recomendaciones por entidad | 10 |
| `ALGORITMOS` | la batería que se mide | los 9 |
| `SELECCION` | `backtest` (gana el mejor por segmento), `rrf`, `ponderado`, o el nombre de uno | `backtest` |
| `METRICA_SELECCION` | `precision`, `usd` o `recall` | `precision` |
| `TIPOS_RECOMENDACION` | CRUZADA, REPOSICION, BRECHA | las tres |
| `FACTOR_REPOSICION` | silencio mayor a esto × su intervalo típico = atrasado | 1,5 |
| `BRECHA_RATIO` | compra menos de esta fracción de lo que le dedican sus pares comparables | 0,5 |
| `PARES_COMPARABLES` | con cuántos se compara una BRECHA: los compradores del ítem de tamaño más parecido, todos listados en la evidencia | 30 |
| `MIN_PARES_COMPARABLES` | con menos no hay BRECHA | 5 |
| `ESTADISTICO_PARES` | `mediana` (no la mueven los extremos), `agregado` (USD del ítem / USD total de todos ellos) o `media` (la inflan los chicos) | mediana |
| `LAMBDA_EASE`, `MAX_ITEMS_EASE` | regularización y tope de ítems de `ease` | 0,5 / 4.000 |
| `DIAS_TENDENCIA` | qué tan reciente es una adopción para `tendencia` | 90 |

### Cómo se estima el valor

Los tres tipos se miden igual: **USD esperados en los próximos `HORIZONTE_DIAS`**, que es
`MT_USD_SI_COMPRA × MT_PROB`.

| Perilla | Qué hace | Default |
|---|---|---|
| `HORIZONTE_DIAS` | el plazo de la estimación y la unidad común de los tres tipos | 90 |
| `PESO_PRIOR_PARES` | cuánta evidencia de los pares se le presta al que tiene poca propia. Con 1 compra manda el segmento; con 20, manda él. 0 = no prestar | 3,0 |
| `USAR_PROBABILIDAD` | multiplicar por la probabilidad de recompra (reposición) o de adopción medida por el backtest (cruzada) | True |
| `MIN_CASOS_RECUPERACION` | casos para creerle a la curva de recuperación de un ítem; con menos se usa la del panel | 30 |
| `ORDENAR_POR` | `esperado` (rinde por visita) o `bruto` (tamaño de la oportunidad, para campañas de recuperación) | esperado |
| `PISO_PROB` | piso de la probabilidad; con 0,05 lo muy atrasado no vale cero | 0,0 |
| `VIDA_MEDIA_AFINIDAD_DIAS` | pesar la afinidad por recencia: lo de hace N días pesa la mitad. 0 = todo igual | 0 |

### Evidencia exigida y topes del potencial

| Perilla | Qué hace | Default |
|---|---|---|
| `MIN_COMPRAS_REPOSICION` | días de compra propios del par. Con 1 no hay ritmo propio, pero el segmento lo presta; 3 es lo conservador | 2 |
| `MAX_CV_INTERVALO` | desvío sobre promedio de los intervalos. Filtra al que compra salteado. `None` = no filtrar | 1,0 |
| `MIN_DIAS_COMPRA_ENTIDAD` | días de compra de la entidad para recomendarle algo | 3 |
| `TOPE_POTENCIAL_POR_HISTORICO` | veces el propio ritmo de compra del ítem. `0` = sin tope | 1,5 |
| `TOPE_POTENCIAL_RELATIVO` | veces su compra total en el mismo lapso. `0` = sin tope | 1,0 |
| `MIN_BASE_PORCENTAJE` | base mínima para calcular un porcentaje (ver abajo) | 1,0 |
| `SUMA_EXACTA`, `MAX_DECIMALES`, `TOLERANCIA_CERO` | cómo se suma el dinero (ver abajo) | True / 6 / 1e-9 |

La evidencia queda en la tabla: `MT_DIAS_COMPRA_ITEM` (cuántas veces lo compró), `MT_INTERVALO_TIPICO`
(su propio ritmo, vacío si no tiene), `MT_INTERVALO_ESPERADO` (el estimado, mezclando con los pares),
`MT_COMPRAS_ESPERADAS`, `MT_PROB`, `MT_USD_SI_COMPRA` y `MT_DIAS_COMPRA_ENTIDAD`.

Sale **una fila por cliente e ítem**: si un ítem califica como reposición y como brecha, queda el tipo
que mejor lo explica.

### La evidencia: con quién se comparó cada recomendación

| Perilla | Qué hace | Default |
|---|---|---|
| `GUARDAR_EVIDENCIA` | escribe `TABLA_EVIDENCIA`. `False` = ni se la pide ni se la toca | True |
| `TABLA_EVIDENCIA` | una fila por cosa que sostiene cada recomendación | `REC_EVIDENCIA` |
| `MAX_EVIDENCIAS` | ejemplos (pares o ítems) por recomendación y por clase. Los totales van siempre completos, y las listas completas en las tablas de abajo | 3 |
| `EVIDENCIA_HASTA_RANKING` | evidencia sólo para las primeras N de cada entidad. 0 = todas | 0 |

`BD_MOTIVO` dice "12 de sus pares más parecidos lo compran"; la evidencia dice **cuáles**. Se une
con `TABLA_DESTINO` por entidad + ítem, y cada recomendación trae:

- **`CALCULO`** (`MT_ORDEN = 0`): el USD en juego desarmado, cada número con de quiénes sale. Un
  ejemplo real de REPOSICION:

  > Lo compró 8 veces, cada 33 días en promedio; lleva 92 días sin comprar. En su segmento lo compran
  > cada 46 días (mediana de 78 compradores con ritmo). Intervalo estimado: (7 x 33.00 + 3 x 45.92) /
  > (7 + 3) = 36.88: lo suyo pesa 7 (sus intervalos) y lo del segmento 3. Ticket: el suyo 24.14 (USD
  > 193 / 8 compras); el del ítem en el segmento 133 por compra (USD 75,698 / 570 días de compra de sus
  > 85 compradores en el segmento), ajustado a su tamaño x0.33 = 44.27. Ticket estimado: (8 x 24.14 +
  > 3 x 44.27) / (8 + 3) = 29.63. En 90 días: 2.4 compras, 72.31. Tope: no más de 1 vez lo que él mismo
  > compra de este ítem en 90 días, 47.61. Lleva 2.5 veces su intervalo sin comprar. De los que llegaron
  > a ese atraso, cuántos volvieron a comprar dentro de 90 días: a 2 veces, 204 de 218 casos de este
  > ítem (94%); a 3 veces, 57 de 67 (85%). Interpolando entre 2 y 3 veces: 89%: 47.61 x 0.893695 =
  > 42.55 esperados.

- La evidencia de su tipo:

| Quién la produjo | `BD_EVIDENCIA` | Qué lista |
|---|---|---|
| `popularidad` | `COMPRADOR_SEGMENTO` | pares del segmento que compran el ítem (los de tamaño más parecido) |
| `coseno_entidad` | `VECINO` | los pares más parecidos que lo compran, con su coseno y su aporte al puntaje |
| `kmeans_valor` | `MIEMBRO_GRUPO` | entidades del mismo grupo de k-means que lo compran |
| `coseno_item` | `ITEM_AFIN` + `CO_COMPRADOR` | lo que ya compra y se compra con el recomendado, y quiénes compran los dos |
| `reglas` | `REGLA` + `CO_COMPRADOR` | las reglas que le aplican (confianza, lift) y quiénes las cumplen |
| `svd` | `FACTOR_LATENTE` + `CO_COMPRADOR` | lo que ya compra y va con el recomendado en el patrón de consumo |
| `ease` | `ITEM_EASE` + `CO_COMPRADOR` | lo que ya compra y su peso en el modelo EASE |
| `secuencia` | `SECUENCIA` + `ADOPTANTE` | las reglas "compró A y después B" y quiénes la hicieron, con las dos fechas |
| `tendencia` | `ADOPTANTE_RECIENTE` | quiénes empezaron a comprarlo hace poco, con la fecha |
| `rrf` / `ponderado` | la de cada uno | cada algoritmo deja la suya; el motivo dice cuál aportó más |
| REPOSICION | `PAR_RITMO` | **ejemplos** de compradores con ritmo; el ritmo del segmento sale de todos (`REC_REFERENCIA`) |
| BRECHA | `PAR_PARTICIPACION` | los 3 pares comparables más parecidos; **todos** (30) están en `REC_PARES_COMPARABLES` |

Cada par trae lo que compró del ítem en la ventana (`MT_USD_PAR_ITEM`, `MT_DIAS_PAR_ITEM`,
`FECHA_PRIMERA_ITEM_PAR`). Para explicar una recomendación:

```sql
SELECT MT_ORDEN, BD_EVIDENCIA, PAR_SK_CLIENTE, REF_BK_SUBMARCA, BD_DETALLE
  FROM REC_EVIDENCIA
 WHERE SK_CLIENTE = :cliente AND BK_SUBMARCA = :submarca
 ORDER BY MT_ORDEN;
```

### Cada número, comprobable

Todo número que dice un motivo o un cálculo nombra de quiénes sale, y esos datos están en una tabla:

| Tabla | Qué tiene | Perilla |
|---|---|---|
| `REC_REFERENCIA` | cada ítem en cada segmento: compradores, USD y días sumados, ticket, ritmo (mediana y de cuántos), probabilidad de adopción y de dónde sale | `GUARDAR_REFERENCIA` |
| `REC_MATRIZ` | lo que compró cada entidad de cada ítem en la ventana: USD, días, primera y última compra, intervalo, participación | `GUARDAR_MATRIZ` |
| `REC_CURVA_RECUPERACION` | por ítem y para todo el panel: de los que llegaron a 1, 1,5, 2… veces su intervalo, cuántos volvieron | `GUARDAR_CURVA` |
| `REC_VECINDARIO` | los vecinos de cada entidad (coseno_entidad) y su grupo de k-means | `GUARDAR_VECINDARIO` |
| `REC_PARES_COMPARABLES` | los 30 pares de cada BRECHA, con lo que compró cada uno: su mediana es el número del motivo (requiere `GUARDAR_EVIDENCIA`) | `GUARDAR_PARES_COMPARABLES` |

| El texto dice | Se comprueba con |
|---|---|
| "sus 30 pares de tamaño parecido le dedican 5,4% (mediana)" | sus 30 filas en `REC_PARES_COMPARABLES` |
| "N de sus pares más parecidos lo compran" | `REC_VECINDARIO` + `REC_MATRIZ` |
| "en su grupo de gasto (k-means #3, 120 pares) lo compran 40%" | `REC_VECINDARIO` + `REC_MATRIZ` |
| "ticket 165 (USD … / … días de compra de sus 142 compradores)" | `REC_REFERENCIA`, que es la suma de `REC_MATRIZ` |
| "cada 37 días (mediana de 133 compradores con ritmo)" | mediana de `MT_INTERVALO_MEDIO` en `REC_MATRIZ` |
| "de los que llegaron a 2 veces, 61 de 100 volvieron" | `REC_CURVA_RECUPERACION` |
| "probabilidad de adopción 5,2%" | `REC_DIAGNOSTICO` y `REC_REFERENCIA` |

Una prueba recalcula desde la fuente, sin usar el motor, cada número de los motivos y cálculos de los
9 algoritmos y los 3 tipos: todos coinciden.

**La BRECHA cambió.** Antes comparaba contra el promedio simple de la participación de *todos* los
compradores del segmento: no se sabía contra quiénes y los clientes chicos (a los que un ítem les pesa
mucho) lo inflaban. En un caso de prueba el motivo decía 11,3% cuando sus pares reales le dedicaban
5,4%. Ahora compara contra los `PARES_COMPARABLES` compradores de tamaño más parecido, con la mediana.

### Algoritmos nuevos

| Algoritmo | Qué hace | Cuándo gana |
|---|---|---|
| `ease` | modelo lineal ítem a ítem (Steck, 2019): cuánto empuja cada ítem propio al candidato, descontando lo que explican los demás | compite con los mejores casi siempre; en la prueba de gustos mezclados quedó 3º a 0,3 puntos del mejor |
| `secuencia` | de los que compraron A, cuántos compraron B **después** (fechas de primera compra) | cuando hay un orden: equipo y repuesto, básico y premium. En la prueba: 46,8% de precisión contra 7,1% del resto |
| `tendencia` | de los que no lo compraban, cuántos empezaron en los últimos `DIAS_TENDENCIA` días | lanzamientos. En la prueba ganó igual `secuencia`; queda en la batería y el backtest decide |

### Volumen

Medido con 60.000 clientes, 1.500 ítems y 600.000 recomendaciones (memoria del proceso, por encima
de lo que ya usa el motor):

| Configuración | Tiempo | Pico de memoria | Filas |
|---|---|---|---|
| sin evidencia ni tablas | 58 s | 1,7 GB | — |
| todo, para las 10 de cada cliente | 118 s | 6,0 GB | evidencia 5,4 M, pares comparables 6,8 M, matriz 1,1 M, vecindario 0,4 M |
| todo, `EVIDENCIA_HASTA_RANKING = 3` | 80 s | 2,8 GB | evidencia 1,6 M |

El texto de cada par (`BD_DETALLE`) no se guarda en memoria: se arma al escribir, de a lotes. Si el
pod queda justo, `EVIDENCIA_HASTA_RANKING = 3` es lo primero; después, apagar `GUARDAR_MATRIZ` (los
números se pueden rehacer igual desde la fuente con SQL).

Variables de entorno: `RC_FECHA_EJECUCION`, `RC_SELECCION`, `RC_DRY_RUN`.
Para calibrar: `RC_NIVELES` y `RC_REJILLA` (JSON) en `calibrar_recomendaciones.ipynb`.

---

## `vig_oracle.py` — Vigilancia

### Cada vigilancia

Una entrada por métrica en `VIGILANCIAS`, con estos campos:

| Campo | Qué hace |
|---|---|
| `nombre` | identifica la vigilancia en todas las tablas |
| `sql` | su propia consulta, con `:desde` y `:hasta` |
| `categorias` | una serie por combinación de valores |
| `metrica` | la columna con el valor |
| `agregacion` | `suma`, `promedio`, `conteo` o `ratio` |
| `denominador` | sólo para `ratio` |
| `unidad` | USD, unidades, galones, % |
| `granos` | `dia`, `semana`, `mes`: los que quieras |
| `direccion` | `ambas`, `baja` (sólo caídas) o `sube` (sólo subas) |
| `escala` | `auto` (log si la métrica es positiva), `log` o `lineal` |
| `atribucion` | columna del hijo con la que se explica quién causó la anomalía |
| `materialidad_minima` | por debajo de esto el evento no pasa de INFO |
| `detectores` | cuáles corren; `None` = los del config |
| `activa` | para apagarla sin borrarla |

### Umbrales y ruido

| Perilla | Qué hace | Default |
|---|---|---|
| `DIAS_HISTORIA` | cuánta historia se guarda y sirve de referencia | 1095 |
| `DIAS_RELECTURA` | cuántos días se releen de la fuente por corrida | 45 |
| `MODO_CORRIDA` | `auto`, `completo`, `incremental` | `auto` |
| `PERIODOS_EVALUADOS` | qué tan atrás se revisa, por grano | 30 / 8 / 6 |
| `PERIODOS_BASE` | referencia de lo normal, por grano | 91 / 26 / 18 |
| `MIN_PERIODOS` | menos datos que esto y el detector no opina | 6 |
| `DETECTORES` | hueco, salto, escalon, tendencia, estacional, racha, nueva, congelado, dia_cerrado | todos |
| `NIVEL_MAXIMO` | tope de nivel por detector | `{"dia_cerrado": "ATENCION"}` |
| `CALENDARIO_SQL`, `CALENDARIO_CONEXION` | tu historia de asuetos, y en qué base está (ver abajo) | `None` / origen |
| `UMBRAL_Z` | desvíos robustos que definen ATENCION / ALERTA / CRITICO | 3 / 5 / 8 |
| `DESVIO_RELATIVO_MINIMO` | diferencia mínima contra lo esperado para que sea un evento | 5% |
| `PISO_SIGMA_RELATIVO` | el ruido nunca se considera menor a esto del nivel | 2% |
| `CRITICO_DESVIO_RELATIVO_MINIMO` | para ser CRÍTICO, apartarse al menos esto de lo esperado | 20% |
| `FACTOR_SOSPECHA_DATO` | tantas veces lo esperado = posible error de carga | 20 |
| `DIA_CERRADO_RELATIVO` | día de la semana casi siempre en cero = cerrado, no se evalúa | 5% |
| `PESO_RELATIVO_MINIMO` | serie que pesa menos que esto × la serie promedio no pasa de ATENCION (0 = todas pesan igual) | 0,2 |
| `VENTANAS_REFERENCIA` | contra cuántos períodos anteriores se compara, por grano | 8 / 8 / 12 |
| `MIN_VENTANAS_REFERENCIA` | con menos, no se puede comprobar: queda en ATENCION | 4 |
| `VECES_POR_ANIO` | si a esa serie le pasó más veces que esto en el año, no es raro para ella | 1 |
| `NOTIFICAR_ULTIMOS_PERIODOS` | sólo se avisa lo que sigue pasando en los últimos N períodos | 3 / 1 / 1 |
| `PERSISTENCIA_SUBE_NIVEL` | períodos seguidos que suben un nivel (sólo detectores de punto) | 3 |
| `NIVEL_MINIMO_EVENTO` | desde qué nivel se guarda un evento (`INFO` = todo) | ATENCION |
| `NIVEL_NOTIFICACION` | desde qué nivel se notifica | ALERTA |
| `CANALES_NOTIFICACION` | `tabla` siempre; sumá `webhook`, `correo`, `teams` | `("tabla",)` |
| `SUMA_EXACTA`, `MAX_DECIMALES`, `TOLERANCIA_CERO` | cómo se suma la métrica (ver abajo) | True / 6 / 1e-9 |

En el motor hay además `detectores_excluidos` (por defecto, `estacional` no corre en grano día),
`desestacionalizar_dia`, `duracion_minima` por detector, `prioridad_detectores` y `unificar_eventos`.

### `ajustes`: los umbrales no pueden ser los mismos para todas las métricas

La mediana y el desvío se calculan **por serie**, así que el "ruido normal" sale de sus propios
datos. Los **umbrales**, en cambio, venían del config global — y ningún par de números sirve para
dos métricas de estabilidad distinta. Medido con un stock que se mueve 0,9% por día y devoluciones
que se mueven 39%, las dos con una caída real del 6%:

| configuración | stock | devoluciones |
|---|---|---|
| global sensible (`DESVIO_RELATIVO_MINIMO = 0,05`) | 1 evento, **0 avisos** | 1 evento, 0 avisos |
| global tolerante (0,30) | **0 eventos** | 1 evento, 0 avisos |
| cada una con su perfil | **8 eventos, 4 avisos** | 0 eventos, 0 avisos |

Cada `Vigilancia` puede pisar cualquier campo de `VigConfig` con `ajustes`:

```python
Vigilancia(nombre="STOCK", sql=SQL_STOCK, categorias=["BD_PLANTA"],
           metrica="MT_STOCK", unidad="unidades", granos=("dia",),
           ajustes={"desvio_relativo_minimo": 0.03, "piso_sigma_relativo": 0.005})

Vigilancia(nombre="DEVOLUCIONES", sql=SQL_DEV, categorias=["BD_CANAL"],
           metrica="MT_DEVOLUCION", unidad="USD", granos=("dia", "semana"),
           ajustes={"desvio_relativo_minimo": 0.35, "piso_sigma_relativo": 0.25,
                    "umbral_z": {"ATENCION": 4.0, "ALERTA": 6.0, "CRITICO": 10.0},
                    "nivel_notificacion": "CRITICO"})
```

Lo que no nombres se hereda del global. Un campo inexistente o umbrales desordenados se rechazan al
validar, antes de correr.

### Perfiles por tipo de dato

El motor detecta lo raro en cualquier variable: el "ruido normal" sale de los datos de cada serie.
Lo que **no** puede saber solo es **qué tan grande tiene que ser un cambio para importar**, porque
eso depende del dominio: un 10% en la venta de un día es ruido, un 10% en el consumo de agua de una
planta es una fuga. Eso se declara por vigilancia con `ajustes`.

Probado con cuatro tipos de variable muy distintos a una venta. Con los valores por defecto, **cero
notificaciones en las 80 series sanas**, y así llega cada falla:

| variable | falla | por defecto | con su perfil |
|---|---|---|---|
| agua (ruido 1,5%) | fuga +10% | ALERTA | **CRÍTICO** |
| | fuga +40% | CRÍTICO | CRÍTICO |
| | medidor en cero | CRÍTICO | CRÍTICO |
| | medidor congelado 7 días | CRÍTICO | CRÍTICO |
| | pico ×1,6 en un día | CRÍTICO | CRÍTICO |
| agua, cierra el domingo | fuga **en domingo** | ATENCION, guardada | **CRÍTICO** |
| valor constante (contrato) | baja −20% | CRÍTICO | CRÍTICO |
| | sube +3% | no avisa (bien) | no avisa |
| temperatura (cruza el cero) | +8 grados 4 días | ALERTA | ALERTA |

Perfil de consumo (agua, energía, gas) — cualquier cambio chico importa y un día cerrado con consumo
es una fuga:

```python
ajustes={"desvio_relativo_minimo": 0.03, "piso_sigma_relativo": 0.005,
         "critico_desvio_relativo_minimo": 0.05,
         "nivel_maximo": {"dia_cerrado": "CRITICO"}}
```

Si la fuente trae la **lectura acumulada** del medidor, no la vigiles así: una lectura acumulada sólo
sube. Calculá el consumo en el SQL (`LECTURA - LAG(LECTURA) OVER (PARTITION BY MEDIDOR ORDER BY FECHA)`).

| tipo de dato | ejemplo | `escala` | `desvio_relativo_minimo` | `piso_sigma_relativo` | detectores | grano |
|---|---|---|---|---|---|---|
| consumo continuo | agua, energía, gas | auto | 0,03 | 0,005 | todos, `dia_cerrado` hasta CRÍTICO | día |
| contador estable | stock, nómina | auto | 0,02–0,03 | 0,005 | escalon, hueco, tendencia | día |
| dinero diario ruidoso | venta por canal | auto | 0,25–0,40 | 0,15–0,25 | salto, escalon, racha | día + semana |
| ratio / porcentaje | tasa de devolución | lineal | 0,10–0,20 | 0,05 | escalon, salto | día + mes |
| puede ser negativa | margen, resultado | **lineal** | 0,10 | 0,05 | escalon, tendencia | mes |
| eventos raros | reclamos | auto | 0,30 | 0,20 | escalon, racha (**sin hueco**) | semana + mes |
| salud del ETL | filas cargadas | auto | 0,05 | 0,02 | **hueco** + escalon | día |

### Asuetos y días sin operación

Un día en que no se opera no se evalúa: un cero ahí es lo normal. El motor ya aprende solo, de la
historia, los días de la semana que casi siempre están en cero. Pero hay dos cosas que no puede saber:

1. **Los asuetos**, que caen en una fecha distinta cada año y son distintos por país.
2. **Los días que no deberían tener venta** en tiendas que igual abren algunos: si una tienda abre 6
   de cada 10 domingos, su domingo no parece cerrado, y cada domingo que cierra parece un apagón.

Sin calendario, la verificación igual evita casi todos los avisos falsos de asuetos: si la tienda
cierra ese día todos los años, no es raro para ella. Pero entonces **esa tienda "cierra seguido"**, y
un apagón real de un día, en un día hábil, pasa como normal. Con el calendario, los asuetos no
cuentan como cierres: medido con tiendas de dos países, ese apagón de un día **no se detectaba sin
calendario y sale CRÍTICO con él**. El asueto de un país no toca al otro. Y un asueto nuevo, que no
está en la historia, sólo se puede saber por el calendario.

**El calendario** es tu tabla de asuetos, en una consulta:

```python
CALENDARIO_SQL = """
    SELECT c.BD_PAIS, c.FECHA, c.BD_MOTIVO
      FROM CAL_ASUETOS c
     WHERE c.FECHA >= :desde AND c.FECHA < :hasta
"""
CALENDARIO_CONEXION = "origen"      # o "destino", según dónde esté la tabla
```

- `FECHA`: el día del asueto.
- Las columnas de alcance (`BD_PAIS`, o las que sean): a quién aplica. Un `*` o vacío vale para todos.
- `BD_MOTIVO` (opcional): aparece en la explicación, "asueto: Día de la Independencia (15-sep-2026)".
- `:desde` y `:hasta` los completa el pipeline con toda la historia que se analiza. Pasale la historia
  completa, no sólo los próximos: los asuetos pasados también se sacan de las medianas.

**En cada vigilancia**, cómo se une:

```python
Vigilancia(nombre="VENTA_TIENDA", ..., categorias=["BD_PAIS", "BD_TIENDA"],
           calendario_por=["BD_PAIS"],                      # cada tienda usa los asuetos de su país
           dias_sin_operacion=("domingo",),                 # para todas
           dias_sin_operacion_por={"SV": ["sabado", "domingo"]})   # distinto por país
```

`calendario_por` tiene que venir en el SQL de la vigilancia, como categoría o como columna extra.
`None` (el default) = esa vigilancia no usa el calendario. `[]` = todo el calendario vale para todas.
Los días aceptan nombre (`"domingo"`, `"sábado"`) o número (0 = lunes ... 6 = domingo).

Un día sin operación **con actividad** sí se ve: es el detector `dia_cerrado` (ATENCION por defecto).

### Semana y mes: días operados (`AJUSTAR_DIAS_OPERADOS`)

Una suma depende de cuántos días se operó. Una semana con asueto vende menos sin que nada ande mal, y
febrero tiene tres días menos que marzo. Por defecto, en semana y mes lo esperado se ajusta a los días
operados de cada período (sólo para `agregacion` `suma` y `conteo`: un promedio o un ratio no dependen
de cuántos días tuvo el período).

| con febrero evaluado, 2.000 series sanas | sin ajuste | con ajuste |
|---|---:|---:|
| avisos falsos en grano mes | 43 (38 de febrero) | **6** |

El resto del año el grano mes queda algo más sensible, porque la diferencia de largo entre meses ya no
se confunde con ruido: en el mismo panel, 12 avisos en vez de 8, por tendencias lentas que existen.

### Cómo se decide que algo es una anomalía

**Los detectores proponen, la verificación decide**, y decide como lo haría una persona mirando los
datos, con números que se pueden rehacer:

1. **Contra las mismas fechas anteriores.** Un día, contra los mismos días de la semana de las 8
   semanas anteriores. Un tramo de varios días, por su **total**, contra los 8 tramos anteriores de
   igual largo con los mismos días de la semana. Una semana, contra las 8 anteriores; un mes, contra
   los 12 anteriores. Los tramos de referencia no se pisan entre sí ni con el evento.
2. **Fuera de todo el rango.** Es anomalía sólo si queda por debajo del mínimo o por encima del máximo
   de esas referencias, y se aparta al menos `DESVIO_RELATIVO_MINIMO` de su mediana.
3. **Descontada la temporada.** Se hace la misma comparación en las mismas fechas de hace un año. Si
   hace un año también estuvo fuera de lo normal (diciembre contra noviembre, el otoño contra el
   verano), eso es temporada y se descuenta. Si hace un año estuvo normal, no hay nada que descontar.
4. **¿Le pasa seguido a esta serie?** Si en el último año tuvo más de `VECES_POR_ANIO` días con un
   desvío así (o tramos así de largos sin movimiento), no es raro **para ella**. Esto es lo que separa
   a una tienda errática, que cierra días sueltos, de una que nunca cierra.
5. **Tendencia**, por su cambio: cuánto cambió el último tramo contra el anterior, frente a cuánto
   venía cambiando de un tramo al siguiente. Un canal que venía creciendo y cae se ve ahí, aunque
   vuelva a un nivel que ya tuvo.

Y para avisar:

- **Sólo lo vigente**: lo que sigue pasando en los últimos 3 días (último mes o semana cerrados). Lo
  que terminó antes queda guardado en `VIG_EVENTO`, pero no se notifica.
- **Un evento es el mismo entre corridas** si sus fechas se pisan o se tocan con uno abierto de la
  misma serie y detector. No se vuelve a avisar, salvo que suba de nivel.
- **"Empeoró"** sólo si el evento ya estaba abierto y subió de nivel. Nunca en la primera corrida.
- Cuando varios detectores marcan lo mismo, **manda el de mayor nivel**; entre iguales, el más
  específico.

Medido con 300 tiendas de dos países, un tercio muy erráticas, corriendo 6 días seguidos:

| | antes | ahora |
|---|---:|---:|
| avisos | 147 | 9 |
| de fallas reales | 12 (8%) | 8 (89%) |
| sobre días viejos | 105 | 0 |

Y en el benchmark de 2.000 series de siempre, los avisos en series sanas bajaron de 11 a 4, con las
20 fallas detectadas igual.

Dos cosas que conviene saber:

- **Una serie chica no pasa de ATENCION** (`PESO_RELATIVO_MINIMO`: pesa menos de 0,2 veces la serie
  promedio de su vigilancia). En tiendas, una que vende la décima parte que el promedio. Si todas te
  importan igual, poné `PESO_RELATIVO_MINIMO = 0` en esa vigilancia con `ajustes`.
- **Un pico en una tienda errática no se avisa** si esa tienda tuvo más de un pico así en el año:
  para ella es normal. Subí `VECES_POR_ANIO` si querés ser más estricto, o bajalo a 0 para que sólo
  salga lo que nunca pasó en el año.

### Incremental: igual que recalcular todo

En modo incremental la vigilancia relee los últimos `DIAS_RELECTURA` días de la fuente y toma el resto
del historial guardado en `VIG_SERIE`. Tiene que dar **exactamente** lo mismo que leer todo, y está
probado así: dos corridas diarias en paralelo con la misma fuente, una incremental y otra completa,
comparando `VIG_SERIE`, `VIG_EVENTO` y `VIG_NOTIFICACION` después de cada día. Dan idéntico:

| escenario | resultado |
|---|---|
| 6 días seguidos, cruzando un cierre de semana | idéntico |
| 6 días seguidos, cruzando un cierre de mes | idéntico |
| 11 días y 70 días sin correr | idéntico |
| BI corrige un día de hace 3 semanas entre corridas | idéntico |

Cómo se logra:

- **La relectura arranca al inicio de la semana y del mes** donde cae. Antes arrancaba en un día
  cualquiera, y la semana y el mes de ese día se rearmaban sólo con los días releídos: julio quedaba en
  13.090 en vez de 509.298, cada día un poco peor, y eso generaba decenas de eventos falsos.
- **Un período releído a medias conserva el valor guardado**, que estaba completo.
- **El período más viejo de la historia, si vino a medias, no se usa** (en los dos modos): la
  comparación contra "la misma semana hace 3 años" caía justo en él.
- **El historial guardado se acota a `DIAS_HISTORIA`**; antes crecía todos los días.

Lo único que el incremental no ve, a propósito: una corrección de la fuente **más vieja que la
relectura** (por defecto, unos 45 días). Si BI recarga algo de hace meses, corré una vez con
`VG_MODO=completo`.

### Qué dice la explicación

Cada evento guarda en `BD_EXPLICACION` los mismos números que lo decidieron, y es lo que se manda:

```
[CRITICO] VENTA_TIENDA / SV | T151 (dia). El miércoles 16-sep-2026: no hubo movimiento
(0 USD). Los 8 miércoles anteriores: 14,307 (9-sep) · 10,533 (2-sep) · 12,079 (26-ago) ·
9,571 (19-ago) · 8,947 (12-ago) · 14,072 (5-ago) · 10,370 (29-jul) · 11,272 (22-jul).
Mediana 10,902, entre 8,947 y 14,307. Hace un año, en las mismas fechas (el miércoles
17-sep-2025), estuvo dentro de lo normal (10,915 contra 10,481): no es temporada. Ninguna de
esas referencias estuvo en cero y el año pasado en esas fechas sí hubo movimiento. En el último
año, esta serie nunca tuvo día sin movimiento fuera de sus días sin operación. En juego: 10,902
USD. Queda en CRITICO porque: desvío 31.8 veces su variación normal (ATENCION desde 3, ALERTA
desde 5, CRITICO desde 8). Suele ser una carga que no llegó, o un cierre que no está en el
calendario de asuetos: si fue asueto, agregalo al calendario y no vuelve a salir.
```

Un tramo de varios días se explica por su total, contra los mismos días de la semana en los tramos
anteriores de igual largo, con la fecha en que empieza cada uno. Todo se puede comprobar sumando esos
días en los datos.

La columna es nueva. Si tu `VIG_EVENTO` ya existe, el pipeline te pide:

```sql
ALTER TABLE VIG_EVENTO ADD (BD_EXPLICACION VARCHAR2(2000));
```

### Dos detectores para datos que no son venta

**`congelado`** — el dato dejó de actualizarse y repite exactamente el mismo valor. Es la falla
típica de cualquier telemetría (un medidor trabado, un ETL que copia el último valor) y ningún otro
detector la ve, porque el valor está en su nivel normal. Sólo avisa si en **esa** serie repetir es
raro: se estima de su propia historia la probabilidad de que un período repita el anterior. Un
contrato fijo repite siempre y nunca salta; un consumo con ruido casi nunca repite. Con los umbrales
por defecto: 3 períodos iguales es ATENCION, 5 es ALERTA, 8 es CRÍTICO. Los ceros los ve `hueco`.

**`dia_cerrado`** — actividad en un día que normalmente está cerrado. Los días cerrados no se evalúan
con los demás detectores (un cero ahí es lo normal), pero lo contrario sí importa: consumo en un día
sin operación. Llega como máximo a ATENCION por defecto, porque en ventas es alguien que abrió un
domingo; en consumo se sube con `nivel_maximo`.

### Qué hace que algo sea CRÍTICO

La calibración ajusta **cuántas** alertas salen. No mira cuánto se movió cada cosa ni si el dato
tiene sentido, así que no alcanza para que un CRÍTICO valga la pena. Eso lo deciden cuatro reglas:

1. **Un día cerrado no es un apagón.** Un día de la semana cuya mediana no llega al 5% del nivel de
   la serie (el domingo de un B2B) está cerrado: sus valores no se evalúan, y un evento que lo cruza
   no se corta. Antes cada domingo cerrado era un CRÍTICO "hueco": en un panel de 2.000 series con un
   tercio cerrando los domingos, eran **2.523 críticos falsos de 2.570**. Ahora son 0.
2. **Raro no es lo mismo que grave.** Una serie muy pareja da un desvío estadístico enorme con un 3%
   de diferencia. Para ser CRÍTICO hace falta apartarse al menos `CRITICO_DESVIO_RELATIVO_MINIMO` (20%)
   de lo esperado; si no, queda en ALERTA. En una métrica muy estable, bajalo con `ajustes`.
3. **"En juego" cuadra con el evento.** Es la diferencia acumulada contra lo esperado **en los
   períodos del propio evento**. Antes, al unificar, el principal heredaba la materialidad más grande
   de los detectores que lo confirmaban, de otro tramo y con otro esperado.
4. **Un valor imposible se marca.** Si algo vale 20 veces lo esperado en un período, casi nunca es
   negocio: es una carga duplicada, un acumulado anual cargado en un día o un error de unidades. Sigue
   siendo CRÍTICO — hay que verlo —, pero el mensaje empieza con *"OJO: posible error de carga"*.

El mensaje dice además **por qué** tiene ese nivel (`Por qué CRITICO: desvío 37.8; posible error
de dato: 341 veces lo esperado`), igual que `BD_MOTIVO_NIVEL`.

### Las fechas de un evento

En grano semana, la fecha es el **lunes** de la semana. Antes se mostraba el jueves anterior (el
1-1-1970, desde donde se cuentan los días, fue jueves); el período guardado siempre fue el correcto.

Un evento es un **rango**, no un punto: `FECHA_INICIO`, `FECHA_FIN` y `MT_PERIODOS`. `MT_OBSERVADO`
y `MT_ESPERADO` son del **último período**, o sea de `FECHA_FIN`. La explicación dice las dos cosas en
palabras: "del domingo 20-sep-2026 al martes 22-sep-2026 (3 días evaluados) [...] cuando lo normal para
un martes [...] es 8,467" (ver *Qué dice la explicación*, más arriba).

`MT_ESPERADO` es siempre un **nivel** comparable con `MT_OBSERVADO`, en cualquier detector. En
`tendencia` es el nivel que daría la recta de la ventana anterior proyectada hasta ese período; en
`nueva` es **nulo**, porque una serie que aparece por primera vez no tiene referencia.

Variables de entorno: `VG_FECHA_EJECUCION`, `VG_MODO`, `VG_DRY_RUN`, `VG_WEBHOOK_URL`.
Para calibrar: `VG_OBJETIVO`, `VG_REJILLA`, `VG_N_FALLAS` en `calibrar_vigilancia.ipynb`.

---

## `MIN_BASE_PORCENTAJE` — un porcentaje necesita una base

**Esto no es precisión, es materialidad, y son dos problemas distintos.** Un cliente con tres
registros —0, 0 y 0,0000064— tiene una venta de `6,449116e-06` y un margen de `-25,04`. Los dos
números son **reales y correctos**. El porcentaje que sale de ahí, no:

```
-25,04219120618 / 0,000006449116 = -388.304.245 %
```

No hay nada que arreglar en la suma: el problema es que un porcentaje sobre una base de seis
millonésimos no significa nada. Por eso hay un mínimo explícito, en la misma moneda que la venta:

| Perilla | Qué hace | Default |
|---|---|---|
| `MIN_BASE_PORCENTAJE` | venta mínima para que se calcule un porcentaje | 1,0 |

Por debajo de esa base:

- **Estadísticas**: `MT_MARGENBRUTO` y `MT_MARGENBRUTO_R12` salen **nulos**, y el mes no entra en
  la recta de `MT_IDDPORCENTUALMARGEN`. La venta y el margen se siguen informando tal cual: lo
  único que se suprime es el porcentaje. La corrida dice cuántos fueron.
- **Recomendación**: el ítem no tiene margen % y la entidad no reparte participaciones.
- **Vigilancia**: el mínimo va en cada `Vigilancia` (`min_denominador`), porque cada métrica tiene
  su propia unidad, y por defecto es 0. El período por debajo queda nulo, como un hueco de la serie.

`0` lo desactiva. Subilo si querés ser más estricto: con `MIN_BASE_PORCENTAJE = 100`, un cliente con
4 USD de venta tampoco reporta margen %.

---

## `SUMA_EXACTA` — por qué una venta de cero no daba cero

Los tres motores lo tienen, con los mismos valores por defecto y el mismo significado.

### El problema

Sumar en punto flotante una venta y su devolución **no da cero exacto**. Cada importe decimal se
guarda redondeado en binario y, al acumular, queda un residuo del orden de 1e-16 veces lo más grande
que pasó por la suma. Con 300.000 USD de venta y 300.000 de devolución, el neto puede quedar en
`-4,7e-10` en lugar de `0`. Como número no molesta. Como **denominador de un porcentaje**, explota:

```
margen bruto = -0,01 / -4,7e-10 = 2.100.000.000 %
```

No es cosa de Python: es IEEE 754, el mismo formato binario de Java, C, Excel y el `BINARY_DOUBLE`
de Oracle. `0.1 + 0.2` da `0.30000000000000004` en todos. El `SUM` del SQL sí da cero porque
**Oracle NUMBER es decimal y exacto**; el problema aparece recién cuando los datos entran a Python.

### La solución: sumar en centavos

`SUMA_EXACTA = True` hace lo mismo que Oracle por dentro: detecta la escala decimal de tus importes
(1.234,56 son 123.456 centavos), suma **enteros** y divide al final. Los enteros se suman sin ningún
error en float64 mientras no pasen de 2^53, así que la venta que se anula con su devolución da cero
**exacto**, sin umbrales ni tolerancias. Cuesta lo mismo: es el mismo `bincount`.

De paso, los números salen limpios: un neto real de 1 USD da `1.0`, no `0.9999999998`, y un margen
del 30% da `30.0`, no `30.000000009749783`.

| Perilla | Qué hace | Default |
|---|---|---|
| `SUMA_EXACTA` | sumar el dinero en su escala decimal | `True` |
| `MAX_DECIMALES` | hasta cuántos decimales busca la escala; más allá redondea | 6 |

La escala se detecta sola de tus datos: se usa la más chica que deje todos los importes enteros. El
tope es 2^53 centavos ≈ 90 billones de USD; si tu bruto lo desbordara, el motor baja de decimales o,
si no alcanza, avisa y pasa al plan B. Cada corrida lo dice en el log:

```
dinero en escala de 2 decimales: las sumas son exactas
```

### Por qué no alcanzaba con sumar "bien"

Un algoritmo de suma exacta (Kahan, `math.fsum`) **no resuelve el caso**:

```
2.559,60 + 4.752,37 - 7.311,97
   float64   : -9,09e-13
   math.fsum : -4,55e-13   <- suma perfecta y aun así no da 0
```

Porque los valores **guardados** ya no se anulan entre sí: el error está en el redondeo binario de
cada uno, no en cómo se suman. Y el módulo `decimal` de Python sí es exacto, pero es 331 veces más
lento: inviable con 7 millones de filas. Por eso, centavos.

### Plan B: `TOLERANCIA_CERO`

Sólo entra si no hay escala decimal usable. Ahí el cero se prueba contra la escala real de lo que se
sumó —la suma de los valores absolutos, con la escala típica del panel como piso:

| Suma | Lo que pasó por ella | Razón | Veredicto |
|---|---|---|---|
| -4,7e-10 | 600.000 | 7,8e-17 | es cero |
| -0,01 | 600.000 | 1,7e-8 | es un neto real, se respeta |

`1e-9` equivale a un centavo en diez millones. `0` lo desactiva. Es una heurística, no una
demostración, y por eso es el plan B y no el principal.

### Lo que NO resuelve

La suma exacta arregla el **cero que no daba cero**. No arregla una base que es chica **de verdad**:
`6,449116e-06` redondeado a 6 decimales sigue sin ser cero, y su porcentaje sigue siendo de cientos
de millones. Para eso está `MIN_BASE_PORCENTAJE`, más arriba. Son dos reglas distintas porque son
dos problemas distintos.

### Qué protege, en cada motor

- **Estadísticas**: `MT_MARGENBRUTO` y `MT_MARGENBRUTO_R12` quedan **nulos** en lugar de dar millones;
  `MT_VOLUMENCOMPRA` y las ventas por ventana salen en `0,00` y no en `-1,8e-11`; los meses que se
  anulan no entran en `MT_IDDPORCENTUALMARGEN`.
- **Recomendación**: el ítem que se vende y se devuelve entero no tiene margen % gigante (daba
  −343.597.384), y la entidad cuya compra neta es cero no participa del promedio de participaciones
  de sus pares.
- **Vigilancia**: un ratio cuyo denominador se anula da **nulo** —un hueco de la serie— en lugar de
  un pico de 1,3e11 que llega al webhook como alerta crítica.

---

## Qué decide el sistema y qué decidís vos

**Se elige solo, con backtest o con reglas de suficiencia de datos:** el algoritmo de recomendación
por segmento, el nivel de segmento de cada entidad, los parámetros del modelo de actividad, la escala
lineal o logarítmica, qué detectores corren en cada grano, la gravedad de cada evento y el ganador del
panel cuando un segmento no tiene con qué medirse.

**Lo calibrás vos con los dos notebooks, sobre tus datos:** el nivel de ítem, los mínimos de soporte y
penetración, canasta o repertorio, y los umbrales de alerta.

**No se puede automatizar, porque es criterio del negocio:** cuántas alertas por día tolera quien las
recibe, qué métricas vale la pena vigilar, y qué tipos de documento son venta, devolución o ruido.
