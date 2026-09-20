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
| `ALGORITMOS` | la batería que se mide | los 6 |
| `SELECCION` | `backtest` (gana el mejor por segmento), `rrf`, `ponderado`, o el nombre de uno | `backtest` |
| `METRICA_SELECCION` | `precision`, `usd` o `recall` | `precision` |
| `TIPOS_RECOMENDACION` | CRUZADA, REPOSICION, BRECHA | las tres |
| `FACTOR_REPOSICION` | silencio mayor a esto × su intervalo típico = atrasado | 1,5 |
| `BRECHA_RATIO` | compra menos de esta fracción de lo que le dedican sus pares | 0,5 |

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

La evidencia queda en la tabla: `MT_DIAS_COMPRA_ITEM` (cuántas veces lo compró), `MT_INTERVALO_TIPICO`
(su propio ritmo, vacío si no tiene), `MT_INTERVALO_ESPERADO` (el estimado, mezclando con los pares),
`MT_COMPRAS_ESPERADAS`, `MT_PROB`, `MT_USD_SI_COMPRA` y `MT_DIAS_COMPRA_ENTIDAD`.

Sale **una fila por cliente e ítem**: si un ítem califica como reposición y como brecha, queda el tipo
que mejor lo explica.

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
| `DETECTORES` | hueco, salto, escalon, tendencia, estacional, racha, nueva | todos |
| `UMBRAL_Z` | desvíos robustos que definen ATENCION / ALERTA / CRITICO | 3 / 5 / 8 |
| `DESVIO_RELATIVO_MINIMO` | diferencia mínima contra lo esperado para que sea un evento | 5% |
| `PISO_SIGMA_RELATIVO` | el ruido nunca se considera menor a esto del nivel | 2% |
| `PESO_RELATIVO_MINIMO` | serie que pesa menos que esto × la serie promedio no pasa de ATENCION | 0,2 |
| `PERSISTENCIA_SUBE_NIVEL` | períodos seguidos que suben un nivel (sólo detectores de punto) | 3 |
| `NIVEL_MINIMO_EVENTO` | desde qué nivel se guarda un evento (`INFO` = todo) | ATENCION |
| `NIVEL_NOTIFICACION` | desde qué nivel se notifica | ALERTA |
| `CANALES_NOTIFICACION` | `tabla` siempre; sumá `webhook`, `correo`, `teams` | `("tabla",)` |

En el motor hay además `detectores_excluidos` (por defecto, `estacional` no corre en grano día),
`desestacionalizar_dia`, `duracion_minima` por detector, `prioridad_detectores` y `unificar_eventos`.

Variables de entorno: `VG_FECHA_EJECUCION`, `VG_MODO`, `VG_DRY_RUN`, `VG_WEBHOOK_URL`.
Para calibrar: `VG_OBJETIVO`, `VG_REJILLA`, `VG_N_FALLAS` en `calibrar_vigilancia.ipynb`.

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
