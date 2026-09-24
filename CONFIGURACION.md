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
| `MIN_BASE_PORCENTAJE` | base mínima para calcular un porcentaje (ver abajo) | 1,0 |
| `SUMA_EXACTA`, `MAX_DECIMALES`, `TOLERANCIA_CERO` | cómo se suma el dinero (ver abajo) | True / 6 / 1e-9 |

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
| `DETECTORES` | hueco, salto, escalon, tendencia, estacional, racha, nueva, congelado, dia_cerrado | todos |
| `NIVEL_MAXIMO` | tope de nivel por detector | `{"dia_cerrado": "ATENCION"}` |
| `CALENDARIO_SQL`, `CALENDARIO_CONEXION` | tu historia de asuetos, y en qué base está (ver abajo) | `None` / origen |
| `UMBRAL_Z` | desvíos robustos que definen ATENCION / ALERTA / CRITICO | 3 / 5 / 8 |
| `DESVIO_RELATIVO_MINIMO` | diferencia mínima contra lo esperado para que sea un evento | 5% |
| `PISO_SIGMA_RELATIVO` | el ruido nunca se considera menor a esto del nivel | 2% |
| `CRITICO_DESVIO_RELATIVO_MINIMO` | para ser CRÍTICO, apartarse al menos esto de lo esperado | 20% |
| `FACTOR_SOSPECHA_DATO` | tantas veces lo esperado = posible error de carga | 20 |
| `DIA_CERRADO_RELATIVO` | día de la semana casi siempre en cero = cerrado, no se evalúa | 5% |
| `PESO_RELATIVO_MINIMO` | serie que pesa menos que esto × la serie promedio no pasa de ATENCION | 0,2 |
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

Medido con tiendas de dos países de asuetos distintos: sin calendario, **56 notificaciones, 31
críticos en asuetos y 22 en domingos**; con calendario, **1 notificación: la caída real**, que sigue
llegando como CRÍTICO. El asueto de un país no toca al otro.

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

### Qué dice la explicación

Cada evento guarda en `BD_EXPLICACION` la razón exacta, y es lo que se manda como mensaje:

```
[CRITICO] VENTA_TIENDA / GT | GT_CAIDA (dia). Del domingo 20-sep-2026 al martes 22-sep-2026
(3 días evaluados): no hubo movimiento (0 USD) cuando lo normal para un martes (la mediana de
los últimos 91 días, comparando cada día con los de su mismo día de la semana) es 8,467. Esta
serie se mueve normalmente ±11% de un día a otro. Este cambio es 35.4 veces esa variación.
En juego: 21,385 USD (la diferencia acumulada contra lo esperado). Queda en CRITICO porque:
desvío 35.4 veces su variación normal (ATENCION desde 3, ALERTA desde 5, CRITICO desde 8);
lleva 3 períodos seguidos: sube un nivel. Suele ser una carga que no llegó, o un cierre que
no está en el calendario de asuetos: si fue asueto, agregalo al calendario y no vuelve a salir.
```

- **Contra qué exactamente** se comparó (la mediana de qué días, ajustada cómo).
- **Cuánto se mueve normalmente** esa serie, y cuántas veces eso fue el cambio. Es la misma escala de
  los umbrales: CRÍTICO es "8 veces su variación normal".
- **Qué días no se evaluaron** y por qué, con el nombre del asueto; en semana y mes, cuántos días se
  operaron contra los habituales.
- **Por qué ese nivel**, con los umbrales a la vista y cada regla que lo subió o lo bajó.
- **Qué suele significar** ese tipo de evento, y qué revisar.

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
