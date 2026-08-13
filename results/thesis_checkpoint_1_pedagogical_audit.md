# Checkpoint 1: auditoria pedagogica y metodologica de la tesis

Fuentes leidas: summaries en `results/`, configs locales de `subtype_classifier/`
y `prediction_sequence/`, y scripts principales de auditoria, geometria,
temporalidad local, dinamica PCA y scoring condicional. No se ejecutaron
experimentos, no se recalcularon embeddings, no se cargo AntigenLM, no se
generaron secuencias y no se modificaron scripts ni archivos `.tex`.

## 1. Resumen en lenguaje claro

1. Ya no estamos en una intuicion vaga: hay un checkpoint local de AntigenLM que carga y produce embeddings/logits sin errores basicos.
2. El resultado mas solido es geometrico: las distancias latentes se parecen mucho mas a distancias moleculares HA/HA+NA que a distancia temporal global.
3. Eso significa que el espacio latente parece guardar informacion biologica real, no solo fechas.
4. PCA muestra que gran parte de la variacion vive en pocas direcciones; TwoNN sugiere algo parecido, pero con mas cautela.
5. Aunque la correlacion temporal global es debil, los vecinos cercanos en el espacio latente suelen estar cerca en meses.
6. Esa temporalidad local persiste despues de quitar duplicados exactos HA+NA.
7. Sobre centroides mensuales en PCA, H3N2 muestra una señal de dinamica lineal mas clara que H1N1.
8. H1N1 parece mas cercano a persistence/random walk, aunque no hay que convertir eso en dogma porque seed7 mostro un drift modesto.
9. La SDE lineal minima es una formulacion puente razonable, no una SDE final.
10. Todavia no hemos reproducido formalmente Figure 3A ni el pipeline oficial de forecasting de AntigenLM.
11. El clasificador de subtipo esta cableado de forma plausible, pero falta label map oficial y pooling oficial.
12. El scoring de `prediction_sequence` muestra señal parcial, no reproduccion completa.
13. La generacion libre no debe ser la replica principal.
14. La SDE queda motivada, no demostrada.
15. El trabajo ya tiene una Etapa 1 defendible: auditoria geometrica y dinamica minima local, separada de la reproduccion oficial del paper.

## 2. Mapa mental del proyecto

| capa | que entra | que sale | pregunta que responde | depende de | confianza |
|---|---|---|---|---|---|
| 1. Datos | GISAID procesado H1N1/H3N2, HA, NA, fechas | Registros por subtipo y mes | Que material biologico y temporal se esta analizando | `preprocess_gisaid.py`, datasets procesados, calidad de metadata | Alta para auditoria local; no equivale a dataset oficial del paper |
| 2. Tokenizer/checkpoints | Carpetas `prediction_sequence/` y `subtype_classifier/` | Modelos que cargan, vocabularios, heads detectadas | Los pesos locales son utilizables tecnicamente | `audit_antigenlm_checkpoints.py`, configs, tokenizer files | Alta para carga/forward; media para protocolo oficial |
| 3. Embeddings | Secuencias HA+NA codificadas y checkpoint local | Embeddings `(n,384)` cacheados | Que representacion continua produce el checkpoint local | Tokenizer, modelo, pooling usado para embeddings | Alta para cache local; condicionada por eleccion de mean pooling |
| 4. Geometria latente | Embeddings, fechas, secuencias | Spearman temporal/Hamming | El espacio preserva similitud biologica | Cache de embeddings, muestreo estratificado, distancia Hamming | Alta para checkpoint local |
| 5. Temporalidad local | Embeddings, fechas, kNN | Distancias temporales vecino vs random | Vecinos latentes estan cerca en tiempo | Cache, deduplicacion, k elegido, densidad de muestreo | Alta para diagnostico local deduplicado |
| 6. PCA/TwoNN | Embeddings cacheados | Varianza explicada, participation ratio, dimension TwoNN | El espacio parece reducible a baja dimension | Cache, normalizacion, deduplicacion, trim | PCA alta; TwoNN media |
| 7. Dinamica minima en PCA | Centroides mensuales PCA | RMSE/MAE de persistence, RW, VAR | Hay señal dinamica simple por subtipo | PCA train-only rolling-origin, centroides mensuales | Media: util y bien controlada, pero no predice secuencias |
| 8. SDE lineal minima | Modelos gaussianos discretos | Formulacion `dx = drift dt + Sigma dW` | Como conectar VAR/RW con una SDE inicial | Resultados PCA, covarianza residual, calibracion | Media-baja como resultado; alta como formulacion metodologica |
| 9. Reproduccion AntigenLM | Checkpoints, tokenizer, protocolo paper | Figure 3A u otras metricas oficiales | Nuestro pipeline reproduce el paper | Informacion oficial de autores, label map, pooling, target, contexto | Gris/pendiente |
| 10. AntigenSDE futuro | Geometria + dinamica + protocolo | SDE con viabilidad/escape y evaluacion prospectiva | Un modelo estocastico mejora forecasting biologico | Todo lo anterior + baselines + validacion prospectiva | Rojo/pendiente como afirmacion fuerte |

## 3. Semaforo de confianza

| resultado | semaforo | lectura honesta |
|---|---|---|
| Spearman Hamming HA/HA+NA | VERDE | Solido para el checkpoint local: Hamming HA promedio 0.6338 y Hamming HA+NA promedio 0.6586 indican preservacion molecular fuerte. No depende de Figure 3A. |
| Spearman temporal | VERDE | Solido como diagnostico negativo-local: rho promedio 0.1436 muestra que el espacio no sigue una linea temporal global simple. No es un fracaso; solo descarta una lectura cronologica lineal. |
| PCA effective dimension | VERDE | Solido para el cache local: 3 PCs explican 90%, 4 explican 95%, 12 explican 99%, PR global 1.89. La interpretacion causal/dinamica sigue siendo limitada. |
| TwoNN | AMARILLO | Util y corregido tras deduplicacion; sugiere dimension 3.5-4.8 con R2 alto, pero es sensible a trimming, duplicados y normalizacion. |
| Temporalidad local deduplicada | VERDE | Resultado fuerte local: H1N1 k=5 mediana 2 meses vs random 58; H3N2 k=5 mediana 3 vs random 72. No prueba SDE, pero si vecindades evolutivamente coherentes. |
| Dinamica PCA rolling-origin | AMARILLO | Metodologicamente util: PCA se ajusta solo con train por corte. H3N2 muestra drift lineal; H1N1 no. Sigue operando sobre centroides, no secuencias. |
| SDE lineal minima | AMARILLO | Buena formulacion puente. No es resultado biologico final ni incluye `F_viab/F_escape`. |
| `subtype_classifier` parcial | AMARILLO | El checkpoint carga y separa H1N1/H3N2 localmente con accuracy hasta 0.875 usando mean pooling. Falta label map oficial, pooling oficial y evaluacion de 12 clases. |
| `prediction_sequence` scoring | ROJO para afirmacion fuerte; AMARILLO como smoke test | Hay señal parcial de ranking, pero top-1 es bajo y la sensibilidad es inestable. No reproduce Figure 3A. |
| Reproduccion Figure 3A | GRIS | Pendiente de protocolo oficial: contexto, target dominante, formato, scoring/generacion, restricciones y metrica exacta. |
| Validacion 2022-2026 | ROJO | No ejecutada. No debe aparecer como resultado, solo como oportunidad/protocolo futuro. |

## 4. Que sigue siendo valido aunque no hayamos replicado el paper

Estos resultados no dependen de haber reproducido Figure 3A:

- La auditoria de carga local de checkpoints: sabemos que los pesos locales cargan y hacen forward minimo.
- La geometria del checkpoint local: Spearman con Hamming HA/HA+NA y temporalidad global.
- La baja dimension efectiva por PCA sobre embeddings cacheados.
- La estimacion preliminar TwoNN, con sus cautelas.
- La temporalidad local deduplicada: vecinos latentes cercanos son temporalmente cercanos.
- La dinamica minima sobre centroides PCA, siempre descrita como dinamica de embeddings locales, no como forecasting oficial.
- La formulacion de SDE lineal minima como puente conceptual.

Esto es importante: aunque Figure 3A siga pendiente, la tesis ya tiene una contribucion local defendible sobre geometria latente. La palabra clave es "checkpoint local": no estamos afirmando que todos los resultados del paper esten reproducidos, sino que el espacio latente disponible tiene propiedades medibles y biologicamente interesantes.

## 5. Que si depende de reproducir AntigenLM

No podemos afirmar todavia:

- "Replicamos el paper".
- "Reprodujimos Figure 3A".
- "AntigenLM forecasting funciona igual que en el paper".
- "Nuestro pipeline de generacion es el pipeline oficial".
- "La generacion libre replica AntigenLM".
- "El scoring condicional local es equivalente al metodo de los autores".
- "AntigenSDE supera AntigenLM".
- "La validacion prospectiva 2022-2026 ya confirma el modelo".

Para esas afirmaciones falta al menos: protocolo oficial de contexto/target, label map o metadatos de clasificacion si aplica, pooling oficial para heads, definicion exacta de dominante mensual, politica de candidatas o generacion, metrica usada en Figure 3A y baselines comparables.

## 6. Auditoria de coherencia

### `summary_type="cls_index"`

Ambos configs (`subtype_classifier` y `prediction_sequence`) conservan campos tipo GPT-2 como `summary_type="cls_index"`, `summary_use_proj=true` y `summary_proj_to_labels=true`. Eso no prueba que nuestro wrapper local este usando exactamente el pooling oficial. En el `subtype_classifier`, la cabeza real `classification_head` existe, pero el resumen mostro que `last`, `mean`, `<HA>` y `<NA>` son tecnicamente plausibles, y que `mean` fue mejor localmente. Falta confirmar el pooling oficial.

### `vocab_size=10` vs `vocab_size=25`

Esto no es necesariamente contradiccion.

- `subtype_classifier` tiene vocabulario de 10 tokens: nucleotidos y marcadores estructurales (`A,C,G,T,N,<pad>,<sep>,<eos>,<HA>,<NA>`). No incluye tokens de subtipo como entrada.
- `prediction_sequence` tiene `vocab_size=25` porque usa el vocabulario base de 10 mas 15 tokens de subtipo en `added_tokens.json`.

La confusion posible es mezclar "tokens de entrada" con "clases de salida". El clasificador tiene 12 logits de salida, pero esas 12 clases no son tokens del vocabulario. Necesitan un label map aparte.

### Vocab/tokenizer vs label map

El vocabulario dice como convertir texto/secuencia en IDs de entrada. El label map dice que significa cada posicion de la cabeza de clasificacion. Que `classification_head.weight` tenga shape `(12,384)` solo dice que hay 12 clases; no dice cual logit es H1N1, H3N2, etc. El mapeo local `H1N1 -> 2`, `H3N2 -> 5` es una inferencia de auditoria, no un mapa oficial.

### Mean pooling vs posible `cls_index`

Los embeddings de geometria usan mean pooling de hidden states. Eso es una decision metodologica razonable para representar secuencias completas, pero no equivale necesariamente al pooling usado por la cabeza de clasificacion. El config menciona `cls_index`, pero en nuestros inputs no hay un token CLS explicito. Por tanto:

- mean pooling para geometria: defendible como representacion local.
- mean pooling para clasificador oficial: no confirmado.

### Ausencia de label map

La ausencia de label map impide convertir el `subtype_classifier` parcial en reproduccion formal. La prueba local de H1N1/H3N2 demuestra que hay señal y que el checkpoint no esta muerto, pero no valida las 12 clases ni el orden oficial de logits.

### Fragilidad del scoring condicional

`prediction_sequence` scoring es tecnicamente mas cercano a forecasting que la generacion libre, pero sigue fragil:

- usa contexto acotado;
- puntua prefijos de continuacion;
- top-1 es bajo;
- top-5 es variable;
- la sensibilidad cambia con contexto y tokens puntuados;
- no sabemos si el paper usa scoring, generacion, restricciones o un pipeline mixto.

Por tanto, sirve como smoke test y como direccion metodologica, no como replica.

### Sobreinterpretaciones a vigilar

- "Baja dimension" no significa automaticamente "SDE valida".
- "Temporalidad local" no significa "prediccion de cepas".
- "H3N2 mejora con VAR(2)" no significa "AntigenSDE ya funciona".
- "H1N1 parece random walk" no significa "H1N1 no tiene drift biologico"; solo que este resumen mensual en PCA no lo captura robustamente.
- "Checkpoint local carga" no significa "pipeline oficial reproducido".

## 7. Explicacion pedagogica de conceptos clave

### Diferencia entre vocab/tokenizer y label map

El tokenizer es el diccionario para leer la entrada: por ejemplo, `A -> 0`, `<HA> -> 8`. El label map es el diccionario para interpretar salidas de clasificacion: por ejemplo, "logit 2 significa H1N1". Puedes tener un vocabulario de 10 tokens y una cabeza de 12 clases sin contradiccion, porque son espacios distintos.

### Diferencia entre checkpoint local y reproduccion del paper

Un checkpoint local que carga bien demuestra que tienes pesos utilizables. Reproducir el paper exige replicar el protocolo: mismo formato de entrada, misma definicion de target, misma metrica, mismo periodo, mismas decisiones de decoding/scoring y mismos baselines.

### Por que Hamming fuerte si importa

Hamming fuerte significa que secuencias molecularmente parecidas quedan cerca en el espacio latente. Eso es exactamente lo minimo que uno quiere antes de usar distancias latentes para hablar de evolucion, vecindad o escape.

### Por que temporalidad global debil no es necesariamente mala

La influenza no avanza como una particula sobre una linea de tiempo ordenada. Puede haber ramas, reintroducciones, clados coexistentes y sesgo de muestreo. Una correlacion temporal global debil dice que "mas separado en fecha" no siempre significa "mas lejos en latente". Eso puede ser realista.

### Que significa temporalidad local

Significa que, aunque el espacio no sea una linea cronologica global, los vecinos cercanos de una cepa o centroide tienden a estar cerca en meses. Es una propiedad local: cerca de un punto, la geometria parece compatible con continuidad evolutiva.

### Que significa PCA n95=4

Significa que cuatro componentes principales explican 95% de la varianza global de los embeddings. No significa que la biologia tenga cuatro dimensiones, sino que la nube de embeddings esta muy comprimida/anistropica en terminos lineales.

### Por que TwoNN no es lo mismo que PCA

PCA mide cuantas direcciones lineales explican varianza. TwoNN intenta estimar dimension intrinseca local usando vecinos cercanos. Si ambos sugieren baja dimension, la evidencia es mas interesante; si discrepan, puede haber curvatura, densidad desigual o problemas de duplicados.

### Que significa predecir centroides

Significa predecir el promedio mensual de embeddings de un subtipo. Es como seguir el "centro de masa" de la poblacion observada en el espacio latente.

### Por que predecir centroides no es lo mismo que predecir mutaciones

Un centroide no es una cepa real. Puede moverse suavemente aunque las secuencias individuales cambien de forma discreta. Predecir centroides evalua dinamica poblacional agregada, no mutaciones especificas ni una HA/NA futura.

### Como una distribucion latente futura podria rankear candidatas reales

Si una SDE produce una distribucion para `x_{t+1}`, entonces cada candidata real del mes futuro puede proyectarse al mismo espacio y recibir una probabilidad o distancia probabilistica. Asi se puede rankear candidatas reales sin generar secuencias nuevas. Esta ruta es mas defendible inicialmente que decodificar libremente.

## 8. Decisiones recomendadas

### A. Congelar

Congelar Checkpoint 1 como estado local de verdad: geometria, PCA/TwoNN, temporalidad local, dinamica PCA y formulacion SDE minima. La razon es evitar mejora circular: ya hay suficiente para una contribucion metodologica parcial.

### B. Documentar

Documentar claramente la separacion entre "resultados del checkpoint local" y "reproduccion oficial del paper". Esta distincion protege la tesis: lo local es valioso, pero no debe disfrazarse de reproduccion completa.

### C. Preguntar a autores

Preguntar a autores por cuatro cosas concretas: label map del `subtype_classifier`, pooling oficial, protocolo de `prediction_sequence` para Figure 3A y definicion exacta de target dominante. Esto puede resolver incertidumbre que no se arregla con mas codigo.

### D. Reproducir minimo

Reproducir minimo solo despues de obtener o fijar protocolo: una version pequena de Figure 3A o scoring equivalente, con ventanas limitadas y metrica definida. No escalar hasta cerrar formato y target.

### E. No tocar por ahora

No tocar por ahora generacion libre, SDE con `F_viab/F_escape`, validacion 2022-2026 ni comparaciones contra AntigenSDE. Son pasos tentadores, pero ahora aumentarian la complejidad antes de cerrar el checkpoint.

## 9. Preguntas mayeuticas para ti

1. Que significa exactamente que la distancia latente preserve Hamming HA/HA+NA?
2. Por que una correlacion temporal global debil no destruye la tesis?
3. Que diferencia hay entre "vecinos temporales locales" y "trayectoria cronologica global"?
4. Por que el vocabulario del tokenizer no es lo mismo que el label map del clasificador?
5. Que afirmacion puedes defender aunque Figure 3A siga pendiente?
6. Que objeto quieres predecir realmente: centroide, distribucion, ranking, secuencia, clado o dominante mensual?
7. Por que predecir centroides mensuales en PCA no equivale a predecir mutaciones HA/NA?
8. Por que H3N2 parece mejor candidato para drift lineal que H1N1 en tus resultados actuales?
9. Como usarias una distribucion latente futura para rankear candidatas reales sin generar secuencias?
10. Que evidencia minima te haria decir "ahora si vale la pena entrenar una SDE con `F_viab` y `F_escape`"?

## 10. Proxima accion unica

Mi recomendacion es congelar Checkpoint 1 y enviar una pregunta tecnica breve a los autores antes de escalar `prediction_sequence`: pedir label map, pooling oficial y protocolo exacto de Figure 3A.

