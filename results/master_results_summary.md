# Resumen maestro de resultados

Fuente: resúmenes existentes en `results/`. No se ejecutaron experimentos nuevos,
no se recalcularon embeddings, no se cargó AntigenLM y no se generaron secuencias
para preparar este documento.

## 1. Resumen ejecutivo

El espacio latente de AntigenLM preserva de forma clara la similitud molecular
de HA y HA+NA: Spearman con Hamming HA alcanza rho promedio 0.6338, y con
Hamming HA+NA alcanza rho promedio 0.6586 en la corrida principal.
En contraste, la correlación con distancia temporal global es débil
(rho promedio 0.1436), lo que sugiere que el espacio no codifica simplemente
orden cronológico lineal.
La temporalidad local sí es fuerte: vecinos cercanos en el espacio latente
están separados por pocos meses, mientras que pares aleatorios del mismo
subtipo están separados por decenas de meses.
Esta señal local persiste después de deduplicar secuencias HA+NA exactas.
PCA y TwoNN sugieren baja dimensión efectiva: PCA alcanza 90% de varianza
global con 3 componentes y 95% con 4; TwoNN sugiere un rango preliminar
aproximado de dimensión intrínseca 3.5-4.8.
La dinámica en PCA space es subtipo-dependiente.
H3N2 muestra señal reproducible de drift lineal: VAR(2) mejora persistence
en seed42 y la señal se mantiene en seed7, aunque VAR(1) gana RMSE en esa
corrida.
H1N1 es más compatible con persistence/random walk o drift débil; seed7
introduce una mejora modesta de VAR(2), por lo que no debe afirmarse que sea
puramente random walk.
La SDE queda motivada, no demostrada.

## 2. Tabla maestra

| bloque experimental | pregunta | método | datos/cache | resultado principal | interpretación | estado | figura/resumen asociado |
|---|---|---|---|---|---|---|---|
| Spearman temporal | ¿La distancia latente sigue el orden temporal global? | Spearman entre distancia latente y diferencia temporal | `embeddings_cache_10k_per_subtype_seed42.pkl`, 50k pares/subtipo | H3N2 rho=0.1172; H1N1 rho=0.1700; promedio=0.1436 | Correlación débil; el espacio no parece ser solo una línea temporal | Resultado principal consolidado | `results/geometry_summary.md` |
| Spearman Hamming HA | ¿La distancia latente preserva similitud molecular de HA? | Spearman entre distancia latente y Hamming HA normalizado | `embeddings_cache_10k_per_subtype_seed42.pkl`, 50k pares/subtipo | H3N2 rho=0.5522; H1N1 rho=0.7154; promedio=0.6338 | Correlación fuerte; evidencia de estructura molecular relevante | Resultado principal consolidado | `results/geometry_summary.md` |
| Spearman Hamming HA+NA | ¿La distancia latente preserva similitud molecular conjunta HA+NA? | Spearman entre distancia latente y Hamming HA+NA normalizado | `embeddings_cache_10k_per_subtype_seed42.pkl`, 50k pares/subtipo | H3N2 rho=0.5820; H1N1 rho=0.7351; promedio=0.6586 | Correlación fuerte; HA+NA es ligeramente superior a HA sola en la corrida principal | Resultado principal consolidado | `results/geometry_summary.md` |
| Robustez seed 7 | ¿La señal molecular fuerte persiste con otro muestreo? | Spearman Hamming HA+NA con menor cache y seed distinto | `embeddings_cache_5k_per_subtype_seed7.pkl`, 30k pares/subtipo | H3N2 rho=0.5348; H1N1 rho=0.6383; promedio=0.5866 | La señal molecular fuerte persiste preliminarmente | Robustez preliminar | `results/geometry_summary.md` |
| PCA effective dimension | ¿La varianza latente está concentrada en pocas direcciones lineales? | PCA global y por subtipo; varianza acumulada; participation ratio | `embeddings_cache_10k_per_subtype_seed42.pkl` | Global: n90=3, n95=4, n99=12, PR=1.89; H1N1 n90=2; H3N2 n90=3 | Evidencia de baja dimensión efectiva y anisotropía fuerte | Resultado principal descriptivo | `results/pca_figures_summary.md`; figuras PCA en `figures/gisaid/` |
| TwoNN intrinsic dimension | ¿La dimensión intrínseca no lineal parece baja? | TwoNN con deduplicación HA+NA, normalizaciones y trimming | `embeddings_cache_10k_per_subtype_seed42.pkl`; 18,093 únicos de 20,000 | Con sample=5000, standard: trim 0.01 da d global 3.49; trim 0.05 da d global 4.83; R2 aprox. 0.97-0.98 | Sugiere rango preliminar 3.5-4.8; no es dimensión absoluta final | Preliminar robusto, requiere MLE/PCA effective dimension adicional | `results/geometry_summary.md` |
| Temporalidad local | ¿Los vecinos latentes están más cerca en el tiempo que pares aleatorios? | kNN latente por subtipo con k=5,10,20 vs baseline random | `embeddings_cache_10k_per_subtype_seed42.pkl` | Sin dedup: vecinos medianos 2-4 meses; random 58-72 meses | Evidencia de estructura temporal local; puede estar influida por duplicados | Diagnóstico descriptivo inicial | `results/temporal_local_neighbors_summary.md` |
| Temporalidad local deduplicada | ¿La temporalidad local persiste sin secuencias HA+NA duplicadas? | kNN latente tras deduplicación exacta HA+NA | `embeddings_cache_10k_per_subtype_seed42.pkl`; H1N1 10,000->9,133; H3N2 10,000->8,960 | H1N1 k=5 mediana vecinos 2 vs random 58; H3N2 k=5 mediana vecinos 3 vs random 72 | Señal fuerte y no explicada solo por duplicados exactos | Resultado fuerte para vecindades locales | `results/temporal_local_neighbors_summary_dedup.md`; `figures/gisaid/temporal_local_neighbors_*_dedup.*` |
| Dinámica PCA piloto | ¿Modelos simples predicen centroides mensuales en PCA space? | PCA global sobre todo el cache; persistence, constant velocity, VAR(1), RW | `embeddings_cache_10k_per_subtype_seed42.pkl`; train hasta 2018; test 2019-2022 | H1N1: persistence difícil de superar; H3N2: VAR(1) mejora persistence en d=3,4 | Piloto útil pero no evaluación predictiva final porque PCA usa todo el cache | Piloto exploratorio | `results/pca_dynamics_summary.md` |
| Rolling-origin PCA | ¿La señal dinámica persiste ajustando PCA solo con train en cada corte? | Rolling-origin mensual; PCA reentrenado por corte; ridge VAR(1/2) | `embeddings_cache_10k_per_subtype_seed42.pkl`; 2019-2022 | H1N1 mejor RMSE persistence d=5, RMSE=0.0695; H3N2 mejor RMSE ridge_var2 d=5, RMSE=0.0670, mejora=0.1548 | Dinámica subtipo-dependiente; H3N2 tiene drift lineal aprovechable | Evaluación PCA retrospectiva más defendible | `results/pca_rolling_dynamics_summary.md`; figuras rolling |
| SDE lineal mínima / modelo gaussiano | ¿Una dinámica probabilística mínima mejora persistence en PCA space? | Gaussian RW, VAR(1), VAR(2) probabilísticos con covarianza residual | `embeddings_cache_10k_per_subtype_seed42.pkl`; rolling-origin 2019-2022 | RMSE: H1N1 persistence d=5 0.0695; H3N2 VAR(2) d=5 0.0670. NLL: VAR(2) gana en ambos subtipos, pero H1N1 empeora RMSE | H3N2 favorece drift lineal; H1N1 requiere cautela por conflicto RMSE/NLL | Modelo probabilístico mínimo en PCA space | `results/pca_sde_summary.md`; figuras SDE PCA |
| Calibración | ¿La incertidumbre gaussiana está bien calibrada? | Barrido de cov-reg, inflación y covarianza full/diagonal | `embeddings_cache_10k_per_subtype_seed42.pkl`; rolling-origin 2019-2022 | Mejor RMSE: H1N1 persistence; H3N2 VAR(2) d=5. Mejor NLL: VAR(2) diagonal d=5. Mejor calibración: H1N1 VAR(2) full d=5; H3N2 VAR(1) full d=4 | Hay tradeoff entre RMSE, NLL y cobertura; H3N2 conserva subdispersión moderada en configs precisas | Calibración preliminar | `results/pca_sde_calibration_summary.md`; `figures/gisaid/pca_sde_calibration_*` |
| Robustez seed42 vs seed7 | ¿La historia dinámica se mantiene en otro cache/seed? | Evaluación acotada en dos caches; full reg=1e-5 y diagonal H3N2 reg=1e-4 | `embeddings_cache_10k_per_subtype_seed42.pkl` y `embeddings_cache_5k_per_subtype_seed7.pkl` | Seed42: H1N1 persistence, H3N2 VAR(2). Seed7: H1N1 VAR(2) mejora RMSE modesto; H3N2 VAR(1) gana RMSE y VAR(2) gana NLL | H3N2 drift lineal robusto; H1N1 se debe describir como drift débil/sensible al muestreo | Robustez externa mínima | `results/pca_sde_robustness_summary.md`; figuras robustez |

## 3. Tabla corta para tesis

| resultado | evidencia cuantitativa | lectura |
|---|---|---|
| Preservación molecular HA | Spearman Hamming HA: H3N2 0.5522, H1N1 0.7154, promedio 0.6338 | El espacio latente preserva similitud molecular de HA. |
| Preservación molecular HA+NA | Spearman Hamming HA+NA: H3N2 0.5820, H1N1 0.7351, promedio 0.6586 | HA+NA confirma y refuerza la señal molecular. |
| Temporalidad global débil | Spearman temporal promedio 0.1436 | La geometría no se reduce a orden cronológico lineal. |
| Baja dimensión efectiva PCA | Global: 3 componentes explican 90%, 4 explican 95%, 12 explican 99%; PR=1.89 | La varianza latente está fuertemente concentrada. |
| Dimensión intrínseca TwoNN | Rango preliminar d=3.49-4.83 con R2 aprox. 0.97-0.98 | Evidencia preliminar de baja dimensión no lineal. |
| Temporalidad local deduplicada | H1N1 k=5: mediana 2 meses vs random 58; H3N2 k=5: mediana 3 vs random 72 | Las vecindades latentes son evolutivamente coherentes. |
| Dinámica PCA rolling-origin | H1N1: persistence RMSE=0.0695; H3N2: VAR(2) RMSE=0.0670, mejora 15.48% | La dinámica es subtipo-dependiente. |
| Robustez dinámica mínima | Seed7 conserva drift lineal en H3N2; H1N1 cambia a mejora modesta de VAR(2) | H3N2 es más robusto; H1N1 requiere más sensibilidad. |

## 4. Contribuciones derivadas

- C1: Auditoría geométrica del espacio latente de AntigenLM usando métricas temporales, moleculares, espectrales e intrínsecas.
- C2: Evidencia de preservación molecular HA/HA+NA, con correlaciones de Spearman fuertes frente a Hamming normalizado.
- C3: Evidencia de temporalidad local no lineal: vecinos latentes cercanos son temporalmente cercanos, incluso tras deduplicación exacta HA+NA.
- C4: Dinámica probabilística mínima en PCA space, evaluada con rolling-origin y modelos gaussianos discretos.
- C5: Diferencia subtipo-dependiente entre H1N1 y H3N2: H3N2 muestra señal de drift lineal más robusta; H1N1 parece más cercano a persistence/random walk o drift débil sensible al muestreo.
- C6: Protocolo reproducible con cache de embeddings, deduplicación, separación por subtipo, métricas moleculares reales y evaluación rolling-origin.

## 5. Limitaciones

- No hay todavía una SDE final con funcionales explícitos `F_viab` y `F_escape`.
- No hay validación en secuencias generadas ni decodificación fiable de trayectorias latentes.
- La dinámica PCA/SDE mínima opera sobre centroides mensuales, no sobre cepas individuales ni sobre secuencias completas.
- Los resultados dependen de GISAID 2000-2022, del checkpoint de AntigenLM usado y de la construcción de los caches.
- Falta validación prospectiva real 2022-2026 con protocolo cerrado antes de mirar resultados.
- Falta comparación contra baselines biológicos fuertes como LBI, modelos filodinámicos o fitness models clásicos.
- TwoNN y PCA sugieren baja dimensión, pero falta consolidación con más seeds, MLE local y pruebas de estabilidad.
- La calibración probabilística todavía muestra tradeoffs entre RMSE, NLL y cobertura; no debe elegirse una configuración solo por una métrica.
- Las figuras PCA son descriptivas; no prueban causalidad, continuidad dinámica ni validez de una SDE.

## 6. Próximos pasos

### Necesario para tesis

- Consolidar la dimensión intrínseca con más seeds y un estimador complementario, por ejemplo MLE local o PCA effective dimension por ventanas temporales.
- Definir un protocolo cerrado de validación retrospectiva/prospectiva antes de usar datos 2022-2026.
- Comparar la dinámica PCA contra baselines biológicos o filodinámicos, al menos LBI si los árboles están disponibles.
- Formalizar la lectura subtipo-dependiente: H3N2 como candidato principal para drift lineal; H1N1 como caso de drift débil o random walk.
- Separar claramente resultados geométricos, dinámica en PCA space y futura SDE completa.

### Opcional para paper

- Repetir la auditoría geométrica con más seeds, tamaños de cache y/o checkpoints si están disponibles.
- Evaluar estabilidad de los centroides mensuales bajo bootstrap por mes.
- Incorporar modelos dinámicos no lineales ligeros en PCA space, siempre contra persistence/random walk.
- Añadir métricas de calibración más formales, como PIT multivariado aproximado o scoring rules adicionales.
- Construir una comparación visual y cuantitativa entre trayectorias PCA y eventos antigénicos conocidos, sin sobreinterpretar causalidad.

### Trabajo futuro

- Implementar una SDE final con `F_viab` y `F_escape`, si los criterios geométricos y dinámicos siguen siendo favorables.
- Diseñar una estrategia defensible para mapear trayectorias latentes a secuencias, probablemente mediante scoring/ranking de cepas reales antes que generación libre.
- Extender la evaluación a predicción prospectiva real 2022-2026 con protocolo predefinido.
- Evaluar si una representación proyectada o regularizada mejora la estabilidad dinámica frente al espacio latente original.
- Integrar incertidumbre epidemiológica y sesgos de muestreo de GISAID en el modelo dinámico.

