# Resumen de geometria latente

Fuente: logs existentes en `results/`. No se ejecutaron experimentos nuevos para preparar este resumen.

| archivo/log | seed | max_per_subtype | pair_samples | metrica | rho H3N2 | rho H1N1 | rho promedio | pares validos/omitidos | interpretacion |
|---|---:|---:|---:|---|---:|---:|---:|---|---|
| `results/geometry_temporal_10k_seed42.log` | 42 | 10000 | 50000 | `temporal` | 0.1172 | 0.1700 | 0.1436 | H3N2: 50,000/0; H1N1: 50,000/0 | Debil |
| `results/geometry_hamming_ha_10k_seed42.log` | 42 | 10000 | 50000 | `hamming_ha` | 0.5522 | 0.7154 | 0.6338 | H3N2: 49,993/7; H1N1: 49,974/26 | Fuerte |
| `results/geometry_hamming_ha_na_10k_seed42.log` | 42 | 10000 | 50000 | `hamming_ha_na` | 0.5820 | 0.7351 | 0.6586 | H3N2: 49,805/195; H1N1: 49,967/33 | Fuerte |
| `results/geometry_hamming_ha_na_5k_seed7.log` | 7 | 5000 | 30000 | `hamming_ha_na` | 0.5348 | 0.6383 | 0.5866 | H3N2: 29,893/107; H1N1: 29,961/39 | Fuerte |

## Lectura metodologica

- La distancia temporal muestra correlacion debil con la distancia latente (`rho` promedio 0.1436). Esto sugiere que el espacio no esta capturando simplemente orden cronologico.
- Hamming HA muestra correlacion fuerte con la distancia latente (`rho` promedio 0.6338), lo que apoya que el espacio preserva similitud molecular relevante en HA.
- Hamming HA+NA tambien muestra correlacion fuerte (`rho` promedio 0.6586), ligeramente superior a HA sola en la corrida principal.
- La corrida con `seed=7`, `max_per_subtype=5000` y `hamming_ha_na` conserva una senal fuerte (`rho` promedio 0.5866). Esto confirma preliminarmente la robustez de la senal, aunque faltan mas semillas y tamanos de muestra.
- La dimension intrinseca ya cuenta con una estimacion TwoNN preliminar separada, resumida abajo. Aun no debe tratarse como valor final absoluto.
- La interpolacion lineal actual no prueba suavidad biologica: los logs advierten que interpolar linealmente dentro del propio espacio latente puede producir un CV tautologico.
- La SDE queda motivada, no demostrada. Estos resultados fortalecen la plausibilidad del espacio latente como variable de estado, pero aun faltan dimension intrinseca, robustez adicional, pruebas de suavidad no tautologicas, baselines y evaluacion predictiva completa.

## Dimension intrinseca preliminar

- Cache usado: `results/embeddings_cache_10k_per_subtype_seed42.pkl`.
- Embeddings originales: 20,000.
- Secuencias HA+NA unicas tras deduplicacion: 18,093.
- Duplicados exactos extra: 1,907.
- Metodo: TwoNN.
- Deduplicacion: si, por secuencia HA+NA exacta.
- Normalizaciones probadas: `none`, `standard`, `l2`.
- Sample sizes probados: 1000, 3000, 5000.
- Trim probado: 0.01 y 0.05.
- R2 observado: aproximadamente 0.968-0.982.

| sample_size | normalizacion | trim | d global | d H1N1 | d H3N2 | R2 global | lectura |
|---:|---|---:|---:|---:|---:|---:|---|
| 5000 | `standard` | 0.01 | 3.49 | 3.57 | 3.53 | 0.9808 | Estimacion conservadora con poco trimming |
| 5000 | `standard` | 0.05 | 4.83 | 4.88 | 4.91 | 0.9705 | Estimacion mas robusta a extremos, pero mas sensible al trimming |

### Lectura metodologica de TwoNN

- TwoNN sugiere dimension intrinseca baja, aproximadamente en el rango 3.5-4.8 bajo las configuraciones principales.
- El resultado es robusto frente a la normalizacion: `none`, `standard` y `l2` dan rangos comparables.
- El resultado es sensible al trimming de extremos de `mu`; por eso debe reportarse como rango, no como un unico valor absoluto.
- Los duplicados exactos explicaban el fallo inicial de dimension menor que 1, R2 negativo y distancias cero.
- Esta estimacion no debe presentarse como dimension final absoluta. Es evidencia preliminar robusta de baja dimension, y motiva analisis adicionales antes de usarla para justificar una SDE.

## PCA y dimension efectiva lineal

| grupo | n | dim | n80 | n90 | n95 | n99 | participation ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| global | 20000 | 384 | 2 | 3 | 4 | 12 | 1.89 |
| H1N1 | 10000 | 384 | 2 | 2 | 3 | 13 | 1.76 |
| H3N2 | 10000 | 384 | 2 | 3 | 6 | 20 | 1.63 |

Top 10 explained variance ratios:

- Global: 0.6975, 0.2024, 0.0458, 0.0163, 0.0089, 0.0053, 0.0039, 0.0029, 0.0027, 0.0019.
- H1N1: 0.7246, 0.2082, 0.0196, 0.0112, 0.0077, 0.0056, 0.0042, 0.0023, 0.0022, 0.0015.
- H3N2: 0.7757, 0.0874, 0.0417, 0.0260, 0.0156, 0.0102, 0.0065, 0.0053, 0.0040, 0.0035.

### Lectura metodologica de PCA

- PCA refuerza la evidencia de dimension efectiva baja.
- El resultado es compatible con TwoNN.
- La participation ratio baja sugiere anisotropia fuerte del espacio latente.
- Esto motiva, pero no demuestra, la viabilidad de una SDE.

## Temporalidad local deduplicada

Fuente: `results/embeddings_cache_10k_per_subtype_seed42.pkl`.

Deduplicacion por hash exacto HA+NA:

| subtipo | antes | despues | removidos |
|---|---:|---:|---:|
| H1N1 | 10000 | 9133 | 867 |
| H3N2 | 10000 | 8960 | 1040 |

| subtipo | k | mediana vecinos | media vecinos | mediana random | media random | razon mediana | razon media |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1N1 | 5 | 2.00 | 8.50 | 58.00 | 66.32 | 0.034 | 0.128 |
| H1N1 | 10 | 3.00 | 10.26 | 58.00 | 66.41 | 0.052 | 0.154 |
| H1N1 | 20 | 4.00 | 12.63 | 57.00 | 66.41 | 0.070 | 0.190 |
| H3N2 | 5 | 3.00 | 9.46 | 72.00 | 82.07 | 0.042 | 0.115 |
| H3N2 | 10 | 3.00 | 11.49 | 71.00 | 82.14 | 0.042 | 0.140 |
| H3N2 | 20 | 5.00 | 14.10 | 72.00 | 82.47 | 0.069 | 0.171 |

Figuras:

- `figures/gisaid/temporal_local_neighbors_h1n1_dedup.pdf`
- `figures/gisaid/temporal_local_neighbors_h3n2_dedup.pdf`

### Lectura metodologica de temporalidad local

- La temporalidad global lineal es debil, como muestra la correlacion temporal de Spearman.
- Sin embargo, la temporalidad local es fuerte: los vecinos latentes estan a pocos meses de distancia, mientras que los pares aleatorios del mismo subtipo estan separados por decenas de meses.
- La senal persiste tras deduplicar HA+NA, por lo que no parece ser solo artefacto de secuencias repetidas.
- Esto apoya la idea de que el espacio latente organiza vecindades evolutivamente coherentes.
- No demuestra por si solo que una SDE funcione, pero fortalece la plausibilidad de modelar dinamicas locales sobre el espacio latente.
