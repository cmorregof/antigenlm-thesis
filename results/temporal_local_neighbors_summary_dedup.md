# Temporalidad local de vecinos latentes

Fuente: `results/embeddings_cache_10k_per_subtype_seed42.pkl`.
No se recalcularon embeddings, no se cargo AntigenLM y no se imprimieron secuencias.

Deduplicacion: activada por hash exacto de HA+NA, conservando el primer representante.

| subtipo | duplicados removidos |
|---|---:|
| H1N1 | 867 |
| H3N2 | 1040 |

La distancia temporal se calcula como diferencia absoluta en meses entre fechas de coleccion.
La baseline aleatoria usa pares del mismo subtipo y el mismo numero de comparaciones que cada valor de k.

| subtipo | k | n | pares vecinos | mediana vecinos | media vecinos | p25 vecinos | p75 vecinos | mediana random | media random | p25 random | p75 random | razon mediana | razon media |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1N1 | 5 | 9133 | 45665 | 2.00 | 8.50 | 1.00 | 8.00 | 58.00 | 66.32 | 26.00 | 98.00 | 0.034 | 0.128 |
| H1N1 | 10 | 9133 | 91330 | 3.00 | 10.26 | 1.00 | 10.00 | 58.00 | 66.41 | 26.00 | 98.00 | 0.052 | 0.154 |
| H1N1 | 20 | 9133 | 182660 | 4.00 | 12.63 | 1.00 | 12.00 | 57.00 | 66.41 | 26.00 | 98.00 | 0.070 | 0.190 |
| H3N2 | 5 | 8960 | 44800 | 3.00 | 9.46 | 1.00 | 10.00 | 72.00 | 82.07 | 33.00 | 121.00 | 0.042 | 0.115 |
| H3N2 | 10 | 8960 | 89600 | 3.00 | 11.49 | 1.00 | 12.00 | 71.00 | 82.14 | 33.00 | 121.00 | 0.042 | 0.140 |
| H3N2 | 20 | 8960 | 179200 | 5.00 | 14.10 | 1.00 | 16.00 | 72.00 | 82.47 | 33.00 | 121.00 | 0.069 | 0.171 |

## Figuras

- `figures/gisaid/temporal_local_neighbors_h1n1_dedup.png`
- `figures/gisaid/temporal_local_neighbors_h1n1_dedup.pdf`
- `figures/gisaid/temporal_local_neighbors_h3n2_dedup.png`
- `figures/gisaid/temporal_local_neighbors_h3n2_dedup.pdf`

## Lectura metodologica

- Si las distancias temporales de vecinos latentes son menores que las de pares aleatorios, hay evidencia de estructura temporal local.
- Si no lo son, el espacio puede preservar similitud molecular sin codificar continuidad temporal simple.
- Esta prueba es local y descriptiva: no concluye que una SDE funcione.
- Esta corrida evalua si la senal persiste tras eliminar duplicados exactos HA+NA.
