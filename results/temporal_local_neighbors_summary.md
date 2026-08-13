# Temporalidad local de vecinos latentes

Fuente: `results/embeddings_cache_10k_per_subtype_seed42.pkl`.
No se recalcularon embeddings, no se cargo AntigenLM y no se imprimieron secuencias.

La distancia temporal se calcula como diferencia absoluta en meses entre fechas de coleccion.
La baseline aleatoria usa pares del mismo subtipo y el mismo numero de comparaciones que cada valor de k.

| subtipo | k | n | pares vecinos | mediana vecinos | media vecinos | p25 vecinos | p75 vecinos | mediana random | media random | p25 random | p75 random | razon mediana | razon media |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1N1 | 5 | 10000 | 50000 | 2.00 | 7.79 | 1.00 | 7.00 | 58.00 | 67.10 | 26.00 | 98.00 | 0.034 | 0.116 |
| H1N1 | 10 | 10000 | 100000 | 3.00 | 9.53 | 1.00 | 9.00 | 58.00 | 66.93 | 26.00 | 98.00 | 0.052 | 0.142 |
| H1N1 | 20 | 10000 | 200000 | 3.00 | 11.89 | 1.00 | 12.00 | 58.00 | 66.91 | 26.00 | 99.00 | 0.052 | 0.178 |
| H3N2 | 5 | 10000 | 50000 | 2.00 | 8.57 | 1.00 | 9.00 | 72.00 | 82.79 | 34.00 | 122.00 | 0.028 | 0.103 |
| H3N2 | 10 | 10000 | 100000 | 3.00 | 10.42 | 1.00 | 11.00 | 72.00 | 82.74 | 33.00 | 122.00 | 0.042 | 0.126 |
| H3N2 | 20 | 10000 | 200000 | 4.00 | 12.87 | 1.00 | 14.00 | 72.00 | 82.38 | 33.00 | 121.00 | 0.056 | 0.156 |

## Figuras

- `figures/gisaid/temporal_local_neighbors_h1n1.png`
- `figures/gisaid/temporal_local_neighbors_h1n1.pdf`
- `figures/gisaid/temporal_local_neighbors_h3n2.png`
- `figures/gisaid/temporal_local_neighbors_h3n2.pdf`

## Lectura metodologica

- Si las distancias temporales de vecinos latentes son menores que las de pares aleatorios, hay evidencia de estructura temporal local.
- Si no lo son, el espacio puede preservar similitud molecular sin codificar continuidad temporal simple.
- Esta prueba es local y descriptiva: no concluye que una SDE funcione.
- La presencia de duplicados exactos en el cache puede aumentar artificialmente la cercania temporal de algunos vecinos; conviene auditar sensibilidad con deduplicacion en una etapa posterior.
