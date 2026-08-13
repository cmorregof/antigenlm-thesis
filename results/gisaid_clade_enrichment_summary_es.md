# Enriquecimiento de clados GISAID en vecindarios latentes de AntigenLM

Este análisis usa el cache deduplicado por HA+NA exacto y el export privado de metadatos GISAID EpiFlu de `EPI_SET_260506bu`. No se redistribuyen secuencias ni metadatos a nivel de accesión.

## Cobertura de metadatos

| grupo | n cache | n unido | % unión | n con clado | % con clado |
| --- | --- | --- | --- | --- | --- |
| H1N1 | 36,753 | 36,723 | 99.92% | 36,156 | 98.38% |
| H3N2 | 45,553 | 45,220 | 99.27% | 40,082 | 87.99% |
| Combined | 82,306 | 81,943 | 99.56% | 76,238 | 92.63% |

## Sensibilidad con control temporal (k=5)

Los candidatos aleatorios se restringen al mismo subtipo y a registros con clado asignado dentro de la ventana temporal indicada. La precisión latente se recalcula sobre el mismo conjunto de consultas elegibles.

| subtipo | k | ventana | precisión real | aleatorio estratificado | enriquecimiento | consultas |
| --- | --- | --- | --- | --- | --- | --- |
| H1N1 | 5 | ±6 meses | 0.9149 | 0.5892 | 1.55x | 36,156 |
| H1N1 | 5 | ±12 meses | 0.9149 | 0.5405 | 1.69x | 36,156 |
| H1N1 | 5 | ±24 meses | 0.9149 | 0.4681 | 1.95x | 36,156 |
| H3N2 | 5 | ±6 meses | 0.8610 | 0.2707 | 3.18x | 40,080 |
| H3N2 | 5 | ±12 meses | 0.8610 | 0.2288 | 3.76x | 40,080 |
| H3N2 | 5 | ±24 meses | 0.8609 | 0.1548 | 5.56x | 40,081 |

## Lectura metodológica

- Los vecindarios latentes están fuertemente enriquecidos por pertenencia al clado GISAID en H1N1 y H3N2.
- El control de permutación de etiquetas dentro de subtipo reduce la precisión al rango basal, por lo que la señal depende de las asignaciones observadas de clado.
- El control aleatorio estratificado por tiempo aumenta el basal, como era esperable, pero el enriquecimiento persiste en las ventanas evaluadas.
- La afirmación válida es coherencia evolutivo-taxonómica local bajo anotaciones GISAID. Esto no equivale a validación de distancia filogenética cuantitativa, antigenicidad, escape inmune, relevancia vacunal ni forecasting prospectivo.
