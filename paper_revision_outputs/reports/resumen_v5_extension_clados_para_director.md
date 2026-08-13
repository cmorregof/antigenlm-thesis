# Resumen v5: extensión con enriquecimiento de clados

## Qué se agregó

El manuscrito ahora incluye una extensión de enriquecimiento de etiquetas de clado usando metadatos privados de GISAID EpiFlu del `EPI_SET_260506bu`. El análisis une el cache deduplicado por HA+NA exacto con metadatos GISAID mediante el identificador de aislado (`epi_isl` / `Isolate_Id`) y reporta únicamente salidas agregadas.

## Entradas y cobertura

- Cache completo de embeddings: `results/embeddings_cache_full_all_available.pkl`
- Tabla privada de unión de metadatos: `data/gisaid_metadata_private/gisaid_epiflu_isolates_2000_2022_epi_set_260506bu_joined_dedup_cache.csv`
- Tamaño del cache deduplicado: 82,306 registros
- Duplicados exactos HA+NA removidos: 29,450 registros
- Cobertura de unión con metadatos: 81,943 / 82,306 registros (99.56%)
- Cobertura con clado asignado no `unassigned`: 76,238 / 82,306 registros (92.63%)
- Cobertura H1N1 con clado asignado: 36,156 / 36,753 registros (98.38%)
- Cobertura H3N2 con clado asignado: 40,082 / 45,553 registros (87.99%)

## Resultado principal de enriquecimiento de clados

En k=5, la precisión de clado de los vecinos latentes es alta en ambos subtipos:

| Subtipo | precision@5 | aleatorio por subtipo | enriquecimiento |
|---|---:|---:|---:|
| H1N1 | 0.9149 | 0.2409 | 3.80x |
| H3N2 | 0.8609 | 0.0689 | 12.49x |

El control de permutación de etiquetas de clado dentro de subtipo reduce la precisión al rango basal:

- H1N1 k=5, media permutada: 0.2406
- H3N2 k=5, media permutada: 0.0690

Esto apoya la afirmación de que los vecindarios latentes están enriquecidos por la pertenencia observada a clados GISAID, no solo por la distribución marginal de clases.

## Control temporal estratificado

Este control es crucial y cambia la interpretación. Los candidatos aleatorios se restringieron a registros del mismo subtipo con clado asignado dentro de la ventana temporal indicada.

| Subtipo | Ventana | precisión real@5 | aleatorio temporal | enriquecimiento |
|---|---:|---:|---:|---:|
| H1N1 | ±6 meses | 0.9149 | 0.5892 | 1.55x |
| H1N1 | ±12 meses | 0.9149 | 0.5405 | 1.69x |
| H1N1 | ±24 meses | 0.9149 | 0.4681 | 1.95x |
| H3N2 | ±6 meses | 0.8610 | 0.2707 | 3.18x |
| H3N2 | ±12 meses | 0.8610 | 0.2288 | 3.76x |
| H3N2 | ±24 meses | 0.8609 | 0.1548 | 5.56x |

Lectura: la señal de clado es real, pero está parcialmente acoplada al recambio temporal de clados, especialmente en H1N1. Por eso el manuscrito no debe afirmar que el embedding captura estructura filogenética independiente del tiempo. La afirmación defendible es coherencia evolutivo-taxonómica local bajo anotaciones de clado GISAID.

## Archivos generados o actualizados

- `paper_revision_outputs/run_clade_enrichment_analysis.py`
- `paper_revision_outputs/logs/clade_enrichment_run.log`
- `results/gisaid_clade_enrichment_results.json`
- `results/gisaid_clade_enrichment_summary.md`
- `results/gisaid_clade_enrichment_summary_es.md`
- `figures/latent_geometry_full/clade_precision_enrichment.pdf`
- `figures/latent_geometry_full/clade_precision_enrichment.png`
- `papers/paper_1_latent_geometry_full/figures/clade_precision_enrichment.pdf`
- `papers/paper_1_latent_geometry_full/main.tex`
- `paper_revision_outputs/revised_main.tex`
- `paper_revision_outputs/latent_geometry_manuscript_compiled_v5_clade.pdf`

## Opinión editorial

La extensión de clados mejora sustancialmente el artículo. Antes, el manuscrito era principalmente una auditoría geométrica de embeddings; ahora tiene una validación biológica externa mediante etiquetas usadas en vigilancia de influenza. Eso aumenta la relevancia para una revista de evolución viral o bioinformática computacional.

Sin embargo, el control temporal impide vender el resultado de forma demasiado fuerte. En H1N1, el enriquecimiento baja a 1.55x dentro de ±6 meses, lo que indica que buena parte de la señal de clado coincide con periodos temporales dominados por ciertos clados. En H3N2, la señal es más convincente incluso bajo control temporal (3.18x en ±6 meses), pero también se atenúa.

Mi recomendación honesta: el paper actual ya es claramente más relevante que la versión BMC inicial. Para BMC Bioinformatics o Scientific Reports está fuerte si la redacción se mantiene conservadora. Para Virus Evolution, ahora tiene una puerta real de entrada, pero todavía sería prudente añadir al menos uno de estos análisis antes de apuntar alto: distancia filogenética cuantitativa, estratificación por geografía/host, o validación con clados/linajes definidos por Nextstrain cuando sea posible. Sin eso, Virus Evolution podría pedir revisión mayor por falta de árbol filogenético.
