# Rolling-origin dynamics in PCA space

Fuente: `results/embeddings_cache_full_FIXED.pkl`.
No se recalcularon embeddings, no se cargo AntigenLM y no se generaron secuencias.

## Diferencias frente al piloto anterior

- El piloto anterior ajustaba PCA una sola vez sobre todo el cache.
- Esta evaluacion ajusta PCA nuevamente en cada corte, usando solo datos disponibles hasta el mes anterior al objetivo.
- La evaluacion es one-step retrospectiva sobre meses 2019-2022 con centroides mensuales.
- Ridge alpha = 1.0.

## Resultados por modelo

| dim | subtipo | modelo | n eval | RMSE | MAE | distancia euclidiana media | RMSE/persistence | mejora vs persistence | mean loglik RW | NLL media RW |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | H1N1 | persistence | 48 | 0.9206 | 0.5424 | 1.3560 | 1.0000 | 0.0000 | NA | NA |
| 3 | H1N1 | constant_velocity | 48 | 1.4536 | 0.8102 | 2.0077 | 1.5790 | -0.5790 | NA | NA |
| 3 | H1N1 | ridge_var1 | 48 | 1.0919 | 0.7491 | 1.6688 | 1.1861 | -0.1861 | NA | NA |
| 3 | H1N1 | ridge_var2 | 48 | 1.0165 | 0.6758 | 1.5430 | 1.1041 | -0.1041 | NA | NA |
| 3 | H1N1 | gaussian_rw_mean | 48 | 0.9241 | 0.5435 | 1.3622 | 1.0038 | -0.0038 | -3.6922 | 3.6922 |
| 3 | H3N2 | persistence | 48 | 0.8470 | 0.5709 | 1.2436 | 1.0000 | 0.0000 | NA | NA |
| 3 | H3N2 | constant_velocity | 48 | 1.2066 | 0.7891 | 1.7544 | 1.4245 | -0.4245 | NA | NA |
| 3 | H3N2 | ridge_var1 | 48 | 1.0775 | 0.8222 | 1.7192 | 1.2720 | -0.2720 | NA | NA |
| 3 | H3N2 | ridge_var2 | 48 | 1.0001 | 0.7504 | 1.5726 | 1.1807 | -0.1807 | NA | NA |
| 3 | H3N2 | gaussian_rw_mean | 48 | 0.8495 | 0.5737 | 1.2511 | 1.0029 | -0.0029 | -3.9221 | 3.9221 |

## Mejor modelo por subtipo

| subtipo | dim | modelo | RMSE | mejora vs persistence |
|---|---:|---|---:|---:|
| H1N1 | 3 | persistence | 0.9206 | 0.0000 |
| H3N2 | 3 | persistence | 0.8470 | 0.0000 |

## Meses evaluados

| subtipo | dim | targets | primer target | ultimo target | targets omitidos |
|---|---:|---:|---|---|---:|
| H1N1 | 3 | 48 | 2016-01 | 2019-12 | 0 |
| H3N2 | 3 | 48 | 2016-01 | 2019-12 | 0 |

## Figuras

- `figures/gisaid/pca_rolling_rmse_by_model.png`
- `figures/gisaid/pca_rolling_rmse_by_model.pdf`
- `figures/gisaid/pca_rolling_relative_improvement.png`
- `figures/gisaid/pca_rolling_relative_improvement.pdf`
- `figures/gisaid/pca_rolling_predictions_h1n1_d3.png`
- `figures/gisaid/pca_rolling_predictions_h1n1_d3.pdf`
- `figures/gisaid/pca_rolling_predictions_h3n2_d3.png`
- `figures/gisaid/pca_rolling_predictions_h3n2_d3.pdf`

## Interpretacion prudente

- Persistence sigue siendo un baseline fuerte si las mejoras relativas son pequenas o negativas.
- Constant velocity puede empeorar cuando los incrementos mensuales no son persistentes o hay sobreoscilacion.
- Ridge VAR(1)/VAR(2) solo aporta senal dinamica si mejora persistence de forma estable por subtipo y dimension.
- La senal dinamica puede ser subtipo-dependiente; no debe promediarse sin revisar H1N1 y H3N2 por separado.
- Esto sigue siendo dinamica en PCA space, no una SDE final ni una evaluacion de generacion de secuencias.
