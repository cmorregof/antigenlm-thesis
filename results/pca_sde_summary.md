# SDE lineal minima / modelo gaussiano dinamico en PCA space

Fuente: `results/embeddings_cache_10k_per_subtype_seed42.pkl`.
No se recalcularon embeddings, no se cargo AntigenLM y no se generaron secuencias.

## Modelo

- Gaussian random walk: `x[t+1] = x[t] + mu_delta + eps`, con `eps ~ N(0,Q)`.
- Linear drift / VAR(1): `x[t+1] = A x[t] + b + eps`.
- VAR(2): `x[t+1] = A1 x[t] + A2 x[t-1] + b + eps`.
- `Q` se estima con residuos de entrenamiento y se regulariza como `Q + cov_reg I`.

## Evaluacion

- Rolling-origin sobre 2019-2022.
- PCA se ajusta en cada corte usando solo datos de entrenamiento.
- Ridge alpha = 1.0.
- Covariance regularization = 1e-05.

## RMSE/MAE

| dim | subtipo | modelo | n eval | RMSE | MAE | distancia euclidiana media | RMSE/persistence | mejora vs persistence |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 3 | H1N1 | persistence | 29 | 0.0883 | 0.0519 | 0.1153 | 1.0000 | 0.0000 |
| 3 | H1N1 | gaussian_rw | 29 | 0.0887 | 0.0522 | 0.1160 | 1.0050 | -0.0050 |
| 3 | H1N1 | linear_drift_var1 | 29 | 0.0987 | 0.0697 | 0.1396 | 1.1185 | -0.1185 |
| 3 | H1N1 | linear_drift_var2 | 29 | 0.1002 | 0.0668 | 0.1427 | 1.1348 | -0.1348 |
| 4 | H1N1 | persistence | 29 | 0.0769 | 0.0414 | 0.1164 | 1.0000 | 0.0000 |
| 4 | H1N1 | gaussian_rw | 29 | 0.0773 | 0.0416 | 0.1171 | 1.0050 | -0.0050 |
| 4 | H1N1 | linear_drift_var1 | 29 | 0.0850 | 0.0549 | 0.1384 | 1.1050 | -0.1050 |
| 4 | H1N1 | linear_drift_var2 | 29 | 0.0871 | 0.0524 | 0.1434 | 1.1316 | -0.1316 |
| 5 | H1N1 | persistence | 29 | 0.0695 | 0.0362 | 0.1183 | 1.0000 | 0.0000 |
| 5 | H1N1 | gaussian_rw | 29 | 0.0698 | 0.0364 | 0.1190 | 1.0050 | -0.0050 |
| 5 | H1N1 | linear_drift_var1 | 29 | 0.0762 | 0.0463 | 0.1395 | 1.0975 | -0.0975 |
| 5 | H1N1 | linear_drift_var2 | 29 | 0.0784 | 0.0446 | 0.1451 | 1.1284 | -0.1284 |
| 3 | H3N2 | persistence | 32 | 0.0964 | 0.0589 | 0.1346 | 1.0000 | 0.0000 |
| 3 | H3N2 | gaussian_rw | 32 | 0.0966 | 0.0590 | 0.1348 | 1.0018 | -0.0018 |
| 3 | H3N2 | linear_drift_var1 | 32 | 0.0851 | 0.0623 | 0.1308 | 0.8829 | 0.1171 |
| 3 | H3N2 | linear_drift_var2 | 32 | 0.0822 | 0.0580 | 0.1255 | 0.8526 | 0.1474 |
| 4 | H3N2 | persistence | 32 | 0.0866 | 0.0517 | 0.1398 | 1.0000 | 0.0000 |
| 4 | H3N2 | gaussian_rw | 32 | 0.0867 | 0.0518 | 0.1401 | 1.0018 | -0.0018 |
| 4 | H3N2 | linear_drift_var1 | 32 | 0.0767 | 0.0544 | 0.1363 | 0.8863 | 0.1137 |
| 4 | H3N2 | linear_drift_var2 | 32 | 0.0739 | 0.0508 | 0.1305 | 0.8530 | 0.1470 |
| 5 | H3N2 | persistence | 32 | 0.0793 | 0.0453 | 0.1431 | 1.0000 | 0.0000 |
| 5 | H3N2 | gaussian_rw | 32 | 0.0794 | 0.0454 | 0.1434 | 1.0018 | -0.0018 |
| 5 | H3N2 | linear_drift_var1 | 32 | 0.0691 | 0.0467 | 0.1378 | 0.8712 | 0.1288 |
| 5 | H3N2 | linear_drift_var2 | 32 | 0.0670 | 0.0437 | 0.1332 | 0.8452 | 0.1548 |

## NLL, Mahalanobis y cobertura

| dim | subtipo | modelo | mean loglik | mean NLL | Mahalanobis medio | cov50 | cov90 | cov95 | trace(Q) media | logdet(Q) medio |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | H1N1 | gaussian_rw | 2.9555 | -2.9555 | 0.9506 | 0.8966 | 0.9310 | 0.9655 | 0.0704 | -13.0378 |
| 3 | H1N1 | linear_drift_var1 | 2.6796 | -2.6796 | 1.2596 | 0.7931 | 0.9655 | 0.9655 | 0.0578 | -13.2085 |
| 3 | H1N1 | linear_drift_var2 | 2.9078 | -2.9078 | 1.2880 | 0.7586 | 0.9310 | 0.9655 | 0.0465 | -13.8042 |
| 4 | H1N1 | gaussian_rw | 5.2681 | -5.2681 | 1.1075 | 0.8276 | 0.9655 | 0.9655 | 0.0721 | -19.9724 |
| 4 | H1N1 | linear_drift_var1 | 4.7990 | -4.7990 | 1.3924 | 0.7931 | 0.9310 | 0.9655 | 0.0587 | -19.6287 |
| 4 | H1N1 | linear_drift_var2 | 5.2116 | -5.2116 | 1.4167 | 0.7931 | 0.9310 | 0.9655 | 0.0476 | -20.6099 |
| 5 | H1N1 | gaussian_rw | 7.4944 | -7.4944 | 1.3847 | 0.7586 | 0.9310 | 0.9310 | 0.0736 | -27.2263 |
| 5 | H1N1 | linear_drift_var1 | 7.1883 | -7.1883 | 1.6050 | 0.8276 | 0.8966 | 0.9655 | 0.0594 | -26.9505 |
| 5 | H1N1 | linear_drift_var2 | 7.6204 | -7.6204 | 1.6433 | 0.7586 | 0.8966 | 0.9310 | 0.0484 | -28.0451 |
| 3 | H3N2 | gaussian_rw | 3.0151 | -3.0151 | 1.4159 | 0.6875 | 0.8438 | 0.8750 | 0.0573 | -14.8725 |
| 3 | H3N2 | linear_drift_var1 | 3.3136 | -3.3136 | 1.5311 | 0.7500 | 0.8750 | 0.9062 | 0.0439 | -15.1814 |
| 3 | H3N2 | linear_drift_var2 | 3.5150 | -3.5150 | 1.4817 | 0.6875 | 0.9062 | 0.9062 | 0.0429 | -15.4473 |
| 4 | H3N2 | gaussian_rw | 3.7146 | -3.7146 | 2.0055 | 0.6562 | 0.7812 | 0.8438 | 0.0581 | -22.0219 |
| 4 | H3N2 | linear_drift_var1 | 4.7870 | -4.7870 | 2.0099 | 0.5312 | 0.8438 | 0.8750 | 0.0446 | -22.2404 |
| 4 | H3N2 | linear_drift_var2 | 4.8992 | -4.8992 | 2.0274 | 0.5625 | 0.8438 | 0.8750 | 0.0434 | -22.6224 |
| 5 | H3N2 | gaussian_rw | 4.8355 | -4.8355 | 2.2937 | 0.6562 | 0.7812 | 0.8438 | 0.0593 | -29.3203 |
| 5 | H3N2 | linear_drift_var1 | 6.8971 | -6.8971 | 2.1844 | 0.5938 | 0.8438 | 0.8750 | 0.0446 | -29.3086 |
| 5 | H3N2 | linear_drift_var2 | 6.9489 | -6.9489 | 2.2258 | 0.5938 | 0.8750 | 0.8750 | 0.0434 | -29.9018 |

## Mejor modelo por subtipo

| criterio | subtipo | dim | modelo | valor |
|---|---|---:|---|---:|
| RMSE | H1N1 | 5 | persistence | 0.0695 |
| RMSE | H3N2 | 5 | linear_drift_var2 | 0.0670 |
| NLL | H1N1 | 5 | linear_drift_var2 | -7.6204 |
| NLL | H3N2 | 5 | linear_drift_var2 | -6.9489 |

## Figuras

- `figures/gisaid/pca_sde_nll_by_model.png`
- `figures/gisaid/pca_sde_nll_by_model.pdf`
- `figures/gisaid/pca_sde_coverage_by_model.png`
- `figures/gisaid/pca_sde_coverage_by_model.pdf`
- `figures/gisaid/pca_sde_mahalanobis_by_model.png`
- `figures/gisaid/pca_sde_mahalanobis_by_model.pdf`
- `figures/gisaid/pca_sde_h3n2_d3_ellipses.png`
- `figures/gisaid/pca_sde_h3n2_d3_ellipses.pdf`

## Interpretacion prudente

- H1N1 debe compararse cuidadosamente contra random walk/persistence, porque el rolling-origin puntual ya sugeria drift debil.
- H3N2 favorece drift lineal solo si VAR(1)/VAR(2) mejora RMSE y NLL frente al random walk.
- Cobertura empirica menor que la nominal sugiere subdispersion; cobertura mayor que la nominal sugiere sobredispersion.
- Este modelo probabilistico sigue operando en PCA space y no es todavia una SDE final ni una evaluacion de generacion de secuencias.
