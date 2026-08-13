# Robustez externa minima de modelo gaussiano dinamico en PCA space

No se recalcularon embeddings, no se cargo AntigenLM y no se generaron secuencias.

## Caches comparados

- `results/embeddings_cache_10k_per_subtype_seed42.pkl`
- `results/embeddings_cache_5k_per_subtype_seed7.pkl`

## Configuracion

- Rolling-origin: 2019-2022.
- Dimensiones: 3, 4, 5.
- Ridge alpha: 1.0.
- Config principal: `cov_type=full`, `cov_reg=1e-5`, `cov_inflation=1.0`.
- Complementaria acotada H3N2: `cov_type=diagonal`, `cov_reg=1e-4`, `cov_inflation=1.0`.

## Tabla comparativa principal

| cache | config | subtipo | d | modelo | n | RMSE | MAE | mean NLL | cov90 | cov95 | mejora vs persistence |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| seed42_max10000 | main_full_reg1e-5 | H1N1 | 3 | gaussian_rw | 29 | 0.0887 | 0.0522 | -2.9555 | 0.9310 | 0.9655 | -0.0050 |
| seed42_max10000 | main_full_reg1e-5 | H1N1 | 3 | linear_drift_var1 | 29 | 0.0987 | 0.0697 | -2.6796 | 0.9655 | 0.9655 | -0.1185 |
| seed42_max10000 | main_full_reg1e-5 | H1N1 | 3 | linear_drift_var2 | 29 | 0.1002 | 0.0668 | -2.9078 | 0.9310 | 0.9655 | -0.1348 |
| seed42_max10000 | main_full_reg1e-5 | H1N1 | 3 | persistence | 29 | 0.0883 | 0.0519 | NA | NA | NA | 0.0000 |
| seed42_max10000 | main_full_reg1e-5 | H1N1 | 4 | gaussian_rw | 29 | 0.0773 | 0.0416 | -5.2681 | 0.9655 | 0.9655 | -0.0050 |
| seed42_max10000 | main_full_reg1e-5 | H1N1 | 4 | linear_drift_var1 | 29 | 0.0850 | 0.0549 | -4.7990 | 0.9310 | 0.9655 | -0.1050 |
| seed42_max10000 | main_full_reg1e-5 | H1N1 | 4 | linear_drift_var2 | 29 | 0.0871 | 0.0524 | -5.2116 | 0.9310 | 0.9655 | -0.1316 |
| seed42_max10000 | main_full_reg1e-5 | H1N1 | 4 | persistence | 29 | 0.0769 | 0.0414 | NA | NA | NA | 0.0000 |
| seed42_max10000 | main_full_reg1e-5 | H1N1 | 5 | gaussian_rw | 29 | 0.0698 | 0.0364 | -7.4944 | 0.9310 | 0.9310 | -0.0050 |
| seed42_max10000 | main_full_reg1e-5 | H1N1 | 5 | linear_drift_var1 | 29 | 0.0762 | 0.0463 | -7.1883 | 0.8966 | 0.9655 | -0.0975 |
| seed42_max10000 | main_full_reg1e-5 | H1N1 | 5 | linear_drift_var2 | 29 | 0.0784 | 0.0446 | -7.6204 | 0.8966 | 0.9310 | -0.1284 |
| seed42_max10000 | main_full_reg1e-5 | H1N1 | 5 | persistence | 29 | 0.0695 | 0.0362 | NA | NA | NA | 0.0000 |
| seed42_max10000 | h3n2_diag_reg1e-4 | H3N2 | 3 | gaussian_rw | 32 | 0.0966 | 0.0590 | -3.2928 | 0.9062 | 0.9375 | -0.0018 |
| seed42_max10000 | h3n2_diag_reg1e-4 | H3N2 | 3 | linear_drift_var1 | 32 | 0.0851 | 0.0623 | -3.4628 | 0.9062 | 0.9062 | 0.1171 |
| seed42_max10000 | h3n2_diag_reg1e-4 | H3N2 | 3 | linear_drift_var2 | 32 | 0.0822 | 0.0580 | -3.6826 | 0.9062 | 0.9062 | 0.1474 |
| seed42_max10000 | main_full_reg1e-5 | H3N2 | 3 | gaussian_rw | 32 | 0.0966 | 0.0590 | -3.0151 | 0.8438 | 0.8750 | -0.0018 |
| seed42_max10000 | main_full_reg1e-5 | H3N2 | 3 | linear_drift_var1 | 32 | 0.0851 | 0.0623 | -3.3136 | 0.8750 | 0.9062 | 0.1171 |
| seed42_max10000 | main_full_reg1e-5 | H3N2 | 3 | linear_drift_var2 | 32 | 0.0822 | 0.0580 | -3.5150 | 0.9062 | 0.9062 | 0.1474 |
| seed42_max10000 | main_full_reg1e-5 | H3N2 | 3 | persistence | 32 | 0.0964 | 0.0589 | NA | NA | NA | 0.0000 |
| seed42_max10000 | h3n2_diag_reg1e-4 | H3N2 | 4 | gaussian_rw | 32 | 0.0867 | 0.0518 | -4.7570 | 0.8438 | 0.8438 | -0.0018 |
| seed42_max10000 | h3n2_diag_reg1e-4 | H3N2 | 4 | linear_drift_var1 | 32 | 0.0767 | 0.0544 | -5.0936 | 0.8438 | 0.8750 | 0.1137 |
| seed42_max10000 | h3n2_diag_reg1e-4 | H3N2 | 4 | linear_drift_var2 | 32 | 0.0739 | 0.0508 | -5.3807 | 0.8750 | 0.8750 | 0.1470 |
| seed42_max10000 | main_full_reg1e-5 | H3N2 | 4 | gaussian_rw | 32 | 0.0867 | 0.0518 | -3.7146 | 0.7812 | 0.8438 | -0.0018 |
| seed42_max10000 | main_full_reg1e-5 | H3N2 | 4 | linear_drift_var1 | 32 | 0.0767 | 0.0544 | -4.7870 | 0.8438 | 0.8750 | 0.1137 |
| seed42_max10000 | main_full_reg1e-5 | H3N2 | 4 | linear_drift_var2 | 32 | 0.0739 | 0.0508 | -4.8992 | 0.8438 | 0.8750 | 0.1470 |
| seed42_max10000 | main_full_reg1e-5 | H3N2 | 4 | persistence | 32 | 0.0866 | 0.0517 | NA | NA | NA | 0.0000 |
| seed42_max10000 | h3n2_diag_reg1e-4 | H3N2 | 5 | gaussian_rw | 32 | 0.0794 | 0.0454 | -6.5882 | 0.8438 | 0.8750 | -0.0018 |
| seed42_max10000 | h3n2_diag_reg1e-4 | H3N2 | 5 | linear_drift_var1 | 32 | 0.0691 | 0.0467 | -7.2421 | 0.8750 | 0.8750 | 0.1288 |
| seed42_max10000 | h3n2_diag_reg1e-4 | H3N2 | 5 | linear_drift_var2 | 32 | 0.0670 | 0.0437 | -7.5580 | 0.8750 | 0.8750 | 0.1548 |
| seed42_max10000 | main_full_reg1e-5 | H3N2 | 5 | gaussian_rw | 32 | 0.0794 | 0.0454 | -4.8355 | 0.7812 | 0.8438 | -0.0018 |
| seed42_max10000 | main_full_reg1e-5 | H3N2 | 5 | linear_drift_var1 | 32 | 0.0691 | 0.0467 | -6.8971 | 0.8438 | 0.8750 | 0.1288 |
| seed42_max10000 | main_full_reg1e-5 | H3N2 | 5 | linear_drift_var2 | 32 | 0.0670 | 0.0437 | -6.9489 | 0.8750 | 0.8750 | 0.1548 |
| seed42_max10000 | main_full_reg1e-5 | H3N2 | 5 | persistence | 32 | 0.0793 | 0.0453 | NA | NA | NA | 0.0000 |
| seed7_max5000 | main_full_reg1e-5 | H1N1 | 3 | gaussian_rw | 28 | 0.1450 | 0.0918 | -2.1262 | 0.8571 | 0.8571 | -0.0028 |
| seed7_max5000 | main_full_reg1e-5 | H1N1 | 3 | linear_drift_var1 | 28 | 0.1410 | 0.0979 | -1.9936 | 0.9286 | 0.9286 | 0.0242 |
| seed7_max5000 | main_full_reg1e-5 | H1N1 | 3 | linear_drift_var2 | 28 | 0.1385 | 0.0915 | -2.2757 | 0.9286 | 0.9286 | 0.0417 |
| seed7_max5000 | main_full_reg1e-5 | H1N1 | 3 | persistence | 28 | 0.1446 | 0.0913 | NA | NA | NA | 0.0000 |
| seed7_max5000 | main_full_reg1e-5 | H1N1 | 4 | gaussian_rw | 28 | 0.1258 | 0.0717 | -4.4990 | 0.8571 | 0.9286 | -0.0028 |
| seed7_max5000 | main_full_reg1e-5 | H1N1 | 4 | linear_drift_var1 | 28 | 0.1221 | 0.0760 | -4.1114 | 0.9286 | 0.9286 | 0.0270 |
| seed7_max5000 | main_full_reg1e-5 | H1N1 | 4 | linear_drift_var2 | 28 | 0.1203 | 0.0714 | -4.5826 | 0.9286 | 0.9286 | 0.0409 |
| seed7_max5000 | main_full_reg1e-5 | H1N1 | 4 | persistence | 28 | 0.1254 | 0.0713 | NA | NA | NA | 0.0000 |
| seed7_max5000 | main_full_reg1e-5 | H1N1 | 5 | gaussian_rw | 28 | 0.1132 | 0.0617 | -6.5228 | 0.8571 | 0.8929 | -0.0028 |
| seed7_max5000 | main_full_reg1e-5 | H1N1 | 5 | linear_drift_var1 | 28 | 0.1096 | 0.0647 | -6.2730 | 0.8929 | 0.9286 | 0.0287 |
| seed7_max5000 | main_full_reg1e-5 | H1N1 | 5 | linear_drift_var2 | 28 | 0.1083 | 0.0611 | -6.7360 | 0.8571 | 0.9286 | 0.0409 |
| seed7_max5000 | main_full_reg1e-5 | H1N1 | 5 | persistence | 28 | 0.1129 | 0.0614 | NA | NA | NA | 0.0000 |
| seed7_max5000 | h3n2_diag_reg1e-4 | H3N2 | 3 | gaussian_rw | 28 | 0.1388 | 0.0852 | -2.9514 | 0.8929 | 0.9286 | -0.0020 |
| seed7_max5000 | h3n2_diag_reg1e-4 | H3N2 | 3 | linear_drift_var1 | 28 | 0.1174 | 0.0800 | -3.1423 | 0.8929 | 0.8929 | 0.1520 |
| seed7_max5000 | h3n2_diag_reg1e-4 | H3N2 | 3 | linear_drift_var2 | 28 | 0.1182 | 0.0755 | -3.1699 | 0.8929 | 0.9286 | 0.1463 |
| seed7_max5000 | main_full_reg1e-5 | H3N2 | 3 | gaussian_rw | 28 | 0.1388 | 0.0852 | -2.9186 | 0.9286 | 0.9286 | -0.0020 |
| seed7_max5000 | main_full_reg1e-5 | H3N2 | 3 | linear_drift_var1 | 28 | 0.1174 | 0.0800 | -3.1098 | 0.8929 | 0.8929 | 0.1520 |
| seed7_max5000 | main_full_reg1e-5 | H3N2 | 3 | linear_drift_var2 | 28 | 0.1182 | 0.0755 | -3.0807 | 0.8929 | 0.9286 | 0.1463 |
| seed7_max5000 | main_full_reg1e-5 | H3N2 | 3 | persistence | 28 | 0.1385 | 0.0850 | NA | NA | NA | 0.0000 |
| seed7_max5000 | h3n2_diag_reg1e-4 | H3N2 | 4 | gaussian_rw | 28 | 0.1233 | 0.0737 | -4.2903 | 0.8929 | 0.8929 | -0.0021 |
| seed7_max5000 | h3n2_diag_reg1e-4 | H3N2 | 4 | linear_drift_var1 | 28 | 0.1030 | 0.0686 | -4.6945 | 0.8929 | 0.8929 | 0.1628 |
| seed7_max5000 | h3n2_diag_reg1e-4 | H3N2 | 4 | linear_drift_var2 | 28 | 0.1041 | 0.0648 | -4.7813 | 0.9286 | 0.9286 | 0.1538 |
| seed7_max5000 | main_full_reg1e-5 | H3N2 | 4 | gaussian_rw | 28 | 0.1233 | 0.0737 | -3.8063 | 0.8214 | 0.8214 | -0.0021 |
| seed7_max5000 | main_full_reg1e-5 | H3N2 | 4 | linear_drift_var1 | 28 | 0.1030 | 0.0686 | -4.5134 | 0.8571 | 0.8929 | 0.1628 |
| seed7_max5000 | main_full_reg1e-5 | H3N2 | 4 | linear_drift_var2 | 28 | 0.1041 | 0.0648 | -4.4281 | 0.8571 | 0.9286 | 0.1538 |
| seed7_max5000 | main_full_reg1e-5 | H3N2 | 4 | persistence | 28 | 0.1230 | 0.0735 | NA | NA | NA | 0.0000 |
| seed7_max5000 | h3n2_diag_reg1e-4 | H3N2 | 5 | gaussian_rw | 28 | 0.1105 | 0.0614 | -6.5301 | 0.8929 | 0.8929 | -0.0020 |
| seed7_max5000 | h3n2_diag_reg1e-4 | H3N2 | 5 | linear_drift_var1 | 28 | 0.0926 | 0.0577 | -6.9548 | 0.8929 | 0.9286 | 0.1610 |
| seed7_max5000 | h3n2_diag_reg1e-4 | H3N2 | 5 | linear_drift_var2 | 28 | 0.0937 | 0.0543 | -7.1194 | 0.9286 | 0.9286 | 0.1502 |
| seed7_max5000 | main_full_reg1e-5 | H3N2 | 5 | gaussian_rw | 28 | 0.1105 | 0.0614 | -5.7594 | 0.8214 | 0.8571 | -0.0020 |
| seed7_max5000 | main_full_reg1e-5 | H3N2 | 5 | linear_drift_var1 | 28 | 0.0926 | 0.0577 | -6.6916 | 0.8929 | 0.8929 | 0.1610 |
| seed7_max5000 | main_full_reg1e-5 | H3N2 | 5 | linear_drift_var2 | 28 | 0.0937 | 0.0543 | -6.6365 | 0.8571 | 0.9286 | 0.1502 |
| seed7_max5000 | main_full_reg1e-5 | H3N2 | 5 | persistence | 28 | 0.1103 | 0.0613 | NA | NA | NA | 0.0000 |

## Mejores modelos por cache

| cache | subtipo | mejor RMSE | d | RMSE | mejor NLL gaussiano | d | mean NLL | cov90 | cov95 |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| seed42_max10000 | H1N1 | persistence | 5 | 0.0695 | linear_drift_var2 (main_full_reg1e-5) | 5 | -7.6204 | 0.8966 | 0.9310 |
| seed42_max10000 | H3N2 | linear_drift_var2 | 5 | 0.0670 | linear_drift_var2 (h3n2_diag_reg1e-4) | 5 | -7.5580 | 0.8750 | 0.8750 |
| seed7_max5000 | H1N1 | linear_drift_var2 | 5 | 0.1083 | linear_drift_var2 (main_full_reg1e-5) | 5 | -6.7360 | 0.8571 | 0.9286 |
| seed7_max5000 | H3N2 | linear_drift_var1 | 5 | 0.0926 | linear_drift_var2 (h3n2_diag_reg1e-4) | 5 | -7.1194 | 0.9286 | 0.9286 |

## Figuras

- `figures/gisaid/pca_sde_robustness_rmse.png`
- `figures/gisaid/pca_sde_robustness_rmse.pdf`
- `figures/gisaid/pca_sde_robustness_nll.png`
- `figures/gisaid/pca_sde_robustness_nll.pdf`

## Lectura metodologica prudente

- Si H1N1 mantiene como mejor RMSE a `persistence` o queda muy cerca de `gaussian_rw`, la lectura de drift debil se conserva.
- Si H3N2 mantiene mejor RMSE y NLL con `linear_drift_var2`, la evidencia de drift lineal util es robusta frente al cambio de cache/seed.
- La corrida diagonal de H3N2 es una prueba acotada de sensibilidad de covarianza, no una nueva calibracion completa.
- Estos resultados siguen siendo dinamica probabilistica en PCA space; no prueban generacion ni una SDE final en el espacio completo.
