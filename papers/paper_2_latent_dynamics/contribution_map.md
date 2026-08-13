# Contribution map

| contribution | evidence | figure/table | risk/limitation |
|---|---|---|---|
| Minimal probabilistic dynamics in AntigenLM PCA space | Gaussian RW, VAR(1), and VAR(2) evaluated with rolling-origin protocol | `results/pca_sde_summary.md` | Operates on PCA centroids, not full latent trajectories or sequences |
| Train-only PCA rolling-origin protocol | PCA is re-fit at each cut using only available training data | `results/pca_rolling_dynamics_summary.md` | Still retrospective and limited to 2019-2022 |
| H3N2 shows useful linear drift | Seed42: VAR(2) d=5 RMSE=0.0670, improvement 15.48% vs persistence | Rolling RMSE figure/table | Needs comparison to biological/phylogenetic baselines |
| H1N1 shows weaker or less stable drift | Seed42: persistence d=5 RMSE=0.0695; seed7 shows modest VAR(2) improvement | Robustness summary | Conclusion is sensitive to cache/seed and should be phrased cautiously |
| Probabilistic evaluation beyond point RMSE | NLL, Mahalanobis distance, coverage, covariance trace/logdet | SDE NLL and coverage figures | Negative NLL values are possible for Gaussian densities; calibration remains imperfect |
| Calibration and covariance sensitivity | Cov-reg, inflation, full vs diagonal covariance grid | `results/pca_sde_calibration_summary.md` | Choosing a model by NLL alone may conflict with RMSE or coverage |
| Robustness across cache/seed | Seed42 vs seed7 comparison preserves H3N2 drift signal more than H1N1 | Robustness RMSE/NLL figures | Only one additional seed/cache; not a full robustness campaign |
| Bridge toward SDEs | Discrete Gaussian dynamics are interpretable as minimal linear SDE discretizations | Discussion synthesis | Does not implement final `F_viab`/`F_escape` SDE or sequence-level validation |

