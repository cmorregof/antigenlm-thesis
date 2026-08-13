# Contribution map

| contribution | evidence | figure/table | risk/limitation |
|---|---|---|---|
| Geometric audit of AntigenLM latent space | Consolidated analysis of temporal distance, Hamming HA/HA+NA, PCA, TwoNN, and local neighbors | `results/master_results_summary.md`; `results/geometry_summary.md` | Audit is tied to one model checkpoint and GISAID-derived records |
| Latent distance preserves molecular similarity | Spearman Hamming HA avg rho=0.6338; Hamming HA+NA avg rho=0.6586 | Spearman table | Hamming is a molecular proxy, not direct antigenic/immunological distance |
| Temporal global distance is weak | Temporal Spearman avg rho=0.1436 | Spearman table | Weak global temporal correlation does not rule out nonlinear or local temporal structure |
| Local temporal neighborhoods are coherent | Dedup H1N1 k=5 median neighbors 2 months vs random 58; H3N2 k=5 median 3 vs random 72 | Temporal-neighbor dedup figures | kNN structure can reflect sampling density and surveillance bias |
| Effective dimension is low | PCA global: n90=3, n95=4, n99=12, PR=1.89 | PCA scree/cumulative | PCA measures linear variance and is affected by anisotropy |
| Intrinsic dimension appears low | TwoNN range approx. 3.5-4.8 with R2 approx. 0.97-0.98 after deduplication | TwoNN/PCA table | TwoNN is preliminary and sensitive to trimming; requires complementary estimators |
| Geometry motivates continuous latent modeling | Molecular preservation, local temporal coherence, and low dimension align with requirements for dynamics | Discussion synthesis | Motivation is not proof that an SDE works |

