# Full-data latent geometry summary

This report audits the geometry of local AntigenLM embeddings. It does not generate sequences, optimize mutations, or reproduce AntigenLM forecasting figures.

## Cache and data

- cache: `results/embeddings_cache_full_FIXED.pkl`
- embeddings: `(111756, 384)`
- deduplicated HA+NA points for TwoNN/temporal locality: `82,306`

## Spearman correlations

Pairwise correlations are estimated by random pair sampling within subtype, not by the full quadratic set of all pairs.

| metric | subtype | rho mean | rho sd | valid pairs mean | omitted pairs mean |
|---|---|---:|---:|---:|---:|
| Temporal | H1N1 | 0.1059 | 0.0019 | 100000 | 0 |
| Temporal | H3N2 | 0.1249 | 0.0053 | 100000 | 0 |
| Hamming HA | H1N1 | 0.8940 | 0.0005 | 99956 | 44 |
| Hamming HA | H3N2 | 0.6512 | 0.0029 | 99996 | 4 |
| Hamming HA+NA | H1N1 | 0.8802 | 0.0008 | 99948 | 52 |
| Hamming HA+NA | H3N2 | 0.6600 | 0.0029 | 99771 | 229 |

## PCA effective dimension

| group | n | n80 | n90 | n95 | n99 | participation ratio | top10 EVR |
|---|---:|---:|---:|---:|---:|---:|---|
| global | 111,756 | 3 | 4 | 5 | 12 | 3.32 | 0.4183, 0.3267, 0.1195, 0.0651, 0.0246, 0.0104, 0.0076, 0.0062, 0.0051, 0.0034 |
| H1N1 | 46,125 | 1 | 2 | 4 | 8 | 1.39 | 0.8432, 0.0796, 0.0247, 0.0198, 0.0116, 0.0063, 0.0034, 0.0027, 0.0014, 0.0011 |
| H3N2 | 65,631 | 2 | 3 | 4 | 10 | 1.80 | 0.7309, 0.1240, 0.0551, 0.0455, 0.0130, 0.0079, 0.0048, 0.0040, 0.0033, 0.0018 |

## TwoNN sensitivity

| sample size | trim | dimension mean | dimension sd | R2 mean |
|---:|---:|---:|---:|---:|
| 5,000 | 0.01 | 2.076 | 0.016 | 0.9532 |
| 5,000 | 0.05 | 3.051 | 0.037 | 0.9645 |
| 10,000 | 0.01 | 2.050 | 0.065 | 0.9518 |
| 10,000 | 0.05 | 3.047 | 0.064 | 0.9638 |
| 20,000 | 0.01 | 2.076 | 0.010 | 0.9517 |
| 20,000 | 0.05 | 3.073 | 0.025 | 0.9620 |

## Temporal locality

| subtype | k | n points | median neighbors | median random | median ratio | mean neighbors | mean random |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1N1 | 5 | 36,753 | 2.00 | 43.00 | 0.047 | 4.74 | 53.93 |
| H1N1 | 10 | 36,753 | 2.00 | 42.00 | 0.048 | 5.60 | 53.66 |
| H1N1 | 20 | 36,753 | 2.00 | 42.00 | 0.048 | 6.67 | 53.66 |
| H3N2 | 5 | 45,553 | 2.00 | 35.00 | 0.057 | 5.21 | 49.01 |
| H3N2 | 10 | 45,553 | 2.00 | 35.00 | 0.057 | 6.04 | 49.10 |
| H3N2 | 20 | 45,553 | 2.00 | 35.00 | 0.057 | 7.16 | 49.05 |

## Figures

- `figures/latent_geometry_full/records_by_year_subtype.pdf`
- `figures/latent_geometry_full/sequence_length_distributions.pdf`
- `figures/latent_geometry_full/spearman_latent_vs_distances.pdf`
- `figures/latent_geometry_full/pca_scree_global.pdf`
- `figures/latent_geometry_full/pca_cumulative_global.pdf`
- `figures/latent_geometry_full/pca_scree_by_subtype.pdf`
- `figures/latent_geometry_full/pca_cumulative_by_subtype.pdf`
- `figures/latent_geometry_full/pca_2d_by_subtype.pdf`
- `figures/latent_geometry_full/pca_2d_by_year.pdf`
- `figures/latent_geometry_full/twonn_sensitivity.pdf`
- `figures/latent_geometry_full/twonn_fit_example.pdf`
- `figures/latent_geometry_full/temporal_local_neighbors_h1n1_dedup.pdf`
- `figures/latent_geometry_full/temporal_local_neighbors_h3n2_dedup.pdf`

## Methodological reading

- Strong Hamming correlations support molecular organization of the local checkpoint embeddings.
- Weak global temporal correlation is not a failure mode by itself, because influenza evolution is branching and nonlinear.
- Strong temporal locality indicates that latent neighborhoods are evolutionarily coherent at local scale.
- Low PCA/TwoNN effective dimension motivates reduced dynamical modeling, but does not validate forecasting or a full SDE.
