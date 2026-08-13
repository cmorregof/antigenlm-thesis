# PCA Anisotropy and Dominant-PC Controls

Outputs are aggregate-only and do not redistribute sequences or accession-level metadata.

## PCA Summary

| representation | n | n80 | n90 | n95 | n99 | participation ratio | PC1 variance | PC2 variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 111,756 | 2 | 2 | 3 | 10 | 1.91 | 0.689 | 0.213 |
| l2_normalized | 111,756 | 2 | 2 | 3 | 10 | 1.81 | 0.720 | 0.181 |
| deduplicated_original | 82,306 | 2 | 2 | 3 | 11 | 1.93 | 0.684 | 0.220 |
| deduplicated_l2_normalized | 82,306 | 2 | 2 | 3 | 11 | 1.82 | 0.715 | 0.187 |

## HA+NA Hamming Spearman After Representation Transformations

| representation | subtype | rho mean | rho sd | valid pairs mean |
|---|---|---:|---:|---:|
| original | H1N1 | 0.8533 | 0.0007 | 99948 |
| original | H3N2 | 0.6704 | 0.0031 | 99747 |
| l2_normalized | H1N1 | 0.8386 | 0.0008 | 99948 |
| l2_normalized | H3N2 | 0.6687 | 0.0031 | 99747 |
| remove_global_PC1 | H1N1 | 0.9096 | 0.0003 | 99948 |
| remove_global_PC1 | H3N2 | 0.7262 | 0.0022 | 99747 |
| remove_global_PC1_PC2 | H1N1 | 0.4073 | 0.0025 | 99948 |
| remove_global_PC1_PC2 | H3N2 | 0.7251 | 0.0019 | 99747 |
| remove_global_PC1_PC2_PC3 | H1N1 | 0.2692 | 0.0027 | 99948 |
| remove_global_PC1_PC2_PC3 | H3N2 | 0.5330 | 0.0011 | 99747 |

These controls test whether the molecular signal is concentrated entirely in dominant global directions. They do not address pooling-token ablations, which require regenerating embeddings.
