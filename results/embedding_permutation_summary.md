# Embedding Permutation Negative Control

Embeddings were permuted among records within subtype before recomputing HA+NA Hamming Spearman correlations. The output is aggregate-only.

| subtype | true rho mean | true rho sd | permuted rho mean | permuted mean sd across pair seeds | valid pairs mean |
|---|---:|---:|---:|---:|---:|
| H1N1 | 0.8533 | 0.0007 | 0.0011 | 0.0009 | 99948 |
| H3N2 | 0.6704 | 0.0031 | 0.0001 | 0.0008 | 99747 |

The near-zero permuted correlations show that the molecular signal depends on the correct record-embedding correspondence, not only on the marginal AntigenLM vector cloud.
