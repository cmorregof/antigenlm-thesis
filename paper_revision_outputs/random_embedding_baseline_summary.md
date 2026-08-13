# Random Embedding Baseline

Created at local run time: `2026-05-05 21:29:49 -05`

This negative control samples independent 384-dimensional Gaussian vectors, normalizes each vector to unit L2 norm, and evaluates HA+NA Hamming correlations with the same subtype-specific pair-sampling protocol used for the main AntigenLM analysis.

It does not reload AntigenLM, regenerate learned embeddings, print complete sequences, or modify raw data.

## Inputs and Parameters

- cache: `results/embeddings_cache_full_all_available.pkl`
- n records: `111,756`
- random embedding dimension: `384`
- random embedding replicates: `10`
- random embedding seeds: `1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009`
- pair samples per subtype/seed: `200,000`
- pair seeds: `42, 7, 123`

## Aggregate Results

| metric | subtype | random rho mean | random rho sd | AntigenLM rho mean | AntigenLM - random | correlations summarized |
|---|---|---:|---:|---:|---:|---:|
| HA+NA Hamming | H1N1 | 0.0001 | 0.0019 | 0.8538 | 0.8537 | 30 |
| HA+NA Hamming | H3N2 | 0.0004 | 0.0021 | 0.6678 | 0.6674 | 30 |

## Interpretation

The random embedding baseline correlations are near zero, as expected for vectors independent of HA/NA sequence content. The AntigenLM HA+NA Hamming correlations are much larger than this negative control, supporting the claim that the observed molecular organization is not a trivial consequence of Euclidean distances in random high-dimensional vectors.
