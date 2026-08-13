# Full Data Audit

This audit reports aggregate counts only. Complete sequences are not printed or redistributed.

Cache: `results/embeddings_cache_full_FIXED.pkl`
Embeddings in cache: `(111756, 384)`

## Checkpoint/cache metadata

- checkpoint path: `prediction_sequence/pytorch_model.bin`
- checkpoint size bytes: `None`
- checkpoint sha256: `6b942a0e2d6af0528a7307ff5754438ad55fdb97390297e9c0f11ffc9803dbff`
- sampling strategy: `all`
- max_per_subtype: `-1`
- seed: `42`
- max_seq_length: `4000`
- embedding_batch_size: `1`

## Records by subtype

| subtype | source paired records | valid HA+NA+date records | cached embeddings | missing from cache |
|---|---:|---:|---:|---:|
| H1N1 | 46,125 | 46,125 | 46,125 | 0 |
| H3N2 | 65,631 | 65,631 | 65,631 | 0 |

## Sequence length summary

| subtype | segment | n | mean | median | p05 | p95 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H1N1 | HA | 46,125 | 1729.0 | 1734.0 | 1701.0 | 1777.0 | 1695 | 1799 |
| H1N1 | NA | 46,125 | 1423.9 | 1420.0 | 1410.0 | 1458.0 | 1350 | 1493 |
| H3N2 | HA | 65,631 | 1723.9 | 1735.0 | 1701.0 | 1762.0 | 1672 | 1800 |
| H3N2 | NA | 65,631 | 1429.4 | 1436.0 | 1410.0 | 1466.0 | 1251 | 1590 |

## Duplicate summary

| subtype | duplicate HA | duplicate NA | duplicate HA+NA |
|---|---:|---:|---:|
| H1N1 | 15,499 | 19,482 | 9,372 |
| H3N2 | 29,344 | 32,455 | 20,078 |

Exact duplicate embeddings in cache: `29,450`.

## Figures

- `figures/latent_geometry_full/records_by_year_subtype.pdf`
- `figures/latent_geometry_full/sequence_length_distributions.pdf`
