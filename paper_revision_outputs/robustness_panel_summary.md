# Robustness Panel Results

Created at local run time: `2026-05-05 09:07:58 -05`

This focused panel reuses the full cached embeddings and does not reload AntigenLM, regenerate embeddings, print sequences, or modify raw data.

## Inputs

- cache: `results/embeddings_cache_full_all_available.pkl`
- pair samples per subtype/seed: `200,000`
- pair seeds: `42, 7, 123`
- temporal k values: `5, 10, 20`
- temporal permutation seeds: `42, 7, 123`

## Deduplicated Pair-Sampled Spearman Correlations

Exact HA+NA duplicates were removed before sampling pairs. Amino-acid distances are simple frame-0 translations with ambiguous codons mapped to `X`; they are robustness proxies, not curated protein alignments.

| metric | subtype | rho mean | rho sd | valid pairs mean | omitted pairs mean |
|---|---|---:|---:|---:|---:|
| aa_hamming_ha | H1N1 | 0.7228 | 0.0019 | 199892 | 108 |
| aa_hamming_ha | H3N2 | 0.6352 | 0.0007 | 199981 | 19 |
| aa_hamming_ha_na | H1N1 | 0.7563 | 0.0013 | 199876 | 124 |
| aa_hamming_ha_na | H3N2 | 0.6131 | 0.0008 | 199352 | 648 |
| aa_hamming_na | H1N1 | 0.6832 | 0.0010 | 199985 | 15 |
| aa_hamming_na | H3N2 | 0.4489 | 0.0015 | 199359 | 641 |
| nt_hamming_ha | H1N1 | 0.8086 | 0.0013 | 199892 | 108 |
| nt_hamming_ha | H3N2 | 0.5883 | 0.0006 | 199981 | 19 |
| nt_hamming_ha_na | H1N1 | 0.8358 | 0.0010 | 199877 | 123 |
| nt_hamming_ha_na | H3N2 | 0.6315 | 0.0015 | 199348 | 652 |
| nt_hamming_na | H1N1 | 0.7677 | 0.0010 | 199985 | 15 |
| nt_hamming_na | H3N2 | 0.5287 | 0.0025 | 199355 | 645 |
| temporal | H1N1 | 0.0690 | 0.0031 | 200000 | 0 |
| temporal | H3N2 | 0.1497 | 0.0017 | 200000 | 0 |

## Temporal Label-Permutation Control

The latent neighbor graph is held fixed, while collection months are permuted within subtype. If temporal coherence were mostly a generic consequence of date distribution alone, permuted labels would give similar neighbor time differences. They do not.

| subtype | k | true median months | permuted median months mean | permuted sd | random median months mean | true/permuted | true/random |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1N1 | 5 | 2.00 | 42.33 | 0.47 | 42.33 | 0.047 | 0.047 |
| H1N1 | 10 | 2.00 | 42.67 | 0.47 | 42.67 | 0.047 | 0.047 |
| H1N1 | 20 | 2.00 | 42.67 | 0.47 | 42.00 | 0.047 | 0.048 |
| H3N2 | 5 | 2.00 | 35.00 | 0.00 | 35.00 | 0.057 | 0.057 |
| H3N2 | 10 | 2.00 | 35.00 | 0.00 | 35.00 | 0.057 | 0.057 |
| H3N2 | 20 | 2.00 | 35.00 | 0.00 | 35.00 | 0.057 | 0.057 |

## Manuscript-Relevant Interpretation

- Molecular correlations remain strong after exact HA+NA deduplication.
- NA-only nucleotide correlations are lower than HA-only in both subtypes in this panel, while combined HA+NA remains strongest among nucleotide proxies.
- Amino-acid Hamming proxies preserve the same broad pattern as nucleotide proxies, but should be treated cautiously because no curated protein alignment was performed.
- Date-label permutation raises neighbor temporal medians from 2 months to tens of months, supporting the claim that local temporal coherence is tied to true collection dates rather than only the marginal date distribution.

## Remaining Caveats

- These controls still do not validate antigenic similarity, phylogenetic distance, clade identity, vaccine relevance, or prospective forecasting.
- Amino-acid distances here are simple translations from available nucleotide strings and do not replace curated HA/NA protein alignments.
- Pairwise sampled distances remain dependent observations; effect sizes and robustness are more important than nominal p-values.
