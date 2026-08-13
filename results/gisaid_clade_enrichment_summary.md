# GISAID Clade Enrichment in AntigenLM Latent Neighborhoods

This analysis uses the exact HA+NA-deduplicated embedding cache and the private GISAID EpiFlu metadata export from `EPI_SET_260506bu`. No sequences or accession-level metadata are redistributed here.

## Inputs

- cache: `results/embeddings_cache_full_all_available.pkl`
- joined metadata: `data/gisaid_metadata_private/gisaid_epiflu_isolates_2000_2022_epi_set_260506bu_joined_dedup_cache.csv`
- deduplicated records: `82,306`
- exact HA+NA duplicates removed: `29,450`

## Metadata Coverage

| group | cache n | joined n | join % | assigned clade n | assigned clade % |
| --- | --- | --- | --- | --- | --- |
| H1N1 | 36,753 | 36,723 | 99.92% | 36,156 | 98.38% |
| H3N2 | 45,553 | 45,220 | 99.27% | 40,082 | 87.99% |
| Combined | 82,306 | 81,943 | 99.56% | 76,238 | 92.63% |

## Clade Precision and Controls

| subtype | k | precision@k | random baseline | permutation mean | enrichment |
| --- | --- | --- | --- | --- | --- |
| H1N1 | 5 | 0.9149 | 0.2409 | 0.2406 | 3.80x |
| H1N1 | 10 | 0.8933 | 0.2403 | 0.2405 | 3.72x |
| H1N1 | 20 | 0.8647 | 0.2404 | 0.2406 | 3.60x |
| H3N2 | 5 | 0.8609 | 0.0689 | 0.0690 | 12.49x |
| H3N2 | 10 | 0.8325 | 0.0689 | 0.0689 | 12.08x |
| H3N2 | 20 | 0.7927 | 0.0688 | 0.0689 | 11.51x |

## Temporal Stratification Sensitivity (k=5)

Random candidates are restricted to assigned-clade records from the same subtype within the stated collection-month window. True precision is recomputed on the same eligible query set.

| subtype | k | window | true precision | stratified random | enrichment | eligible queries |
| --- | --- | --- | --- | --- | --- | --- |
| H1N1 | 5 | ±6 mo | 0.9149 | 0.5892 | 1.55x | 36,156 |
| H1N1 | 5 | ±12 mo | 0.9149 | 0.5405 | 1.69x | 36,156 |
| H1N1 | 5 | ±24 mo | 0.9149 | 0.4681 | 1.95x | 36,156 |
| H3N2 | 5 | ±6 mo | 0.8610 | 0.2707 | 3.18x | 40,080 |
| H3N2 | 5 | ±12 mo | 0.8610 | 0.2288 | 3.76x | 40,080 |
| H3N2 | 5 | ±24 mo | 0.8609 | 0.1548 | 5.56x | 40,081 |

## Interpretation

- Latent neighborhoods are strongly enriched for GISAID clade membership in both subtypes.
- The within-subtype clade-label permutation control drops precision to the random-baseline range, indicating that enrichment depends on the observed clade assignments rather than only on class imbalance.
- Temporally stratified random baselines are higher than global random baselines, as expected because clades are temporally structured; enrichment remains above the stratified baseline across the tested windows.
- The valid claim is local evolutionary-taxonomic coherence under GISAID clade annotations. This is not quantitative phylogenetic-distance validation, antigenic validation, immune-escape validation, vaccine-strain relevance, or forecasting validation.
