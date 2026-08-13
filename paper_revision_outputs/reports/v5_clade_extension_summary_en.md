# v5 Clade Extension Summary

## What Was Added

The manuscript now includes a clade-label enrichment extension using private GISAID EpiFlu metadata from `EPI_SET_260506bu`. The analysis joins the exact HA+NA-deduplicated embedding cache to GISAID metadata by isolate identifier (`epi_isl` / `Isolate_Id`) and reports only aggregate outputs.

## Inputs and Coverage

- Full embedding cache: `results/embeddings_cache_full_all_available.pkl`
- Metadata join table: `data/gisaid_metadata_private/gisaid_epiflu_isolates_2000_2022_epi_set_260506bu_joined_dedup_cache.csv`
- Deduplicated cache size: 82,306 records
- Exact HA+NA duplicates removed: 29,450 records
- Metadata join coverage: 81,943 / 82,306 records (99.56%)
- Assigned non-unassigned clade coverage: 76,238 / 82,306 records (92.63%)
- H1N1 assigned clade coverage: 36,156 / 36,753 records (98.38%)
- H3N2 assigned clade coverage: 40,082 / 45,553 records (87.99%)

## Main Clade Enrichment Result

At k=5, latent nearest-neighbor clade precision is high in both subtypes:

| Subtype | precision@5 | subtype random | enrichment |
|---|---:|---:|---:|
| H1N1 | 0.9149 | 0.2409 | 3.80x |
| H3N2 | 0.8609 | 0.0689 | 12.49x |

The within-subtype clade-label permutation control reduces precision to the random-baseline range:

- H1N1 k=5 permutation mean: 0.2406
- H3N2 k=5 permutation mean: 0.0690

This supports the claim that latent neighborhoods are enriched for observed GISAID clade membership, not merely for the marginal clade distribution.

## Temporal Stratification Control

The temporal control is important and changes the interpretation. Random candidates were restricted to assigned-clade records from the same subtype within the stated collection-month window.

| Subtype | Window | true precision@5 | stratified random | enrichment |
|---|---:|---:|---:|---:|
| H1N1 | ±6 months | 0.9149 | 0.5892 | 1.55x |
| H1N1 | ±12 months | 0.9149 | 0.5405 | 1.69x |
| H1N1 | ±24 months | 0.9149 | 0.4681 | 1.95x |
| H3N2 | ±6 months | 0.8610 | 0.2707 | 3.18x |
| H3N2 | ±12 months | 0.8610 | 0.2288 | 3.76x |
| H3N2 | ±24 months | 0.8609 | 0.1548 | 5.56x |

Interpretation: the clade signal is real but partially coupled to temporal clade turnover, especially in H1N1. The manuscript therefore should not claim that the embedding captures time-independent phylogenetic structure. The supported claim is local evolutionary-taxonomic coherence under GISAID clade annotations.

## Files Generated or Updated

- `paper_revision_outputs/run_clade_enrichment_analysis.py`
- `paper_revision_outputs/logs/clade_enrichment_run.log`
- `results/gisaid_clade_enrichment_results.json`
- `results/gisaid_clade_enrichment_summary.md`
- `results/gisaid_clade_enrichment_summary_es.md`
- `figures/latent_geometry_full/clade_precision_enrichment.pdf`
- `figures/latent_geometry_full/clade_precision_enrichment.png`
- `papers/paper_1_latent_geometry_full/figures/clade_precision_enrichment.pdf`
- `papers/paper_1_latent_geometry_full/main.tex`
- `paper_revision_outputs/revised_main.tex`
- `paper_revision_outputs/latent_geometry_manuscript_compiled_v5_clade.pdf`

## Editorial Reading

The clade extension materially improves the paper. It turns the manuscript from a mostly geometric embedding audit into a stronger computational virology audit because it connects latent neighborhoods to an external surveillance annotation. However, the temporal stratification control prevents overclaiming: H1N1 clade enrichment is strongly time-coupled, and H3N2 remains more robust but is also attenuated. The paper is stronger than the BMC Bioinformatics version, but for Virus Evolution it still needs careful framing and ideally an additional phylogenetic-distance or geography/host-stratified analysis.
