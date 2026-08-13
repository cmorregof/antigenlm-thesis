# Robustness Panel Update

Date: 2026-05-05

This update continues from the previous pre-submission pass rather than repeating the full five-phase audit.

## Files Added

- `paper_revision_outputs/run_robustness_panel.py`
- `paper_revision_outputs/robustness_panel_run.log`
- `paper_revision_outputs/robustness_panel_results.json`
- `paper_revision_outputs/robustness_panel_summary.md`

## Inputs Reused

- Full cached embeddings: `results/embeddings_cache_full_all_available.pkl`
- Existing helper functions from `latent_geometry_full_analysis.py`
- Existing deduplication rule: exact HA+NA duplicate removal, retaining first representative
- Same pair-sampling scale as the manuscript: 200,000 requested pairs per subtype per seed
- Seeds: 42, 7, 123

No embeddings were regenerated, no model checkpoint was loaded, no complete sequences were printed, and no raw data were modified.

## Analyses Completed

1. Exact HA+NA-deduplicated pair-sampled Spearman correlations.
2. NA-only nucleotide Hamming correlations.
3. Simple frame-0 translated amino-acid Hamming proxy correlations.
4. Date-label permutation control for local temporal-neighborhood enrichment.

## Key Results

### Deduplicated Spearman Correlations

| metric | H1N1 rho mean | H3N2 rho mean |
|---|---:|---:|
| Temporal distance | 0.0690 | 0.1497 |
| Nucleotide HA Hamming | 0.8086 | 0.5883 |
| Nucleotide NA Hamming | 0.7677 | 0.5287 |
| Nucleotide HA+NA Hamming | 0.8358 | 0.6315 |
| Amino-acid HA+NA Hamming | 0.7563 | 0.6131 |

Interpretation: molecular organization remains strong after exact HA+NA deduplication. The HA+NA nucleotide proxy remains strongest among nucleotide proxies in both subtypes.

### Date-Label Permutation

| subtype | k | true median months | permuted median months |
|---|---:|---:|---:|
| H1N1 | 5 | 2.00 | 42.33 |
| H1N1 | 10 | 2.00 | 42.67 |
| H1N1 | 20 | 2.00 | 42.67 |
| H3N2 | 5 | 2.00 | 35.00 |
| H3N2 | 10 | 2.00 | 35.00 |
| H3N2 | 20 | 2.00 | 35.00 |

Interpretation: fixing the latent neighbor graph but permuting collection months within subtype removes the short-range temporal enrichment. This supports the claim that local temporal coherence is tied to observed temporal labels, not only to the marginal date distribution.

## Manuscript Changes Made

Updated:

- `papers/paper_1_latent_geometry_full/main.tex`
- `paper_revision_outputs/revised_main.tex`
- `paper_revision_outputs/03_revision_summary.md`
- `paper_revision_outputs/05_submission_strategy.md`

Manuscript additions:

- Methods now mention the robustness panel, NA-only distances, simple amino-acid translation proxies, deduplicated pair correlations, and date-label permutation.
- Results now include a focused robustness subsection with two new tables:
  - deduplicated Spearman robustness table;
  - date-label permutation table.
- Discussion now explicitly notes that robustness checks strengthen the molecular and local temporal interpretations.
- Limitations now distinguish completed controls from still-missing broader rank-preservation diagnostics.
- Reproducibility section now names the robustness script and output files.

## Remaining Scientific Risks

- No antigenic assay, HI, neutralization, or antigenic-cartography validation.
- No phylogenetic distance, clade, lineage, geographic, or host-stratified validation.
- Amino-acid distances are simple translation proxies, not curated protein alignments.
- No kNN molecular retrieval precision or trustworthiness/continuity analysis yet.
- No prospective forecasting validation.

## Current Submission Readiness

The paper is now closer to a credible BMC Bioinformatics or Scientific Reports submission candidate. Remaining work is mainly editorial/reproducibility cleanup plus journal-compliant data availability and checkpoint provenance language.
