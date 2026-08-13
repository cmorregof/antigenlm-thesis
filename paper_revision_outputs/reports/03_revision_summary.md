# Revision Summary

Date: 2026-05-05

Revised manuscript:

- `papers/paper_1_latent_geometry_full/main.tex`
- Copy: `paper_revision_outputs/revised_main.tex`
- Original generated draft backup: `paper_revision_outputs/original_main_backup.tex`

## Sections Rewritten

- Title:
  - Changed from a broad "geometric structure" thesis-checkpoint title to a more precise audit title: molecular and temporal geometry in AntigenLM latent representations.
- Abstract:
  - Added dataset size, embedding definition, core quantitative results, interpretation, and explicit limitations.
  - Stated what the work does not establish: antigenic similarity, phylogenetic validity, sequence generation, vaccine relevance, and prospective forecasting.
- Introduction:
  - Reframed the paper around representation auditing before downstream state-space modeling.
  - Positioned AntigenLM within DNA/protein language models and influenza evolution.
  - Added molecular versus antigenic distance distinction.
- Methods:
  - Expanded data source, local embedding extraction, tokenizer/input format, checkpoint provenance, distance definitions, pair sampling, deduplication, PCA, TwoNN, and local temporal-neighborhood methodology.
  - Separated full-cache analyses from deduplicated local analyses.
  - Added caution about dependent pairwise distances and p-value interpretation.
- Results:
  - Reorganized around claims supported by current outputs.
  - Kept all numerical values tied to `results/latent_geometry_full_metrics.json`.
  - Explained why molecular correlation is strong while global temporal correlation is weak.
  - Interpreted PCA and TwoNN as complementary diagnostics rather than identical dimension estimates.
- Discussion:
  - Separated validated findings from plausible interpretation and future use.
  - Framed "candidate state space" as a geometric precondition, not validation of a predictive model.
- Limitations:
  - Replaced generic limitations with specific biological, methodological, intrinsic-dimension, surveillance-bias, and forecasting limitations.
- Reproducibility and Data Availability:
  - Added a section naming the main scripts and result files.
  - Stated raw sequence redistribution limitations and the need for rebuild instructions for authorized users.
- Ethics, Biosafety, and Responsible Use:
  - Added an explicit statement that the work does not generate sequences, optimize mutations, or make clinical/public-health recommendations.
- Conclusion:
  - Rewritten to preserve the supported claims while avoiding overreach.

## Claims Softened

- "Biologically meaningful latent space" was replaced by molecular organization under HA/HA+NA nucleotide Hamming proxies.
- "Evolutionary state space" was softened to a plausible candidate state space for future probabilistic modeling.
- "Low-dimensional manifold" was softened to low effective structure under PCA and TwoNN diagnostics.
- "Temporal structure" was split into weak global chronological association and strong local temporal-neighborhood enrichment.
- Antigenic, vaccine-strain, and forecasting claims were explicitly excluded.

## Claims Strengthened

- The molecular-distance result is now more clearly stated as the strongest supported finding.
- The local temporal-neighborhood result is now interpreted as an important local result despite weak global temporal correlation.
- Exact HA+NA deduplication is more prominent in methods and results.
- PCA and TwoNN are presented as qualitatively concordant but non-equivalent evidence of low-dimensional structure.
- The manuscript now states why p-values are not emphasized for sampled pairwise distances.

## Citations Added or Verified

Updated bibliography:

- `papers/paper_1_latent_geometry_full/references.bib`

Major citation clusters added:

- AntigenLM: `pei2026antigenlm`
- DNA/genomic language models: `ji2021dnabert`, `dallatorre2025nucleotide`, `nguyen2023hyenadna`
- Protein/viral language models: `rives2021biological`, `hie2021viral`
- GISAID: `shu2017gisaid`
- Influenza antigenic evolution: `smith2004mapping`, `koel2013substitutions`, `bedford2014integrating`
- Forecasting/phylodynamics context: `luksza2014fitness`, `neher2014predicting`, `hadfield2018nextstrain`
- Intrinsic dimension alternatives: `levina2005maximum`, `ceruti2014danco`
- Metric/neighborhood validation: `mantel1967detection`, `szekely2007distance`, `lee2009quality`
- Existing PCA, Spearman, and TwoNN references were retained and improved where needed.

## Figures and Tables

No figures were fabricated or regenerated during the revision.

Figures currently used:

- `papers/paper_1_latent_geometry_full/figures/records_by_year_subtype.pdf`
- `papers/paper_1_latent_geometry_full/figures/sequence_length_distributions.pdf`
- `papers/paper_1_latent_geometry_full/figures/spearman_latent_vs_distances.pdf`
- `papers/paper_1_latent_geometry_full/figures/pca_scree_global.pdf`
- `papers/paper_1_latent_geometry_full/figures/pca_cumulative_global.pdf`
- `papers/paper_1_latent_geometry_full/figures/pca_2d_by_subtype.pdf`
- `papers/paper_1_latent_geometry_full/figures/pca_2d_by_year.pdf`
- `papers/paper_1_latent_geometry_full/figures/twonn_sensitivity.pdf`
- `papers/paper_1_latent_geometry_full/figures/twonn_fit_example.pdf`
- `papers/paper_1_latent_geometry_full/figures/temporal_local_neighbors_h1n1_dedup.pdf`
- `papers/paper_1_latent_geometry_full/figures/temporal_local_neighbors_h3n2_dedup.pdf`

Figures that could be added without rerunning analysis:

- `figures/latent_geometry_full/pca_scree_by_subtype.pdf`
- `figures/latent_geometry_full/pca_cumulative_by_subtype.pdf`

Figures/tables that would require new analysis:

- Deduplicated Spearman correlations.
- NA-only Hamming correlations.
- Amino-acid HA/NA Hamming correlations.
- Date-permutation controls for local temporal coherence.
- Random-embedding or matched-spectrum controls.
- kNN molecular retrieval precision.
- Trustworthiness/continuity or local rank-preservation curves.
- Alternative intrinsic-dimension estimator results.

## Remaining Blockers Before Journal Submission

- The local checkpoint provenance should be described more precisely: official release, locally reconstructed checkpoint, or derived task-specific checkpoint.
- Raw data access and GISAID acknowledgement text need to be made journal-compliant.
- The manuscript has now gained one negative control and additional molecular-distance analyses through the robustness panel; broader rank-preservation and biological validation are still missing.
- Exact HA+NA-deduplicated pairwise Spearman correlations have now been run and folded into the manuscript.
- No antigenic, phylogenetic, clade, or prospective forecasting validation is available.
- Software versions and exact commands should be added to a supplement or reproducibility appendix.
- The LaTeX manuscript still needs compilation verification after citation expansion.

## Originally Recommended Immediate Next Analysis

The original pre-robustness recommendation was to run a small robustness bundle before submission:

1. Exact HA+NA-deduplicated Spearman correlations.
2. NA-only and amino-acid HA/NA Hamming correlations.
3. Date-permutation control for local temporal-neighbor enrichment.

This has now been completed; see the update below and `paper_revision_outputs/06_robustness_panel_update.md`.

## Update After Robustness Panel

Completed after the original revision pass:

- Script added: `paper_revision_outputs/run_robustness_panel.py`
- Run log: `paper_revision_outputs/robustness_panel_run.log`
- JSON results: `paper_revision_outputs/robustness_panel_results.json`
- Summary: `paper_revision_outputs/robustness_panel_summary.md`

The panel reused `results/embeddings_cache_full_all_available.pkl` and did not reload AntigenLM, regenerate embeddings, print complete sequences, or modify raw data.

Supported additions now folded into `papers/paper_1_latent_geometry_full/main.tex`:

- Exact HA+NA-deduplicated pair-sampled Spearman correlations.
- NA-only nucleotide Hamming correlations.
- Simple frame-0 translated amino-acid Hamming robustness proxies.
- Date-label permutation control for local temporal-neighbor enrichment.

Main robustness results:

- Deduplicated nucleotide HA+NA Hamming correlations remain strong: H1N1 rho mean `0.8358`; H3N2 rho mean `0.6315`.
- Deduplicated nucleotide NA-only correlations are also substantial: H1N1 rho mean `0.7677`; H3N2 rho mean `0.5287`.
- Simple amino-acid HA+NA proxies remain correlated with latent distance: H1N1 rho mean `0.7563`; H3N2 rho mean `0.6131`.
- Date-label permutation moves local-neighbor temporal medians from `2` months to `42.33-42.67` months in H1N1 and `35.00` months in H3N2, matching subtype-random baselines.

Remaining blockers are reduced but not eliminated. The paper still lacks antigenic, phylogenetic, clade, geographic/host-stratified, and prospective forecasting validation.
