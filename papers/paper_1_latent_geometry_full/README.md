# Paper 1: full latent geometry audit

This folder contains an Overleaf-ready draft generated from:

- `results/latent_geometry_full_metrics.json`
- `results/latent_geometry_full_summary.md`
- `results/full_data_audit_summary.md`
- `figures/latent_geometry_full/`

## What this paper claims

- It audits the geometry of local AntigenLM embeddings for Influenza A.
- It reports aggregate molecular, temporal, PCA, TwoNN, and local-neighborhood diagnostics.
- It uses all cached valid embeddings for global analyses and pair sampling for pairwise correlations.

## What this paper does not claim

- It does not reproduce AntigenLM Figure 3A or Figure 3B.
- It does not generate sequences.
- It does not optimize mutations.
- It does not make clinical or vaccine recommendations.
- It does not validate a full predictive SDE.

## Overleaf

Upload the contents of this folder to Overleaf and compile `main.tex` with `pdfLaTeX` and BibTeX.

## Figures copied

- `records_by_year_subtype.pdf`
- `sequence_length_distributions.pdf`
- `spearman_latent_vs_distances.pdf`
- `pca_scree_global.pdf`
- `pca_cumulative_global.pdf`
- `pca_2d_by_subtype.pdf`
- `pca_2d_by_year.pdf`
- `twonn_sensitivity.pdf`
- `twonn_fit_example.pdf`
- `temporal_local_neighbors_h1n1_dedup.pdf`
- `temporal_local_neighbors_h3n2_dedup.pdf`

## Figures missing

- None

## Rebuild

From the repository root:

```bash
venv_antigenlm/bin/python make_paper_1_latent_geometry_full.py
```
