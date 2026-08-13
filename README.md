# Latent Geometry Audit of AntigenLM Influenza Representations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains code and aggregate outputs for:

> **A Geometric Audit of AntigenLM Latent Representations for Influenza A HA/NA Sequences**  
> Carlos Manuel Orrego-Franco, Juan Carlos Riaño-Rojas  
> *arXiv / pre-journal manuscript, 2026*

We present a systematic geometric audit of cached AntigenLM embeddings for 111,756 Influenza A HA/NA records from H1N1 and H3N2. The analysis asks whether latent Euclidean distances preserve molecular sequence similarity, whether temporal structure is present locally or globally, whether the embedding cloud has low effective dimension, and whether latent neighborhoods are enriched for GISAID clade labels relative to random and simple sequence-derived baselines.

**Preprint**: arXiv link to be added upon upload  
**Paper DOI**: to be added upon publication

## Key Findings

| Property | Result |
|---|---:|
| HA+NA Hamming correlation (H1N1) | rho = 0.854 |
| HA+NA Hamming correlation (H3N2) | rho = 0.668 |
| Random embedding baseline (H1N1) | rho = 0.0001 |
| Random embedding baseline (H3N2) | rho = 0.0004 |
| Embedding permutation control | near-zero rho within subtype |
| Global PCA n95 | 3 components |
| TwoNN intrinsic dimension | 3.9-5.5 (trim-dependent) |
| Median latent neighbor time difference (H1N1) | 2 months vs. 43 months (random) |
| Median latent neighbor time difference (H3N2) | 2 months vs. 35 months (random) |
| Clade precision@5 (H1N1) | 0.9149, 3.80x over subtype-matched random |
| Clade precision@5 (H3N2) | 0.8609, 12.49x over subtype-matched random |
| Sequence baseline context | 5-mer/Hamming neighborhoods exceed AntigenLM for sampled clade retrieval |
| Time-stratified clade enrichment at +/-6 months | 1.55x H1N1, 3.18x H3N2 |

The embedding cloud is strongly sequence-aware, low-dimensional under the tested diagnostics, locally temporally coherent, and enriched for sequence-derived evolutionary-taxonomic clade labels. It does **not** establish antigenic similarity, quantitative phylogenetic preservation, vaccine-strain relevance, sequence-generation reliability, clinical/public-health utility, or forecasting performance.

## Repository Structure

```text
antigenlm-thesis/
├── thesis/                               # Thesis document, drafts, and notes
│   ├── current/
│   ├── drafts/
│   └── notes/
├── docs/                                 # Setup and organization notes
├── latent_geometry.py                    # Embedding extraction and exploratory geometry
├── latent_geometry_full_analysis.py      # Main analysis: correlations, PCA, TwoNN, temporal
├── make_paper_1_latent_geometry_full.py  # Reproduce manuscript tables and figures
├── papers/
│   └── paper_1_latent_geometry_full/     # LaTeX manuscript package
├── paper_revision_outputs/
│   ├── run_robustness_panel.py           # Deduplication and robustness correlations
│   ├── run_random_embedding_baseline.py  # Random embedding negative control
│   ├── run_clade_enrichment_analysis.py  # GISAID clade-label enrichment controls
│   ├── run_preprocessing_counts.py       # Aggregate preprocessing/cache counts
│   ├── run_sequence_baselines.py         # k-mer/Hamming sequence-neighborhood baselines
│   ├── run_embedding_permutation_control.py
│   ├── run_alignment_robustness.py       # Small Biopython pairwise-alignment panel
│   ├── run_pca_anisotropy_controls.py
│   ├── run_temporal_dedup_sensitivity.py
│   ├── reports/                          # Revision reports and editorial notes
│   ├── manuscript_versions/              # Compiled/intermediate manuscript versions
│   └── logs/                             # Build and run logs
├── results/                              # Aggregate outputs and local embedding caches
├── figures/
│   └── latent_geometry_full/             # Analysis figures in PDF/PNG format
├── data/                                 # Local data only; not redistributed
├── prediction_sequence/                  # Local model checkpoint files
├── subtype_classifier/                   # Local auxiliary checkpoint files
└── archive/                              # Old Overleaf/ZIP exports kept for traceability
```

See `docs/PROJECT_STRUCTURE.md` for the working organization map.

## Data Availability

**Raw sequences are not included in this repository.**

The analysis uses Influenza A HA/NA sequences from the [GISAID EpiFlu database](https://www.gisaid.org/). Access requires a GISAID account and agreement to GISAID data-use terms. Complete sequences cannot be redistributed here.

This repository provides:

- aggregate results as JSON and Markdown summaries;
- analysis code to reproduce the reported aggregate outputs from an authorized local cache;
- manuscript figures and LaTeX source;
- scripts for deduplication robustness checks, the random embedding negative control, and aggregate clade-label enrichment controls.
- scripts for preprocessing counts, sequence baselines, embedding permutation controls, PCA anisotropy controls, temporal duplicate-representative sensitivity, and a limited alignment-aware robustness panel.

The AntigenLM checkpoint used for embedding extraction is documented in the manuscript with SHA256:

```text
6b942a0e2d6af0528a7307ff5754438ad55fdb97390297e9c0f11ffc9803dbff
```

The expected local fine-tuned checkpoint path in this workflow is `prediction_sequence/pytorch_model.bin`.
The `.bin` weights are not redistributed in this repository. Authorized users should obtain model
weights only from the official release associated with the original AntigenLM paper:
Pei, Chi, and Kang, *AntigenLM: Structure-Aware DNA Language Modeling for Influenza*
([arXiv:2602.09067](https://arxiv.org/abs/2602.09067);
[OpenReview/ICLR 2026](https://openreview.net/forum?id=Y0zPlHDO5p)). The ICLR version states
that code, preprocessing scripts, and trained checkpoints are to be released by the original
authors. After download, verify that the local checkpoint matches the SHA256 documented above.

A full GISAID acknowledgement table with accession identifiers, originating laboratories, and
submitting laboratories should be included as Supplementary Table S1 in the journal submission
package according to GISAID requirements.

The clade-label enrichment analysis uses local GISAID EpiFlu metadata exports associated with
`EPI_SET_260506bu`. This repository redistributes only aggregate clade-enrichment results; it
does not redistribute the private metadata export or accession-level join table.

## Reproducing the Analysis

### Requirements

Python 3.9+ is recommended. The geometric analysis scripts do not require a GPU once embeddings are cached.

```bash
pip install -r requirements.txt
```

### Step 1: Build or load the embedding cache

Embedding extraction requires authorized GISAID-derived inputs and the local AntigenLM checkpoint. The exact command may depend on the local preprocessing layout. The manuscript cache was produced with `latent_geometry.py` and saved as:

```text
results/embeddings_cache_full_all_available.pkl
```

For exploratory runs, inspect available options with:

```bash
python latent_geometry.py --help
```

### Step 2: Run the full geometric analysis

```bash
python latent_geometry_full_analysis.py \
  --cache-path results/embeddings_cache_full_all_available.pkl \
  --pair-samples-per-subtype 200000 \
  --pair-seeds 42,7,123
```

Primary outputs:

- `results/latent_geometry_full_metrics.json`
- `results/latent_geometry_full_summary.md`
- `figures/latent_geometry_full/`

### Step 3: Run the robustness panel

```bash
python paper_revision_outputs/run_robustness_panel.py \
  --cache-path results/embeddings_cache_full_all_available.pkl \
  --pair-samples-per-subtype 200000 \
  --pair-seeds 42,7,123 \
  --temporal-k-values 5,10,20 \
  --temporal-permutation-seeds 42,7,123
```

### Step 4: Run the random embedding negative control

```bash
python paper_revision_outputs/run_random_embedding_baseline.py \
  --cache-path results/embeddings_cache_full_all_available.pkl \
  --pair-samples-per-subtype 200000 \
  --pair-seeds 42,7,123 \
  --random-replicates 10
```

### Step 5: Run the GISAID clade-label enrichment analysis

This step requires the private local GISAID metadata join table. It writes only aggregate outputs.

```bash
python paper_revision_outputs/run_clade_enrichment_analysis.py \
  --cache-path results/embeddings_cache_full_all_available.pkl \
  --metadata-join-path data/gisaid_metadata_private/gisaid_epiflu_isolates_2000_2022_epi_set_260506bu_joined_dedup_cache.csv
```

Primary outputs:

- `results/gisaid_clade_enrichment_results.json`
- `results/gisaid_clade_enrichment_summary.md`
- `results/gisaid_clade_enrichment_summary_es.md`
- `figures/latent_geometry_full/clade_precision_enrichment.pdf`

### Step 6: Run added audit controls

```bash
python paper_revision_outputs/run_preprocessing_counts.py

python paper_revision_outputs/run_sequence_baselines.py \
  --cache-path results/embeddings_cache_full_all_available.pkl \
  --metadata-join-path data/gisaid_metadata_private/gisaid_epiflu_isolates_2000_2022_epi_set_260506bu_joined_dedup_cache.csv

python paper_revision_outputs/run_embedding_permutation_control.py \
  --cache-path results/embeddings_cache_full_all_available.pkl

python paper_revision_outputs/run_alignment_robustness.py \
  --cache-path results/embeddings_cache_full_all_available.pkl

python paper_revision_outputs/run_pca_anisotropy_controls.py \
  --cache-path results/embeddings_cache_full_all_available.pkl

python paper_revision_outputs/run_temporal_dedup_sensitivity.py \
  --cache-path results/embeddings_cache_full_all_available.pkl
```

Added aggregate outputs:

- `results/preprocessing_counts.json` and `.md`
- `results/sequence_baseline_results.json` and `.md`
- `results/embedding_permutation_results.json` and `.md`
- `results/alignment_robustness_results.json` and `.md`
- `results/pca_anisotropy_controls.json` and `.md`
- `results/temporal_dedup_sensitivity_results.json` and `.md`
- `figures/latent_geometry_full/sequence_baseline_clade_precision.pdf`

### Step 7: Rebuild the manuscript package

```bash
python make_paper_1_latent_geometry_full.py
```

The current arXiv/pre-journal package can be compiled from `latent_geometry_final/` with Tectonic:

```bash
cd latent_geometry_final
tectonic antigenlm_latent_geometry_audit.tex
```

## Results Summary

Human-readable summaries are available in:

- `results/latent_geometry_full_summary.md`
- `paper_revision_outputs/robustness_panel_summary.md`
- `paper_revision_outputs/random_embedding_baseline_summary.md`
- `results/gisaid_clade_enrichment_summary.md`
- `results/gisaid_clade_enrichment_summary_es.md`
- `results/preprocessing_counts.md`
- `results/sequence_baseline_summary.md`
- `results/embedding_permutation_summary.md`
- `results/alignment_robustness_summary.md`
- `results/pca_anisotropy_controls_summary.md`
- `results/temporal_dedup_sensitivity_summary.md`

## Citation

If you use this code or aggregate results, please cite:

```bibtex
@article{orrego2026latent,
  title   = {A Geometric Audit of AntigenLM Latent Representations
             for Influenza A HA/NA Sequences},
  author  = {Orrego-Franco, Carlos Manuel and Ria{\~n}o-Rojas, Juan Carlos},
  journal = {arXiv},
  year    = {2026},
  note    = {Pre-journal manuscript}
}
```

Please also cite GISAID:

```bibtex
@article{shu2017gisaid,
  title   = {GISAID: Global initiative on sharing all influenza data -- from vision to reality},
  author  = {Shu, Yuelong and McCauley, John},
  journal = {Euro Surveillance},
  volume  = {22},
  number  = {13},
  pages   = {30494},
  year    = {2017},
  doi     = {10.2807/1560-7917.ES.2017.22.13.30494}
}
```

## License

Code in this repository is released under the [MIT License](LICENSE).

Aggregate result files (JSON and Markdown summaries) may be reused with attribution under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Raw sequence data are subject to GISAID data-use terms and are not covered by this repository license.

## Contact

Carlos Manuel Orrego-Franco  
Universidad Nacional de Colombia  
ORCID: https://orcid.org/0009-0001-9163-5137  
[GitHub: @cmorregof](https://github.com/cmorregof)

Juan Carlos Riaño-Rojas  
Universidad Nacional de Colombia  
ORCID: https://orcid.org/0000-0002-5719-2854
