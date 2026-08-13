# v5 Clade Integration Report

## Analysis Status

The clade-label enrichment extension was implemented and run locally. The analysis used the exact HA+NA-deduplicated cache and the private GISAID EpiFlu metadata join from `EPI_SET_260506bu`. No raw sequences, `.pkl` cache contents, or accession-level metadata were exported into public-facing aggregate outputs.

## Commands Run

```bash
source venv_antigenlm/bin/activate
python paper_revision_outputs/run_clade_enrichment_analysis.py 2>&1 | tee paper_revision_outputs/logs/clade_enrichment_run.log
cd papers/paper_1_latent_geometry_full
tectonic main.tex
```

## Files Created or Updated

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
- `paper_revision_outputs/v5_clade_extension_summary_en.md`
- `paper_revision_outputs/resumen_v5_extension_clados_para_director.md`

## Main Numerical Results

### Coverage

| Group | cache n | joined n | join % | assigned clade n | assigned clade % |
|---|---:|---:|---:|---:|---:|
| H1N1 | 36,753 | 36,723 | 99.92% | 36,156 | 98.38% |
| H3N2 | 45,553 | 45,220 | 99.27% | 40,082 | 87.99% |
| Combined | 82,306 | 81,943 | 99.56% | 76,238 | 92.63% |

### Global Subtype-Matched Clade Enrichment

| Subtype | k | precision@k | random baseline | permutation mean | enrichment |
|---|---:|---:|---:|---:|---:|
| H1N1 | 5 | 0.9149 | 0.2409 | 0.2406 | 3.80x |
| H1N1 | 10 | 0.8933 | 0.2403 | 0.2405 | 3.72x |
| H1N1 | 20 | 0.8647 | 0.2404 | 0.2406 | 3.60x |
| H3N2 | 5 | 0.8609 | 0.0689 | 0.0690 | 12.49x |
| H3N2 | 10 | 0.8325 | 0.0689 | 0.0689 | 12.08x |
| H3N2 | 20 | 0.7927 | 0.0688 | 0.0689 | 11.51x |

### Temporally Stratified Control at k=5

| Subtype | window | true precision | stratified random | enrichment |
|---|---:|---:|---:|---:|
| H1N1 | ±6 months | 0.9149 | 0.5892 | 1.55x |
| H1N1 | ±12 months | 0.9149 | 0.5405 | 1.69x |
| H1N1 | ±24 months | 0.9149 | 0.4681 | 1.95x |
| H3N2 | ±6 months | 0.8610 | 0.2707 | 3.18x |
| H3N2 | ±12 months | 0.8610 | 0.2288 | 3.76x |
| H3N2 | ±24 months | 0.8609 | 0.1548 | 5.56x |

## Manuscript Changes

- Title updated to include evolutionary-taxonomic geometry.
- Abstract updated with clade coverage, enrichment, permutation, and temporal-stratification interpretation.
- Introduction updated from three/four preconditions to include local clade-label coherence.
- Methods now include `Clade-Label Enrichment Analysis`.
- Results now include:
  - clade metadata coverage table,
  - clade enrichment table,
  - clade precision figure,
  - temporally stratified clade sensitivity table.
- Discussion now separates molecular organization, temporal coherence, and clade-label coherence.
- Limitations now explicitly state that clade labels are not quantitative phylogenetic distances and that temporal co-occurrence attenuates the signal.
- Reproducibility/Data Availability now names the clade script, aggregate JSON/Markdown outputs, and `EPI_SET_260506bu`.
- Acknowledgments now mention the local metadata export in relation to Supplementary Table S1.

## Compilation

Tectonic compilation succeeded.

Compiled PDF:

`paper_revision_outputs/latent_geometry_manuscript_compiled_v5_clade.pdf`

Warnings:

- Underfull hbox in the conclusion paragraph.
- Underfull hbox in the bibliography.

No missing figures, duplicate labels, or fatal LaTeX errors remain after replacing a fragile `\path{...}` command inside a table caption.

## Editorial Interpretation

The v5 clade result materially increases the biological relevance of the paper, but the temporal control changes the strongest possible claim. The global clade enrichment is very strong, especially in H3N2, and the label-permutation control is convincing. However, the time-stratified baseline shows that clade enrichment is partly explained by temporal clade turnover, especially for H1N1. The manuscript has therefore been written conservatively: it claims local evolutionary-taxonomic coherence under GISAID clade annotations, not phylogenetic-distance preservation or antigenic validity.

Practical journal reading:

- BMC Bioinformatics: strong fit after v5.
- Scientific Reports: strong fit if data availability/GISAID supplement is complete.
- Virus Evolution: now plausible, but likely to request quantitative phylogenetic validation or stronger sampling-bias controls.
- Bioinformatics: possible only if framed as a reusable embedding-audit workflow, not just a case study.
