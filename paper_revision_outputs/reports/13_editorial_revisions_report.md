# Editorial Revisions Report

Date: 2026-05-05

## Scope

Implemented the requested pre-submission editorial revisions in the English manuscript:

- `papers/paper_1_latent_geometry_full/main.tex`
- synchronized copy: `paper_revision_outputs/revised_main.tex`

No raw data were modified. The English manuscript was backed up before editing:

- `paper_revision_outputs/main_before_editorial_revisions.tex`

## Random embedding baseline

The prompt suggested placeholder random-baseline correlations around 0.08--0.12. Those values were not used. A real negative-control analysis was run instead.

Script created:

- `paper_revision_outputs/run_random_embedding_baseline.py`

Outputs created:

- `paper_revision_outputs/random_embedding_baseline_results.json`
- `paper_revision_outputs/random_embedding_baseline_summary.md`
- `paper_revision_outputs/random_embedding_baseline_run.log`

Actual random-baseline aggregate results:

| metric | subtype | random rho mean | random rho sd | AntigenLM rho mean | AntigenLM - random |
|---|---|---:|---:|---:|---:|
| HA+NA Hamming | H1N1 | 0.0001 | 0.0019 | 0.8538 | 0.8537 |
| HA+NA Hamming | H3N2 | 0.0004 | 0.0021 | 0.6678 | 0.6674 |

Design:

- 111,756 records from `results/embeddings_cache_full_all_available.pkl`
- 384-dimensional Gaussian random vectors
- unit `l2` normalization of each random vector
- 10 independent random embedding replicates, seeds 1000--1009
- three pair-sampling seeds: 42, 7, 123
- 200,000 requested non-self pairs per subtype per pair seed
- HA+NA Hamming distances computed with the same 5% length-tolerance/shared-prefix rule used in the main paper

## BEFORE/AFTER revision map

### Abstract

BEFORE:

- Reported the main molecular, PCA, TwoNN, and temporal-neighborhood results.
- Ended with a general caution that the work did not establish antigenic similarity, phylogenetic validity, sequence generation, vaccine relevance, or forecasting.

AFTER:

- Adds the real random embedding negative-control result: random HA+NA Hamming correlations near zero.
- Reframes the contribution as a geometric precondition.
- Explicitly states that the work does not include HI assays, neutralization titers, phylogenetic distances, or clade labels.
- Ends by saying the paper is a geometric audit, not biological or functional validation.

### Introduction

BEFORE:

- Motivated influenza evolution, sequence language models, and geometric checking.

AFTER:

- Moves the audit framing earlier.
- States that molecular preservation, low-dimensional structure, and evolutionary-history structure are preconditions rather than substitutes for biological validation.
- Adds explicit scope boundaries: no antigenic validity, no phylogenetic grounding, no forecasting performance, and no reproduction of the full AntigenLM pipeline.

### Methods

BEFORE:

- Included data source, embeddings, distances, pair-sampled correlations, deduplication, PCA, TwoNN, and temporal-neighborhood methods.
- No random embedding negative control.

AFTER:

- Adds `Random Embedding Control`.
- Describes 10 independent Gaussian random embedding replicates, unit `l2` normalization, 384 dimensions, three pair-sampling seeds, and 200,000 requested pairs per subtype per seed.
- States the expected interpretation: correlations near zero if the learned molecular signal is not present.

### Results

BEFORE:

- Molecular correlation results were followed directly by PCA results.

AFTER:

- Adds `Random Embeddings Do Not Reproduce the Molecular Signal`.
- Adds `Table~\ref{tab:random-baseline}` with actual random baseline values.
- States that random embeddings do not reproduce the AntigenLM molecular signal.

### Discussion

BEFORE:

- Said the results validated a narrow geometric precondition and motivated future work.

AFTER:

- Quantifies the bounded geometric interpretation using HA+NA correlations, PCA/TwoNN ranges, and local temporal medians.
- Adds that the properties are not reproduced by the random embedding control.
- Explicitly states that geometry alone does not establish antigenic similarity, phylogenetic validity, immune escape, functional conservation, or forecasting performance.

### Limitations

BEFORE:

- Covered biological limitations, methodological limits, amino-acid proxy limits, intrinsic-dimension caveats, surveillance bias, and non-reproduction of forecasting.

AFTER:

- Expands biological limitations: no HI, neutralization, antigenic cartography, curated protein alignments, structural annotations, antigenic-site analysis, clade labels, phylogenetic distances, or fitness measurements.
- Expands rank-correlation limitations: pairwise dependence, seed standard deviations not bootstrap confidence intervals, and missing Mantel/distance-correlation/kNN/trustworthiness diagnostics.
- Expands amino-acid proxy limitations.
- Expands intrinsic-dimension limitations, including near-duplicates, density, boundary effects, finite-sample bias, standardization, and trimming sensitivity.
- Expands surveillance-bias discussion.
- Reframes the work as not a forecasting, antigenic, or functional validation.

### Reproducibility and Data Availability

BEFORE:

- Listed the main scripts and aggregate outputs.

AFTER:

- Adds the random embedding script and result files.
- Adds a formal `Data Availability Statement`.
- States that complete GISAID-derived sequences are not redistributed.
- Notes that public repository URL and final data-access instructions still need to be populated before submission.

### Conclusion

BEFORE:

- Said the findings make the embedding cloud a plausible candidate state space for future probabilistic modeling.

AFTER:

- Removes the stronger state-space phrasing.
- Says the findings show useful geometric structure under tested diagnostics.
- Explicitly says they do not establish antigenic prediction, phylogenetic grounding, immune-escape interpretation, vaccine-strain relevance, sequence-generation reliability, or forecasting skill.
- States that HI/neutralization data, phylogenetic distances, clade annotations, and prospective post-2022 benchmarks remain essential before evolutionary or clinical relevance claims.

## Compilation

Compilation succeeded with Tectonic.

Commands:

```bash
cd papers/paper_1_latent_geometry_full
tectonic main.tex
tectonic --keep-logs --keep-intermediates main.tex
```

Compiled PDF:

- `paper_revision_outputs/latent_geometry_manuscript_compiled_v3_editorial.pdf`

Compile logs:

- `paper_revision_outputs/main_compile_tectonic_v3_editorial.log`
- `paper_revision_outputs/main_compile_bibtex_v3_editorial.blg`
- `paper_revision_outputs/main_compile_references_v3_editorial.bbl`

Remaining warnings:

```text
Package inputenc Warning: inputenc package ignored with utf8 based engines.
Underfull \hbox (badness 1888) in paragraph at lines 18--25
```

No overfull `hbox`, fatal errors, undefined references, or missing citations were detected in the preserved compile log.

## Remaining manual checks before submission

- Full visual inspection of the v3 PDF.
- Add public repository URL or final code/data-access instructions.
- Finalize GISAID acknowledgement/data-use language.
- Confirm checkpoint provenance/access instructions are sufficient for the target journal.
- Advisor review of whether the new random-control analysis should be included as a table only or also as a small supplementary output.
