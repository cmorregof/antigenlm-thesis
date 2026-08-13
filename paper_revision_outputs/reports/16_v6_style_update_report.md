# v6 LaTeX Style Update Report

Date: 2026-05-06  
Target journal framing: Scientific Reports  
Main manuscript: `papers/paper_1_latent_geometry_full/main.tex`  
Backup before editing: `paper_revision_outputs/main_before_v6_style_update.tex`  
Synchronized copy: `paper_revision_outputs/revised_main.tex`  
Compiled PDF: `paper_revision_outputs/latent_geometry_manuscript_compiled_v6_style.pdf`  
Compile log: `paper_revision_outputs/main_compile_tectonic_v6_style.log`

## Scope

Implemented presentation/style changes only. No numerical results, data tables, scientific conclusions, raw data, or experimental outputs were changed. No new experiments were run.

## Implemented Changes With BEFORE/AFTER

### Change 1: Abstract Replaced And Shortened

BEFORE: The v5 abstract was a single long paragraph of approximately 408 words, including expanded negations and detailed clade-control framing.

AFTER: Replaced the complete abstract with the v6 shortened Scientific Reports version. One terminology-normalization edit was applied after insertion: `random-baseline levels` became `subtype-matched random-baseline levels` to satisfy the global terminology rule.

Final abstract word count from `.tex`: 287 words.

### Change 2: Generalizable Audit Claim Added To Introduction

BEFORE:

```latex
This paper conducts a systematic geometric audit of cached local AntigenLM
representations for influenza HA/NA sequences. We ask whether these embeddings
satisfy four minimal preconditions: molecular organization, avoidance of trivial
time/subtype collapse, low-dimensional structure under explicit diagnostics,
and local coherence with available evolutionary-taxonomic clade labels.
```

AFTER:

```latex
We propose a reproducible geometric audit framework applicable to any viral
sequence language model before deployment as a biological state space, and
demonstrate it on AntigenLM embeddings for influenza~A HA/NA sequences as
a case study. This paper asks whether these embeddings satisfy four minimal
preconditions: molecular organization, avoidance of trivial time/subtype
collapse, low-dimensional structure under explicit diagnostics, and local
coherence with available evolutionary-taxonomic clade labels.
```

### Change 3A: Methods Distances And Pair-Sampled Correlations Fused

BEFORE:

```latex
\subsection{Distances}
...
\subsection{Pair-Sampled Correlations}
```

AFTER:

```latex
\subsection{Distances and Pair-Sampled Correlations}
```

Transition added:

```latex
Using these distance definitions, we computed Spearman rank correlations
between latent and molecular or temporal distances via subtype-specific
pair sampling.
```

### Change 3B: Methods PCA And TwoNN Fused

BEFORE:

```latex
\subsection{PCA Effective Dimension}
...
\subsection{TwoNN Intrinsic-Dimension Diagnostic}
```

AFTER:

```latex
\subsection{Effective Dimension Diagnostics: PCA and TwoNN}
```

Transition added:

```latex
We complement the linear PCA spectrum with a local nonlinear diagnostic,
TwoNN, to assess whether the low effective dimension is also apparent
under nearest-neighbor geometry.
```

Final Methods subsection count: 8.

### Change 4: Results Dataset Composition Subsection Removed

BEFORE:

```latex
\subsection{Dataset Composition}

The cached dataset contains 111,756 embeddings of dimension 384, with 46,125
H1N1 and 65,631 H3N2 records. ...
```

AFTER: Removed the standalone subsection and inserted the dataset context at the start of the first Results subsection:

```latex
The cache contains 111,756 embeddings in $\R^{384}$, comprising 46,125 H1N1
and 65,631 H3N2 records collected between 2000 and 2022 (Figure~\ref{fig:data-year});
exact HA+NA deduplication retains 82,306 records for neighborhood and robustness
analyses (Figure~\ref{fig:lengths}).
```

Final Results subsection count: 7.

### Change 5: Table 5 Caption Clarified

BEFORE:

```latex
\caption{Temporal stratification sensitivity for clade-precision@5.
Random candidates are assigned-clade records from the same subtype
within the stated collection-month window; true latent precision is
recomputed on the same eligible query set.}
```

AFTER:

```latex
\caption{Temporal stratification sensitivity for clade-precision@5.
Random candidates are assigned-clade records from the same subtype
within the stated collection-month window; true latent precision is
recomputed on the same eligible query set. True latent clade precision
is constant across window sizes for a given subtype; only the stratified
random baseline and enrichment ratio vary with window width.}
```

### Change 6: Figure 6 Caption Expanded

BEFORE:

```latex
\caption{Two-dimensional global PCA projection. The projection is
descriptive and should not be read as a complete representation of
evolutionary or antigenic relationships.}
```

AFTER:

```latex
\caption{Two-dimensional global PCA projection. The projection is
descriptive and should not be read as a complete representation of
evolutionary or antigenic relationships. In particular, this
two-dimensional view does not constitute quantitative phylogenetic,
antigenic, or forecasting evidence.}
```

### Change 7: Repeated Negations Reduced

BEFORE, Results clade interpretation:

```latex
The interpretation remains bounded: this is clade-label enrichment, not
quantitative phylogenetic-distance preservation, antigenic similarity,
immune-escape structure, or forecasting validation.
```

AFTER:

```latex
The interpretation remains bounded: this is clade-label enrichment,
not evidence of quantitative phylogenetic preservation, antigenic
similarity, immune escape, or forecasting performance.
```

BEFORE, Discussion synthesis:

```latex
However, molecular organization, dimensionality, temporal coherence, and
clade-label enrichment are still audit findings. They do not, by themselves,
establish antigenic similarity, quantitative phylogenetic-distance preservation,
immune-escape patterns, functional conservation, vaccine-strain relevance,
or forecasting performance.
```

AFTER:

```latex
These findings remain audit-level evidence: they support molecular,
dimensional, temporal, and clade-label structure in the cached embeddings,
but not antigenic, phylogenetic, functional, vaccine-strain, or
forecasting validity.
```

BEFORE, Conclusion:

```latex
They do not establish antigenic prediction, quantitative phylogenetic
grounding, immune-escape interpretation, vaccine-strain relevance,
sequence-generation reliability, or forecasting skill. The findings
should be treated as a geometric and label-enrichment precondition,
favorable but not sufficient, for future probabilistic modeling of
influenza evolution.
```

AFTER:

```latex
These findings should be treated as favorable geometric and label-enrichment
preconditions for future probabilistic modeling of influenza evolution,
not as evidence of antigenic, phylogenetic, vaccine-strain,
sequence-generation, or forecasting validity.
```

BEFORE, Introduction operational scope:

```latex
We intentionally do not reproduce the AntigenLM forecasting pipeline,
generate sequences, optimize mutations, infer vaccine strains, or
evaluate prospective forecasts.
```

AFTER:

```latex
We intentionally do not evaluate sequence generation, mutation
optimization, vaccine-strain inference, or prospective forecasting.
```

### Change 8: Terminology Normalized

BEFORE examples:

```latex
clade enrichment
random baseline
clade-enrichment results
```

AFTER examples:

```latex
clade-label enrichment
subtype-matched random baseline
clade-label enrichment results
```

Applied only where the context referred to the formal clade-label analysis or an otherwise unqualified subtype-matched baseline. Narrative use of `clade coherence` was preserved as requested. Random-embedding-control wording was left intact where it specifically referred to the random embedding baseline rather than the subtype-matched clade baseline.

### Change 9: Bibliography Verification Checklist

No bibliography edits were made because this item was specified as a checklist/no-edition step, and the GISAID DOI was already present.

- `pei2026antigenlm`: OpenReview lists AntigenLM as an ICLR 2026 poster, published 2026-01-26, with authors Yue Pei, Xuebin Chi, and Yu Kang. Existing entry is consistent with accepted ICLR 2026 status; no proceedings DOI was found/added.
- `dallatorre2025nucleotide`: Nature Methods page verifies volume 22, pages 287--297 (2025), DOI `10.1038/s41592-024-02523-z`. Existing entry is consistent.
- `nguyen2023hyenadna`: Official NeurIPS proceedings page verifies NeurIPS 2023 / Advances in Neural Information Processing Systems 36, but the current `.bib` author list appears to differ from the official proceedings author list. This should be corrected before final journal submission.
- `shu2017gisaid`: DOI `10.2807/1560-7917.ES.2017.22.13.30494` is already present in `references.bib`.

Verification sources consulted:

- AntigenLM OpenReview: https://openreview.net/forum?id=Y0zPlHDO5p
- Nucleotide Transformer DOI/Nature page: https://doi.org/10.1038/s41592-024-02523-z
- HyenaDNA NeurIPS proceedings: https://proceedings.neurips.cc/paper_files/paper/2023/hash/86ab6927ee4ae9bde4247793c46797c7-Abstract-Conference.html
- GISAID DOI page/search result: https://doi.org/10.2807/1560-7917.ES.2017.22.13.30494

## Labels And References

No `\label{...}` keys were changed, removed, or added. No `\ref{...}` keys required updating, because section renumbering is handled automatically and subsection labels were not used.

Labels present:

- `fig:data-year`
- `fig:lengths`
- `tab:spearman`
- `fig:spearman`
- `tab:random-baseline`
- `tab:clade-coverage`
- `tab:clade-enrichment`
- `fig:clade-enrichment`
- `tab:clade-temporal-stratified`
- `tab:pca`
- `fig:pca-spectrum`
- `fig:pca-2d`
- `tab:twonn`
- `fig:twonn`
- `tab:temporal-local`
- `fig:temporal-local`
- `tab:robust-spearman`
- `tab:temporal-permutation`

Sanity check results:

- Duplicate labels: none.
- Missing refs: none.
- Missing cited bibliography keys: none.
- Missing figure files: none.
- Uncited bibliography keys: `hadfield2018nextstrain`, `mantel1967detection`.

## Final Structure Checks

- Abstract: 287 words.
- Methods: 8 subsections.
- Results: 7 subsections.
- Dataset Composition is no longer a separate Results subsection.
- Figures 1 and 2 remain referenced at the beginning of Results.
- Table 5 caption includes the clarification about constant true precision across window widths.
- Figure 6 caption includes the additional caution against reading the PCA projection as phylogenetic, antigenic, or forecasting evidence.

## Compilation

Compile command:

```bash
cd papers/paper_1_latent_geometry_full
tectonic main.tex
```

Compilation succeeded with Tectonic.

Generated PDF:

```text
paper_revision_outputs/latent_geometry_manuscript_compiled_v6_style.pdf
```

Page count: 21 pages, counted with bundled `pypdf`.

Remaining LaTeX warnings:

- `main.tex:542`: Underfull `\hbox` in the final conclusion paragraph.
- `main.bbl:25`: Underfull `\hbox` in the bibliography.

No fatal errors, missing figures, missing citations, duplicate labels, or broken cross-references were detected.

## Ambiguities And Resolutions

- The v6 instruction said to paste the abstract exactly, while the later terminology-normalization rule required qualified uses of `random baseline`. I preserved the supplied abstract content and made the minimal terminology edit from `random-baseline levels` to `subtype-matched random-baseline levels`.
- A prior editorial note suggested including the H3N2 `3.18-fold` temporally stratified enrichment in the abstract. The v6 prompt supplied an exact replacement abstract that only names H1N1 `1.55-fold` in the abstract. I did not add the H3N2 value there, to respect the exact replacement instruction and avoid additional content edits.
- The HyenaDNA bibliography entry appears to need metadata correction against the official NeurIPS proceedings page, but the v6 prompt specified bibliography verification as checklist/no edition. I therefore reported the issue without editing `references.bib`.
