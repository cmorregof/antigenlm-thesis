# Final Reviewer-Proofing Summary

Date: 2026-05-05

## Inputs inspected

- `paper_revision_outputs/07_robustness_method_audit.md`
- `papers/paper_1_latent_geometry_full/main.tex`
- `paper_revision_outputs/robustness_panel_results.json`
- `paper_revision_outputs/robustness_panel_summary.md`
- `paper_revision_outputs/05_submission_strategy.md`
- `papers/paper_1_latent_geometry_full/references.bib`

The repository does not contain a root-level `references.bib`; the manuscript bibliography used for this pass is `papers/paper_1_latent_geometry_full/references.bib`.

## Backup and revised copy

- Backup preserved before edits: `paper_revision_outputs/main_before_final_reviewer_proofing.tex`
- Edited manuscript: `papers/paper_1_latent_geometry_full/main.tex`
- Synchronized revised copy: `paper_revision_outputs/revised_main.tex`

## Manuscript sections changed

- Abstract: clarified that the robustness panel supports only the tested molecular-distance and temporal-label controls, and does not establish antigenic, phylogenetic, vaccine-strain, sequence-generation, or prospective forecasting validity.
- Methods, data/embeddings: retained the local-cache framing and avoided language implying official full AntigenLM forecasting reproduction.
- Methods, distance definitions: clarified the 5% length-tolerance rule, shared-prefix comparison, subtype-specific molecular sequence proxies, HA+NA length-weighted Hamming distance, and limitations of the simple translated amino-acid proxy.
- Methods, pair-sampled correlations: clarified subtype-specific sampling, 200,000 requested non-self pairs per subtype per seed, random seeds 42, 7, and 123, and seed-to-seed standard deviations rather than confidence intervals.
- Methods, exact deduplication: clarified exact HA+NA sequence-pair deduplication by retaining the first representative in cache order; near-duplicates, one-segment duplicates, and amino-acid-equivalent nucleotide variants are not removed by this rule.
- Methods, local temporal neighborhoods: clarified that self-matches are excluded and that the date-label control permutes collection month-index labels within subtype while keeping the latent nearest-neighbor graph fixed.
- Results, Spearman and robustness captions: clarified that Hamming distances are molecular sequence-distance proxies and that reported uncertainty values are seed-to-seed standard deviations.
- Results, robustness panel: reframed amino-acid HA+NA results as a translated sensitivity proxy, not curated protein-level validation.
- Discussion: limited robustness claims to the controls actually performed.
- Limitations: strengthened statements on missing curated multiple sequence alignment, antigenic assay validation, phylogenetic validation, prospective forecasting validation, bootstrap confidence intervals, and random-embedding or matched-spectrum controls.
- Reproducibility and data/code availability: clarified that aggregate outputs, scripts, tables, and figures are documented; full sequences should not be redistributed if restricted; local checkpoint provenance and applicable GISAID acknowledgement/data terms must be finalized before submission.
- Conclusion: clarified that the results support a geometric precondition for future modeling rather than antigenic, vaccine, or forecasting validity.

## Risky phrases removed or softened

- "shared aligned prefix" was softened to "shared prefix" to avoid implying curated alignment.
- Amino-acid HA+NA language was changed to "simple translated amino-acid proxy" or "sensitivity proxy".
- Amino-acid robustness results are no longer framed as curated protein alignment, antigenic-site analysis, antigenic distance, or protein-level validation.
- Seed standard deviations are explicitly described as seed-to-seed standard deviations, not bootstrap confidence intervals.
- Hamming distances are explicitly described as molecular sequence-distance proxies, not biological alignments or antigenic distances.
- Date permutation is described as within-subtype permutation of collection month-index labels with the latent neighbor graph held fixed.
- Robustness claims are limited to exact HA+NA deduplication, subtype-specific pair sampling, three random seeds, simple translated amino-acid sensitivity distances, and fixed-graph within-subtype date-label permutation.
- Forecasting, vaccine-strain, antigenic, phylogenetic, and sequence-generation implications are stated as not established.

## Remaining submission blockers

- Local LaTeX compilation could not be performed because `pdflatex`, `bibtex`, `latexmk`, and `tectonic` are not available on PATH in this environment.
- The final submission must document the local AntigenLM checkpoint provenance, access path or accession, version, and ideally checksum in a way suitable for readers/reviewers.
- The data availability statement must be checked against the actual source data licenses and any applicable GISAID acknowledgement and data-use requirements. The current wording deliberately avoids inventing exact legal language.
- No curated multiple sequence alignment is included.
- No HI, neutralization, antigenic cartography, or other antigenic assay validation is included.
- No phylogenetic tree, clade-label, or phylodynamic validation is included.
- No prospective forecasting validation is included.
- No bootstrap confidence intervals are reported.
- No random-embedding, matched-spectrum random-embedding, PCA-whitened, or untrained-checkpoint control is included.
- A software/runtime version table would strengthen reproducibility for journal submission.

## Readiness assessment

- Ready for advisor review: yes. The manuscript is now substantially more conservative, traceable to repository outputs, and explicit about methodological limits.
- Ready for journal submission: not quite. It is a plausible submission candidate after advisor review, local LaTeX compilation, checkpoint/data-provenance finalization, and data-availability/legal wording checks. For a stronger submission, the next highest-value analysis remains a negative-control embedding or matched-spectrum/random-projection control, plus bootstrap confidence intervals or a clearly justified uncertainty framework.
