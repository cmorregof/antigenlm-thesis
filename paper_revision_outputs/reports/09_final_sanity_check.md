# Final Sanity Check

Date: 2026-05-05

Scope: final non-experimental reviewer-proofing checks for `papers/paper_1_latent_geometry_full/main.tex` after the robustness-method audit updates. No new experiments were run and no raw data were modified.

## Check results

| Check | Result | Notes |
|---|---:|---|
| All cited keys exist in bibliography | PASS | `\cite{...}` keys in `main.tex` all resolve to entries in `papers/paper_1_latent_geometry_full/references.bib`. |
| Referenced figure files exist | PASS | All `\includegraphics{figures/...}` targets exist under `papers/paper_1_latent_geometry_full/figures/`. |
| Referenced table labels exist | PASS | Tables are embedded in `main.tex`; all table labels referenced by `\ref{...}` are present. |
| No duplicate labels | PASS | No duplicate `\label{...}` keys found. |
| No unresolved internal references detectable by static scan | PASS | All `\ref`, `\autoref`, and `\pageref` targets found among manuscript labels. |
| No obvious TODO/FIXME markers | PASS | No `TODO` or `FIXME` markers found in `main.tex` or current markdown outputs. |
| Revised copy synchronized | PASS | `papers/paper_1_latent_geometry_full/main.tex` and `paper_revision_outputs/revised_main.tex` are identical. |
| Local LaTeX toolchain available | BLOCKED | `pdflatex`, `bibtex`, `latexmk`, and `tectonic` are not available on PATH. Compilation was therefore not run in this environment. |

## Risk-language scan

Static scans were run for potentially overclaiming terms:

- `confidence interval`, `confidence intervals`, `bootstrap`
- `curated protein`, `protein-level validation`
- `antigenic validation`, `antigenic similarity`, `antigenic distance`
- `forecasting validation`, `prospective forecasting`, `forecast`
- `vaccine`

Result: PASS with expected cautionary hits. These terms occur as scope limitations, negative claims, or explicit non-goals, not as unsupported positive results. In particular:

- Seed standard deviations are explicitly stated to be seed-to-seed standard deviations, not confidence intervals.
- The manuscript states that bootstrap confidence intervals were not computed.
- Amino-acid HA+NA distances are described as simple translated sensitivity proxies, not curated protein alignments, antigenic-site distances, antigenic measurements, or protein-level validation.
- Molecular Hamming distances are described as molecular sequence-distance proxies, not antigenic distances.
- Prospective forecasting, vaccine-strain relevance, sequence-generation quality, antigenic validity, and phylogenetic validity are explicitly not claimed.

## Robustness-claim traceability

The robustness claims retained in `main.tex` are traceable to `paper_revision_outputs/robustness_panel_results.json` and `paper_revision_outputs/robustness_panel_summary.md`:

- Input cache: `results/embeddings_cache_full_all_available.pkl`
- Exact HA+NA deduplicated records: 82,306 records retained after removing exact HA+NA pair duplicates.
- Pair sampling: 200,000 requested non-self pairs per subtype per seed.
- Seeds: 42, 7, and 123.
- Subtypes: H1N1 and H3N2 analyzed separately.
- Molecular distance controls: HA nucleotide Hamming, NA nucleotide Hamming, HA+NA nucleotide Hamming, and simple translated amino-acid HA+NA proxy.
- Temporal permutation control: collection month-index labels permuted within subtype while holding the latent nearest-neighbor graph fixed.
- Temporal nearest-neighbor k values: 5, 10, and 20.

## Unsupported-claim check

No unsupported positive manuscript claims were found for:

- Confidence intervals or bootstrap uncertainty.
- Antigenic validation, HI validation, neutralization validation, or antigenic cartography.
- Curated protein alignment or protein-level validation.
- Phylogenetic validation or clade-level validation.
- Prospective forecasting validation.
- Vaccine recommendation or vaccine-strain selection.
- Sequence generation or mutation optimization.

## Remaining sanity caveats

- Static checks do not replace a full LaTeX compile. The manuscript should be compiled in an environment with a TeX toolchain before submission.
- Static citation checks confirm key presence, not bibliographic style correctness or DOI validity.
- Static figure checks confirm file existence, not visual quality, resolution, or journal formatting compliance.
- Data/code availability wording still needs final review against actual data-source licenses and GISAID terms if GISAID-derived records are used.
- The local AntigenLM checkpoint provenance still needs submission-grade documentation.
