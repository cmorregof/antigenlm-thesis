# Formatting and Author Update Report

Date: 2026-05-05

## Author list status

Both required full author names are now present in the manuscript and visible on the rendered first page:

```latex
\author{
Carlos Manuel Orrego-Franco
\and
Juan Carlos Riaño-Rojas
}
% TODO before submission: add institutional affiliations, corresponding author email, and ORCID IDs if required by the target journal.
```

Juan Carlos Riaño-Rojas was added during this pass. The previous author block contained only `Carlos Morrego`, which was incomplete and did not include the coauthor.

No affiliations, emails, ORCID IDs, or corresponding-author metadata were invented. A non-rendered TODO comment was added near the author block for journal-required metadata.

## Backup and synchronization

- Backup before editing: `paper_revision_outputs/main_before_formatting_author_update.tex`
- Edited manuscript: `papers/paper_1_latent_geometry_full/main.tex`
- Synchronized manuscript copy: `paper_revision_outputs/revised_main.tex`

`main.tex` and `paper_revision_outputs/revised_main.tex` are synchronized after the edits.

## Layout issues found

The previous compile report flagged overfull boxes at:

- Lines 56--57: long input-template display in the embedding-extraction method.
- Lines 63--66: long checkpoint path/hash paragraph.
- Lines 367--368: long reproducibility paths and result-file paths.

The previous compile also flagged:

- A harmless `inputenc` warning under Tectonic/XeTeX.
- One underfull bibliography box in `main.bbl`.

## LaTeX formatting changes made

Minimal formatting changes were applied:

- Added `\usepackage{xurl}` before `hyperref` to improve line breaking in paths/URLs.
- Added `\emergencystretch=3em` to give TeX modest flexibility before producing overfull lines.
- Replaced raw monospaced long paths and the SHA256 hash with `\path{...}`:
  - `prediction_sequence/pytorch_model.bin`
  - checkpoint SHA256 hash
  - reproducibility script and result paths
- Broke the long input-template display into a two-line `aligned` display.
- Kept `inputenc` and `fontenc` for compatibility with common journal LaTeX workflows. The `inputenc` warning remains harmless under Tectonic/XeTeX.

No numerical results, scientific claims, raw data, figures, tables, or bibliography entries were changed.

## Compilation

Compile commands run from `papers/paper_1_latent_geometry_full`:

```bash
tectonic main.tex
tectonic --keep-logs --keep-intermediates main.tex
```

Compilation succeeded.

New compiled PDF:

- `paper_revision_outputs/latent_geometry_manuscript_compiled_v2.pdf`

New compile artifacts:

- `paper_revision_outputs/main_compile_tectonic_v2.log`
- `paper_revision_outputs/main_compile_bibtex_v2.blg`
- `paper_revision_outputs/main_compile_references_v2.bbl`

## Remaining warnings

The overfull `\hbox` warnings were eliminated.

Remaining warnings:

```text
Package inputenc Warning: inputenc package ignored with utf8 based engines.
Underfull \hbox (badness 1888) in paragraph at lines 18--25 of main.bbl
```

These are not fatal. The `inputenc` warning is expected under Tectonic/XeTeX; the bibliography underfull box is a minor formatting warning.

## Visual-inspection concerns

The first page was rendered with Quick Look and visually checked. The author line renders as:

- Carlos Manuel Orrego-Franco
- Juan Carlos Riaño-Rojas

Remaining concerns before submission:

- Perform a full-page visual pass through the PDF, especially tables and figure captions.
- Add journal-specific affiliations, corresponding author email, and ORCID IDs before submission if required.
- Check whether the target journal prefers a different bibliography style or author/affiliation format.
