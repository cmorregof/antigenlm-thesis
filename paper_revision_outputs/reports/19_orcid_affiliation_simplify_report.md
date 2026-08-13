# ORCID And Affiliation Simplification Report

Date: 2026-05-06

## Files Updated

- `papers/paper_1_latent_geometry_full/main.tex`
- `paper_revision_outputs/revised_main.tex`

## Backup

- `paper_revision_outputs/main_before_orcid_affiliation_simplify.tex`

## Author Block After Update

```latex
\author{
Carlos Manuel Orrego-Franco\\
Juan Carlos Riaño-Rojas\\
\vspace{0.4em}
\small PCM Computational Applications Research Group\\
\small Universidad Nacional de Colombia\\
\small \texttt{corregof@unal.edu.co}; \texttt{jcrianoro@unal.edu.co}\\
\small ORCID: \url{https://orcid.org/0009-0001-9163-5137}
}
% TODO before submission: add Juan Carlos Riaño-Rojas ORCID ID if required by the target journal.
```

## Changes Made

- Removed `Sede Manizales, Campus La Nubia`.
- Removed `Manizales, Colombia`.
- Kept the institutional affiliation as `Universidad Nacional de Colombia`.
- Kept the research group as `PCM Computational Applications Research Group`.
- Added Carlos Manuel Orrego-Franco's ORCID: `https://orcid.org/0009-0001-9163-5137`.
- Did not add an ORCID for Juan Carlos Riaño-Rojas because none was provided.

## Compilation

Command:

```bash
cd papers/paper_1_latent_geometry_full
tectonic main.tex
```

Compilation succeeded.

Compiled PDF:

- `paper_revision_outputs/latent_geometry_manuscript_compiled_v6_orcid_affiliation.pdf`

Page count: 21 pages.

Remaining warnings:

- `main.tex:546`: Underfull `\hbox` in the final conclusion paragraph.
- `main.bbl:25`: Underfull `\hbox` in the bibliography.

These warnings are minor layout warnings and do not block PDF generation.
