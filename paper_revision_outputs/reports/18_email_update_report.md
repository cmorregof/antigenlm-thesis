# Email Update Report

Date: 2026-05-06

## Files Updated

- `papers/paper_1_latent_geometry_full/main.tex`
- `paper_revision_outputs/revised_main.tex`

## Backup

- `paper_revision_outputs/main_before_email_update.tex`

## Author Block After Update

```latex
\author{
Carlos Manuel Orrego-Franco\\
Juan Carlos Riaño-Rojas\\
\vspace{0.4em}
\small PCM Computational Applications Research Group\\
\small Universidad Nacional de Colombia, Sede Manizales, Campus La Nubia\\
\small Manizales, Colombia\\
\small \texttt{corregof@unal.edu.co}; \texttt{jcrianoro@unal.edu.co}
}
% TODO before submission: add ORCID IDs if required by the target journal.
```

## Notes

- Added both author emails provided by the author.
- Did not add ORCID IDs because none were provided.
- Left a non-printing LaTeX TODO for ORCID IDs only.

## Compilation

Command:

```bash
cd papers/paper_1_latent_geometry_full
tectonic main.tex
```

Compilation succeeded.

Compiled PDF:

- `paper_revision_outputs/latent_geometry_manuscript_compiled_v6_email.pdf`

Page count: 21 pages.

Remaining warnings:

- `main.tex:546`: Underfull `\hbox` in the final conclusion paragraph.
- `main.bbl:25`: Underfull `\hbox` in the bibliography.

These warnings are minor layout warnings and do not block PDF generation.
