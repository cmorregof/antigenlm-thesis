# Affiliation Update Report

Date: 2026-05-06

## Files Updated

- `papers/paper_1_latent_geometry_full/main.tex`
- `paper_revision_outputs/revised_main.tex`

## Backup

- `paper_revision_outputs/main_before_affiliation_update.tex`

## Affiliation Source

The PCM website identifies the group as the PCM Computational Applications Research Group and states that it is located at Universidad Nacional de Colombia, Sede Manizales, Campus La Nubia:

- https://pcm-ca.github.io/

## Author Block Used

```latex
\author{
Carlos Manuel Orrego-Franco\\
Juan Carlos Riaño-Rojas\\
\vspace{0.4em}
\small PCM Computational Applications Research Group\\
\small Universidad Nacional de Colombia, Sede Manizales, Campus La Nubia\\
\small Manizales, Colombia
}
% TODO before submission: add corresponding author email and ORCID IDs if required by the target journal.
```

## Notes

- Both authors are listed exactly as required.
- The affiliation is shared and compact to avoid adding an extra page.
- No email, ORCID, department, faculty, or corresponding-author metadata was added because these were not yet provided.
- The Colciencias category and descriptive group history were not added to the author block because they are not standard affiliation metadata for a manuscript title page.

## Compilation

Command:

```bash
cd papers/paper_1_latent_geometry_full
tectonic main.tex
```

Compilation succeeded.

Compiled PDF:

- `paper_revision_outputs/latent_geometry_manuscript_compiled_v6_affiliation.pdf`

Page count: 21 pages.

Remaining warnings:

- `main.tex:545`: Underfull `\hbox` in the final conclusion paragraph.
- `main.bbl:25`: Underfull `\hbox` in the bibliography.

These warnings are minor layout warnings and do not block PDF generation.
