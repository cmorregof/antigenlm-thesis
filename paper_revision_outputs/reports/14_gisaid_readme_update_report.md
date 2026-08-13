# GISAID, Repository URL, and README Update Report

Date: 2026-05-05

## Files changed or created

- Updated manuscript: `papers/paper_1_latent_geometry_full/main.tex`
- Synchronized manuscript copy: `paper_revision_outputs/revised_main.tex`
- Backup before this pass: `paper_revision_outputs/main_before_gisaid_readme_update.tex`
- New compiled PDF: `paper_revision_outputs/latent_geometry_manuscript_compiled_v4_gisaid_readme.pdf`
- Compile log: `paper_revision_outputs/main_compile_tectonic_v4_gisaid_readme.log`
- Repository README: `README.md`
- License file: `LICENSE`
- Updated ignore rules: `.gitignore`

## Manuscript updates

Added an unnumbered `Acknowledgments` section before the references. The section acknowledges:

- Authors and Originating laboratories responsible for specimens.
- Submitting laboratories responsible for sequence/metadata generation and GISAID sharing.
- The GISAID Initiative.
- Shu and McCauley 2017 via `\cite{shu2017gisaid}`.
- Supplementary Table S1 as the submission-package location for contributing laboratories and sequence accession identifiers.

Updated `Reproducibility and Data Availability` and the formal `Data Availability Statement` with:

- public repository URL: `https://github.com/cmorregof/latent_geometry`
- aggregate JSON/code availability wording
- documented AntigenLM checkpoint SHA256
- GISAID access/data-use requirements
- Supplementary Table S1 accession-list requirement

No methods, results, tables, numerical values, or scientific claims were changed.

## README

Replaced the previous upstream AntigenLM placeholder README with a manuscript-focused repository README. It now includes:

- manuscript title and authors
- overview and key findings
- repository structure
- GISAID-aware data availability statement
- reproducibility commands using the actual local CLI options where available
- result-summary file pointers
- citation block
- license and contact sections

## License and ignore rules

Created `LICENSE` with the standard MIT license text for the code.

Updated `.gitignore` to exclude:

- model/checkpoint binaries and large cache formats
- raw sequence formats
- raw/restricted data directories
- Python caches/environments
- OS/build artifacts

## Clade/lineage metadata check

The cached metadata/records were inspected without printing sequences. No clade, lineage, Nextstrain, genotype, or similar annotation field is present in `results/embeddings_cache_full_all_available.pkl`.

Available record keys are:

- `day`
- `epi_isl`
- `ha_sequence`
- `month`
- `na_sequence`
- `strain_name`
- `subtype`
- `subtype_token`
- `year`

Therefore, clade/lineage validation cannot be added from the current cached metadata alone. It would require external annotation, for example from Nextstrain clades, phylogenetic inference, or an additional GISAID/metadata export containing lineage/clade fields.

## Compilation

Compilation succeeded with Tectonic:

```bash
cd papers/paper_1_latent_geometry_full
tectonic main.tex
tectonic --keep-logs --keep-intermediates main.tex
```

Remaining warnings:

```text
Package inputenc Warning: inputenc package ignored with utf8 based engines.
Underfull \hbox (badness 1888) in paragraph at lines 18--25
```

No fatal errors, missing citations, unresolved references, or overfull boxes were detected.

## Checklist status

- Acknowledgments section added with GISAID text: done.
- GISAID citation appears in the Acknowledgments section: done.
- Data Availability Statement updated with repository URL and GISAID access language: done.
- README.md written at repository root: done.
- LICENSE added: done.
- `.gitignore` updated to exclude raw sequence data, checkpoints, and large binary caches: done.
- Referenced aggregate JSON/summary files exist: done.
- Supplementary Table S1: still needs to be prepared from the official GISAID acknowledgement/accession export before journal submission.
