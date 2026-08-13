# LaTeX Install and Compile Report

Date: 2026-05-05

## Environment detection

Detected OS:

```text
Darwin MacBook-Air-de-Carlos.local 25.4.0 Darwin Kernel Version 25.4.0: Thu Mar 19 19:33:43 PDT 2026; root:xnu-12377.101.15~1/RELEASE_ARM64_T8142 arm64
```

Detected package managers and TeX tools before installation:

```text
which brew: /opt/homebrew/bin/brew
which apt: apt not found
which tlmgr: tlmgr not found
which pdflatex: pdflatex not found
which bibtex: bibtex not found
which latexmk: latexmk not found
which tectonic: tectonic not found
```

## Package manager used

Homebrew was available on macOS and was used.

## LaTeX tool installed

Installed lightweight TeX toolchain:

```bash
brew install tectonic
```

Installed tool:

```text
/opt/homebrew/bin/tectonic
Tectonic 0.16.9
```

After installation, the classic TeX tools were still not present on PATH:

```text
pdflatex not found
bibtex not found
latexmk not found
tlmgr not found
```

BasicTeX was not installed because Tectonic successfully compiled the manuscript.

## Compile commands run

Initial compile from the manuscript directory:

```bash
cd papers/paper_1_latent_geometry_full
tectonic main.tex
```

Result: success. Tectonic downloaded the needed bundle files, ran BibTeX automatically, reran TeX as needed, and wrote `main.pdf`.

Second compile with log/intermediate retention:

```bash
cd papers/paper_1_latent_geometry_full
tectonic --keep-logs --keep-intermediates main.tex
```

Result: success. This preserved `main.log`, `main.blg`, `main.bbl`, `main.aux`, and `main.out`.

Copied compiled outputs:

```bash
cp papers/paper_1_latent_geometry_full/main.pdf paper_revision_outputs/latent_geometry_manuscript_compiled.pdf
cp papers/paper_1_latent_geometry_full/main.log paper_revision_outputs/main_compile_tectonic.log
cp papers/paper_1_latent_geometry_full/main.blg paper_revision_outputs/main_compile_bibtex.blg
cp papers/paper_1_latent_geometry_full/main.bbl paper_revision_outputs/main_compile_references.bbl
```

## Compilation result

Compilation succeeded.

Generated PDF:

- `papers/paper_1_latent_geometry_full/main.pdf`
- `paper_revision_outputs/latent_geometry_manuscript_compiled.pdf`

PDF metadata from macOS:

```text
kMDItemFSSize        = 2025732
kMDItemNumberOfPages = 14
```

The generated PDF is a valid PDF 1.5 document.

## Warnings and errors

No fatal LaTeX errors occurred.

Warnings reported in `main.log`:

```text
Package inputenc Warning: inputenc package ignored with utf8 based engines.
Overfull \hbox (114.54889pt too wide) in paragraph at lines 56--57
Overfull \hbox (171.56691pt too wide) in paragraph at lines 63--66
Overfull \hbox (214.29105pt too wide) in paragraph at lines 367--368
Overfull \hbox (142.1656pt too wide) in paragraph at lines 367--368
Underfull \hbox (badness 1888) in paragraph at lines 18--25 of `main.bbl`
```

Interpretation:

- The `inputenc` warning is expected because Tectonic uses an engine with UTF-8 support.
- The overfull boxes are layout warnings, mainly from long inline model/checkpoint paths, hashes, and reproducibility-file paths.
- The underfull box occurs in the bibliography formatting.
- None of these warnings prevented PDF generation.

## Missing packages installed

No `tlmgr` package installs were run. Tectonic fetched required TeX bundle resources automatically. BasicTeX was not needed.

Homebrew installed Tectonic and its runtime dependencies, including libraries such as `libpng`, `freetype`, `fontconfig`, `glib`, `cairo`, `icu4c@78`, and `harfbuzz`.

## Manuscript edits made for compilation

No manuscript edits were made during compilation. The compile succeeded without changing `papers/paper_1_latent_geometry_full/main.tex`.

## Preserved compile artifacts

- `paper_revision_outputs/main_compile_tectonic.log`
- `paper_revision_outputs/main_compile_bibtex.blg`
- `paper_revision_outputs/main_compile_references.bbl`
- `paper_revision_outputs/latent_geometry_manuscript_compiled.pdf`

Intermediate compile artifacts also exist in `papers/paper_1_latent_geometry_full/` from the retained Tectonic run.

## Visual inspection status

The PDF is ready for visual inspection. Before submission, visually inspect pages with the reported overfull boxes, especially the local checkpoint/hash paragraph and the reproducibility/data-availability paragraph, because they may show text extending beyond the margin.
