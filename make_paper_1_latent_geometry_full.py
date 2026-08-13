# -*- coding: utf-8 -*-
"""
Build an Overleaf-ready Paper 1 draft after the full geometry analysis.

Input:
  results/latent_geometry_full_metrics.json

Output:
  papers/paper_1_latent_geometry_full/
    main.tex
    references.bib
    README.md
    figures/*.pdf
  archive/exports/paper_1_latent_geometry_full.zip
"""

import json
import os
import shutil
import zipfile
from collections import defaultdict

import numpy as np


METRICS_PATH = "results/latent_geometry_full_metrics.json"
FIGURES_SRC = "figures/latent_geometry_full"
PAPER_DIR = "papers/paper_1_latent_geometry_full"
PAPER_FIGURES = os.path.join(PAPER_DIR, "figures")
ZIP_PATH = "archive/exports/paper_1_latent_geometry_full.zip"
SUBTYPES = ("H1N1", "H3N2")
METRIC_LABELS = {
    "temporal": "Temporal distance",
    "hamming_ha": "HA Hamming",
    "hamming_ha_na": "HA+NA Hamming",
}
FIGURES = [
    "records_by_year_subtype.pdf",
    "sequence_length_distributions.pdf",
    "spearman_latent_vs_distances.pdf",
    "pca_scree_global.pdf",
    "pca_cumulative_global.pdf",
    "pca_2d_by_subtype.pdf",
    "pca_2d_by_year.pdf",
    "twonn_sensitivity.pdf",
    "twonn_fit_example.pdf",
    "temporal_local_neighbors_h1n1_dedup.pdf",
    "temporal_local_neighbors_h3n2_dedup.pdf",
]


def fmt(value, digits=3):
    try:
        if value is None or not np.isfinite(float(value)):
            return "--"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "--"


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"No existe {METRICS_PATH}. Ejecuta primero latent_geometry_full_analysis.py."
        )
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_spearman(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["metric"], row["subtype"])].append(row)
    out = {}
    for key, values in grouped.items():
        rhos = np.asarray([v["rho"] for v in values], dtype=float)
        valid = np.asarray([v["valid_pairs"] for v in values], dtype=float)
        out[key] = {
            "rho_mean": float(np.nanmean(rhos)),
            "rho_sd": float(np.nanstd(rhos)),
            "valid_mean": float(np.nanmean(valid)),
        }
    return out


def latex_spearman_table(metrics):
    agg = aggregate_spearman(metrics["spearman"])
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Pair-sampled Spearman correlations between latent Euclidean distance and temporal or molecular distances. Values are summarized across random seeds.}",
        "\\label{tab:spearman}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Metric & Subtype & $\\rho$ mean & $\\rho$ sd & valid pairs \\\\",
        "\\midrule",
    ]
    for metric in ("temporal", "hamming_ha", "hamming_ha_na"):
        for subtype in SUBTYPES:
            row = agg.get((metric, subtype), {})
            lines.append(
                f"{METRIC_LABELS[metric]} & {subtype} & {fmt(row.get('rho_mean'), 4)} & "
                f"{fmt(row.get('rho_sd'), 4)} & {row.get('valid_mean', 0):.0f} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def latex_pca_table(metrics):
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{PCA effective dimension of AntigenLM embeddings. $n_q$ denotes the number of principal components required to explain fraction $q$ of variance.}",
        "\\label{tab:pca}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Group & n & $n_{80}$ & $n_{90}$ & $n_{95}$ & $n_{99}$ & PR \\\\",
        "\\midrule",
    ]
    for group in ("global", "H1N1", "H3N2"):
        row = metrics["pca"].get(group)
        if not row:
            continue
        thr = row["n_components_by_threshold"]
        lines.append(
            f"{group} & {row['n_samples']:,} & {thr['0.8']} & {thr['0.9']} & "
            f"{thr['0.95']} & {thr['0.99']} & {fmt(row['participation_ratio'], 2)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def latex_twonn_table(metrics):
    grouped = defaultdict(list)
    for row in metrics["twonn"]["rows"]:
        if np.isfinite(float(row.get("dimension", np.nan))):
            grouped[(row["sample_size_used"], row["trim"])].append(row)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{TwoNN intrinsic-dimension sensitivity after exact HA+NA deduplication.}",
        "\\label{tab:twonn}",
        "\\begin{tabular}{rrrr}",
        "\\toprule",
        "Sample size & trim & dimension mean & $R^2$ mean \\\\",
        "\\midrule",
    ]
    for key in sorted(grouped):
        values = grouped[key]
        dims = np.asarray([v["dimension"] for v in values], dtype=float)
        r2s = np.asarray([v["r2"] for v in values], dtype=float)
        lines.append(f"{key[0]:,} & {key[1]:.2f} & {fmt(np.mean(dims), 2)} & {fmt(np.mean(r2s), 3)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def latex_temporal_table(metrics):
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Local temporal coherence of latent nearest neighbors after exact HA+NA deduplication.}",
        "\\label{tab:temporal-local}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Subtype & k & median NN months & median random months & ratio \\\\",
        "\\midrule",
    ]
    for row in metrics["temporal_locality"]["rows"]:
        lines.append(
            f"{row['subtype']} & {row['k']} & {fmt(row['neighbor']['median'], 2)} & "
            f"{fmt(row['random']['median'], 2)} & {fmt(row['median_ratio'], 3)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def copy_figures():
    os.makedirs(PAPER_FIGURES, exist_ok=True)
    copied = []
    missing = []
    for name in FIGURES:
        src = os.path.join(FIGURES_SRC, name)
        dst = os.path.join(PAPER_FIGURES, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied.append(name)
        else:
            missing.append(name)
    return copied, missing


def write_references():
    bib = r"""@inproceedings{pei2026antigenlm,
  author = {Pei, Y. and Chi, X. and Kang, Y.},
  title = {AntigenLM: Structure-Aware DNA Language Modeling for Influenza},
  booktitle = {International Conference on Learning Representations},
  year = {2026},
  note = {Metadata should be verified against the official publication record}
}

@article{facco2017intrinsic,
  author = {Facco, Elena and d'Errico, Maria and Rodriguez, Alex and Laio, Alessandro},
  title = {Estimating the intrinsic dimension of datasets by a minimal neighborhood information},
  journal = {Scientific Reports},
  volume = {7},
  pages = {12140},
  year = {2017},
  doi = {10.1038/s41598-017-11873-y}
}

@book{jolliffe2002pca,
  author = {Jolliffe, I. T.},
  title = {Principal Component Analysis},
  publisher = {Springer},
  edition = {2},
  year = {2002}
}

@article{jolliffe2016pca,
  author = {Jolliffe, Ian T. and Cadima, Jorge},
  title = {Principal component analysis: a review and recent developments},
  journal = {Philosophical Transactions of the Royal Society A},
  volume = {374},
  number = {2065},
  pages = {20150202},
  year = {2016},
  doi = {10.1098/rsta.2015.0202}
}

@article{spearman1904,
  author = {Spearman, C.},
  title = {The proof and measurement of association between two things},
  journal = {The American Journal of Psychology},
  volume = {15},
  number = {1},
  pages = {72--101},
  year = {1904}
}

@article{luksza2014fitness,
  author = {\L{}uksza, Marta and L\"assig, Michael},
  title = {A predictive fitness model for influenza},
  journal = {Nature},
  volume = {507},
  pages = {57--61},
  year = {2014}
}

@article{neher2014predicting,
  author = {Neher, Richard A. and Russell, Colin A. and Shraiman, Boris I.},
  title = {Predicting evolution from the shape of genealogical trees},
  journal = {eLife},
  volume = {3},
  pages = {e03568},
  year = {2014}
}
"""
    with open(os.path.join(PAPER_DIR, "references.bib"), "w", encoding="utf-8") as f:
        f.write(bib)


def write_main(metrics):
    data = metrics["data_audit"]
    n_total = data["embedding_shape"][0]
    n_dim = data["embedding_shape"][1]
    h1 = data["by_subtype"].get("H1N1", {})
    h3 = data["by_subtype"].get("H3N2", {})
    tex = rf"""\documentclass[11pt]{{article}}

\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[english]{{babel}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{subcaption}}
\usepackage{{booktabs}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{float}}

\title{{Geometric Structure of AntigenLM Latent Representations for Influenza A Evolution}}
\author{{Carlos Morrego \and Thesis checkpoint draft}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
We audit the geometry of local AntigenLM latent representations for Influenza A using HA/NA sequence records from H1N1 and H3N2. Each record is represented as
\[
z_i = Enc_\theta(HA_i, NA_i) \in \mathbb{{R}}^{{384}}.
\]
The analysis evaluates whether latent Euclidean distances preserve molecular similarity, whether global and local temporal structure are present, and whether the embedding cloud has low effective dimension. Pairwise molecular and temporal correlations are estimated by stratified random sampling of pairs within subtype, avoiding the infeasible quadratic comparison over all records. PCA and TwoNN are used as complementary linear and local intrinsic-dimension diagnostics, and nearest-neighbor temporal differences are compared against subtype-matched random baselines. This work does not generate sequences, optimize mutations, make vaccine recommendations, or reproduce AntigenLM forecasting figures. The results support using the local AntigenLM embedding space as a candidate state space for future probabilistic models, while leaving forecasting validation for subsequent work.
\end{{abstract}}

\section{{Introduction}}
Influenza A evolves through the accumulation and selection of mutations in surface proteins, especially hemagglutinin (HA) and neuraminidase (NA). Sequence language models may provide compact representations of this evolutionary variation, but a latent space should not be used as a dynamical state variable without first auditing its geometry. This checkpoint asks whether local AntigenLM embeddings preserve molecular similarity, contain interpretable temporal organization, and exhibit low effective dimension.

\section{{Background}}
AntigenLM is used here as a released local checkpoint and embedding function, not as a fully reproduced forecasting pipeline \cite{{pei2026antigenlm}}. For two records $i,j$, latent distance is
\[
d_Z(i,j)=\|z_i-z_j\|_2.
\]
Temporal distance is
\[
d_T(i,j)=|12(y_i-y_j)+(m_i-m_j)|,
\]
and normalized Hamming distance is
\[
d_H(x,y)=\frac{{1}}{{L}}\sum_{{\ell=1}}^L \mathbf{{1}}\{{x_\ell\neq y_\ell\}}.
\]
We estimate rank correlation using Spearman's $\rho$ \cite{{spearman1904}}.

PCA effective dimension is summarized by $n_{{80}}, n_{{90}}, n_{{95}}, n_{{99}}$ and by the participation ratio
\[
PR=\frac{{(\sum_i \lambda_i)^2}}{{\sum_i \lambda_i^2}}.
\]
TwoNN estimates local intrinsic dimension using
\[
\mu_i=\frac{{r_{{2,i}}}}{{r_{{1,i}}}},\qquad
-\log(1-F_i)=d\log(\mu_i),
\]
where $r_1$ and $r_2$ are first and second nearest-neighbor distances \cite{{facco2017intrinsic}}.

\section{{Methods}}
\subsection{{Data and embeddings}}
The processed local dataset contains {n_total:,} cached embeddings of dimension {n_dim}. The cache includes {h1.get('cached_embeddings', 0):,} H1N1 and {h3.get('cached_embeddings', 0):,} H3N2 records. Complete sequences are not printed or redistributed; outputs contain only aggregate counts, metrics, tables, and figures.

\subsection{{Pair sampling}}
All valid records are used for embedding extraction and global PCA-style analyses. Pairwise correlations are estimated by random pair sampling within subtype because all pairwise comparisons would scale quadratically. Multiple random seeds are used to assess sampling sensitivity.

\subsection{{Deduplication}}
TwoNN and local temporal-neighborhood analyses are performed after exact HA+NA deduplication, retaining the first representative of each exact sequence pair.

\section{{Results}}
\subsection{{Data audit}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.78\textwidth]{{figures/records_by_year_subtype.pdf}}
\caption{{Processed records by year and subtype.}}
\label{{fig:data-year}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\textwidth]{{figures/sequence_length_distributions.pdf}}
\caption{{HA and NA sequence-length distributions by subtype.}}
\label{{fig:lengths}}
\end{{figure}}

\subsection{{Latent distance preserves molecular similarity}}
Table~\ref{{tab:spearman}} summarizes pair-sampled Spearman correlations. The main methodological comparison is between temporal distance and molecular Hamming distances.

{latex_spearman_table(metrics)}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.85\textwidth]{{figures/spearman_latent_vs_distances.pdf}}
\caption{{Spearman correlation summary for latent distance versus temporal and molecular distances.}}
\label{{fig:spearman}}
\end{{figure}}

\subsection{{Effective dimension by PCA}}
PCA is used as a linear effective-dimension diagnostic \cite{{jolliffe2002pca,jolliffe2016pca}}.

{latex_pca_table(metrics)}

\begin{{figure}}[H]
\centering
\begin{{subfigure}}{{0.48\textwidth}}
\includegraphics[width=\textwidth]{{figures/pca_scree_global.pdf}}
\caption{{Global scree plot.}}
\end{{subfigure}}
\begin{{subfigure}}{{0.48\textwidth}}
\includegraphics[width=\textwidth]{{figures/pca_cumulative_global.pdf}}
\caption{{Global cumulative variance.}}
\end{{subfigure}}
\caption{{Global PCA spectrum.}}
\label{{fig:pca-spectrum}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\begin{{subfigure}}{{0.48\textwidth}}
\includegraphics[width=\textwidth]{{figures/pca_2d_by_subtype.pdf}}
\caption{{Colored by subtype.}}
\end{{subfigure}}
\begin{{subfigure}}{{0.48\textwidth}}
\includegraphics[width=\textwidth]{{figures/pca_2d_by_year.pdf}}
\caption{{Colored by year.}}
\end{{subfigure}}
\caption{{Two-dimensional global PCA projection.}}
\label{{fig:pca-2d}}
\end{{figure}}

\subsection{{TwoNN intrinsic-dimension sensitivity}}
TwoNN provides a local, non-PCA intrinsic-dimension diagnostic. It should be interpreted as a range rather than a single absolute truth.

{latex_twonn_table(metrics)}

\begin{{figure}}[H]
\centering
\begin{{subfigure}}{{0.48\textwidth}}
\includegraphics[width=\textwidth]{{figures/twonn_sensitivity.pdf}}
\caption{{Sensitivity.}}
\end{{subfigure}}
\begin{{subfigure}}{{0.48\textwidth}}
\includegraphics[width=\textwidth]{{figures/twonn_fit_example.pdf}}
\caption{{Example fit.}}
\end{{subfigure}}
\caption{{TwoNN intrinsic-dimension diagnostics.}}
\label{{fig:twonn}}
\end{{figure}}

\subsection{{Temporal distance is weak globally but strong locally}}
Local temporal structure is evaluated by comparing latent nearest-neighbor time differences against random subtype-matched pairs.

{latex_temporal_table(metrics)}

\begin{{figure}}[H]
\centering
\begin{{subfigure}}{{0.48\textwidth}}
\includegraphics[width=\textwidth]{{figures/temporal_local_neighbors_h1n1_dedup.pdf}}
\caption{{H1N1.}}
\end{{subfigure}}
\begin{{subfigure}}{{0.48\textwidth}}
\includegraphics[width=\textwidth]{{figures/temporal_local_neighbors_h3n2_dedup.pdf}}
\caption{{H3N2.}}
\end{{subfigure}}
\caption{{Temporal differences of latent nearest neighbors versus random baselines after exact HA+NA deduplication.}}
\label{{fig:temporal-local}}
\end{{figure}}

\section{{Discussion}}
The audit separates three claims. First, molecular organization is assessed by Hamming correlations, which test whether nearby points in latent space tend to have similar HA/NA sequences. Second, global temporal correlation asks whether latent distance behaves like a simple chronological axis; weak global temporal correlation is not necessarily problematic because influenza evolution is branching and nonlinear. Third, temporal locality asks whether latent neighborhoods are temporally coherent, which is more relevant for local dynamical modeling.

Low PCA effective dimension and stable TwoNN ranges motivate reduced probabilistic dynamics in latent or PCA space. However, these geometric findings do not prove forecasting performance, do not validate a full stochastic differential equation, and do not establish vaccine-strain prediction.

\section{{Limitations}}
This checkpoint is not a reproduction of AntigenLM forecasting figures. It does not generate or decode sequences, does not evaluate prospective 2022--2026 forecasting, and does not compare against an official AntigenLM inference protocol. Hamming distance is a molecular proxy, not an experimental antigenic distance. Pairwise correlations are estimated by sampled pairs rather than complete all-pairs computation.

\section{{Conclusion}}
The local AntigenLM latent space exhibits molecular organization, low effective dimension, and coherent local temporal neighborhoods. These results support its use as a candidate state space for future probabilistic models of influenza evolution, but they do not by themselves validate a full predictive SDE or vaccine-strain forecasting model.

\bibliographystyle{{plain}}
\bibliography{{references}}

\end{{document}}
"""
    with open(os.path.join(PAPER_DIR, "main.tex"), "w", encoding="utf-8") as f:
        f.write(tex)


def write_readme(copied, missing):
    text = f"""# Paper 1: full latent geometry audit

This folder contains an Overleaf-ready draft generated from:

- `results/latent_geometry_full_metrics.json`
- `results/latent_geometry_full_summary.md`
- `results/full_data_audit_summary.md`
- `figures/latent_geometry_full/`

## What this paper claims

- It audits the geometry of local AntigenLM embeddings for Influenza A.
- It reports aggregate molecular, temporal, PCA, TwoNN, and local-neighborhood diagnostics.
- It uses all cached valid embeddings for global analyses and pair sampling for pairwise correlations.

## What this paper does not claim

- It does not reproduce AntigenLM Figure 3A or Figure 3B.
- It does not generate sequences.
- It does not optimize mutations.
- It does not make clinical or vaccine recommendations.
- It does not validate a full predictive SDE.

## Overleaf

Upload the contents of this folder to Overleaf and compile `main.tex` with `pdfLaTeX` and BibTeX.

## Figures copied

{chr(10).join(f'- `{name}`' for name in copied)}

## Figures missing

{chr(10).join(f'- `{name}`' for name in missing) if missing else '- None'}

## Rebuild

From the repository root:

```bash
venv_antigenlm/bin/python make_paper_1_latent_geometry_full.py
```
"""
    with open(os.path.join(PAPER_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


def make_zip():
    os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(PAPER_DIR):
            for filename in files:
                path = os.path.join(root, filename)
                zf.write(path, arcname=path)


def main():
    metrics = load_metrics()
    os.makedirs(PAPER_DIR, exist_ok=True)
    copied, missing = copy_figures()
    write_main(metrics)
    write_references()
    write_readme(copied, missing)
    make_zip()
    print(f"Paper draft: {PAPER_DIR}/main.tex")
    print(f"References: {PAPER_DIR}/references.bib")
    print(f"Copied figures: {len(copied)}")
    print(f"Missing figures: {len(missing)}")
    print(f"Zip: {ZIP_PATH}")


if __name__ == "__main__":
    main()
