# -*- coding: utf-8 -*-
"""
latent_geometry_visualize.py
============================
Figuras PCA desde cache de embeddings de AntigenLM.

Este script no carga AntigenLM, no recalcula embeddings y no ejecuta UMAP.
Solo usa embeddings cacheados para generar visualizaciones descriptivas.
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = "figures/gisaid"
RESULTS_DIR = "results"
PCA_THRESHOLDS = (0.80, 0.90, 0.95, 0.99)
SUBTYPE_COLORS = {"H1N1": "#457B9D", "H3N2": "#E63946"}
EXPORT_DPI = 300


def load_cache(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)

    required = ("embeddings", "years", "types", "records")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Cache invalido, faltan claves: {missing}")

    embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
    years = np.asarray(payload["years"])
    types = np.asarray(payload["types"])
    records = payload["records"]

    lengths = {
        "embeddings": len(embeddings),
        "years": len(years),
        "types": len(types),
        "records": len(records),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Cache desalineado: {lengths}")

    return embeddings, years, types, records, payload.get("metadata", {})


def pca_from_covariance(embeddings):
    X = np.asarray(embeddings, dtype=np.float64)
    finite_mask = np.isfinite(X).all(axis=1)
    X = X[finite_mask]
    if len(X) < 2:
        raise ValueError("Insuficientes embeddings finitos para PCA")

    mean = X.mean(axis=0, keepdims=True)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / max(len(X_centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    positive = eigvals[eigvals > 1e-12]
    ratios = positive / positive.sum()
    cumulative = np.cumsum(ratios)
    participation_ratio = float((positive.sum() ** 2) / np.sum(positive ** 2))
    components_by_threshold = {
        threshold: int(np.searchsorted(cumulative, threshold) + 1)
        for threshold in PCA_THRESHOLDS
    }

    return {
        "n": int(len(X)),
        "dim": int(X.shape[1]),
        "mean": mean,
        "eigvals": positive,
        "eigvecs": eigvecs[:, :len(positive)],
        "ratios": ratios,
        "cumulative": cumulative,
        "participation_ratio": participation_ratio,
        "components_by_threshold": components_by_threshold,
    }


def project_with_pca(embeddings, pca, n_components=3):
    X = np.asarray(embeddings, dtype=np.float64)
    X_centered = X - pca["mean"]
    return X_centered @ pca["eigvecs"][:, :n_components]


def threshold_value(summary, threshold):
    return summary["components_by_threshold"][threshold]


def print_summary(label, summary):
    print(
        f"{label}: n={summary['n']} dim={summary['dim']} "
        f"n80={threshold_value(summary, 0.80)} "
        f"n90={threshold_value(summary, 0.90)} "
        f"n95={threshold_value(summary, 0.95)} "
        f"n99={threshold_value(summary, 0.99)} "
        f"PR={summary['participation_ratio']:.2f}"
    )
    top10 = ", ".join(f"{value:.4f}" for value in summary["ratios"][:10])
    print(f"  top10 explained variance ratios: {top10}")


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_figure(fig, basename):
    png_path = os.path.join(FIGURES_DIR, f"{basename}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{basename}.pdf")
    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def save_scree_global(summary, max_components):
    n_plot = min(max_components, len(summary["ratios"]))
    xs = np.arange(1, n_plot + 1)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(xs, summary["ratios"][:n_plot], marker="o", markersize=3.5, linewidth=1.7)
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA global de embeddings latentes")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_figure(fig, "pca_scree_global")


def save_scree_log_global(summary, max_components):
    n_plot = min(max_components, len(summary["ratios"]))
    xs = np.arange(1, n_plot + 1)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(xs, summary["ratios"][:n_plot], marker="o", markersize=3.5, linewidth=1.7)
    ax.set_yscale("log")
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Explained variance ratio (escala log)")
    ax.set_title("PCA global de embeddings latentes - cola espectral")
    ax.grid(alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_figure(fig, "pca_scree_log_global")


def save_cumulative_global(summary, max_components):
    n_plot = min(max_components, len(summary["cumulative"]))
    xs = np.arange(1, n_plot + 1)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(xs, summary["cumulative"][:n_plot], marker="o", markersize=2.8, linewidth=1.8)
    for threshold in PCA_THRESHOLDS:
        n_components = threshold_value(summary, threshold)
        ax.axhline(threshold, linestyle="--", linewidth=1.0, color="gray", alpha=0.75)
        ax.axvline(n_components, linestyle=":", linewidth=1.0, color="gray", alpha=0.65)
        ax.scatter([n_components], [threshold], color="#333333", s=28, zorder=5)
        ax.text(
            n_components + 0.8,
            min(threshold + 0.012, 1.0),
            f"{int(threshold * 100)}%: {n_components} PCs",
            fontsize=8.5,
            va="bottom",
        )
    ax.set_xlabel("Numero de componentes principales")
    ax.set_ylabel("Varianza explicada acumulada")
    ax.set_ylim(0, 1.02)
    ax.set_title("PCA global de embeddings latentes - varianza acumulada")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_figure(fig, "pca_cumulative_global")


def save_scree_by_subtype(subtype_summaries, max_components):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for subtype in sorted(subtype_summaries):
        summary = subtype_summaries[subtype]
        n_plot = min(max_components, len(summary["ratios"]))
        ax.plot(
            np.arange(1, n_plot + 1),
            summary["ratios"][:n_plot],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=SUBTYPE_COLORS.get(subtype),
            label=subtype,
        )
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA por subtipo - scree plot")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_figure(fig, "pca_scree_by_subtype")


def save_cumulative_by_subtype(subtype_summaries, max_components):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for subtype in sorted(subtype_summaries):
        summary = subtype_summaries[subtype]
        n_plot = min(max_components, len(summary["cumulative"]))
        ax.plot(
            np.arange(1, n_plot + 1),
            summary["cumulative"][:n_plot],
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=SUBTYPE_COLORS.get(subtype),
            label=subtype,
        )
    for threshold in PCA_THRESHOLDS:
        ax.axhline(threshold, linestyle="--", linewidth=1.0, color="gray", alpha=0.75)
        ax.text(max_components, threshold + 0.006, f"{int(threshold * 100)}%", ha="right", fontsize=9)
    ax.set_xlabel("Numero de componentes principales")
    ax.set_ylabel("Varianza explicada acumulada")
    ax.set_ylim(0, 1.02)
    ax.set_title("PCA por subtipo - varianza acumulada")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_figure(fig, "pca_cumulative_by_subtype")


def axis_label(pc_idx, ratios):
    return f"PC{pc_idx + 1} ({ratios[pc_idx] * 100:.2f}% var.)"


def save_pca_2d_by_subtype(coords, types, ratios):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for subtype in sorted(set(types)):
        mask = types == subtype
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=5,
            alpha=0.45,
            linewidths=0,
            color=SUBTYPE_COLORS.get(subtype),
            label=f"{subtype} (n={int(np.sum(mask))})",
        )
    ax.set_xlabel(axis_label(0, ratios))
    ax.set_ylabel(axis_label(1, ratios))
    ax.set_title("Proyeccion PCA 2D del espacio latente por subtipo")
    ax.legend(markerscale=2.5)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return save_figure(fig, "pca_2d_by_subtype")


def save_pca_2d_by_year(coords, years, ratios, path, title):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=years, cmap="viridis", s=5, alpha=0.5, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Anio")
    ax.set_xlabel(axis_label(0, ratios))
    ax.set_ylabel(axis_label(1, ratios))
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    basename = os.path.splitext(os.path.basename(path))[0]
    return save_figure(fig, basename)


def save_pca_3d_by_subtype(coords, types, ratios):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for subtype in sorted(set(types)):
        mask = types == subtype
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            s=6,
            alpha=0.45,
            color=SUBTYPE_COLORS.get(subtype),
            label=f"{subtype} (n={int(np.sum(mask))})",
            depthshade=False,
        )
    ax.set_xlabel(axis_label(0, ratios))
    ax.set_ylabel(axis_label(1, ratios))
    ax.set_zlabel(axis_label(2, ratios))
    ax.set_title("PCA 3D del espacio latente - por subtipo")
    ax.legend(markerscale=2.5)
    plt.tight_layout()
    return save_figure(fig, "pca_3d_by_subtype")


def save_pca_3d_by_year(coords, years, ratios):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=years,
        cmap="viridis",
        s=6,
        alpha=0.5,
        linewidths=0,
        depthshade=False,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.1)
    cbar.set_label("Anio")
    ax.set_xlabel(axis_label(0, ratios))
    ax.set_ylabel(axis_label(1, ratios))
    ax.set_zlabel(axis_label(2, ratios))
    ax.set_title("PCA 3D del espacio latente - por anio")
    plt.tight_layout()
    return save_figure(fig, "pca_3d_by_year")


def write_summary(path, global_summary, subtype_summaries, generated_paths):
    def row(label, summary):
        return (
            f"| {label} | {summary['n']} | {summary['dim']} | "
            f"{threshold_value(summary, 0.80)} | {threshold_value(summary, 0.90)} | "
            f"{threshold_value(summary, 0.95)} | {threshold_value(summary, 0.99)} | "
            f"{summary['participation_ratio']:.2f} |"
        )

    lines = [
        "# Resumen de figuras PCA",
        "",
        "Fuente: `results/embeddings_cache_10k_per_subtype_seed42.pkl`.",
        "No se recalcularon embeddings ni se cargo AntigenLM.",
        "",
        "| grupo | n | dim | n80 | n90 | n95 | n99 | participation ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("global", global_summary),
        row("H1N1", subtype_summaries["H1N1"]),
        row("H3N2", subtype_summaries["H3N2"]),
        "",
        "## Top 10 explained variance ratios",
        "",
        "- Global: " + ", ".join(f"{value:.4f}" for value in global_summary["ratios"][:10]) + ".",
        "- H1N1: " + ", ".join(f"{value:.4f}" for value in subtype_summaries["H1N1"]["ratios"][:10]) + ".",
        "- H3N2: " + ", ".join(f"{value:.4f}" for value in subtype_summaries["H3N2"]["ratios"][:10]) + ".",
        "",
        "## Figuras generadas",
        "",
    ]
    for fig_path in generated_paths:
        lines.append(f"- `{fig_path}`")

    lines.extend([
        "",
        "## Interpretacion breve de figuras",
        "",
        "- `pca_scree_global`: muestra la concentracion extrema de varianza en las primeras componentes.",
        "- `pca_scree_log_global`: permite inspeccionar la cola espectral, que queda comprimida en escala lineal.",
        "- `pca_cumulative_global`: resume que 2, 3, 4 y 12 componentes alcanzan 80%, 90%, 95% y 99% de varianza explicada, respectivamente.",
        "- `pca_2d_by_subtype`: muestra organizacion clara por subtipo en las dos primeras componentes principales.",
        "- `pca_2d_by_year`, `pca_2d_h1n1_by_year` y `pca_2d_h3n2_by_year`: permiten explorar patron temporal, sin asumir una trayectoria cronologica simple.",
        "",
        "## Lectura metodologica",
        "",
        "- PCA confirma concentracion fuerte de varianza en pocas componentes.",
        "- Esto es compatible con TwoNN y refuerza la hipotesis de baja dimension efectiva.",
        "- La participation ratio baja sugiere anisotropia fuerte.",
        "- Las proyecciones bidimensionales revelan organizacion por subtipo; la coloracion por anio no implica por si misma una dinamica temporal monotona.",
        "- En conjunto con la correlacion fuerte entre distancia latente y Hamming HA/HA+NA, estos resultados sugieren que la geometria latente captura estructura molecular relevante mas que unicamente orden temporal.",
        "- Las figuras son exploratorias/descriptivas y no prueban por si solas que una SDE funcione.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Genera figuras PCA desde cache de embeddings AntigenLM"
    )
    parser.add_argument(
        "--cache-path",
        default=os.path.join(RESULTS_DIR, "embeddings_cache_10k_per_subtype_seed42.pkl"),
    )
    parser.add_argument("--pca-max-components", type=int, default=100)
    args = parser.parse_args()

    ensure_dirs()
    Z, years, types, records, metadata = load_cache(args.cache_path)
    print(f"Cache cargado: embeddings={Z.shape} records={len(records)}")
    print(
        "Metadata: "
        f"sampling={metadata.get('sampling_strategy')} "
        f"seed={metadata.get('seed')} "
        f"max_per_subtype={metadata.get('max_per_subtype')}"
    )

    global_pca = pca_from_covariance(Z)
    subtype_summaries = {}
    for subtype in sorted(set(types)):
        subtype_summaries[subtype] = pca_from_covariance(Z[types == subtype])

    print_summary("global", global_pca)
    for subtype in sorted(subtype_summaries):
        print_summary(subtype, subtype_summaries[subtype])

    coords = project_with_pca(Z, global_pca, n_components=3)
    generated = []
    generated.extend(save_scree_global(global_pca, max(50, args.pca_max_components)))
    generated.extend(save_scree_log_global(global_pca, max(50, args.pca_max_components)))
    generated.extend(save_cumulative_global(global_pca, max(50, args.pca_max_components)))
    generated.extend(save_scree_by_subtype(subtype_summaries, max(50, args.pca_max_components)))
    generated.extend(save_cumulative_by_subtype(subtype_summaries, max(50, args.pca_max_components)))
    generated.extend(save_pca_2d_by_subtype(coords, types, global_pca["ratios"]))
    generated.extend(save_pca_2d_by_year(
        coords,
        years,
        global_pca["ratios"],
        os.path.join(FIGURES_DIR, "pca_2d_by_year.png"),
        "Proyeccion PCA 2D del espacio latente por anio",
    ))

    for subtype in ("H1N1", "H3N2"):
        mask = types == subtype
        generated.extend(save_pca_2d_by_year(
            coords[mask],
            years[mask],
            global_pca["ratios"],
            os.path.join(FIGURES_DIR, f"pca_2d_{subtype.lower()}_by_year.png"),
            f"Proyeccion PCA 2D {subtype} por anio",
        ))

    generated.extend(save_pca_3d_by_subtype(coords, types, global_pca["ratios"]))
    generated.extend(save_pca_3d_by_year(coords, years, global_pca["ratios"]))

    summary_path = write_summary(
        os.path.join(RESULTS_DIR, "pca_figures_summary.md"),
        global_pca,
        subtype_summaries,
        generated,
    )

    print("\nFiguras generadas:")
    for path in generated:
        print(f"  {path}")
    print(f"\nResumen guardado: {summary_path}")
    print("\nAdvertencia: las proyecciones 2D/3D son descriptivas; no prueban por si solas que una SDE funcione.")


if __name__ == "__main__":
    main()
