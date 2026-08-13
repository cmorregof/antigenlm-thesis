# -*- coding: utf-8 -*-
"""
latent_temporal_neighbors.py
============================
Auditoria de temporalidad local en el espacio latente de AntigenLM.

El script carga embeddings cacheados, no carga AntigenLM, no recalcula
embeddings y no imprime secuencias. Evalua si vecinos cercanos en el
espacio latente tienden a estar mas cerca temporalmente que pares
aleatorios dentro del mismo subtipo.
"""

import argparse
import hashlib
import os
import pickle
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = "figures/gisaid"
RESULTS_DIR = "results"
DEFAULT_K_VALUES = (5, 10, 20)
SUBTYPE_COLORS = {"H1N1": "#457B9D", "H3N2": "#E63946"}
RANDOM_COLOR = "#6B7280"
EXPORT_DPI = 300


def load_cache(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)

    required = ("embeddings", "years", "months", "types", "records")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Cache invalido, faltan claves: {missing}")

    embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
    years = np.asarray(payload["years"], dtype=int)
    months = np.asarray(payload["months"], dtype=int)
    types = np.asarray(payload["types"])
    records = payload["records"]

    lengths = {
        "embeddings": len(embeddings),
        "years": len(years),
        "months": len(months),
        "types": len(types),
        "records": len(records),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Cache desalineado: {lengths}")

    return embeddings, years, months, types, records, payload.get("metadata", {})


def temporal_month_index(years, months):
    return years * 12 + np.clip(months, 1, 12)


def ha_na_hash(record):
    ha = record.get("ha_sequence", "") if record else ""
    na = record.get("na_sequence", "") if record else ""
    return hashlib.sha1(f"{ha}|{na}".encode("utf-8")).hexdigest()


def deduplicate_records(embeddings, years, months, types, records):
    """Deduplica por HA+NA exactos conservando el primer representante."""
    seen = set()
    keep = []
    removed_by_subtype = Counter()

    for idx, record in enumerate(records):
        key = ha_na_hash(record)
        if key in seen:
            removed_by_subtype[str(types[idx])] += 1
            continue
        seen.add(key)
        keep.append(idx)

    keep = np.asarray(keep, dtype=int)
    dedup_records = [records[i] for i in keep]
    return (
        embeddings[keep],
        years[keep],
        months[keep],
        types[keep],
        dedup_records,
        removed_by_subtype,
    )


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n_pairs": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def random_temporal_deltas(time_months, n_pairs, seed):
    rng = np.random.default_rng(seed)
    n = len(time_months)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n - 1, size=n_pairs)
    # Evita pares i==j sin bucles costosos.
    j = j + (j >= i)
    return np.abs(time_months[i] - time_months[j])


def nearest_neighbor_temporal_deltas(embeddings, time_months, k_values):
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError("scikit-learn es necesario para vecinos cercanos") from exc

    max_k = max(k_values)
    finite_mask = np.isfinite(embeddings).all(axis=1)
    X = embeddings[finite_mask]
    t = time_months[finite_mask]
    if len(X) <= max_k:
        raise ValueError(f"Insuficientes puntos finitos ({len(X)}) para k={max_k}")

    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_indices = indices[:, 1:]

    deltas_by_k = {}
    for k in k_values:
        idx = neighbor_indices[:, :k]
        deltas = np.abs(t[:, None] - t[idx]).reshape(-1)
        deltas_by_k[k] = deltas
    return deltas_by_k, int(np.sum(~finite_mask))


def export_figure(fig, basename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    png_path = os.path.join(FIGURES_DIR, f"{basename}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{basename}.pdf")
    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_subtype(subtype, neighbor_deltas, random_deltas, suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    color = SUBTYPE_COLORS.get(subtype, "#457B9D")

    labels = []
    data = []
    for k in sorted(neighbor_deltas):
        labels.append(f"k={k}")
        data.append(neighbor_deltas[k])
    labels.append("random")
    data.append(random_deltas[max(neighbor_deltas)])

    box = axes[0].boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(RANDOM_COLOR if label == "random" else color)
        patch.set_alpha(0.55)
    axes[0].set_ylabel("Diferencia temporal absoluta (meses)")
    axes[0].set_title(f"{subtype}: vecinos latentes vs pares aleatorios")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].spines[["top", "right"]].set_visible(False)

    bins = np.linspace(
        0,
        max(np.percentile(random_deltas[max(neighbor_deltas)], 99), 1),
        45,
    )
    for k in sorted(neighbor_deltas):
        axes[1].hist(
            neighbor_deltas[k],
            bins=bins,
            density=True,
            alpha=0.30,
            label=f"vecinos k={k}",
        )
    axes[1].hist(
        random_deltas[max(neighbor_deltas)],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        color=RANDOM_COLOR,
        label="random",
    )
    axes[1].set_xlabel("Diferencia temporal absoluta (meses)")
    axes[1].set_ylabel("Densidad")
    axes[1].set_title("Distribucion de distancias temporales")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Temporalidad local en espacio latente: "
        f"{subtype}{' (deduplicado)' if suffix else ''}",
        y=1.02,
        fontsize=13,
    )
    plt.tight_layout()
    return export_figure(fig, f"temporal_local_neighbors_{subtype.lower()}{suffix}")


def analyze_subtype(embeddings, years, months, types, subtype, k_values, seed,
                    figure_suffix=""):
    mask = types == subtype
    X = embeddings[mask]
    time_months = temporal_month_index(years[mask], months[mask])
    neighbor_deltas, omitted_nonfinite = nearest_neighbor_temporal_deltas(
        X, time_months, k_values
    )

    random_deltas = {}
    results = {}
    for k in k_values:
        random_deltas[k] = random_temporal_deltas(
            time_months,
            n_pairs=len(neighbor_deltas[k]),
            seed=seed + k,
        )
        neighbor_stats = stats(neighbor_deltas[k])
        random_stats = stats(random_deltas[k])
        results[k] = {
            "neighbor": neighbor_stats,
            "random": random_stats,
            "median_ratio": neighbor_stats["median"] / random_stats["median"] if random_stats["median"] else np.nan,
            "mean_ratio": neighbor_stats["mean"] / random_stats["mean"] if random_stats["mean"] else np.nan,
        }

    fig_paths = plot_subtype(
        subtype, neighbor_deltas, random_deltas, suffix=figure_suffix
    )
    return {
        "subtype": subtype,
        "n": int(len(X)),
        "omitted_nonfinite": omitted_nonfinite,
        "results": results,
        "figures": fig_paths,
    }


def write_summary(path, analyses, cache_path, deduplicate=False,
                  removed_by_subtype=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = [
        "# Temporalidad local de vecinos latentes",
        "",
        f"Fuente: `{cache_path}`.",
        "No se recalcularon embeddings, no se cargo AntigenLM y no se imprimieron secuencias.",
        "",
    ]
    if deduplicate:
        lines.extend([
            "Deduplicacion: activada por hash exacto de HA+NA, conservando el primer representante.",
            "",
            "| subtipo | duplicados removidos |",
            "|---|---:|",
        ])
        removed_by_subtype = removed_by_subtype or {}
        for subtype in ("H1N1", "H3N2"):
            lines.append(f"| {subtype} | {int(removed_by_subtype.get(subtype, 0))} |")
        lines.append("")

    lines.extend([
        "La distancia temporal se calcula como diferencia absoluta en meses entre fechas de coleccion.",
        "La baseline aleatoria usa pares del mismo subtipo y el mismo numero de comparaciones que cada valor de k.",
        "",
        "| subtipo | k | n | pares vecinos | mediana vecinos | media vecinos | p25 vecinos | p75 vecinos | mediana random | media random | p25 random | p75 random | razon mediana | razon media |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for analysis in analyses:
        for k, result in analysis["results"].items():
            nstats = result["neighbor"]
            rstats = result["random"]
            lines.append(
                f"| {analysis['subtype']} | {k} | {analysis['n']} | {nstats['n_pairs']} | "
                f"{nstats['median']:.2f} | {nstats['mean']:.2f} | {nstats['p25']:.2f} | {nstats['p75']:.2f} | "
                f"{rstats['median']:.2f} | {rstats['mean']:.2f} | {rstats['p25']:.2f} | {rstats['p75']:.2f} | "
                f"{result['median_ratio']:.3f} | {result['mean_ratio']:.3f} |"
            )

    lines.extend([
        "",
        "## Figuras",
        "",
    ])
    for analysis in analyses:
        for fig_path in analysis["figures"]:
            lines.append(f"- `{fig_path}`")

    lines.extend([
        "",
        "## Lectura metodologica",
        "",
        "- Si las distancias temporales de vecinos latentes son menores que las de pares aleatorios, hay evidencia de estructura temporal local.",
        "- Si no lo son, el espacio puede preservar similitud molecular sin codificar continuidad temporal simple.",
        "- Esta prueba es local y descriptiva: no concluye que una SDE funcione.",
    ])
    if deduplicate:
        lines.append("- Esta corrida evalua si la senal persiste tras eliminar duplicados exactos HA+NA.")
    else:
        lines.append("- La presencia de duplicados exactos en el cache puede aumentar artificialmente la cercania temporal de algunos vecinos; conviene auditar sensibilidad con deduplicacion en una etapa posterior.")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Analiza temporalidad local de vecinos latentes desde cache"
    )
    parser.add_argument(
        "--cache-path",
        default=os.path.join(RESULTS_DIR, "embeddings_cache_10k_per_subtype_seed42.pkl"),
    )
    parser.add_argument("--k-values", default="5,10,20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate", action="store_true",
                        help="Elimina duplicados exactos HA+NA antes de calcular vecinos")
    args = parser.parse_args()

    k_values = tuple(int(k.strip()) for k in args.k_values.split(",") if k.strip())
    if not k_values:
        raise ValueError("Debe especificarse al menos un k")

    embeddings, years, months, types, records, metadata = load_cache(args.cache_path)
    print(f"Cache cargado: embeddings={embeddings.shape} records={len(records)}")
    print(
        "Metadata: "
        f"sampling={metadata.get('sampling_strategy')} "
        f"seed={metadata.get('seed')} "
        f"max_per_subtype={metadata.get('max_per_subtype')}"
    )

    removed_by_subtype = Counter()
    figure_suffix = ""
    summary_name = "temporal_local_neighbors_summary.md"
    if args.deduplicate:
        before_counts = Counter(types)
        embeddings, years, months, types, records, removed_by_subtype = deduplicate_records(
            embeddings, years, months, types, records
        )
        after_counts = Counter(types)
        figure_suffix = "_dedup"
        summary_name = "temporal_local_neighbors_summary_dedup.md"
        print("\nDeduplicacion HA+NA exacta activada:")
        for subtype in ("H1N1", "H3N2"):
            print(
                f"  {subtype}: {before_counts.get(subtype, 0):,} -> "
                f"{after_counts.get(subtype, 0):,} "
                f"(removidos={removed_by_subtype.get(subtype, 0):,})"
            )

    analyses = []
    for subtype in ("H1N1", "H3N2"):
        analysis = analyze_subtype(
            embeddings, years, months, types,
            subtype=subtype,
            k_values=k_values,
            seed=args.seed,
            figure_suffix=figure_suffix,
        )
        analyses.append(analysis)
        print(f"\n{subtype}: n={analysis['n']} omitidos_no_finitos={analysis['omitted_nonfinite']}")
        for k, result in analysis["results"].items():
            nstats = result["neighbor"]
            rstats = result["random"]
            print(
                f"  k={k}: vecinos mediana={nstats['median']:.2f} media={nstats['mean']:.2f} "
                f"p25/p75={nstats['p25']:.2f}/{nstats['p75']:.2f} | "
                f"random mediana={rstats['median']:.2f} media={rstats['mean']:.2f} "
                f"p25/p75={rstats['p25']:.2f}/{rstats['p75']:.2f} | "
                f"ratio_mediana={result['median_ratio']:.3f} ratio_media={result['mean_ratio']:.3f}"
            )
        for fig_path in analysis["figures"]:
            print(f"  figura: {fig_path}")

    summary_path = write_summary(
        os.path.join(RESULTS_DIR, summary_name),
        analyses,
        args.cache_path,
        deduplicate=args.deduplicate,
        removed_by_subtype=removed_by_subtype,
    )
    print(f"\nResumen guardado: {summary_path}")
    print("Advertencia: prueba descriptiva local; no valida por si sola una SDE.")


if __name__ == "__main__":
    main()
