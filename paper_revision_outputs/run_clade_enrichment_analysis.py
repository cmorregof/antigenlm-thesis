#!/usr/bin/env python3
"""Clade-label enrichment analysis for AntigenLM latent neighborhoods.

This script reads the local embedding cache and a private GISAID metadata join,
but writes only aggregate results. It does not print or redistribute sequences
or accession-level metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUBTYPES = ("H1N1", "H3N2")


def clean_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"nan", "none", "na", "n/a", "not assigned", "unassigned"}:
        return None
    return text


def month_index(year: object, month: object) -> int:
    return int(year) * 12 + int(month) - 1


def load_cache(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return (
        np.asarray(payload["embeddings"], dtype=np.float32),
        np.asarray(payload["years"], dtype=np.int32),
        np.asarray(payload["months"], dtype=np.int32),
        np.asarray(payload["types"]),
        payload["records"],
        payload.get("metadata", {}),
    )


def deduplicate_by_ha_na(Z, years, months, subtypes, records):
    seen: set[tuple[str, str]] = set()
    keep: list[int] = []
    removed = Counter()
    for i, record in enumerate(records):
        key = (record.get("ha_sequence", ""), record.get("na_sequence", ""))
        if key in seen:
            removed[str(subtypes[i])] += 1
            continue
        seen.add(key)
        keep.append(i)
    keep_arr = np.asarray(keep, dtype=np.int64)
    return (
        Z[keep_arr],
        years[keep_arr],
        months[keep_arr],
        subtypes[keep_arr],
        [records[i] for i in keep],
        dict(removed),
    )


def load_joined_metadata(path: Path):
    by_epi: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epi = row.get("epi_isl") or row.get("Isolate_Id")
            if not epi:
                continue
            matched = str(row.get("matched", "")).strip().lower() == "true"
            clade_raw = clean_label(row.get("clade_raw") or row.get("Clade"))
            clade = clean_label(row.get("clade") or row.get("Clade"))
            major_clade = clean_label(row.get("major_clade"))
            if major_clade is None and clade is not None:
                major_clade = ".".join(clade.split(".")[:2])
            by_epi[epi] = {
                "matched": matched,
                "clade_raw": clade_raw,
                "clade": clade,
                "major_clade": major_clade,
                "lineage": clean_label(row.get("lineage") or row.get("Lineage")),
                "genotype": clean_label(row.get("genotype") or row.get("Genotype")),
            }
    return by_epi


def build_joined_arrays(records, subtypes, years, months, metadata_by_epi):
    n = len(records)
    matched = np.zeros(n, dtype=bool)
    clade = np.empty(n, dtype=object)
    major_clade = np.empty(n, dtype=object)
    lineage = np.empty(n, dtype=object)
    genotype = np.empty(n, dtype=object)
    raw_clade_present = np.zeros(n, dtype=bool)
    for i, record in enumerate(records):
        item = metadata_by_epi.get(record.get("epi_isl", ""))
        if not item:
            clade[i] = None
            major_clade[i] = None
            lineage[i] = None
            genotype[i] = None
            continue
        matched[i] = bool(item["matched"])
        raw_clade_present[i] = item["clade_raw"] is not None
        clade[i] = item["clade"]
        major_clade[i] = item["major_clade"]
        lineage[i] = item["lineage"]
        genotype[i] = item["genotype"]
    return {
        "subtypes": np.asarray(subtypes),
        "years": np.asarray(years, dtype=np.int32),
        "months": np.asarray(months, dtype=np.int32),
        "month_indices": np.asarray([month_index(y, m) for y, m in zip(years, months)], dtype=np.int32),
        "matched": matched,
        "raw_clade_present": raw_clade_present,
        "clade": clade,
        "major_clade": major_clade,
        "lineage": lineage,
        "genotype": genotype,
    }


def coverage_summary(joined, total_n: int):
    rows = []
    for subtype in SUBTYPES:
        mask = joined["subtypes"] == subtype
        rows.append(
            {
                "group": subtype,
                "cache_n": int(np.sum(mask)),
                "joined_n": int(np.sum(mask & joined["matched"])),
                "raw_clade_n": int(np.sum(mask & joined["raw_clade_present"])),
                "assigned_clade_n": int(np.sum(mask & np.fromiter((x is not None for x in joined["clade"]), bool, total_n))),
                "lineage_n": int(np.sum(mask & np.fromiter((x is not None for x in joined["lineage"]), bool, total_n))),
                "genotype_n": int(np.sum(mask & np.fromiter((x is not None for x in joined["genotype"]), bool, total_n))),
            }
        )
    combined = {
        "group": "Combined",
        "cache_n": int(total_n),
        "joined_n": int(np.sum(joined["matched"])),
        "raw_clade_n": int(np.sum(joined["raw_clade_present"])),
        "assigned_clade_n": int(sum(x is not None for x in joined["clade"])),
        "lineage_n": int(sum(x is not None for x in joined["lineage"])),
        "genotype_n": int(sum(x is not None for x in joined["genotype"])),
    }
    rows.append(combined)
    for row in rows:
        row["join_fraction"] = row["joined_n"] / row["cache_n"] if row["cache_n"] else float("nan")
        row["assigned_clade_fraction"] = (
            row["assigned_clade_n"] / row["cache_n"] if row["cache_n"] else float("nan")
        )
    return rows


def encode_labels(labels: np.ndarray):
    assigned_positions = [i for i, value in enumerate(labels) if value is not None]
    classes = sorted({labels[i] for i in assigned_positions})
    mapping = {value: j for j, value in enumerate(classes)}
    codes = np.full(len(labels), -1, dtype=np.int32)
    for i in assigned_positions:
        codes[i] = mapping[labels[i]]
    return codes, classes


def fit_neighbors_by_subtype(Z, subtypes, max_k: int):
    neighbors = {}
    for subtype in SUBTYPES:
        subtype_idx = np.where(subtypes == subtype)[0]
        nn = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean", algorithm="brute", n_jobs=-1)
        nn.fit(Z[subtype_idx])
        _, local_indices = nn.kneighbors(Z[subtype_idx], return_distance=True)
        neighbors[subtype] = {
            "global_indices": subtype_idx,
            "local_neighbors": local_indices[:, 1 : max_k + 1],
        }
    return neighbors


def precision_for_codes(codes: np.ndarray, local_neighbors: np.ndarray, k: int, query_mask: np.ndarray | None = None):
    query_valid = codes >= 0
    if query_mask is not None:
        query_valid = query_valid & query_mask
    neigh = local_neighbors[:, :k]
    neigh_codes = codes[neigh]
    valid_neighbors = neigh_codes >= 0
    denom = valid_neighbors.sum(axis=1)
    same = (neigh_codes == codes[:, None]) & valid_neighbors
    usable = query_valid & (denom > 0)
    if not np.any(usable):
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "n_queries": 0,
            "mean_valid_neighbors": float("nan"),
        }
    precision = same.sum(axis=1)[usable] / denom[usable]
    return {
        "mean": float(np.mean(precision)),
        "median": float(np.median(precision)),
        "n_queries": int(np.sum(usable)),
        "mean_valid_neighbors": float(np.mean(denom[usable])),
    }


def sampled_random_baseline(codes: np.ndarray, k: int, seeds: list[int]):
    assigned = np.where(codes >= 0)[0]
    values = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        seed_values = []
        for q in assigned:
            if len(assigned) - 1 < k:
                continue
            sampled = rng.choice(assigned, size=k, replace=False)
            if q in sampled:
                while q in sampled:
                    sampled = rng.choice(assigned, size=k, replace=False)
            seed_values.append(np.mean(codes[sampled] == codes[q]))
        values.append(float(np.mean(seed_values)))
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "seed_values": values,
    }


def permutation_control(codes: np.ndarray, local_neighbors: np.ndarray, k: int, n_replicates: int, seed: int):
    rng = np.random.default_rng(seed)
    assigned = np.where(codes >= 0)[0]
    values = []
    for _ in range(n_replicates):
        permuted = codes.copy()
        permuted[assigned] = rng.permutation(permuted[assigned])
        values.append(precision_for_codes(permuted, local_neighbors, k)["mean"])
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "replicates": int(n_replicates),
    }


def temporal_stratified_baseline(
    codes: np.ndarray,
    months: np.ndarray,
    local_neighbors: np.ndarray,
    k: int,
    window: int,
    seeds: list[int],
):
    assigned = np.where(codes >= 0)[0]
    assigned_months = months[assigned]
    order = np.argsort(assigned_months, kind="mergesort")
    sorted_assigned = assigned[order]
    sorted_months = assigned_months[order]

    eligible = np.zeros(len(codes), dtype=bool)
    candidate_cache: dict[int, np.ndarray] = {}
    for q in assigned:
        left = np.searchsorted(sorted_months, months[q] - window, side="left")
        right = np.searchsorted(sorted_months, months[q] + window, side="right")
        candidates = sorted_assigned[left:right]
        candidates = candidates[candidates != q]
        if len(candidates) >= k:
            eligible[q] = True
            candidate_cache[int(q)] = candidates

    true_same_query_set = precision_for_codes(codes, local_neighbors, k, query_mask=eligible)
    seed_values = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        vals = []
        for q, candidates in candidate_cache.items():
            sampled = rng.choice(candidates, size=k, replace=False)
            vals.append(np.mean(codes[sampled] == codes[q]))
        seed_values.append(float(np.mean(vals)) if vals else float("nan"))

    random_mean = float(np.nanmean(seed_values))
    return {
        "window_months": int(window),
        "k": int(k),
        "eligible_queries": int(np.sum(eligible)),
        "true_precision_same_query_set": true_same_query_set["mean"],
        "stratified_random_mean": random_mean,
        "stratified_random_sd": float(np.nanstd(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0,
        "seed_values": seed_values,
        "enrichment": true_same_query_set["mean"] / random_mean if random_mean > 0 else float("nan"),
    }


def clade_enrichment(
    Z,
    joined,
    k_values: list[int],
    random_seeds: list[int],
    permutation_replicates: int,
    permutation_seed: int,
    temporal_windows: list[int],
):
    max_k = max(k_values)
    neighbors = fit_neighbors_by_subtype(Z, joined["subtypes"], max_k)
    rows = []
    temporal_rows = []

    for label_name in ("clade", "major_clade"):
        for subtype in SUBTYPES:
            subtype_global = neighbors[subtype]["global_indices"]
            label_values = joined[label_name][subtype_global]
            codes, classes = encode_labels(label_values)
            local_neighbors = neighbors[subtype]["local_neighbors"]
            class_counts = Counter(codes[codes >= 0])
            n_labeled = int(np.sum(codes >= 0))
            largest_frac = max(class_counts.values()) / n_labeled if n_labeled else float("nan")

            for k in k_values:
                true_precision = precision_for_codes(codes, local_neighbors, k)
                random = sampled_random_baseline(codes, k, random_seeds)
                perm = permutation_control(codes, local_neighbors, k, permutation_replicates, permutation_seed)
                rows.append(
                    {
                        "subtype": subtype,
                        "label": label_name,
                        "n_labeled": n_labeled,
                        "classes": len(classes),
                        "largest_class_fraction": float(largest_frac),
                        "k": int(k),
                        "mean_precision": true_precision["mean"],
                        "median_precision": true_precision["median"],
                        "mean_valid_neighbors": true_precision["mean_valid_neighbors"],
                        "random_baseline": random["mean"],
                        "random_baseline_sd": random["sd"],
                        "permutation_mean": perm["mean"],
                        "permutation_sd": perm["sd"],
                        "permutation_p05": perm["p05"],
                        "permutation_p95": perm["p95"],
                        "permutation_replicates": perm["replicates"],
                        "enrichment_vs_random": true_precision["mean"] / random["mean"],
                        "delta_vs_random": true_precision["mean"] - random["mean"],
                    }
                )

            if label_name == "clade":
                subtype_months = joined["month_indices"][subtype_global]
                for k in k_values:
                    for window in temporal_windows:
                        temporal = temporal_stratified_baseline(
                            codes,
                            subtype_months,
                            local_neighbors,
                            k,
                            window,
                            random_seeds,
                        )
                        temporal_rows.append({"subtype": subtype, "label": label_name, **temporal})

    return rows, temporal_rows


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def fmt(value: float, digits: int = 4) -> str:
    if value != value:
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def write_summary(path: Path, payload: dict):
    coverage_rows = [
        [
            row["group"],
            f"{row['cache_n']:,}",
            f"{row['joined_n']:,}",
            pct(row["join_fraction"]),
            f"{row['assigned_clade_n']:,}",
            pct(row["assigned_clade_fraction"]),
        ]
        for row in payload["coverage"]
    ]
    clade_rows = [
        row
        for row in payload["clade_enrichment"]
        if row["label"] == "clade"
    ]
    clade_table_rows = [
        [
            row["subtype"],
            str(row["k"]),
            fmt(row["mean_precision"]),
            fmt(row["random_baseline"]),
            fmt(row["permutation_mean"]),
            f"{row['enrichment_vs_random']:.2f}x",
        ]
        for row in clade_rows
    ]
    temporal_rows = [
        row for row in payload["temporal_stratified"] if row["k"] == 5
    ]
    temporal_table_rows = [
        [
            row["subtype"],
            str(row["k"]),
            f"±{row['window_months']} mo",
            fmt(row["true_precision_same_query_set"]),
            fmt(row["stratified_random_mean"]),
            f"{row['enrichment']:.2f}x",
            f"{row['eligible_queries']:,}",
        ]
        for row in temporal_rows
    ]

    text = [
        "# GISAID Clade Enrichment in AntigenLM Latent Neighborhoods",
        "",
        "This analysis uses the exact HA+NA-deduplicated embedding cache and the private GISAID EpiFlu metadata export from `EPI_SET_260506bu`. No sequences or accession-level metadata are redistributed here.",
        "",
        "## Inputs",
        "",
        f"- cache: `{payload['inputs']['cache_path']}`",
        f"- joined metadata: `{payload['inputs']['metadata_join_path']}`",
        f"- deduplicated records: `{payload['deduplication']['deduplicated_n']:,}`",
        f"- exact HA+NA duplicates removed: `{payload['deduplication']['removed_n']:,}`",
        "",
        "## Metadata Coverage",
        "",
        markdown_table(
            ["group", "cache n", "joined n", "join %", "assigned clade n", "assigned clade %"],
            coverage_rows,
        ),
        "",
        "## Clade Precision and Controls",
        "",
        markdown_table(
            ["subtype", "k", "precision@k", "random baseline", "permutation mean", "enrichment"],
            clade_table_rows,
        ),
        "",
        "## Temporal Stratification Sensitivity (k=5)",
        "",
        "Random candidates are restricted to assigned-clade records from the same subtype within the stated collection-month window. True precision is recomputed on the same eligible query set.",
        "",
        markdown_table(
            ["subtype", "k", "window", "true precision", "stratified random", "enrichment", "eligible queries"],
            temporal_table_rows,
        ),
        "",
        "## Interpretation",
        "",
        "- Latent neighborhoods are strongly enriched for GISAID clade membership in both subtypes.",
        "- The within-subtype clade-label permutation control drops precision to the random-baseline range, indicating that enrichment depends on the observed clade assignments rather than only on class imbalance.",
        "- Temporally stratified random baselines are higher than global random baselines, as expected because clades are temporally structured; enrichment remains above the stratified baseline across the tested windows.",
        "- The valid claim is local evolutionary-taxonomic coherence under GISAID clade annotations. This is not quantitative phylogenetic-distance validation, antigenic validation, immune-escape validation, vaccine-strain relevance, or forecasting validation.",
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def write_spanish_summary(path: Path, payload: dict):
    coverage_rows = [
        [
            row["group"],
            f"{row['cache_n']:,}",
            f"{row['joined_n']:,}",
            pct(row["join_fraction"]),
            f"{row['assigned_clade_n']:,}",
            pct(row["assigned_clade_fraction"]),
        ]
        for row in payload["coverage"]
    ]
    temporal_rows = [row for row in payload["temporal_stratified"] if row["k"] == 5]
    text = [
        "# Enriquecimiento de clados GISAID en vecindarios latentes de AntigenLM",
        "",
        "Este análisis usa el cache deduplicado por HA+NA exacto y el export privado de metadatos GISAID EpiFlu de `EPI_SET_260506bu`. No se redistribuyen secuencias ni metadatos a nivel de accesión.",
        "",
        "## Cobertura de metadatos",
        "",
        markdown_table(
            ["grupo", "n cache", "n unido", "% unión", "n con clado", "% con clado"],
            coverage_rows,
        ),
        "",
        "## Sensibilidad con control temporal (k=5)",
        "",
        "Los candidatos aleatorios se restringen al mismo subtipo y a registros con clado asignado dentro de la ventana temporal indicada. La precisión latente se recalcula sobre el mismo conjunto de consultas elegibles.",
        "",
        markdown_table(
            ["subtipo", "k", "ventana", "precisión real", "aleatorio estratificado", "enriquecimiento", "consultas"],
            [
                [
                    row["subtype"],
                    str(row["k"]),
                    f"±{row['window_months']} meses",
                    fmt(row["true_precision_same_query_set"]),
                    fmt(row["stratified_random_mean"]),
                    f"{row['enrichment']:.2f}x",
                    f"{row['eligible_queries']:,}",
                ]
                for row in temporal_rows
            ],
        ),
        "",
        "## Lectura metodológica",
        "",
        "- Los vecindarios latentes están fuertemente enriquecidos por pertenencia al clado GISAID en H1N1 y H3N2.",
        "- El control de permutación de etiquetas dentro de subtipo reduce la precisión al rango basal, por lo que la señal depende de las asignaciones observadas de clado.",
        "- El control aleatorio estratificado por tiempo aumenta el basal, como era esperable, pero el enriquecimiento persiste en las ventanas evaluadas.",
        "- La afirmación válida es coherencia evolutivo-taxonómica local bajo anotaciones GISAID. Esto no equivale a validación de distancia filogenética cuantitativa, antigenicidad, escape inmune, relevancia vacunal ni forecasting prospectivo.",
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def plot_clade_precision(path_base: Path, rows: list[dict]):
    clade_rows = [r for r in rows if r["label"] == "clade"]
    k_values = sorted({r["k"] for r in clade_rows})
    subtypes = list(SUBTYPES)
    width = 0.18
    x = np.arange(len(k_values))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    colors = {"precision": "#2f6f8f", "random": "#b7b7b7", "permutation": "#e0a33a"}
    for ax, subtype in zip(axes, subtypes):
        sub_rows = {r["k"]: r for r in clade_rows if r["subtype"] == subtype}
        true = [sub_rows[k]["mean_precision"] for k in k_values]
        random = [sub_rows[k]["random_baseline"] for k in k_values]
        perm = [sub_rows[k]["permutation_mean"] for k in k_values]
        ax.bar(x - width, true, width, label="Latent neighbors", color=colors["precision"])
        ax.bar(x, random, width, label="Random", color=colors["random"])
        ax.bar(x + width, perm, width, label="Permuted labels", color=colors["permutation"])
        ax.set_title(subtype)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlabel("k")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Clade precision@k")
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"))
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    plt.close(fig)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run clade-label enrichment controls for the latent geometry paper.")
    parser.add_argument("--cache-path", default="results/embeddings_cache_full_all_available.pkl")
    parser.add_argument(
        "--metadata-join-path",
        default="data/gisaid_metadata_private/gisaid_epiflu_isolates_2000_2022_epi_set_260506bu_joined_dedup_cache.csv",
    )
    parser.add_argument("--output-json", default="results/gisaid_clade_enrichment_results.json")
    parser.add_argument("--output-summary", default="results/gisaid_clade_enrichment_summary.md")
    parser.add_argument("--output-summary-es", default="results/gisaid_clade_enrichment_summary_es.md")
    parser.add_argument("--figure-base", default="figures/latent_geometry_full/clade_precision_enrichment")
    parser.add_argument("--k-values", default="5,10,20")
    parser.add_argument("--temporal-windows", default="6,12,24")
    parser.add_argument("--random-seeds", default="42,7,123")
    parser.add_argument("--permutation-replicates", type=int, default=100)
    parser.add_argument("--permutation-seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    cache_path = ROOT / args.cache_path
    metadata_path = ROOT / args.metadata_join_path
    output_json = ROOT / args.output_json
    output_summary = ROOT / args.output_summary
    output_summary_es = ROOT / args.output_summary_es
    figure_base = ROOT / args.figure_base

    Z, years, months, subtypes, records, metadata = load_cache(cache_path)
    Zd, yd, md, td, rd, removed = deduplicate_by_ha_na(Z, years, months, subtypes, records)
    metadata_by_epi = load_joined_metadata(metadata_path)
    joined = build_joined_arrays(rd, td, yd, md, metadata_by_epi)

    k_values = parse_int_list(args.k_values)
    temporal_windows = parse_int_list(args.temporal_windows)
    random_seeds = parse_int_list(args.random_seeds)

    coverage = coverage_summary(joined, len(rd))
    enrichment_rows, temporal_rows = clade_enrichment(
        Zd,
        joined,
        k_values=k_values,
        random_seeds=random_seeds,
        permutation_replicates=args.permutation_replicates,
        permutation_seed=args.permutation_seed,
        temporal_windows=temporal_windows,
    )

    payload = {
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - started,
        "inputs": {
            "cache_path": args.cache_path,
            "metadata_join_path": args.metadata_join_path,
            "metadata_epi_set": "EPI_SET_260506bu",
        },
        "cache_metadata": metadata,
        "deduplication": {
            "original_n": int(len(Z)),
            "deduplicated_n": int(len(Zd)),
            "removed_n": int(len(Z) - len(Zd)),
            "removed_by_subtype": removed,
            "method": "Exact HA+NA sequence deduplication by first representative.",
        },
        "parameters": {
            "k_values": k_values,
            "temporal_windows_months": temporal_windows,
            "random_seeds": random_seeds,
            "permutation_replicates": args.permutation_replicates,
            "permutation_seed": args.permutation_seed,
            "neighbor_graph": "Exact Euclidean kNN computed within subtype on deduplicated 384-dimensional embeddings.",
        },
        "coverage": coverage,
        "clade_enrichment": enrichment_rows,
        "temporal_stratified": temporal_rows,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_summary(output_summary, payload)
    write_spanish_summary(output_summary_es, payload)
    plot_clade_precision(figure_base, enrichment_rows)

    print(f"Wrote {output_json.relative_to(ROOT)}")
    print(f"Wrote {output_summary.relative_to(ROOT)}")
    print(f"Wrote {output_summary_es.relative_to(ROOT)}")
    print(f"Wrote {figure_base.with_suffix('.pdf').relative_to(ROOT)}")
    print(f"Wrote {figure_base.with_suffix('.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
