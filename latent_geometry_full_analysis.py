# -*- coding: utf-8 -*-
"""
Full-data latent geometry audit for AntigenLM embeddings.

This script is read-only with respect to sequences and checkpoints: it consumes
an embeddings cache, computes aggregate metrics, and writes summaries/figures.
It never prints complete sequences and never saves raw sequences to outputs.
"""

import argparse
import json
import os
import pickle
import hashlib
import shutil
import time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


RESULTS_DIR = "results"
FIGURES_DIR = "figures/latent_geometry_full"
PROCESSED_DIR = "data/processed_gisaid"
PCA_THRESHOLDS = (0.80, 0.90, 0.95, 0.99)
SUBTYPE_ORDER = ("H1N1", "H3N2")
SUBTYPE_COLORS = {"H1N1": "#457B9D", "H3N2": "#E63946"}
METRIC_LABELS = {
    "temporal": "Temporal",
    "hamming_ha": "Hamming HA",
    "hamming_ha_na": "Hamming HA+NA",
}


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def load_cache(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    required = ("embeddings", "years", "months", "types", "records")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Cache invalido: faltan claves {missing}")
    Z = np.asarray(payload["embeddings"], dtype=np.float64)
    years = np.asarray(payload["years"], dtype=int)
    months = np.asarray(payload["months"], dtype=int)
    types = np.asarray(payload["types"]).astype(str)
    records = payload["records"]
    n = {len(Z), len(years), len(months), len(types), len(records)}
    if len(n) != 1:
        raise ValueError("Cache desalineado entre embeddings/metadatos/records")
    return Z, years, months, types, records, payload.get("metadata", {})


def load_source_records():
    datasets = {}
    for subtype in SUBTYPE_ORDER:
        path = os.path.join(PROCESSED_DIR, f"dataset_{subtype}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        datasets[subtype] = data.get("paired_strains", [])
    return datasets


def sha1_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ha_na_key(record):
    return sha1_text(f"{record.get('ha_sequence', '')}|{record.get('na_sequence', '')}")


def sequence_array(sequence):
    return np.frombuffer((sequence or "").upper().encode("ascii", errors="ignore"), dtype=np.uint8)


def build_record_features(records):
    features = []
    for record in records:
        features.append(
            {
                "ha": sequence_array(record.get("ha_sequence", "")),
                "na": sequence_array(record.get("na_sequence", "")),
                "ha_len": len(record.get("ha_sequence", "") or ""),
                "na_len": len(record.get("na_sequence", "") or ""),
            }
        )
    return features


def normalized_hamming_arrays(a, b, tolerance=0.05):
    len_a = len(a)
    len_b = len(b)
    if len_a == 0 or len_b == 0:
        return None
    max_len = max(len_a, len_b)
    min_len = min(len_a, len_b)
    if (max_len - min_len) / max_len > tolerance:
        return None
    return float(np.count_nonzero(a[:min_len] != b[:min_len]) / min_len)


def hamming_distance(features, i, j, metric):
    if metric == "hamming_ha":
        return normalized_hamming_arrays(features[i]["ha"], features[j]["ha"])
    if metric == "hamming_ha_na":
        d_ha = normalized_hamming_arrays(features[i]["ha"], features[j]["ha"])
        d_na = normalized_hamming_arrays(features[i]["na"], features[j]["na"])
        if d_ha is None or d_na is None:
            return None
        ha_len = min(features[i]["ha_len"], features[j]["ha_len"])
        na_len = min(features[i]["na_len"], features[j]["na_len"])
        denom = ha_len + na_len
        if denom == 0:
            return None
        return float((d_ha * ha_len + d_na * na_len) / denom)
    raise ValueError(metric)


def month_index(years, months):
    return years * 12 + np.clip(months, 1, 12)


def audit_source_and_cache(Z, years, months, types, records, metadata, cache_path):
    source = load_source_records()
    lines = [
        "# Full Data Audit",
        "",
        "This audit reports aggregate counts only. Complete sequences are not printed or redistributed.",
        "",
        f"Cache: `{cache_path}`",
        f"Embeddings in cache: `{Z.shape}`",
        "",
        "## Checkpoint/cache metadata",
        "",
    ]
    ckpt = metadata.get("checkpoint", {})
    if ckpt:
        lines.extend([
            f"- checkpoint path: `{ckpt.get('path')}`",
            f"- checkpoint size bytes: `{ckpt.get('size_bytes')}`",
            f"- checkpoint sha256: `{ckpt.get('sha256', 'not recorded')}`",
        ])
    lines.extend([
        f"- sampling strategy: `{metadata.get('sampling_strategy')}`",
        f"- max_per_subtype: `{metadata.get('max_per_subtype')}`",
        f"- seed: `{metadata.get('seed')}`",
        f"- max_seq_length: `{metadata.get('max_seq_length')}`",
        f"- embedding_batch_size: `{metadata.get('embedding_batch_size', 'not recorded')}`",
        "",
        "## Records by subtype",
        "",
        "| subtype | source paired records | valid HA+NA+date records | cached embeddings | missing from cache |",
        "|---|---:|---:|---:|---:|",
    ])

    audit = {
        "cache_path": cache_path,
        "embedding_shape": list(Z.shape),
        "metadata": metadata,
        "by_subtype": {},
    }
    source_year_counts = defaultdict(Counter)
    source_month_counts = defaultdict(Counter)
    length_rows = []

    for subtype in SUBTYPE_ORDER:
        recs = source.get(subtype, [])
        valid = [
            r for r in recs
            if r.get("year") is not None
            and r.get("month") is not None
            and r.get("ha_sequence")
            and r.get("na_sequence")
        ]
        cached = int(np.sum(types == subtype))
        missing = max(0, len(valid) - cached)
        lines.append(f"| {subtype} | {len(recs):,} | {len(valid):,} | {cached:,} | {missing:,} |")

        ha_lengths = [len(r.get("ha_sequence", "") or "") for r in valid]
        na_lengths = [len(r.get("na_sequence", "") or "") for r in valid]
        for r in valid:
            source_year_counts[subtype][int(r.get("year"))] += 1
            source_month_counts[subtype][f"{int(r.get('year'))}-{int(r.get('month')):02d}"] += 1
        length_rows.append((subtype, ha_lengths, na_lengths))

        audit["by_subtype"][subtype] = {
            "source_paired_records": len(recs),
            "valid_ha_na_date_records": len(valid),
            "cached_embeddings": cached,
            "missing_from_cache": missing,
            "ha_length": describe_values(ha_lengths),
            "na_length": describe_values(na_lengths),
            "duplicates_ha": count_duplicates_hash(r.get("ha_sequence", "") for r in valid),
            "duplicates_na": count_duplicates_hash(r.get("na_sequence", "") for r in valid),
            "duplicates_ha_na": count_duplicates_hash(f"{r.get('ha_sequence', '')}|{r.get('na_sequence', '')}" for r in valid),
        }

    lines.extend([
        "",
        "## Sequence length summary",
        "",
        "| subtype | segment | n | mean | median | p05 | p95 | min | max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for subtype, ha_lengths, na_lengths in length_rows:
        for segment, values in (("HA", ha_lengths), ("NA", na_lengths)):
            s = describe_values(values)
            lines.append(
                f"| {subtype} | {segment} | {s['n']:,} | {s['mean']:.1f} | {s['median']:.1f} | "
                f"{s['p05']:.1f} | {s['p95']:.1f} | {s['min']:.0f} | {s['max']:.0f} |"
            )

    embedding_duplicates = count_embedding_duplicates(Z)
    audit["embedding_exact_duplicates"] = embedding_duplicates
    lines.extend([
        "",
        "## Duplicate summary",
        "",
        "| subtype | duplicate HA | duplicate NA | duplicate HA+NA |",
        "|---|---:|---:|---:|",
    ])
    for subtype in SUBTYPE_ORDER:
        row = audit["by_subtype"][subtype]
        lines.append(
            f"| {subtype} | {row['duplicates_ha']:,} | {row['duplicates_na']:,} | {row['duplicates_ha_na']:,} |"
        )
    lines.extend([
        "",
        f"Exact duplicate embeddings in cache: `{embedding_duplicates:,}`.",
        "",
        "## Figures",
        "",
        "- `figures/latent_geometry_full/records_by_year_subtype.pdf`",
        "- `figures/latent_geometry_full/sequence_length_distributions.pdf`",
    ])

    plot_records_by_year(source_year_counts)
    plot_length_distributions(length_rows)

    path = os.path.join(RESULTS_DIR, "full_data_audit_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return audit


def describe_values(values):
    arr = np.asarray(list(values), dtype=np.float64)
    if len(arr) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p05": np.nan, "p95": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def count_duplicates_hash(values):
    counts = Counter(sha1_text(v or "") for v in values)
    return int(sum(count - 1 for count in counts.values() if count > 1))


def count_embedding_duplicates(Z):
    counts = Counter()
    for row in np.asarray(Z):
        counts[hashlib.sha1(np.ascontiguousarray(row).tobytes()).hexdigest()] += 1
    return int(sum(count - 1 for count in counts.values() if count > 1))


def plot_records_by_year(source_year_counts):
    years = sorted({year for counts in source_year_counts.values() for year in counts})
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.42
    x = np.arange(len(years))
    for offset, subtype in zip((-width / 2, width / 2), SUBTYPE_ORDER):
        vals = [source_year_counts[subtype].get(year, 0) for year in years]
        ax.bar(x + offset, vals, width=width, label=subtype, color=SUBTYPE_COLORS[subtype], alpha=0.75)
    ax.set_xticks(x[::max(1, len(x) // 12)])
    ax.set_xticklabels([str(y) for y in years[::max(1, len(x) // 12)]], rotation=45, ha="right")
    ax.set_ylabel("Records")
    ax.set_title("Processed records by year and subtype")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "records_by_year_subtype")


def plot_length_distributions(length_rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for subtype, ha_lengths, na_lengths in length_rows:
        axes[0].hist(ha_lengths, bins=50, alpha=0.45, label=subtype, color=SUBTYPE_COLORS[subtype])
        axes[1].hist(na_lengths, bins=50, alpha=0.45, label=subtype, color=SUBTYPE_COLORS[subtype])
    for ax, title in zip(axes, ("HA length", "NA length")):
        ax.set_title(title)
        ax.set_xlabel("Nucleotides")
        ax.set_ylabel("Records")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "sequence_length_distributions")


def sample_pairs(n, n_pairs, rng):
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n - 1, size=n_pairs)
    j = j + (j >= i)
    return i, j


def spearman_full(Z, years, months, types, records, pair_samples, seeds):
    print("\n[Full Spearman] Pair-sampled latent vs temporal/molecular distances")
    features = build_record_features(records)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for subtype in SUBTYPE_ORDER:
            idx = np.where(types == subtype)[0]
            if len(idx) < 2:
                continue
            local_i, local_j = sample_pairs(len(idx), pair_samples, rng)
            global_i = idx[local_i]
            global_j = idx[local_j]
            latent = np.linalg.norm(Z[global_i] - Z[global_j], axis=1)
            t_months = month_index(years, months)
            temporal = np.abs(t_months[global_i] - t_months[global_j])
            rows.append(spearman_row(seed, subtype, "temporal", latent, temporal, pair_samples, 0))

            for metric in ("hamming_ha", "hamming_ha_na"):
                latent_valid = []
                biological = []
                omitted = 0
                for gi, gj, dl in zip(global_i, global_j, latent):
                    value = hamming_distance(features, int(gi), int(gj), metric)
                    if value is None:
                        omitted += 1
                        continue
                    latent_valid.append(float(dl))
                    biological.append(value)
                rows.append(
                    spearman_row(
                        seed, subtype, metric,
                        np.asarray(latent_valid), np.asarray(biological),
                        pair_samples, omitted,
                    )
                )
                print(
                    f"  seed={seed} {subtype} {metric}: "
                    f"rho={rows[-1]['rho']:.4f} valid={rows[-1]['valid_pairs']:,} omitted={omitted:,}"
                )
    plot_spearman_summary(rows)
    return rows


def spearman_row(seed, subtype, metric, latent, metric_values, requested, omitted):
    finite = np.isfinite(latent) & np.isfinite(metric_values)
    latent = latent[finite]
    metric_values = metric_values[finite]
    if len(latent) < 10:
        rho, pvalue = np.nan, np.nan
    else:
        rho, pvalue = spearmanr(latent, metric_values)
    return {
        "seed": int(seed),
        "subtype": subtype,
        "metric": metric,
        "rho": float(rho),
        "pvalue": float(pvalue),
        "requested_pairs": int(requested),
        "valid_pairs": int(len(latent)),
        "omitted_pairs": int(omitted + np.sum(~finite)),
        "latent_mean": float(np.mean(latent)) if len(latent) else np.nan,
        "latent_std": float(np.std(latent)) if len(latent) else np.nan,
        "metric_mean": float(np.mean(metric_values)) if len(metric_values) else np.nan,
        "metric_std": float(np.std(metric_values)) if len(metric_values) else np.nan,
    }


def plot_spearman_summary(rows):
    labels = []
    means = []
    stds = []
    colors = []
    for metric in ("temporal", "hamming_ha", "hamming_ha_na"):
        for subtype in SUBTYPE_ORDER:
            vals = [r["rho"] for r in rows if r["metric"] == metric and r["subtype"] == subtype and np.isfinite(r["rho"])]
            labels.append(f"{METRIC_LABELS[metric]}\n{subtype}")
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)
            colors.append(SUBTYPE_COLORS[subtype])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(np.arange(len(labels)), means, yerr=stds, color=colors, alpha=0.75, capsize=4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Latent distance correlation with temporal and molecular distances")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "spearman_latent_vs_distances")


def pca_spectrum(X):
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    X = X[finite]
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0, None)
    eigvecs = eigvecs[:, order]
    positive = eigvals[eigvals > 1e-12]
    ratios = positive / np.sum(positive)
    cumulative = np.cumsum(ratios)
    thresholds = {str(t): int(np.searchsorted(cumulative, t) + 1) for t in PCA_THRESHOLDS}
    pr = float((np.sum(positive) ** 2) / np.sum(positive ** 2))
    return {
        "n_samples": int(len(X)),
        "embedding_dim": int(X.shape[1]),
        "mean": mean.ravel(),
        "components": eigvecs[:, :max(2, min(100, eigvecs.shape[1]))],
        "explained_variance": positive,
        "explained_variance_ratio": ratios,
        "cumulative_variance_ratio": cumulative,
        "n_components_by_threshold": thresholds,
        "participation_ratio": pr,
        "top10_explained_variance_ratio": [float(x) for x in ratios[:10]],
    }


def pca_full(Z, years, types, plot_max_points, seed):
    print("\n[Full PCA] Exact covariance spectrum")
    results = {"global": pca_spectrum(Z)}
    for subtype in SUBTYPE_ORDER:
        results[subtype] = pca_spectrum(Z[types == subtype])
    plot_pca_scree(results)
    plot_pca_projection(Z, years, types, results["global"], plot_max_points, seed)
    return {k: compact_pca(v) for k, v in results.items()}


def compact_pca(result):
    return {
        "n_samples": result["n_samples"],
        "embedding_dim": result["embedding_dim"],
        "n_components_by_threshold": result["n_components_by_threshold"],
        "participation_ratio": result["participation_ratio"],
        "top10_explained_variance_ratio": result["top10_explained_variance_ratio"],
    }


def plot_pca_scree(results):
    global_result = results["global"]
    for name, cumulative in (
        ("pca_scree_global", False),
        ("pca_cumulative_global", True),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        values = global_result["cumulative_variance_ratio"] if cumulative else global_result["explained_variance_ratio"]
        n = min(80, len(values))
        ax.plot(np.arange(1, n + 1), values[:n], marker="o", markersize=3, linewidth=1.4)
        if cumulative:
            for threshold in PCA_THRESHOLDS:
                ax.axhline(threshold, linestyle="--", color="gray", linewidth=0.8)
            ax.set_ylim(0, 1.02)
            ax.set_ylabel("Cumulative explained variance")
            ax.set_title("Global PCA cumulative variance")
        else:
            ax.set_ylabel("Explained variance ratio")
            ax.set_title("Global PCA scree plot")
        ax.set_xlabel("PCA component")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save_fig(fig, name)

    for name, cumulative in (
        ("pca_scree_by_subtype", False),
        ("pca_cumulative_by_subtype", True),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for subtype in SUBTYPE_ORDER:
            result = results[subtype]
            values = result["cumulative_variance_ratio"] if cumulative else result["explained_variance_ratio"]
            n = min(80, len(values))
            ax.plot(np.arange(1, n + 1), values[:n], marker="o", markersize=3, linewidth=1.4, label=subtype)
        if cumulative:
            for threshold in PCA_THRESHOLDS:
                ax.axhline(threshold, linestyle="--", color="gray", linewidth=0.8)
            ax.set_ylim(0, 1.02)
            ax.set_ylabel("Cumulative explained variance")
            ax.set_title("PCA cumulative variance by subtype")
        else:
            ax.set_ylabel("Explained variance ratio")
            ax.set_title("PCA scree plot by subtype")
        ax.set_xlabel("PCA component")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        save_fig(fig, name)


def plot_pca_projection(Z, years, types, global_pca, plot_max_points, seed):
    rng = np.random.default_rng(seed)
    if plot_max_points and len(Z) > plot_max_points:
        idx = rng.choice(len(Z), size=plot_max_points, replace=False)
    else:
        idx = np.arange(len(Z))
    Z2 = (Z[idx] - global_pca["mean"]) @ global_pca["components"][:, :2]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for subtype in SUBTYPE_ORDER:
        mask = types[idx] == subtype
        ax.scatter(Z2[mask, 0], Z2[mask, 1], s=4, alpha=0.35, linewidths=0, label=subtype, color=SUBTYPE_COLORS[subtype])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Global PCA projection by subtype")
    ax.legend(markerscale=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "pca_2d_by_subtype")

    fig, ax = plt.subplots(figsize=(7.8, 6.3))
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=years[idx], cmap="viridis", s=4, alpha=0.4, linewidths=0)
    fig.colorbar(sc, ax=ax, label="Year")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Global PCA projection by collection year")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "pca_2d_by_year")


def deduplicate_by_ha_na(Z, years, months, types, records):
    seen = set()
    keep = []
    removed = Counter()
    for i, record in enumerate(records):
        key = ha_na_key(record)
        if key in seen:
            removed[str(types[i])] += 1
            continue
        seen.add(key)
        keep.append(i)
    keep = np.asarray(keep, dtype=int)
    return Z[keep], years[keep], months[keep], types[keep], [records[i] for i in keep], dict(removed)


def standardize(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-12] = 1.0
    return (X - mu) / sigma


def twonn_estimate(X, trim=0.01, return_fit=False):
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    X = standardize(X)
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    valid = np.isfinite(r1) & np.isfinite(r2) & (r1 > 1e-12) & (r2 > r1)
    mu = r2[valid] / r1[valid]
    zero_distances = int(np.sum(~valid))
    before_trim = len(mu)
    if trim > 0 and len(mu) > 10:
        lo = np.quantile(mu, trim)
        hi = np.quantile(mu, 1 - trim)
        mu = mu[(mu >= lo) & (mu <= hi)]
    n_trimmed = before_trim - len(mu)
    mu = np.sort(mu)
    if len(mu) < 10:
        result = {"dimension": np.nan, "r2": np.nan, "n_used": int(len(mu)), "zero_or_invalid_distances": zero_distances, "n_trimmed": n_trimmed}
        return (result, None) if return_fit else result
    x = np.log(mu)
    F = (np.arange(1, len(mu) + 1) - 0.5) / len(mu)
    y = -np.log1p(-F)
    A = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = slope * x + intercept
    r2_score = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    result = {
        "dimension": float(slope),
        "intercept": float(intercept),
        "r2": float(r2_score),
        "n_used": int(len(mu)),
        "zero_or_invalid_distances": zero_distances,
        "n_trimmed": int(n_trimmed),
        "mu_min": float(np.min(mu)),
        "mu_median": float(np.median(mu)),
        "mu_max": float(np.max(mu)),
    }
    if return_fit:
        return result, {"x": x, "y": y, "yhat": yhat}
    return result


def twonn_sensitivity(Z, years, months, types, records, sample_sizes, trims, seeds):
    print("\n[TwoNN] Sensitivity on HA+NA deduplicated embeddings")
    Zd, yd, md, td, rd, removed = deduplicate_by_ha_na(Z, years, months, types, records)
    rows = []
    rng_by_seed = {seed: np.random.default_rng(seed) for seed in seeds}
    for sample_size in sample_sizes:
        effective_size = min(sample_size, len(Zd))
        for seed in seeds:
            rng = rng_by_seed[seed]
            idx = rng.choice(len(Zd), size=effective_size, replace=False) if len(Zd) > effective_size else np.arange(len(Zd))
            Xs = Zd[idx]
            for trim in trims:
                start = time.time()
                try:
                    result = twonn_estimate(Xs, trim=trim)
                    status = "ok"
                except Exception as exc:
                    result = {"dimension": np.nan, "r2": np.nan, "n_used": 0, "zero_or_invalid_distances": 0, "n_trimmed": 0}
                    status = f"failed: {type(exc).__name__}"
                rows.append({
                    "sample_size_requested": int(sample_size),
                    "sample_size_used": int(effective_size),
                    "seed": int(seed),
                    "trim": float(trim),
                    "status": status,
                    "runtime_seconds": float(time.time() - start),
                    **result,
                })
                print(f"  n={effective_size:,} seed={seed} trim={trim:.2f}: d={rows[-1]['dimension']:.3f} R2={rows[-1]['r2']:.4f} {status}")
    plot_twonn_sensitivity(rows)
    plot_twonn_fit_example(Zd, sample_sizes, trims, seeds)
    return {"deduplicated_n": int(len(Zd)), "removed_duplicates": removed, "rows": rows}


def plot_twonn_sensitivity(rows):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for trim in sorted({r["trim"] for r in rows}):
        subset = [r for r in rows if r["trim"] == trim and np.isfinite(r["dimension"])]
        sizes = sorted({r["sample_size_used"] for r in subset})
        means = []
        stds = []
        for size in sizes:
            vals = [r["dimension"] for r in subset if r["sample_size_used"] == size]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=4, label=f"trim={trim:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("TwoNN intrinsic dimension")
    ax.set_title("TwoNN sensitivity to sample size and trimming")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "twonn_sensitivity")


def plot_twonn_fit_example(Zd, sample_sizes, trims, seeds):
    sample_size = min(max(sample_sizes), len(Zd))
    trim = trims[-1]
    seed = seeds[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Zd), size=sample_size, replace=False) if len(Zd) > sample_size else np.arange(len(Zd))
    result, fit = twonn_estimate(Zd[idx], trim=trim, return_fit=True)
    if fit is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    step = max(1, len(fit["x"]) // 8000)
    ax.scatter(fit["x"][::step], fit["y"][::step], s=3, alpha=0.25, label="empirical")
    ax.plot(fit["x"], fit["yhat"], color="#E63946", linewidth=2, label=f"fit d={result['dimension']:.2f}, R2={result['r2']:.3f}")
    ax.set_xlabel("log(mu)")
    ax.set_ylabel("-log(1 - F(mu))")
    ax.set_title(f"TwoNN fit example (n={sample_size:,}, trim={trim:.2f})")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "twonn_fit_example")


def temporal_local(Z, years, months, types, records, k_values, seed, max_points_per_subtype=0):
    from sklearn.neighbors import NearestNeighbors

    print("\n[Temporal locality] Nearest-neighbor time deltas after HA+NA deduplication")
    Zd, yd, md, td, rd, removed = deduplicate_by_ha_na(Z, years, months, types, records)
    rng = np.random.default_rng(seed)
    rows = []
    for subtype in SUBTYPE_ORDER:
        idx = np.where(td == subtype)[0]
        if max_points_per_subtype and len(idx) > max_points_per_subtype:
            idx = rng.choice(idx, size=max_points_per_subtype, replace=False)
        X = Zd[idx]
        t = month_index(yd[idx], md[idx])
        nn = NearestNeighbors(n_neighbors=max(k_values) + 1, metric="euclidean", n_jobs=-1)
        nn.fit(X)
        _, neighbor_idx = nn.kneighbors(X)
        neighbor_idx = neighbor_idx[:, 1:]
        random_cache = {}
        deltas_for_plot = {}
        random_for_plot = {}
        for k in k_values:
            deltas = np.abs(t[:, None] - t[neighbor_idx[:, :k]]).reshape(-1)
            ri, rj = sample_pairs(len(t), len(deltas), rng)
            random_deltas = np.abs(t[ri] - t[rj])
            random_cache[k] = random_deltas
            deltas_for_plot[k] = deltas
            random_for_plot[k] = random_deltas
            ns = stats(deltas)
            rs = stats(random_deltas)
            rows.append({
                "subtype": subtype,
                "n_points": int(len(X)),
                "k": int(k),
                "neighbor": ns,
                "random": rs,
                "median_ratio": float(ns["median"] / rs["median"]) if rs["median"] else np.nan,
                "mean_ratio": float(ns["mean"] / rs["mean"]) if rs["mean"] else np.nan,
            })
            print(f"  {subtype} k={k}: median neighbors={ns['median']:.2f} random={rs['median']:.2f}")
        plot_temporal_subtype(subtype, deltas_for_plot, random_for_plot)
    return {"deduplicated_n": int(len(Zd)), "removed_duplicates": removed, "rows": rows}


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def plot_temporal_subtype(subtype, deltas_by_k, random_by_k):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    labels = [f"k={k}" for k in sorted(deltas_by_k)] + ["random"]
    data = [deltas_by_k[k] for k in sorted(deltas_by_k)] + [random_by_k[max(deltas_by_k)]]
    box = axes[0].boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor("#6B7280" if label == "random" else SUBTYPE_COLORS[subtype])
        patch.set_alpha(0.55)
    axes[0].set_ylabel("Absolute temporal difference (months)")
    axes[0].set_title(f"{subtype}: latent neighbors vs random")
    axes[0].spines[["top", "right"]].set_visible(False)

    bins = np.linspace(0, max(1, np.percentile(random_by_k[max(deltas_by_k)], 99)), 45)
    for k in sorted(deltas_by_k):
        axes[1].hist(deltas_by_k[k], bins=bins, density=True, alpha=0.30, label=f"k={k}")
    axes[1].hist(random_by_k[max(deltas_by_k)], bins=bins, density=True, histtype="step", linewidth=2.0, color="#6B7280", label="random")
    axes[1].set_xlabel("Absolute temporal difference (months)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Temporal distance distribution")
    axes[1].legend()
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, f"temporal_local_neighbors_{subtype.lower()}_dedup")


def save_fig(fig, basename):
    pdf = os.path.join(FIGURES_DIR, f"{basename}.pdf")
    png = os.path.join(FIGURES_DIR, f"{basename}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(metrics):
    path = os.path.join(RESULTS_DIR, "latent_geometry_full_summary.md")
    lines = [
        "# Full-data latent geometry summary",
        "",
        "This report audits the geometry of local AntigenLM embeddings. It does not generate sequences, optimize mutations, or reproduce AntigenLM forecasting figures.",
        "",
        "## Cache and data",
        "",
        f"- cache: `{metrics['cache_path']}`",
        f"- embeddings: `{tuple(metrics['data_audit']['embedding_shape'])}`",
        f"- deduplicated HA+NA points for TwoNN/temporal locality: `{metrics['twonn']['deduplicated_n']:,}`",
        "",
        "## Spearman correlations",
        "",
        "Pairwise correlations are estimated by random pair sampling within subtype, not by the full quadratic set of all pairs.",
        "",
        "| metric | subtype | rho mean | rho sd | valid pairs mean | omitted pairs mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for metric in ("temporal", "hamming_ha", "hamming_ha_na"):
        for subtype in SUBTYPE_ORDER:
            subset = [r for r in metrics["spearman"] if r["metric"] == metric and r["subtype"] == subtype]
            if not subset:
                continue
            rhos = np.asarray([r["rho"] for r in subset], dtype=float)
            valid = np.asarray([r["valid_pairs"] for r in subset], dtype=float)
            omitted = np.asarray([r["omitted_pairs"] for r in subset], dtype=float)
            lines.append(
                f"| {METRIC_LABELS[metric]} | {subtype} | {np.nanmean(rhos):.4f} | {np.nanstd(rhos):.4f} | "
                f"{np.nanmean(valid):.0f} | {np.nanmean(omitted):.0f} |"
            )

    lines.extend([
        "",
        "## PCA effective dimension",
        "",
        "| group | n | n80 | n90 | n95 | n99 | participation ratio | top10 EVR |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for group, result in metrics["pca"].items():
        thr = result["n_components_by_threshold"]
        top10 = ", ".join(f"{v:.4f}" for v in result["top10_explained_variance_ratio"][:10])
        lines.append(
            f"| {group} | {result['n_samples']:,} | {thr['0.8']} | {thr['0.9']} | {thr['0.95']} | {thr['0.99']} | "
            f"{result['participation_ratio']:.2f} | {top10} |"
        )

    lines.extend([
        "",
        "## TwoNN sensitivity",
        "",
        "| sample size | trim | dimension mean | dimension sd | R2 mean |",
        "|---:|---:|---:|---:|---:|",
    ])
    for size in sorted({r["sample_size_used"] for r in metrics["twonn"]["rows"]}):
        for trim in sorted({r["trim"] for r in metrics["twonn"]["rows"]}):
            subset = [r for r in metrics["twonn"]["rows"] if r["sample_size_used"] == size and r["trim"] == trim and np.isfinite(r["dimension"])]
            if not subset:
                continue
            dims = np.asarray([r["dimension"] for r in subset])
            r2s = np.asarray([r["r2"] for r in subset])
            lines.append(f"| {size:,} | {trim:.2f} | {np.mean(dims):.3f} | {np.std(dims):.3f} | {np.mean(r2s):.4f} |")

    lines.extend([
        "",
        "## Temporal locality",
        "",
        "| subtype | k | n points | median neighbors | median random | median ratio | mean neighbors | mean random |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in metrics["temporal_locality"]["rows"]:
        lines.append(
            f"| {row['subtype']} | {row['k']} | {row['n_points']:,} | "
            f"{row['neighbor']['median']:.2f} | {row['random']['median']:.2f} | {row['median_ratio']:.3f} | "
            f"{row['neighbor']['mean']:.2f} | {row['random']['mean']:.2f} |"
        )

    lines.extend([
        "",
        "## Figures",
        "",
        "- `figures/latent_geometry_full/records_by_year_subtype.pdf`",
        "- `figures/latent_geometry_full/sequence_length_distributions.pdf`",
        "- `figures/latent_geometry_full/spearman_latent_vs_distances.pdf`",
        "- `figures/latent_geometry_full/pca_scree_global.pdf`",
        "- `figures/latent_geometry_full/pca_cumulative_global.pdf`",
        "- `figures/latent_geometry_full/pca_scree_by_subtype.pdf`",
        "- `figures/latent_geometry_full/pca_cumulative_by_subtype.pdf`",
        "- `figures/latent_geometry_full/pca_2d_by_subtype.pdf`",
        "- `figures/latent_geometry_full/pca_2d_by_year.pdf`",
        "- `figures/latent_geometry_full/twonn_sensitivity.pdf`",
        "- `figures/latent_geometry_full/twonn_fit_example.pdf`",
        "- `figures/latent_geometry_full/temporal_local_neighbors_h1n1_dedup.pdf`",
        "- `figures/latent_geometry_full/temporal_local_neighbors_h3n2_dedup.pdf`",
        "",
        "## Methodological reading",
        "",
        "- Strong Hamming correlations support molecular organization of the local checkpoint embeddings.",
        "- Weak global temporal correlation is not a failure mode by itself, because influenza evolution is branching and nonlinear.",
        "- Strong temporal locality indicates that latent neighborhoods are evolutionarily coherent at local scale.",
        "- Low PCA/TwoNN effective dimension motivates reduced dynamical modeling, but does not validate forecasting or a full SDE.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_metrics_json(metrics):
    serializable = json.loads(json.dumps(metrics, default=to_jsonable))
    path = os.path.join(RESULTS_DIR, "latent_geometry_full_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return path


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Counter):
        return dict(obj)
    return str(obj)


def maybe_make_paper(script_path):
    if not script_path:
        return
    import subprocess
    subprocess.run(["venv_antigenlm/bin/python", script_path], check=True)


def main():
    parser = argparse.ArgumentParser(description="Full-data geometry analysis from AntigenLM embedding cache")
    parser.add_argument("--cache-path", default=os.path.join(RESULTS_DIR, "embeddings_cache_full_all_available.pkl"))
    parser.add_argument("--pair-samples-per-subtype", type=int, default=100000)
    parser.add_argument("--pair-seeds", default="42,7,123")
    parser.add_argument("--twonn-sample-sizes", default="5000,10000,20000,50000")
    parser.add_argument("--twonn-trims", default="0.01,0.05")
    parser.add_argument("--twonn-seeds", default="42,7,123")
    parser.add_argument("--temporal-k-values", default="5,10,20")
    parser.add_argument("--temporal-max-points-per-subtype", type=int, default=0)
    parser.add_argument("--plot-max-points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make-paper-script", default="")
    args = parser.parse_args()

    ensure_dirs()
    Z, years, months, types, records, metadata = load_cache(args.cache_path)
    metrics = {
        "cache_path": args.cache_path,
        "parameters": vars(args),
    }
    metrics["data_audit"] = audit_source_and_cache(Z, years, months, types, records, metadata, args.cache_path)
    metrics["spearman"] = spearman_full(
        Z, years, months, types, records,
        pair_samples=args.pair_samples_per_subtype,
        seeds=parse_int_list(args.pair_seeds),
    )
    metrics["pca"] = pca_full(Z, years, types, plot_max_points=args.plot_max_points, seed=args.seed)
    metrics["twonn"] = twonn_sensitivity(
        Z, years, months, types, records,
        sample_sizes=parse_int_list(args.twonn_sample_sizes),
        trims=parse_float_list(args.twonn_trims),
        seeds=parse_int_list(args.twonn_seeds),
    )
    metrics["temporal_locality"] = temporal_local(
        Z, years, months, types, records,
        k_values=parse_int_list(args.temporal_k_values),
        seed=args.seed,
        max_points_per_subtype=args.temporal_max_points_per_subtype,
    )
    metrics_path = write_metrics_json(metrics)
    summary_path = write_summary(metrics)
    print(f"\nMetrics JSON: {metrics_path}")
    print(f"Summary: {summary_path}")
    maybe_make_paper(args.make_paper_script)


if __name__ == "__main__":
    main()
