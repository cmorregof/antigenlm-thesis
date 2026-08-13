#!/usr/bin/env python3
"""PCA anisotropy and dominant-PC sensitivity controls.

Outputs aggregate PCA summaries and pair-sampled HA+NA Hamming correlations
after L2 normalization and after subtracting the leading global PCs. It writes
no sequences or accession-level metadata.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
SUBTYPES = ("H1N1", "H3N2")
PCA_THRESHOLDS = (0.80, 0.90, 0.95, 0.99)


def load_cache(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return (
        np.asarray(payload["embeddings"], dtype=np.float64),
        np.asarray(payload["types"]).astype(str),
        payload["records"],
        payload.get("metadata", {}),
    )


def deduplicate_by_ha_na(Z, types, records):
    seen: set[tuple[str, str]] = set()
    keep: list[int] = []
    removed = Counter()
    for i, record in enumerate(records):
        key = (record.get("ha_sequence", "") or "", record.get("na_sequence", "") or "")
        if key in seen:
            removed[str(types[i])] += 1
            continue
        seen.add(key)
        keep.append(i)
    keep_arr = np.asarray(keep, dtype=np.int64)
    return Z[keep_arr], types[keep_arr], [records[i] for i in keep], dict(removed)


def pca_summary(X: np.ndarray) -> dict:
    X = np.asarray(X, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0, None)
    eigvecs = eigvecs[:, order]
    positive = eigvals[eigvals > 1e-12]
    ratios = positive / np.sum(positive)
    cumulative = np.cumsum(ratios)
    return {
        "n_samples": int(len(X)),
        "embedding_dim": int(X.shape[1]),
        "n_components_by_threshold": {str(t): int(np.searchsorted(cumulative, t) + 1) for t in PCA_THRESHOLDS},
        "participation_ratio": float((np.sum(positive) ** 2) / np.sum(positive ** 2)),
        "top10_explained_variance_ratio": [float(x) for x in ratios[:10]],
        "components": eigvecs,
        "mean": X.mean(axis=0),
    }


def compact(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k not in {"components", "mean"}}


def sample_pairs(n: int, n_pairs: int, rng: np.random.Generator):
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n - 1, size=n_pairs)
    j = j + (j >= i)
    return i, j


def normalized_hamming_arrays(a: str, b: str, tolerance: float = 0.05) -> float | None:
    len_a = len(a)
    len_b = len(b)
    if len_a == 0 or len_b == 0:
        return None
    max_len = max(len_a, len_b)
    min_len = min(len_a, len_b)
    if (max_len - min_len) / max_len > tolerance:
        return None
    aa = np.frombuffer(a[:min_len].encode("ascii", errors="ignore"), dtype=np.uint8)
    bb = np.frombuffer(b[:min_len].encode("ascii", errors="ignore"), dtype=np.uint8)
    return float(np.count_nonzero(aa != bb) / min_len)


def hamming_ha_na(records: list[dict], i: int, j: int) -> float | None:
    a = records[i]
    b = records[j]
    ha_i = a.get("ha_sequence", "") or ""
    ha_j = b.get("ha_sequence", "") or ""
    na_i = a.get("na_sequence", "") or ""
    na_j = b.get("na_sequence", "") or ""
    d_ha = normalized_hamming_arrays(ha_i, ha_j)
    d_na = normalized_hamming_arrays(na_i, na_j)
    if d_ha is None or d_na is None:
        return None
    ha_len = min(len(ha_i), len(ha_j))
    na_len = min(len(na_i), len(na_j))
    denom = ha_len + na_len
    if denom == 0:
        return None
    return float((d_ha * ha_len + d_na * na_len) / denom)


def transform_remove_pcs(Z: np.ndarray, mean: np.ndarray, components: np.ndarray, n_remove: int) -> np.ndarray:
    C = components[:, :n_remove]
    Xc = Z - mean[None, :]
    return Xc - (Xc @ C) @ C.T


def correlations_for_representations(
    representations: dict[str, np.ndarray],
    types: np.ndarray,
    records: list[dict],
    n_pairs: int,
    seeds: list[int],
) -> list[dict]:
    rows = []
    for subtype in SUBTYPES:
        idx = np.where(types == subtype)[0]
        subtype_records = [records[int(i)] for i in idx]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            li, lj = sample_pairs(len(idx), n_pairs, rng)
            molecular = np.full(len(li), np.nan, dtype=np.float64)
            for row, (a, b) in enumerate(zip(li, lj)):
                value = hamming_ha_na(subtype_records, int(a), int(b))
                if value is not None:
                    molecular[row] = value
            valid = np.isfinite(molecular)
            for name, Zt in representations.items():
                Xt = Zt[idx]
                latent = np.linalg.norm(Xt[li] - Xt[lj], axis=1)
                finite = valid & np.isfinite(latent)
                rho = spearmanr(latent[finite], molecular[finite]).statistic if np.sum(finite) > 10 else np.nan
                rows.append(
                    {
                        "representation": name,
                        "subtype": subtype,
                        "seed": int(seed),
                        "rho": float(rho),
                        "valid_pairs": int(np.sum(finite)),
                        "requested_pairs": int(n_pairs),
                    }
                )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    for representation in sorted({row["representation"] for row in rows}):
        for subtype in SUBTYPES:
            vals = [row for row in rows if row["representation"] == representation and row["subtype"] == subtype]
            rhos = np.asarray([row["rho"] for row in vals], dtype=float)
            out.append(
                {
                    "representation": representation,
                    "subtype": subtype,
                    "rho_mean": float(np.nanmean(rhos)),
                    "rho_sd": float(np.nanstd(rhos, ddof=1)),
                    "valid_pairs_mean": float(np.mean([row["valid_pairs"] for row in vals])),
                }
            )
    return out


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    Z, types, records, metadata = load_cache(ROOT / args.cache_path)
    seeds = [int(x) for x in args.pair_seeds.split(",") if x.strip()]
    original = pca_summary(Z)
    Z_l2 = normalize(Z, norm="l2")
    l2_summary = pca_summary(Z_l2)
    Zd, td, rd, removed = deduplicate_by_ha_na(Z, types, records)
    dedup_summary = pca_summary(Zd)
    Zd_l2 = normalize(Zd, norm="l2")
    dedup_l2_summary = pca_summary(Zd_l2)
    representations = {
        "original": Z,
        "l2_normalized": Z_l2,
        "remove_global_PC1": transform_remove_pcs(Z, original["mean"], original["components"], 1),
        "remove_global_PC1_PC2": transform_remove_pcs(Z, original["mean"], original["components"], 2),
        "remove_global_PC1_PC2_PC3": transform_remove_pcs(Z, original["mean"], original["components"], 3),
    }
    corr_rows = correlations_for_representations(representations, types, records, args.pair_samples_per_subtype, seeds)
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "runtime_seconds": float(time.time() - started),
        "inputs": {"cache_path": args.cache_path},
        "cache_metadata": metadata,
        "parameters": {
            "pair_samples_per_subtype": int(args.pair_samples_per_subtype),
            "pair_seeds": seeds,
        },
        "pca": {
            "original": compact(original),
            "l2_normalized": compact(l2_summary),
            "deduplicated_original": compact(dedup_summary),
            "deduplicated_l2_normalized": compact(dedup_l2_summary),
            "deduplication": {"deduplicated_n": int(len(Zd)), "removed_duplicates": removed},
        },
        "correlations": {"rows": corr_rows, "aggregate": aggregate(corr_rows)},
    }


def fmt(value: float, digits: int = 4) -> str:
    if value != value:
        return "NA"
    return f"{value:.{digits}f}"


def write_summary(payload: dict, path: Path) -> None:
    pca = payload["pca"]
    lines = [
        "# PCA Anisotropy and Dominant-PC Controls",
        "",
        "Outputs are aggregate-only and do not redistribute sequences or accession-level metadata.",
        "",
        "## PCA Summary",
        "",
        "| representation | n | n80 | n90 | n95 | n99 | participation ratio | PC1 variance | PC2 variance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("original", "l2_normalized", "deduplicated_original", "deduplicated_l2_normalized"):
        row = pca[name]
        n = row["n_components_by_threshold"]
        top = row["top10_explained_variance_ratio"]
        lines.append(
            f"| {name} | {row['n_samples']:,} | {n['0.8']} | {n['0.9']} | {n['0.95']} | {n['0.99']} | "
            f"{fmt(row['participation_ratio'], 2)} | {fmt(top[0], 3)} | {fmt(top[1], 3)} |"
        )
    lines.extend([
        "",
        "## HA+NA Hamming Spearman After Representation Transformations",
        "",
        "| representation | subtype | rho mean | rho sd | valid pairs mean |",
        "|---|---|---:|---:|---:|",
    ])
    order = ["original", "l2_normalized", "remove_global_PC1", "remove_global_PC1_PC2", "remove_global_PC1_PC2_PC3"]
    for name in order:
        for row in [r for r in payload["correlations"]["aggregate"] if r["representation"] == name]:
            lines.append(
                f"| {row['representation']} | {row['subtype']} | {fmt(row['rho_mean'])} | "
                f"{fmt(row['rho_sd'])} | {row['valid_pairs_mean']:.0f} |"
            )
    lines.extend([
        "",
        "These controls test whether the molecular signal is concentrated entirely in dominant global directions. They do not address pooling-token ablations, which require regenerating embeddings.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PCA anisotropy controls.")
    parser.add_argument("--cache-path", default="results/embeddings_cache_full_all_available.pkl")
    parser.add_argument("--output-json", default="results/pca_anisotropy_controls.json")
    parser.add_argument("--output-summary", default="results/pca_anisotropy_controls_summary.md")
    parser.add_argument("--pair-samples-per-subtype", type=int, default=100000)
    parser.add_argument("--pair-seeds", default="42,7,123")
    args = parser.parse_args()
    payload = run(args)
    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_summary
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(payload, out_md)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Runtime seconds: {payload['runtime_seconds']:.1f}")


if __name__ == "__main__":
    main()
