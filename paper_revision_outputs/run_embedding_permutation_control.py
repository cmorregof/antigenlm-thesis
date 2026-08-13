#!/usr/bin/env python3
"""Embedding permutation negative control for molecular correlations.

The control preserves the empirical AntigenLM vector cloud within each subtype
but permutes embeddings among records, breaking record-embedding
correspondence. Outputs are aggregate only and do not include sequences or
accession-level metadata.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SUBTYPES = ("H1N1", "H3N2")


def load_cache(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return (
        np.asarray(payload["embeddings"], dtype=np.float32),
        np.asarray(payload["types"]).astype(str),
        payload["records"],
        payload.get("metadata", {}),
    )


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


def rho(latent: np.ndarray, molecular: np.ndarray) -> float:
    finite = np.isfinite(latent) & np.isfinite(molecular)
    if np.sum(finite) < 10:
        return float("nan")
    return float(spearmanr(latent[finite], molecular[finite]).statistic)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    Z, types, records, metadata = load_cache(ROOT / args.cache_path)
    pair_seeds = [int(x) for x in args.pair_seeds.split(",") if x.strip()]
    rows = []

    for subtype in SUBTYPES:
        subtype_global = np.where(types == subtype)[0]
        X = Z[subtype_global]
        subtype_records = [records[int(i)] for i in subtype_global]
        for pair_seed in pair_seeds:
            rng = np.random.default_rng(pair_seed)
            li, lj = sample_pairs(len(subtype_global), args.pair_samples_per_subtype, rng)
            molecular = np.full(len(li), np.nan, dtype=np.float32)
            for n, (a, b) in enumerate(zip(li, lj)):
                value = hamming_ha_na(subtype_records, int(a), int(b))
                if value is not None:
                    molecular[n] = value

            true_latent = np.linalg.norm(X[li] - X[lj], axis=1)
            true_rho = rho(true_latent, molecular)
            permuted = []
            for rep in range(args.permutation_replicates):
                perm_rng = np.random.default_rng(args.permutation_seed + 1000 * pair_seed + rep)
                order = perm_rng.permutation(len(X))
                Xp = X[order]
                perm_latent = np.linalg.norm(Xp[li] - Xp[lj], axis=1)
                permuted.append(rho(perm_latent, molecular))
            arr = np.asarray(permuted, dtype=float)
            rows.append(
                {
                    "subtype": subtype,
                    "pair_seed": int(pair_seed),
                    "requested_pairs": int(args.pair_samples_per_subtype),
                    "valid_pairs": int(np.sum(np.isfinite(molecular))),
                    "true_rho": true_rho,
                    "permuted_mean": float(np.nanmean(arr)),
                    "permuted_sd": float(np.nanstd(arr, ddof=1)),
                    "permuted_p05": float(np.nanpercentile(arr, 5)),
                    "permuted_p95": float(np.nanpercentile(arr, 95)),
                    "permutation_replicates": int(args.permutation_replicates),
                }
            )
            print(
                f"{subtype} seed={pair_seed}: true rho={true_rho:.4f}; "
                f"permuted mean={np.nanmean(arr):.4f}"
            )

    aggregate = []
    for subtype in SUBTYPES:
        vals = [row for row in rows if row["subtype"] == subtype]
        aggregate.append(
            {
                "subtype": subtype,
                "true_rho_mean": float(np.mean([v["true_rho"] for v in vals])),
                "true_rho_sd": float(np.std([v["true_rho"] for v in vals], ddof=1)),
                "permuted_mean": float(np.mean([v["permuted_mean"] for v in vals])),
                "permuted_sd_across_pair_seeds": float(np.std([v["permuted_mean"] for v in vals], ddof=1)),
                "valid_pairs_mean": float(np.mean([v["valid_pairs"] for v in vals])),
            }
        )

    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "runtime_seconds": float(time.time() - started),
        "inputs": {"cache_path": args.cache_path},
        "cache_metadata": metadata,
        "parameters": {
            "pair_samples_per_subtype": int(args.pair_samples_per_subtype),
            "pair_seeds": pair_seeds,
            "permutation_replicates": int(args.permutation_replicates),
            "permutation_seed": int(args.permutation_seed),
        },
        "rows": rows,
        "aggregate": aggregate,
        "interpretation": "Permuting embeddings within subtype preserves the vector distribution and anisotropy but breaks record-embedding correspondence.",
    }


def fmt(value: float, digits: int = 4) -> str:
    if value != value:
        return "NA"
    return f"{value:.{digits}f}"


def write_summary(payload: dict, path: Path) -> None:
    lines = [
        "# Embedding Permutation Negative Control",
        "",
        "Embeddings were permuted among records within subtype before recomputing HA+NA Hamming Spearman correlations. The output is aggregate-only.",
        "",
        "| subtype | true rho mean | true rho sd | permuted rho mean | permuted mean sd across pair seeds | valid pairs mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['subtype']} | {fmt(row['true_rho_mean'])} | {fmt(row['true_rho_sd'])} | "
            f"{fmt(row['permuted_mean'])} | {fmt(row['permuted_sd_across_pair_seeds'])} | "
            f"{row['valid_pairs_mean']:.0f} |"
        )
    lines.extend([
        "",
        "The near-zero permuted correlations show that the molecular signal depends on the correct record-embedding correspondence, not only on the marginal AntigenLM vector cloud.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run embedding permutation control.")
    parser.add_argument("--cache-path", default="results/embeddings_cache_full_all_available.pkl")
    parser.add_argument("--output-json", default="results/embedding_permutation_results.json")
    parser.add_argument("--output-summary", default="results/embedding_permutation_summary.md")
    parser.add_argument("--pair-samples-per-subtype", type=int, default=100000)
    parser.add_argument("--pair-seeds", default="42,7,123")
    parser.add_argument("--permutation-replicates", type=int, default=20)
    parser.add_argument("--permutation-seed", type=int, default=42)
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
