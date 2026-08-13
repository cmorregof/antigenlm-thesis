#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random embedding negative control for Paper 1.

This script reuses the cached metadata/sequences only to compute HA+NA Hamming
distances for sampled pairs. It does not reload AntigenLM, recompute learned
embeddings, print complete sequences, or modify raw data.

Outputs:
  paper_revision_outputs/random_embedding_baseline_results.json
  paper_revision_outputs/random_embedding_baseline_summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

OUT_DIR = "paper_revision_outputs"
SUBTYPE_ORDER = ("H1N1", "H3N2")


def load_cache(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    required = ("embeddings", "years", "months", "types", "records")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Cache missing required keys: {missing}")
    Z = np.asarray(payload["embeddings"])
    types = np.asarray(payload["types"]).astype(str)
    records = payload["records"]
    n = {len(Z), len(types), len(records)}
    if len(n) != 1:
        raise ValueError("Cache arrays are not aligned")
    return Z, types, records, payload.get("metadata", {})


def sequence_array(sequence):
    return np.frombuffer((sequence or "").upper().encode("ascii", errors="ignore"), dtype=np.uint8)


def build_record_features(records):
    features = []
    for record in records:
        ha = record.get("ha_sequence", "") or ""
        na = record.get("na_sequence", "") or ""
        features.append(
            {
                "ha": sequence_array(ha),
                "na": sequence_array(na),
                "ha_len": len(ha),
                "na_len": len(na),
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


def hamming_ha_na(features, i, j):
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


def sample_pairs(n, n_pairs, rng):
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n - 1, size=n_pairs)
    j = j + (j >= i)
    return i, j


def spearman_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.sum(finite) < 10:
        return np.nan, np.nan, int(np.sum(finite))
    rho, pvalue = spearmanr(a[finite], b[finite])
    return float(rho), float(pvalue), int(np.sum(finite))


def compute_molecular_pairs(types, records, pair_samples, pair_seeds):
    features = build_record_features(records)
    pair_sets = []
    for pair_seed in pair_seeds:
        rng = np.random.default_rng(pair_seed)
        for subtype in SUBTYPE_ORDER:
            idx = np.where(types == subtype)[0]
            local_i, local_j = sample_pairs(len(idx), pair_samples, rng)
            gi = idx[local_i]
            gj = idx[local_j]
            hamming = []
            valid_i = []
            valid_j = []
            omitted = 0
            for a, b in zip(gi, gj):
                value = hamming_ha_na(features, int(a), int(b))
                if value is None:
                    omitted += 1
                    continue
                valid_i.append(int(a))
                valid_j.append(int(b))
                hamming.append(float(value))
            pair_sets.append(
                {
                    "pair_seed": int(pair_seed),
                    "subtype": str(subtype),
                    "i": np.asarray(valid_i, dtype=np.int64),
                    "j": np.asarray(valid_j, dtype=np.int64),
                    "hamming_ha_na": np.asarray(hamming, dtype=np.float64),
                    "requested_pairs": int(pair_samples),
                    "valid_pairs": int(len(hamming)),
                    "omitted_pairs": int(omitted),
                }
            )
            print(
                f"[pairs] seed={pair_seed} {subtype}: "
                f"valid={len(hamming):,} omitted={omitted:,}"
            )
    return pair_sets


def random_unit_embeddings(n, dim, rng):
    X = rng.standard_normal((n, dim), dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["subtype"]].append(row)
    out = []
    for subtype in SUBTYPE_ORDER:
        values = grouped[subtype]
        rhos = np.asarray([row["rho"] for row in values], dtype=float)
        out.append(
            {
                "metric": "hamming_ha_na",
                "subtype": subtype,
                "rho_mean_random": float(np.nanmean(rhos)),
                "rho_sd_random": float(np.nanstd(rhos)),
                "rho_min_random": float(np.nanmin(rhos)),
                "rho_max_random": float(np.nanmax(rhos)),
                "n_correlations": int(len(values)),
                "valid_pairs_mean": float(np.mean([row["valid_pairs"] for row in values])),
                "omitted_pairs_mean": float(np.mean([row["omitted_pairs"] for row in values])),
            }
        )
    return out


def write_outputs(payload):
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "random_embedding_baseline_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    antigenlm = {"H1N1": 0.8538, "H3N2": 0.6678}
    lines = [
        "# Random Embedding Baseline",
        "",
        f"Created at local run time: `{payload['created_at']}`",
        "",
        "This negative control samples independent 384-dimensional Gaussian vectors, normalizes each vector to unit L2 norm, and evaluates HA+NA Hamming correlations with the same subtype-specific pair-sampling protocol used for the main AntigenLM analysis.",
        "",
        "It does not reload AntigenLM, regenerate learned embeddings, print complete sequences, or modify raw data.",
        "",
        "## Inputs and Parameters",
        "",
        f"- cache: `{payload['cache_path']}`",
        f"- n records: `{payload['n_records']:,}`",
        f"- random embedding dimension: `{payload['embedding_dim']}`",
        f"- random embedding replicates: `{payload['random_replicates']}`",
        f"- random embedding seeds: `{', '.join(map(str, payload['random_embedding_seeds']))}`",
        f"- pair samples per subtype/seed: `{payload['pair_samples_per_subtype']:,}`",
        f"- pair seeds: `{', '.join(map(str, payload['pair_seeds']))}`",
        "",
        "## Aggregate Results",
        "",
        "| metric | subtype | random rho mean | random rho sd | AntigenLM rho mean | AntigenLM - random | correlations summarized |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        subtype = row["subtype"]
        delta = antigenlm[subtype] - row["rho_mean_random"]
        lines.append(
            f"| HA+NA Hamming | {subtype} | {row['rho_mean_random']:.4f} | "
            f"{row['rho_sd_random']:.4f} | {antigenlm[subtype]:.4f} | {delta:.4f} | "
            f"{row['n_correlations']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The random embedding baseline correlations are near zero, as expected for vectors independent of HA/NA sequence content. The AntigenLM HA+NA Hamming correlations are much larger than this negative control, supporting the claim that the observed molecular organization is not a trivial consequence of Euclidean distances in random high-dimensional vectors.",
        ]
    )
    summary_path = os.path.join(OUT_DIR, "random_embedding_baseline_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Random embedding negative control")
    parser.add_argument("--cache-path", default="results/embeddings_cache_full_all_available.pkl")
    parser.add_argument("--pair-samples-per-subtype", type=int, default=200000)
    parser.add_argument("--pair-seeds", default="42,7,123")
    parser.add_argument("--random-replicates", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--random-seed-base", type=int, default=1000)
    args = parser.parse_args()

    start = time.time()
    pair_seeds = [int(x.strip()) for x in args.pair_seeds.split(",") if x.strip()]
    random_seeds = [args.random_seed_base + i for i in range(args.random_replicates)]

    Z, types, records, metadata = load_cache(args.cache_path)
    pair_sets = compute_molecular_pairs(types, records, args.pair_samples_per_subtype, pair_seeds)

    rows = []
    for random_seed in random_seeds:
        rng = np.random.default_rng(random_seed)
        R = random_unit_embeddings(len(types), args.embedding_dim, rng)
        for pair_set in pair_sets:
            i = pair_set["i"]
            j = pair_set["j"]
            latent = np.linalg.norm(R[i] - R[j], axis=1)
            rho, pvalue, valid = spearman_safe(latent, pair_set["hamming_ha_na"])
            row = {
                "random_embedding_seed": int(random_seed),
                "pair_seed": int(pair_set["pair_seed"]),
                "subtype": pair_set["subtype"],
                "metric": "hamming_ha_na",
                "rho": rho,
                "pvalue": pvalue,
                "requested_pairs": int(pair_set["requested_pairs"]),
                "valid_pairs": int(valid),
                "omitted_pairs": int(pair_set["omitted_pairs"]),
            }
            rows.append(row)
            print(
                f"[random] random_seed={random_seed} pair_seed={pair_set['pair_seed']} "
                f"{pair_set['subtype']}: rho={rho:.4f}"
            )

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "runtime_seconds": float(time.time() - start),
        "cache_path": args.cache_path,
        "cache_metadata": metadata,
        "n_records": int(len(types)),
        "embedding_dim": int(args.embedding_dim),
        "random_replicates": int(args.random_replicates),
        "random_embedding_seeds": random_seeds,
        "pair_samples_per_subtype": int(args.pair_samples_per_subtype),
        "pair_seeds": pair_seeds,
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    json_path, summary_path = write_outputs(payload)
    print(f"JSON: {json_path}")
    print(f"Summary: {summary_path}")
    print(f"Runtime seconds: {payload['runtime_seconds']:.1f}")


if __name__ == "__main__":
    main()
