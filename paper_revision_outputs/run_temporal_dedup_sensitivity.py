#!/usr/bin/env python3
"""Temporal neighborhood summaries under duplicate-representative choices.

Exact HA+NA duplicates can have different collection dates. This script keeps
one representative per exact HA+NA group using several policies and recomputes
local temporal-neighborhood summaries. It writes aggregate results only.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
SUBTYPES = ("H1N1", "H3N2")


def load_cache(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return (
        np.asarray(payload["embeddings"], dtype=np.float32),
        np.asarray(payload["years"], dtype=np.int32),
        np.asarray(payload["months"], dtype=np.int32),
        np.asarray(payload["types"]).astype(str),
        payload["records"],
        payload.get("metadata", {}),
    )


def month_index(years, months):
    return years * 12 + np.clip(months, 1, 12)


def group_duplicates(records: list[dict]):
    groups = defaultdict(list)
    for i, record in enumerate(records):
        key = (record.get("ha_sequence", "") or "", record.get("na_sequence", "") or "")
        groups[key].append(i)
    return list(groups.values())


def choose_indices(groups: list[list[int]], years: np.ndarray, months: np.ndarray, policy: str, seed: int | None = None):
    rng = np.random.default_rng(seed)
    t = month_index(years, months)
    keep = []
    for group in groups:
        if policy == "first":
            keep.append(group[0])
        elif policy == "earliest":
            keep.append(min(group, key=lambda i: (int(t[i]), i)))
        elif policy == "latest":
            keep.append(max(group, key=lambda i: (int(t[i]), -i)))
        elif policy == "random":
            keep.append(int(rng.choice(group)))
        else:
            raise ValueError(policy)
    return np.asarray(sorted(keep), dtype=np.int64)


def dist(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "fraction_le_1_month": float(np.mean(values <= 1)),
        "fraction_le_3_months": float(np.mean(values <= 3)),
        "fraction_le_6_months": float(np.mean(values <= 6)),
        "fraction_le_12_months": float(np.mean(values <= 12)),
    }


def temporal_rows(Z, years, months, types, keep, k_values, random_seed):
    rows = []
    Zd = Z[keep]
    td = types[keep]
    months_idx = month_index(years[keep], months[keep])
    rng = np.random.default_rng(random_seed)
    for subtype in SUBTYPES:
        idx = np.where(td == subtype)[0]
        X = Zd[idx]
        t = months_idx[idx]
        nn = NearestNeighbors(n_neighbors=max(k_values) + 1, metric="euclidean", algorithm="brute", n_jobs=-1)
        nn.fit(X)
        _, neigh = nn.kneighbors(X)
        neigh = neigh[:, 1:]
        for k in k_values:
            neighbor_deltas = np.abs(t[:, None] - t[neigh[:, :k]]).reshape(-1)
            ri = rng.integers(0, len(t), size=len(neighbor_deltas))
            rj = rng.integers(0, len(t) - 1, size=len(neighbor_deltas))
            rj = rj + (rj >= ri)
            random_deltas = np.abs(t[ri] - t[rj])
            rows.append(
                {
                    "subtype": subtype,
                    "n_points": int(len(idx)),
                    "k": int(k),
                    "neighbor": dist(neighbor_deltas),
                    "subtype_random": dist(random_deltas),
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    Z, years, months, types, records, metadata = load_cache(ROOT / args.cache_path)
    groups = group_duplicates(records)
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    rows = []
    policies = ["first", "earliest", "latest"]
    for policy in policies:
        keep = choose_indices(groups, years, months, policy)
        for row in temporal_rows(Z, years, months, types, keep, k_values, args.random_seed):
            rows.append({"representative_policy": policy, "seed": None, **row})
        print(f"{policy}: kept {len(keep):,}")
    for seed in [int(x) for x in args.random_representative_seeds.split(",") if x.strip()]:
        keep = choose_indices(groups, years, months, "random", seed)
        for row in temporal_rows(Z, years, months, types, keep, k_values, args.random_seed + seed):
            rows.append({"representative_policy": "random", "seed": int(seed), **row})
        print(f"random seed={seed}: kept {len(keep):,}")
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "runtime_seconds": float(time.time() - started),
        "inputs": {"cache_path": args.cache_path},
        "cache_metadata": metadata,
        "parameters": {
            "k_values": k_values,
            "random_representative_seeds": [int(x) for x in args.random_representative_seeds.split(",") if x.strip()],
            "random_baseline_seed": int(args.random_seed),
        },
        "duplicate_groups": {
            "total_groups": int(len(groups)),
            "multi_record_groups": int(sum(len(g) > 1 for g in groups)),
            "records_in_multi_record_groups": int(sum(len(g) for g in groups if len(g) > 1)),
        },
        "rows": rows,
    }


def fmt(value: float, digits: int = 2) -> str:
    if value != value:
        return "NA"
    return f"{value:.{digits}f}"


def write_summary(payload: dict, path: Path) -> None:
    lines = [
        "# Temporal Deduplication Representative Sensitivity",
        "",
        "Exact HA+NA duplicate groups were collapsed using different representative-date policies before recomputing latent temporal-neighborhood summaries.",
        "",
        "| policy | seed | subtype | k | n points | neighbor median | neighbor IQR | neighbor p10-p90 | random median | <=6 mo |",
        "|---|---:|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in payload["rows"]:
        n = row["neighbor"]
        r = row["subtype_random"]
        seed = "" if row["seed"] is None else str(row["seed"])
        lines.append(
            f"| {row['representative_policy']} | {seed} | {row['subtype']} | {row['k']} | {row['n_points']:,} | "
            f"{fmt(n['median'])} | {fmt(n['p25'])}--{fmt(n['p75'])} | {fmt(n['p10'])}--{fmt(n['p90'])} | "
            f"{fmt(r['median'])} | {fmt(n['fraction_le_6_months'], 3)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal dedup representative sensitivity.")
    parser.add_argument("--cache-path", default="results/embeddings_cache_full_all_available.pkl")
    parser.add_argument("--output-json", default="results/temporal_dedup_sensitivity_results.json")
    parser.add_argument("--output-summary", default="results/temporal_dedup_sensitivity_summary.md")
    parser.add_argument("--k-values", default="5,10,20")
    parser.add_argument("--random-representative-seeds", default="42,7,123")
    parser.add_argument("--random-seed", type=int, default=42)
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
