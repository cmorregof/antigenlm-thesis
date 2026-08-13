#!/usr/bin/env python3
"""Small pairwise-alignment robustness panel.

This script computes Biopython PairwiseAligner global alignments on a modest
sample of within-subtype pairs from the exact HA+NA-deduplicated cache. It is
intended as a limited robustness panel, not as a curated HA/NA multiple
sequence alignment or protein-level antigenic-site analysis.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
from Bio import Align
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


def sample_pairs(n: int, n_pairs: int, rng: np.random.Generator):
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n - 1, size=n_pairs)
    j = j + (j >= i)
    return i, j


def normalized_prefix_hamming(a: str, b: str, tolerance: float = 0.05) -> float | None:
    if not a or not b:
        return None
    max_len = max(len(a), len(b))
    min_len = min(len(a), len(b))
    if (max_len - min_len) / max_len > tolerance:
        return None
    aa = np.frombuffer(a[:min_len].encode("ascii", errors="ignore"), dtype=np.uint8)
    bb = np.frombuffer(b[:min_len].encode("ascii", errors="ignore"), dtype=np.uint8)
    return float(np.count_nonzero(aa != bb) / min_len)


def alignment_distance(aligner: Align.PairwiseAligner, a: str, b: str) -> float | None:
    if not a or not b:
        return None
    alignment = aligner.align(a, b)[0]
    counts = alignment.counts()
    denom = counts.aligned + counts.gaps
    if denom <= 0:
        return None
    return float((counts.mismatches + counts.gaps) / denom)


def weighted_pair_distance(d_ha: float | None, d_na: float | None, rec_a: dict, rec_b: dict) -> float | None:
    if d_ha is None or d_na is None:
        return None
    ha_len = min(len(rec_a.get("ha_sequence", "") or ""), len(rec_b.get("ha_sequence", "") or ""))
    na_len = min(len(rec_a.get("na_sequence", "") or ""), len(rec_b.get("na_sequence", "") or ""))
    denom = ha_len + na_len
    if denom == 0:
        return None
    return float((d_ha * ha_len + d_na * na_len) / denom)


def rho(latent: list[float], values: list[float]) -> float:
    x = np.asarray(latent, dtype=float)
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.sum(finite) < 10:
        return float("nan")
    return float(spearmanr(x[finite], y[finite]).statistic)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    Z, types, records, metadata = load_cache(ROOT / args.cache_path)
    Zd, td, rd, removed = deduplicate_by_ha_na(Z, types, records)
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []

    for subtype in SUBTYPES:
        idx = np.where(td == subtype)[0]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            li, lj = sample_pairs(len(idx), args.pairs_per_subtype, rng)
            latent = np.linalg.norm(Zd[idx[li]] - Zd[idx[lj]], axis=1)
            prefix = {"HA": [], "NA": [], "HA+NA": []}
            aligned = {"HA": [], "NA": [], "HA+NA": []}
            latent_by_metric = {("prefix", "HA"): [], ("prefix", "NA"): [], ("prefix", "HA+NA"): [], ("aligned", "HA"): [], ("aligned", "NA"): [], ("aligned", "HA+NA"): []}
            for a_local, b_local, dl in zip(li, lj, latent):
                a = rd[int(idx[int(a_local)])]
                b = rd[int(idx[int(b_local)])]
                p_ha = normalized_prefix_hamming(a.get("ha_sequence", "") or "", b.get("ha_sequence", "") or "")
                p_na = normalized_prefix_hamming(a.get("na_sequence", "") or "", b.get("na_sequence", "") or "")
                p_both = weighted_pair_distance(p_ha, p_na, a, b)
                a_ha = alignment_distance(aligner, a.get("ha_sequence", "") or "", b.get("ha_sequence", "") or "")
                a_na = alignment_distance(aligner, a.get("na_sequence", "") or "", b.get("na_sequence", "") or "")
                a_both = weighted_pair_distance(a_ha, a_na, a, b)
                values = {
                    ("prefix", "HA"): p_ha,
                    ("prefix", "NA"): p_na,
                    ("prefix", "HA+NA"): p_both,
                    ("aligned", "HA"): a_ha,
                    ("aligned", "NA"): a_na,
                    ("aligned", "HA+NA"): a_both,
                }
                for key, value in values.items():
                    if value is not None and np.isfinite(value):
                        if key[0] == "prefix":
                            prefix[key[1]].append(value)
                        else:
                            aligned[key[1]].append(value)
                        latent_by_metric[key].append(float(dl))

            for proxy in ("prefix", "aligned"):
                source = prefix if proxy == "prefix" else aligned
                for segment in ("HA", "NA", "HA+NA"):
                    values = source[segment]
                    rows.append(
                        {
                            "distance_proxy": proxy,
                            "segment": segment,
                            "subtype": subtype,
                            "seed": int(seed),
                            "requested_pairs": int(args.pairs_per_subtype),
                            "valid_pairs": int(len(values)),
                            "rho": rho(latent_by_metric[(proxy, segment)], values),
                        }
                    )
                    print(f"{subtype} seed={seed} {proxy} {segment}: rho={rows[-1]['rho']:.4f} n={len(values)}")

    aggregate = []
    for proxy in ("prefix", "aligned"):
        for segment in ("HA", "NA", "HA+NA"):
            for subtype in SUBTYPES:
                vals = [row for row in rows if row["distance_proxy"] == proxy and row["segment"] == segment and row["subtype"] == subtype]
                rhos = np.asarray([row["rho"] for row in vals], dtype=float)
                aggregate.append(
                    {
                        "distance_proxy": proxy,
                        "segment": segment,
                        "subtype": subtype,
                        "rho_mean": float(np.nanmean(rhos)),
                        "rho_sd": float(np.nanstd(rhos, ddof=1)) if len(rhos) > 1 else 0.0,
                        "valid_pairs_mean": float(np.mean([row["valid_pairs"] for row in vals])),
                    }
                )

    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "runtime_seconds": float(time.time() - started),
        "inputs": {"cache_path": args.cache_path},
        "cache_metadata": metadata,
        "parameters": {
            "pairs_per_subtype": int(args.pairs_per_subtype),
            "seeds": seeds,
            "aligner": {
                "library": "Biopython PairwiseAligner",
                "mode": "global",
                "match_score": 1.0,
                "mismatch_score": 0.0,
                "open_gap_score": -2.0,
                "extend_gap_score": -0.5,
            },
        },
        "deduplication": {"deduplicated_n": int(len(Zd)), "removed_duplicates": removed},
        "rows": rows,
        "aggregate": aggregate,
        "limitations": [
            "This is a small pairwise global-alignment subsample, not a full MSA.",
            "It does not curate HA/NA protein reading frames, antigenic sites, insertions, deletions, or subtype-specific numbering.",
        ],
    }


def fmt(value: float, digits: int = 4) -> str:
    if value != value:
        return "NA"
    return f"{value:.{digits}f}"


def write_summary(payload: dict, path: Path) -> None:
    lines = [
        "# Alignment-Aware Robustness Panel",
        "",
        "This panel uses Biopython global pairwise alignments on a modest sampled set of deduplicated within-subtype pairs. It is a robustness check, not a curated multiple sequence alignment or protein-level antigenic analysis.",
        "",
        "| distance proxy | segment | subtype | rho mean | rho sd | valid pairs mean |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['distance_proxy']} | {row['segment']} | {row['subtype']} | "
            f"{fmt(row['rho_mean'])} | {fmt(row['rho_sd'])} | {row['valid_pairs_mean']:.0f} |"
        )
    lines.extend([
        "",
        "Interpretation: persistence of positive correlations under the aligned proxy supports that the scalable shared-prefix Hamming result is not solely a prefix-length artifact. The small sample and simple global alignment keep this as a limited robustness panel.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small alignment-aware robustness panel.")
    parser.add_argument("--cache-path", default="results/embeddings_cache_full_all_available.pkl")
    parser.add_argument("--output-json", default="results/alignment_robustness_results.json")
    parser.add_argument("--output-summary", default="results/alignment_robustness_summary.md")
    parser.add_argument("--pairs-per-subtype", type=int, default=250)
    parser.add_argument("--seeds", default="42,7,123")
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
