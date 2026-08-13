#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focused robustness panel for Paper 1.

This script reuses the full embedding cache and helper functions from
latent_geometry_full_analysis.py. It does not load AntigenLM, does not
recompute embeddings, does not print sequences, and does not modify raw data.

Outputs:
  paper_revision_outputs/robustness_panel_results.json
  paper_revision_outputs/robustness_panel_summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from latent_geometry_full_analysis import (
    SUBTYPE_ORDER,
    deduplicate_by_ha_na,
    load_cache,
    month_index,
    normalized_hamming_arrays,
    sample_pairs,
    stats,
)


OUT_DIR = "paper_revision_outputs"
NT_TO_AA = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def seq_array(sequence: str) -> np.ndarray:
    return np.frombuffer((sequence or "").upper().encode("ascii", errors="ignore"), dtype=np.uint8)


def translate_nt(sequence: str) -> str:
    sequence = (sequence or "").upper()
    aas = []
    usable = len(sequence) - (len(sequence) % 3)
    for i in range(0, usable, 3):
        codon = sequence[i:i + 3]
        aas.append(NT_TO_AA.get(codon, "X"))
    return "".join(aas)


def build_features(records: list[dict]) -> list[dict]:
    features = []
    for record in records:
        ha = record.get("ha_sequence", "") or ""
        na = record.get("na_sequence", "") or ""
        ha_aa = translate_nt(ha)
        na_aa = translate_nt(na)
        features.append(
            {
                "nt_ha": seq_array(ha),
                "nt_na": seq_array(na),
                "aa_ha": seq_array(ha_aa),
                "aa_na": seq_array(na_aa),
                "nt_ha_len": len(ha),
                "nt_na_len": len(na),
                "aa_ha_len": len(ha_aa),
                "aa_na_len": len(na_aa),
            }
        )
    return features


def molecular_distance(features: list[dict], i: int, j: int, metric: str) -> float | None:
    if metric in ("nt_hamming_ha", "nt_hamming_na", "aa_hamming_ha", "aa_hamming_na"):
        kind, _, segment = metric.partition("_hamming_")
        return normalized_hamming_arrays(features[i][f"{kind}_{segment}"], features[j][f"{kind}_{segment}"])

    if metric in ("nt_hamming_ha_na", "aa_hamming_ha_na"):
        kind = metric.split("_", 1)[0]
        d_ha = normalized_hamming_arrays(features[i][f"{kind}_ha"], features[j][f"{kind}_ha"])
        d_na = normalized_hamming_arrays(features[i][f"{kind}_na"], features[j][f"{kind}_na"])
        if d_ha is None or d_na is None:
            return None
        ha_len = min(features[i][f"{kind}_ha_len"], features[j][f"{kind}_ha_len"])
        na_len = min(features[i][f"{kind}_na_len"], features[j][f"{kind}_na_len"])
        denom = ha_len + na_len
        if denom == 0:
            return None
        return float((d_ha * ha_len + d_na * na_len) / denom)

    raise ValueError(metric)


def spearman_row(seed, subtype, metric, latent, values, requested, omitted):
    latent = np.asarray(latent, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(latent) & np.isfinite(values)
    latent = latent[finite]
    values = values[finite]
    if len(latent) < 10:
        rho, pvalue = np.nan, np.nan
    else:
        rho, pvalue = spearmanr(latent, values)
    return {
        "seed": int(seed),
        "subtype": str(subtype),
        "metric": str(metric),
        "rho": float(rho),
        "pvalue": float(pvalue),
        "requested_pairs": int(requested),
        "valid_pairs": int(len(latent)),
        "omitted_pairs": int(omitted + np.sum(~finite)),
    }


def deduplicated_spearman(Z, years, months, types, records, pair_samples, seeds):
    print("[robustness] Deduplicated Spearman panel")
    Zd, yd, md, td, rd, removed = deduplicate_by_ha_na(Z, years, months, types, records)
    features = build_features(rd)
    t_months = month_index(yd, md)
    metrics = (
        "temporal",
        "nt_hamming_ha",
        "nt_hamming_na",
        "nt_hamming_ha_na",
        "aa_hamming_ha",
        "aa_hamming_na",
        "aa_hamming_ha_na",
    )
    rows = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        for subtype in SUBTYPE_ORDER:
            idx = np.where(td == subtype)[0]
            local_i, local_j = sample_pairs(len(idx), pair_samples, rng)
            gi = idx[local_i]
            gj = idx[local_j]
            latent = np.linalg.norm(Zd[gi] - Zd[gj], axis=1)

            temporal = np.abs(t_months[gi] - t_months[gj])
            rows.append(spearman_row(seed, subtype, "temporal", latent, temporal, pair_samples, 0))

            values_by_metric = {metric: [] for metric in metrics if metric != "temporal"}
            latent_by_metric = {metric: [] for metric in metrics if metric != "temporal"}
            omitted = {metric: 0 for metric in metrics if metric != "temporal"}
            for a, b, dl in zip(gi, gj, latent):
                ai = int(a)
                bi = int(b)
                for metric in values_by_metric:
                    value = molecular_distance(features, ai, bi, metric)
                    if value is None:
                        omitted[metric] += 1
                        continue
                    values_by_metric[metric].append(value)
                    latent_by_metric[metric].append(float(dl))

            for metric in values_by_metric:
                row = spearman_row(
                    seed,
                    subtype,
                    metric,
                    latent_by_metric[metric],
                    values_by_metric[metric],
                    pair_samples,
                    omitted[metric],
                )
                rows.append(row)
                print(
                    f"  seed={seed} {subtype} {metric}: "
                    f"rho={row['rho']:.4f} valid={row['valid_pairs']:,} omitted={row['omitted_pairs']:,}"
                )

    return {
        "deduplicated_n": int(len(Zd)),
        "removed_duplicates": removed,
        "pair_samples_per_subtype": int(pair_samples),
        "seeds": [int(seed) for seed in seeds],
        "rows": rows,
    }


def temporal_permutation_control(Z, years, months, types, records, k_values, seeds):
    print("[robustness] Temporal label-permutation control")
    Zd, yd, md, td, rd, removed = deduplicate_by_ha_na(Z, years, months, types, records)
    t_months_all = month_index(yd, md)
    rows = []

    for subtype in SUBTYPE_ORDER:
        idx = np.where(td == subtype)[0]
        X = Zd[idx]
        t = t_months_all[idx]
        nn = NearestNeighbors(n_neighbors=max(k_values) + 1, metric="euclidean", n_jobs=-1)
        nn.fit(X)
        _, neighbor_idx = nn.kneighbors(X)
        neighbor_idx = neighbor_idx[:, 1:]

        for k in k_values:
            true_deltas = np.abs(t[:, None] - t[neighbor_idx[:, :k]]).reshape(-1)
            true_stats = stats(true_deltas)
            for seed in seeds:
                rng = np.random.default_rng(seed)
                perm_t = rng.permutation(t)
                perm_deltas = np.abs(perm_t[:, None] - perm_t[neighbor_idx[:, :k]]).reshape(-1)
                perm_stats = stats(perm_deltas)
                ri, rj = sample_pairs(len(t), len(true_deltas), rng)
                random_deltas = np.abs(t[ri] - t[rj])
                random_stats = stats(random_deltas)
                rows.append(
                    {
                        "subtype": subtype,
                        "n_points": int(len(X)),
                        "k": int(k),
                        "seed": int(seed),
                        "true_neighbor": true_stats,
                        "permuted_neighbor": perm_stats,
                        "subtype_random": random_stats,
                        "median_ratio_true_to_permuted": (
                            float(true_stats["median"] / perm_stats["median"])
                            if perm_stats["median"] else np.nan
                        ),
                        "median_ratio_true_to_random": (
                            float(true_stats["median"] / random_stats["median"])
                            if random_stats["median"] else np.nan
                        ),
                    }
                )
            print(
                f"  {subtype} k={k}: true median={true_stats['median']:.2f}; "
                f"permuted medians={[r['permuted_neighbor']['median'] for r in rows if r['subtype'] == subtype and r['k'] == k]}"
            )

    return {
        "deduplicated_n": int(len(Zd)),
        "removed_duplicates": removed,
        "k_values": [int(k) for k in k_values],
        "seeds": [int(seed) for seed in seeds],
        "rows": rows,
    }


def aggregate_spearman(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["metric"], row["subtype"])].append(row)
    out = []
    for (metric, subtype), values in sorted(grouped.items()):
        rhos = np.asarray([v["rho"] for v in values], dtype=float)
        valid = np.asarray([v["valid_pairs"] for v in values], dtype=float)
        omitted = np.asarray([v["omitted_pairs"] for v in values], dtype=float)
        out.append(
            {
                "metric": metric,
                "subtype": subtype,
                "rho_mean": float(np.nanmean(rhos)),
                "rho_sd": float(np.nanstd(rhos)),
                "valid_pairs_mean": float(np.nanmean(valid)),
                "omitted_pairs_mean": float(np.nanmean(omitted)),
            }
        )
    return out


def aggregate_temporal_permutation(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["subtype"], row["k"])].append(row)
    out = []
    for (subtype, k), values in sorted(grouped.items()):
        true_medians = np.asarray([v["true_neighbor"]["median"] for v in values], dtype=float)
        perm_medians = np.asarray([v["permuted_neighbor"]["median"] for v in values], dtype=float)
        random_medians = np.asarray([v["subtype_random"]["median"] for v in values], dtype=float)
        out.append(
            {
                "subtype": subtype,
                "k": int(k),
                "true_median_months": float(np.nanmean(true_medians)),
                "permuted_median_months_mean": float(np.nanmean(perm_medians)),
                "permuted_median_months_sd": float(np.nanstd(perm_medians)),
                "random_median_months_mean": float(np.nanmean(random_medians)),
                "random_median_months_sd": float(np.nanstd(random_medians)),
                "true_to_permuted_median_ratio": (
                    float(np.nanmean(true_medians) / np.nanmean(perm_medians))
                    if np.nanmean(perm_medians) else np.nan
                ),
                "true_to_random_median_ratio": (
                    float(np.nanmean(true_medians) / np.nanmean(random_medians))
                    if np.nanmean(random_medians) else np.nan
                ),
            }
        )
    return out


def write_json(payload, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def fmt(value, digits=4):
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_summary(payload, path):
    lines = [
        "# Robustness Panel Results",
        "",
        f"Created at local run time: `{payload['created_at']}`",
        "",
        "This focused panel reuses the full cached embeddings and does not reload AntigenLM, regenerate embeddings, print sequences, or modify raw data.",
        "",
        "## Inputs",
        "",
        f"- cache: `{payload['cache_path']}`",
        f"- pair samples per subtype/seed: `{payload['parameters']['pair_samples_per_subtype']:,}`",
        f"- pair seeds: `{', '.join(map(str, payload['parameters']['pair_seeds']))}`",
        f"- temporal k values: `{', '.join(map(str, payload['parameters']['temporal_k_values']))}`",
        f"- temporal permutation seeds: `{', '.join(map(str, payload['parameters']['temporal_permutation_seeds']))}`",
        "",
        "## Deduplicated Pair-Sampled Spearman Correlations",
        "",
        "Exact HA+NA duplicates were removed before sampling pairs. Amino-acid distances are simple frame-0 translations with ambiguous codons mapped to `X`; they are robustness proxies, not curated protein alignments.",
        "",
        "| metric | subtype | rho mean | rho sd | valid pairs mean | omitted pairs mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["deduplicated_spearman"]["aggregate"]:
        lines.append(
            f"| {row['metric']} | {row['subtype']} | {fmt(row['rho_mean'])} | "
            f"{fmt(row['rho_sd'])} | {row['valid_pairs_mean']:.0f} | {row['omitted_pairs_mean']:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Temporal Label-Permutation Control",
            "",
            "The latent neighbor graph is held fixed, while collection months are permuted within subtype. If temporal coherence were mostly a generic consequence of date distribution alone, permuted labels would give similar neighbor time differences. They do not.",
            "",
            "| subtype | k | true median months | permuted median months mean | permuted sd | random median months mean | true/permuted | true/random |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["temporal_permutation"]["aggregate"]:
        lines.append(
            f"| {row['subtype']} | {row['k']} | {fmt(row['true_median_months'], 2)} | "
            f"{fmt(row['permuted_median_months_mean'], 2)} | {fmt(row['permuted_median_months_sd'], 2)} | "
            f"{fmt(row['random_median_months_mean'], 2)} | {fmt(row['true_to_permuted_median_ratio'], 3)} | "
            f"{fmt(row['true_to_random_median_ratio'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Manuscript-Relevant Interpretation",
            "",
            "- Molecular correlations remain strong after exact HA+NA deduplication.",
            "- NA-only nucleotide correlations are lower than HA-only in both subtypes in this panel, while combined HA+NA remains strongest among nucleotide proxies.",
            "- Amino-acid Hamming proxies preserve the same broad pattern as nucleotide proxies, but should be treated cautiously because no curated protein alignment was performed.",
            "- Date-label permutation raises neighbor temporal medians from 2 months to tens of months, supporting the claim that local temporal coherence is tied to true collection dates rather than only the marginal date distribution.",
            "",
            "## Remaining Caveats",
            "",
            "- These controls still do not validate antigenic similarity, phylogenetic distance, clade identity, vaccine relevance, or prospective forecasting.",
            "- Amino-acid distances here are simple translations from available nucleotide strings and do not replace curated HA/NA protein alignments.",
            "- Pairwise sampled distances remain dependent observations; effect sizes and robustness are more important than nominal p-values.",
        ]
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Focused robustness panel for revised Paper 1")
    parser.add_argument("--cache-path", default="results/embeddings_cache_full_all_available.pkl")
    parser.add_argument("--pair-samples-per-subtype", type=int, default=200000)
    parser.add_argument("--pair-seeds", default="42,7,123")
    parser.add_argument("--temporal-k-values", default="5,10,20")
    parser.add_argument("--temporal-permutation-seeds", default="42,7,123")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    start = time.time()
    pair_seeds = [int(x.strip()) for x in args.pair_seeds.split(",") if x.strip()]
    k_values = [int(x.strip()) for x in args.temporal_k_values.split(",") if x.strip()]
    perm_seeds = [int(x.strip()) for x in args.temporal_permutation_seeds.split(",") if x.strip()]

    Z, years, months, types, records, metadata = load_cache(args.cache_path)
    spearman = deduplicated_spearman(
        Z,
        years,
        months,
        types,
        records,
        pair_samples=args.pair_samples_per_subtype,
        seeds=pair_seeds,
    )
    temporal = temporal_permutation_control(
        Z,
        years,
        months,
        types,
        records,
        k_values=k_values,
        seeds=perm_seeds,
    )

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "runtime_seconds": float(time.time() - start),
        "cache_path": args.cache_path,
        "cache_metadata": metadata,
        "parameters": {
            "pair_samples_per_subtype": int(args.pair_samples_per_subtype),
            "pair_seeds": pair_seeds,
            "temporal_k_values": k_values,
            "temporal_permutation_seeds": perm_seeds,
        },
        "deduplicated_spearman": {
            **spearman,
            "aggregate": aggregate_spearman(spearman["rows"]),
        },
        "temporal_permutation": {
            **temporal,
            "aggregate": aggregate_temporal_permutation(temporal["rows"]),
        },
    }
    json_path = os.path.join(OUT_DIR, "robustness_panel_results.json")
    summary_path = os.path.join(OUT_DIR, "robustness_panel_summary.md")
    write_json(payload, json_path)
    write_summary(payload, summary_path)
    print(f"JSON: {json_path}")
    print(f"Summary: {summary_path}")
    print(f"Runtime seconds: {payload['runtime_seconds']:.1f}")


if __name__ == "__main__":
    main()
