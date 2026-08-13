#!/usr/bin/env python3
"""Sequence-neighborhood baselines for the AntigenLM geometry paper.

This script compares sampled AntigenLM latent neighborhoods against simple
sequence-derived baselines on the exact HA+NA-deduplicated cache:

* k-mer TF-IDF cosine kNN.
* k-mer random projection to 384 dimensions.
* candidate-retrieved Hamming HA+NA kNN, where k-mer neighbors provide a
  finite candidate set that is reranked by exact shared-prefix Hamming.
* subtype-matched random assigned-clade records.

The output is aggregate JSON/Markdown plus a PDF/PNG summary figure. It does
not write sequences, accessions, isolate names, or accession-level metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
SUBTYPES = ("H1N1", "H3N2")


def clean_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a", "not assigned", "unassigned"}:
        return None
    return text


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


def deduplicate_by_ha_na(Z, years, months, subtypes, records):
    seen: set[tuple[str, str]] = set()
    keep: list[int] = []
    removed = {subtype: 0 for subtype in SUBTYPES}
    for i, record in enumerate(records):
        key = (record.get("ha_sequence", "") or "", record.get("na_sequence", "") or "")
        if key in seen:
            removed[str(subtypes[i])] = removed.get(str(subtypes[i]), 0) + 1
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
        removed,
    )


def load_joined_metadata(path: Path):
    by_epi: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epi = row.get("epi_isl") or row.get("Isolate_Id")
            if not epi:
                continue
            by_epi[epi] = {
                "matched": str(row.get("matched", "")).strip().lower() == "true",
                "clade": clean_label(row.get("clade") or row.get("Clade")),
            }
    return by_epi


def month_index(years, months):
    return years * 12 + np.clip(months, 1, 12)


def build_joined_arrays(records, subtypes, years, months, metadata_by_epi):
    labels = np.empty(len(records), dtype=object)
    matched = np.zeros(len(records), dtype=bool)
    for i, record in enumerate(records):
        item = metadata_by_epi.get(record.get("epi_isl", ""))
        labels[i] = item["clade"] if item else None
        matched[i] = bool(item and item["matched"])
    return {
        "subtypes": np.asarray(subtypes).astype(str),
        "years": np.asarray(years, dtype=np.int32),
        "months": np.asarray(months, dtype=np.int32),
        "month_indices": month_index(np.asarray(years, dtype=np.int32), np.asarray(months, dtype=np.int32)),
        "matched": matched,
        "clade": labels,
    }


def encode_labels(labels: np.ndarray):
    classes = sorted({value for value in labels if value is not None})
    mapping = {value: i for i, value in enumerate(classes)}
    codes = np.full(len(labels), -1, dtype=np.int32)
    for i, value in enumerate(labels):
        if value is not None:
            codes[i] = mapping[value]
    return codes, classes


def sequence_text(record: dict, k: int) -> str:
    # The N spacer prevents direct HA/NA junction k-mers composed solely from
    # adjacent segment ends while keeping the representation character based.
    return (record.get("ha_sequence", "") or "").upper() + ("N" * k) + (record.get("na_sequence", "") or "").upper()


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


def hamming_ha_na(records: list[dict], i: int, j: int) -> float:
    a = records[i]
    b = records[j]
    ha_i = a.get("ha_sequence", "") or ""
    ha_j = b.get("ha_sequence", "") or ""
    na_i = a.get("na_sequence", "") or ""
    na_j = b.get("na_sequence", "") or ""
    d_ha = normalized_hamming_arrays(ha_i, ha_j)
    d_na = normalized_hamming_arrays(na_i, na_j)
    if d_ha is None or d_na is None:
        return float("inf")
    ha_len = min(len(ha_i), len(ha_j))
    na_len = min(len(na_i), len(na_j))
    denom = ha_len + na_len
    if denom == 0:
        return float("inf")
    return float((d_ha * ha_len + d_na * na_len) / denom)


def remove_self_neighbors(local_neighbors: np.ndarray, query_positions: np.ndarray, max_k: int) -> np.ndarray:
    out = np.empty((len(query_positions), max_k), dtype=np.int64)
    for row, (query, neigh) in enumerate(zip(query_positions, local_neighbors)):
        filtered = [int(x) for x in neigh if int(x) != int(query)]
        if len(filtered) < max_k:
            filtered.extend([filtered[-1] if filtered else int(query)] * (max_k - len(filtered)))
        out[row] = filtered[:max_k]
    return out


def distribution(values: np.ndarray) -> dict:
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


def evaluate_neighbors(
    method: str,
    subtype: str,
    codes: np.ndarray,
    months: np.ndarray,
    query_positions: np.ndarray,
    neighbors: np.ndarray,
    k_values: list[int],
    extra: dict | None = None,
) -> list[dict]:
    rows = []
    for k in k_values:
        neigh = neighbors[:, :k]
        neigh_codes = codes[neigh]
        valid = neigh_codes >= 0
        denom = valid.sum(axis=1)
        same = (neigh_codes == codes[query_positions, None]) & valid
        usable = (codes[query_positions] >= 0) & (denom > 0)
        precision = same.sum(axis=1)[usable] / denom[usable]
        deltas = np.abs(months[query_positions, None] - months[neigh]).reshape(-1)
        row = {
            "method": method,
            "subtype": subtype,
            "k": int(k),
            "query_sample_n": int(len(query_positions)),
            "usable_queries": int(np.sum(usable)),
            "precision_mean": float(np.mean(precision)) if precision.size else float("nan"),
            "precision_median": float(np.median(precision)) if precision.size else float("nan"),
            "mean_valid_neighbors": float(np.mean(denom[usable])) if np.any(usable) else float("nan"),
            "temporal_months": distribution(deltas),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def random_neighbors(
    n_points: int,
    assigned: np.ndarray,
    query_positions: np.ndarray,
    max_k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.empty((len(query_positions), max_k), dtype=np.int64)
    for row, q in enumerate(query_positions):
        candidates = assigned[assigned != q]
        out[row] = rng.choice(candidates, size=max_k, replace=False)
    return out


def hamming_rerank_neighbors(
    records: list[dict],
    query_positions: np.ndarray,
    candidate_neighbors: np.ndarray,
    max_k: int,
) -> np.ndarray:
    out = np.empty((len(query_positions), max_k), dtype=np.int64)
    for row, query in enumerate(query_positions):
        scored = []
        seen = set()
        for cand in candidate_neighbors[row]:
            cand = int(cand)
            if cand == int(query) or cand in seen:
                continue
            seen.add(cand)
            scored.append((hamming_ha_na(records, int(query), cand), cand))
        scored.sort(key=lambda item: (item[0], item[1]))
        chosen = [cand for dist, cand in scored if np.isfinite(dist)]
        if len(chosen) < max_k:
            chosen.extend([cand for dist, cand in scored if not np.isfinite(dist)])
        if len(chosen) < max_k:
            chosen.extend([chosen[-1] if chosen else int(query)] * (max_k - len(chosen)))
        out[row] = chosen[:max_k]
    return out


def plot_precision(rows: list[dict], out_base: Path) -> None:
    rows5 = [row for row in rows if row["k"] == 5]
    methods = [
        "AntigenLM latent",
        "k-mer TF-IDF",
        "k-mer random projection",
        "candidate-retrieved Hamming",
        "subtype random",
    ]
    labels = {
        "AntigenLM latent": "AntigenLM",
        "k-mer TF-IDF": "k-mer",
        "k-mer random projection": "k-mer RP",
        "candidate-retrieved Hamming": "Hamming",
        "subtype random": "Random",
    }
    colors = ["#2f6f8f", "#5f9e6e", "#9a7bbd", "#d98c3a", "#b7b7b7"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    for ax, subtype in zip(axes, SUBTYPES):
        vals = []
        for method in methods:
            match = [row for row in rows5 if row["subtype"] == subtype and row["method"] == method]
            vals.append(match[0]["precision_mean"] if match else np.nan)
        ax.bar(np.arange(len(methods)), vals, color=colors, alpha=0.82)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels([labels[m] for m in methods], rotation=25, ha="right")
        ax.set_title(subtype)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Clade precision@5")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    Z, years, months, subtypes, records, metadata = load_cache(ROOT / args.cache_path)
    Zd, yd, md, td, rd, removed = deduplicate_by_ha_na(Z, years, months, subtypes, records)
    metadata_by_epi = load_joined_metadata(ROOT / args.metadata_join_path)
    joined = build_joined_arrays(rd, td, yd, md, metadata_by_epi)
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    max_k = max(k_values)
    rng = np.random.default_rng(args.seed)
    rows = []
    subtype_details = {}

    for subtype in SUBTYPES:
        print(f"[sequence-baselines] {subtype}")
        subtype_global = np.where(joined["subtypes"] == subtype)[0]
        X = Zd[subtype_global]
        subtype_records = [rd[int(i)] for i in subtype_global]
        subtype_months = joined["month_indices"][subtype_global]
        codes, classes = encode_labels(joined["clade"][subtype_global])
        assigned = np.where(codes >= 0)[0]
        query_n = min(args.max_queries_per_subtype, len(assigned))
        query_positions = np.sort(rng.choice(assigned, size=query_n, replace=False))
        print(f"  assigned={len(assigned):,} query_sample={len(query_positions):,} classes={len(classes)}")

        # AntigenLM latent neighbors.
        nn_latent = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean", algorithm="brute", n_jobs=-1)
        nn_latent.fit(X)
        _, latent_raw = nn_latent.kneighbors(X[query_positions])
        latent_neighbors = remove_self_neighbors(latent_raw, query_positions, max_k)
        rows.extend(evaluate_neighbors("AntigenLM latent", subtype, codes, subtype_months, query_positions, latent_neighbors, k_values))

        # k-mer TF-IDF baseline.
        texts = [sequence_text(record, args.kmer) for record in subtype_records]
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(args.kmer, args.kmer), lowercase=False, dtype=np.float32, norm="l2")
        Xk = vectorizer.fit_transform(texts)
        nn_kmer = NearestNeighbors(n_neighbors=max(max_k + 1, args.hamming_candidate_pool + 1), metric="cosine", algorithm="brute", n_jobs=-1)
        nn_kmer.fit(Xk)
        _, kmer_raw = nn_kmer.kneighbors(Xk[query_positions])
        kmer_neighbors = remove_self_neighbors(kmer_raw[:, : max_k + 1], query_positions, max_k)
        rows.extend(evaluate_neighbors("k-mer TF-IDF", subtype, codes, subtype_months, query_positions, kmer_neighbors, k_values, {"kmer": args.kmer}))

        # Candidate-retrieved Hamming baseline.
        candidate_neighbors = remove_self_neighbors(kmer_raw, query_positions, min(args.hamming_candidate_pool, kmer_raw.shape[1] - 1))
        hamming_neighbors = hamming_rerank_neighbors(subtype_records, query_positions, candidate_neighbors, max_k)
        rows.extend(
            evaluate_neighbors(
                "candidate-retrieved Hamming",
                subtype,
                codes,
                subtype_months,
                query_positions,
                hamming_neighbors,
                k_values,
                {"candidate_pool": int(candidate_neighbors.shape[1]), "candidate_retrieval": "k-mer TF-IDF cosine"},
            )
        )

        # Random projection of sequence features to 384 dimensions.
        rp = SparseRandomProjection(n_components=args.random_projection_dim, random_state=args.seed, dense_output=True)
        Xrp = rp.fit_transform(Xk)
        Xrp = normalize(Xrp, norm="l2", copy=False)
        nn_rp = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean", algorithm="brute", n_jobs=-1)
        nn_rp.fit(Xrp)
        _, rp_raw = nn_rp.kneighbors(Xrp[query_positions])
        rp_neighbors = remove_self_neighbors(rp_raw, query_positions, max_k)
        rows.extend(
            evaluate_neighbors(
                "k-mer random projection",
                subtype,
                codes,
                subtype_months,
                query_positions,
                rp_neighbors,
                k_values,
                {"kmer": args.kmer, "projection_dim": args.random_projection_dim},
            )
        )

        random_neigh = random_neighbors(len(subtype_global), assigned, query_positions, max_k, np.random.default_rng(args.seed + 100))
        rows.extend(evaluate_neighbors("subtype random", subtype, codes, subtype_months, query_positions, random_neigh, k_values))

        subtype_details[subtype] = {
            "deduplicated_records": int(len(subtype_global)),
            "assigned_clade_records": int(len(assigned)),
            "classes": int(len(classes)),
            "query_sample_n": int(query_n),
            "kmer": int(args.kmer),
            "kmer_vocabulary_size": int(len(vectorizer.vocabulary_)),
        }

    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "runtime_seconds": float(time.time() - started),
        "inputs": {
            "cache_path": args.cache_path,
            "metadata_join_path": args.metadata_join_path,
        },
        "cache_metadata": metadata,
        "parameters": {
            "seed": int(args.seed),
            "k_values": k_values,
            "max_queries_per_subtype": int(args.max_queries_per_subtype),
            "kmer": int(args.kmer),
            "hamming_candidate_pool": int(args.hamming_candidate_pool),
            "random_projection_dim": int(args.random_projection_dim),
        },
        "deduplication": {
            "deduplicated_n": int(len(Zd)),
            "removed_duplicates": removed,
        },
        "subtypes": subtype_details,
        "rows": rows,
        "interpretation": [
            "All metrics are computed on sampled assigned-clade query records from the exact HA+NA-deduplicated cache.",
            "The Hamming baseline is candidate-retrieved: k-mer TF-IDF cosine neighbors define the candidate pool, then candidates are reranked by exact shared-prefix HA+NA Hamming distance.",
            "The baselines use only sequence-derived information and therefore contextualize whether AntigenLM neighborhoods add to or recapitulate raw sequence structure.",
        ],
    }


def fmt(value: float, digits: int = 4) -> str:
    if value != value:
        return "NA"
    return f"{value:.{digits}f}"


def write_summary(payload: dict, path: Path) -> None:
    lines = [
        "# Sequence Baseline Neighborhood Results",
        "",
        "This analysis writes aggregate results only. It does not redistribute sequences, isolate identifiers, accessions, or accession-level metadata.",
        "",
        "## Parameters",
        "",
        f"- query sample per subtype: `{payload['parameters']['max_queries_per_subtype']}` assigned-clade records",
        f"- k values: `{payload['parameters']['k_values']}`",
        f"- nucleotide k-mer size: `{payload['parameters']['kmer']}`",
        f"- Hamming candidate pool: `{payload['parameters']['hamming_candidate_pool']}` k-mer neighbors",
        f"- random projection dimension: `{payload['parameters']['random_projection_dim']}`",
        "",
        "## Clade Precision and Temporal Neighborhoods",
        "",
        "| method | subtype | k | queries | precision@k | valid neighbors | median months | IQR months | p10-p90 months | <=6 mo |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in payload["rows"]:
        t = row["temporal_months"]
        lines.append(
            f"| {row['method']} | {row['subtype']} | {row['k']} | {row['query_sample_n']:,} | "
            f"{fmt(row['precision_mean'])} | {fmt(row['mean_valid_neighbors'], 2)} | "
            f"{fmt(t.get('median', float('nan')), 1)} | "
            f"{fmt(t.get('p25', float('nan')), 1)}--{fmt(t.get('p75', float('nan')), 1)} | "
            f"{fmt(t.get('p10', float('nan')), 1)}--{fmt(t.get('p90', float('nan')), 1)} | "
            f"{fmt(t.get('fraction_le_6_months', float('nan')), 3)} |"
        )
    lines.extend([
        "",
        "## Interpretation Guardrails",
        "",
        "- If AntigenLM matches or trails Hamming/k-mer neighborhoods, the correct interpretation is that it recapitulates substantial sequence-derived structure in a learned compressed representation.",
        "- If AntigenLM exceeds these baselines, the improvement is only under these local clade/temporal metrics and is not antigenic, phenotypic, or forecasting validation.",
        "- Candidate-retrieved Hamming is not an exhaustive all-vs-all Hamming kNN baseline; it is a practical approximation reranking a k-mer candidate set.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequence-neighborhood baselines.")
    parser.add_argument("--cache-path", default="results/embeddings_cache_full_all_available.pkl")
    parser.add_argument(
        "--metadata-join-path",
        default="data/gisaid_metadata_private/gisaid_epiflu_isolates_2000_2022_epi_set_260506bu_joined_dedup_cache.csv",
    )
    parser.add_argument("--output-json", default="results/sequence_baseline_results.json")
    parser.add_argument("--output-summary", default="results/sequence_baseline_summary.md")
    parser.add_argument("--figure-base", default="figures/latent_geometry_full/sequence_baseline_clade_precision")
    parser.add_argument("--k-values", default="5,10,20")
    parser.add_argument("--max-queries-per-subtype", type=int, default=2000)
    parser.add_argument("--kmer", type=int, default=5)
    parser.add_argument("--hamming-candidate-pool", type=int, default=500)
    parser.add_argument("--random-projection-dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = run(args)
    out_json = ROOT / args.output_json
    out_md = ROOT / args.output_summary
    fig_base = ROOT / args.figure_base
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(payload, out_md)
    plot_precision(payload["rows"], fig_base)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {fig_base.with_suffix('.pdf')}")
    print(f"Runtime seconds: {payload['runtime_seconds']:.1f}")


if __name__ == "__main__":
    main()
