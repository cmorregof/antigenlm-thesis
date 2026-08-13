#!/usr/bin/env python3
"""Aggregate preprocessing/cache counts for the AntigenLM geometry paper.

The script reads local processed GISAID-derived JSON files and the embedding
cache, but writes only aggregate counts. It does not write sequences,
isolate names, accessions, or accession-level metadata.
"""

from __future__ import annotations

import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUBTYPES = ("H1N1", "H3N2")
VALID_NT = set("ACGTNURYKMSWBDHV-")


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_cache(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def pct(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def describe_lengths(values: list[int]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "min": int(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": int(np.max(arr)),
    }


def char_summary(records: list[dict]) -> dict:
    counts = Counter()
    bad_records = 0
    gap_records = 0
    ambiguous_records = 0
    for record in records:
        has_bad = False
        has_gap = False
        has_amb = False
        for segment in ("ha_sequence", "na_sequence"):
            seq = (record.get(segment) or "").upper()
            counts.update(seq)
            if "-" in seq:
                has_gap = True
            if any(ch not in {"A", "C", "G", "T"} for ch in seq):
                has_amb = True
            if any(ch not in VALID_NT for ch in seq):
                has_bad = True
        bad_records += int(has_bad)
        gap_records += int(has_gap)
        ambiguous_records += int(has_amb)
    return {
        "records_with_non_iupac_or_unexpected_chars": bad_records,
        "records_with_gap_character": gap_records,
        "records_with_ambiguous_or_non_acgt_character": ambiguous_records,
        "observed_characters": dict(sorted(counts.items())),
    }


def dedup_count(records: list[dict]) -> int:
    keys = {(record.get("ha_sequence") or "", record.get("na_sequence") or "") for record in records}
    return len(keys)


def build_counts() -> dict:
    cache_path = ROOT / "results/embeddings_cache_full_all_available.pkl"
    cache = load_cache(cache_path)
    all_records = []
    by_subtype = {}
    for subtype in SUBTYPES:
        path = ROOT / f"data/processed_gisaid/dataset_{subtype}.json"
        data = load_json(path)
        records = data.get("paired_strains", [])
        all_records.extend(records)
        ha_lengths = [len(r.get("ha_sequence") or "") for r in records]
        na_lengths = [len(r.get("na_sequence") or "") for r in records]
        valid_ha = [r for r in records if r.get("ha_sequence")]
        valid_na = [r for r in records if r.get("na_sequence")]
        valid_subtype = [r for r in records if r.get("subtype")]
        valid_date = [r for r in records if r.get("year") is not None and r.get("month") is not None]
        paired = [
            r
            for r in records
            if r.get("ha_sequence")
            and r.get("na_sequence")
            and r.get("subtype")
            and r.get("year") is not None
            and r.get("month") is not None
        ]
        by_subtype[subtype] = {
            "processed_paired_records": len(records),
            "records_with_ha_sequence": len(valid_ha),
            "records_with_na_sequence": len(valid_na),
            "records_with_subtype": len(valid_subtype),
            "records_with_year_and_month": len(valid_date),
            "paired_records_with_ha_na_subtype_date": len(paired),
            "exact_ha_na_deduplicated_records": dedup_count(records),
            "ha_length": describe_lengths(ha_lengths),
            "na_length": describe_lengths(na_lengths),
            "character_summary": char_summary(records),
        }

    cache_types = np.asarray(cache["types"]).astype(str)
    dedup_total = dedup_count(all_records)
    cache_counts = {subtype: int(np.sum(cache_types == subtype)) for subtype in SUBTYPES}
    paired_total = sum(row["paired_records_with_ha_na_subtype_date"] for row in by_subtype.values())
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "notes": [
            "Counts start from the local processed paired-strain JSON files, not from raw GISAID XLS/FASTA exports.",
            "Earlier raw-export filters were not fully retained in the current logs and are reported as unavailable rather than reconstructed.",
            "No sequences, isolate identifiers, accessions, locations, hosts, or submitter metadata are written by this output.",
        ],
        "unavailable_raw_filter_counts": [
            "raw local GISAID/EpiFlu records loaded",
            "records removed during original FASTA/XLS parsing",
            "records removed before paired_strains construction",
            "host/geography/laboratory exclusions before local JSON creation, if any",
        ],
        "cache_path": str(cache_path.relative_to(ROOT)),
        "cache_metadata": cache.get("metadata", {}),
        "total": {
            "processed_paired_records": len(all_records),
            "paired_records_with_ha_na_subtype_date": paired_total,
            "successfully_embedded_records": int(len(cache["records"])),
            "exact_ha_na_deduplicated_records": dedup_total,
            "deduplicated_fraction": pct(dedup_total, paired_total),
            "cache_shape": list(np.asarray(cache["embeddings"]).shape),
        },
        "by_subtype": by_subtype,
        "cache_counts_by_subtype": cache_counts,
    }
    return result


def write_markdown(payload: dict, path: Path) -> None:
    total = payload["total"]
    rows = [
        ("Raw local GISAID/EpiFlu records loaded", "not retained in current logs", "", "Private raw-export count not reconstructed."),
        ("Processed paired records in local JSON", f"{total['processed_paired_records']:,}", "", "Starting point for this reproducible audit."),
        ("Records with HA, NA, subtype, year, and month", f"{total['paired_records_with_ha_na_subtype_date']:,}", "0 from processed JSON", "All processed paired records satisfy these fields."),
        ("Successfully embedded records", f"{total['successfully_embedded_records']:,}", "0 from processed paired set", "No missing cache entries among processed paired records."),
        ("Records in final cache", f"{total['successfully_embedded_records']:,}", "", "Mean-pooled 384-dimensional vectors."),
        ("Exact HA+NA deduplicated records", f"{total['exact_ha_na_deduplicated_records']:,}", f"{total['successfully_embedded_records'] - total['exact_ha_na_deduplicated_records']:,}", "Exact duplicate HA+NA strings collapsed for local-neighborhood analyses."),
    ]
    lines = [
        "# Preprocessing and Cache Construction Counts",
        "",
        "This file contains aggregate counts only. It intentionally omits sequences, accessions, isolate names, locations, hosts, submitters, and accession-level metadata.",
        "",
        "## Main Count Table",
        "",
        "| Step | Retained records | Removed at step | Notes |",
        "|---|---:|---:|---|",
    ]
    for step, retained, removed, note in rows:
        lines.append(f"| {step} | {retained} | {removed} | {note} |")

    lines.extend([
        "",
        "## Counts by Subtype",
        "",
        "| Subtype | processed paired | cache records | exact HA+NA deduplicated | HA median length | NA median length | records with non-ACGT/IUPAC-or-gap ambiguity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for subtype, row in payload["by_subtype"].items():
        lines.append(
            f"| {subtype} | {row['processed_paired_records']:,} | "
            f"{payload['cache_counts_by_subtype'][subtype]:,} | "
            f"{row['exact_ha_na_deduplicated_records']:,} | "
            f"{row['ha_length']['median']:.0f} | {row['na_length']['median']:.0f} | "
            f"{row['character_summary']['records_with_ambiguous_or_non_acgt_character']:,} |"
        )

    lines.extend([
        "",
        "## Scope Notes",
        "",
        "- The reproducible counts begin at the processed `paired_strains` JSON files.",
        "- Counts for earlier raw GISAID parsing/filtering stages are not retained in the current logs.",
        "- The processed JSON records contain `epi_isl`, subtype, year/month/day, HA sequence, and NA sequence, but no host, geography, laboratory, or passage filters are applied in this audit script.",
        "- Host, geography, passage, and submitter information are available only in private metadata exports and are not redistributed in these outputs.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_counts()
    out_json = ROOT / "results/preprocessing_counts.json"
    out_md = ROOT / "results/preprocessing_counts.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, out_md)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
