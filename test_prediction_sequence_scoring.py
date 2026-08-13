#!/usr/bin/env python3
"""
Read-only conditional scoring smoke test for the local AntigenLM prediction checkpoint.

This script does not train, generate, optimize, mutate, print, or save sequences.
It scores small candidate sets with a causal LM objective over an explicitly
reported continuation prefix, using a bounded historical context so the audit is
small enough to run on CPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from antigen_model import GPTForFluMultiTask
from influ_tokenizer import InfluTokenizer


ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "prediction_sequence"
DATA_DIR = ROOT / "data" / "processed_gisaid"
REPORT_PATH = ROOT / "results" / "prediction_sequence_scoring_smoke_summary.md"
SUBTYPES = ("H1N1", "H3N2")
N_POSITIONS_MAX = 13000


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(cell).replace("\n", "<br>") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def remap_transformer_to_backbone(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("transformer."):
            remapped[key.replace("transformer.", "backbone.", 1)] = value
        else:
            remapped[key] = value
    return remapped


def is_causal_mask_buffer(key: str) -> bool:
    return "attn.bias" in key or "attn.masked_bias" in key


def month_index(year: int, month: int) -> int:
    return int(year) * 12 + int(month) - 1


def month_label(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def record_fingerprint(record: dict[str, Any]) -> str:
    payload = (record["ha_sequence"] + "|" + record["na_sequence"]).encode("ascii", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


def token_fingerprint(token_ids: list[int]) -> str:
    payload = ",".join(str(int(token)) for token in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:12]


def find_dataset_path(subtype: str) -> Path | None:
    exact = DATA_DIR / f"dataset_{subtype}.json"
    if exact.exists():
        return exact
    matches = sorted(DATA_DIR.glob(f"*{subtype}*.json"))
    return matches[0] if matches else None


def load_dataset(subtype: str) -> tuple[Path, dict[str, Any]] | None:
    path = find_dataset_path(subtype)
    if path is None:
        return None
    return path, read_json(path)


def encode_record(
    tokenizer: InfluTokenizer,
    record: dict[str, Any],
    subtype_token: str,
) -> list[int]:
    return tokenizer.encode_strain(
        ha_sequence=record["ha_sequence"],
        na_sequence=record["na_sequence"],
        subtype=subtype_token,
        add_eos=True,
    )


def encode_record_bounded(
    tokenizer: InfluTokenizer,
    record: dict[str, Any],
    subtype_token: str,
    max_tokens: int | None,
) -> list[int]:
    full = encode_record(tokenizer, record, subtype_token)
    if max_tokens is None or len(full) <= max_tokens:
        return full
    if max_tokens < 12:
        raise ValueError("context-token-budget debe ser al menos 12")

    sequence_budget = max_tokens - 5
    ha_budget = max(1, int(sequence_budget * 0.55))
    na_budget = max(1, sequence_budget - ha_budget)
    return tokenizer.encode_strain(
        ha_sequence=record["ha_sequence"][:ha_budget],
        na_sequence=record["na_sequence"][:na_budget],
        subtype=subtype_token,
        add_eos=True,
    )


def build_month_records(dataset: dict[str, Any], subtype: str) -> dict[int, list[dict[str, Any]]]:
    records_by_month: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source_idx, record in enumerate(dataset.get("paired_strains", [])):
        if not record.get("ha_sequence") or not record.get("na_sequence"):
            continue
        idx = month_index(record["year"], record["month"])
        records_by_month[idx].append(
            {
                "subtype": subtype,
                "source_idx": int(source_idx),
                "record": record,
                "year": int(record["year"]),
                "month": int(record["month"]),
                "month_index": idx,
                "fingerprint": record_fingerprint(record),
            }
        )
    return records_by_month


def get_record_info(dataset: dict[str, Any], subtype: str, source_idx: int) -> dict[str, Any]:
    record = dataset["paired_strains"][int(source_idx)]
    return {
        "subtype": subtype,
        "source_idx": int(source_idx),
        "record": record,
        "year": int(record["year"]),
        "month": int(record["month"]),
        "month_index": month_index(record["year"], record["month"]),
        "fingerprint": record_fingerprint(record),
    }


def select_windows(dataset: dict[str, Any], num_windows: int, seed: int, debug_one: bool) -> list[tuple[int, dict[str, Any]]]:
    windows = [
        (idx, window)
        for idx, window in enumerate(dataset.get("windows", []))
        if window.get("context") and window.get("target")
    ]
    if debug_one:
        return windows[:1]
    rng = random.Random(seed)
    if len(windows) <= num_windows:
        return windows
    return sorted(rng.sample(windows, num_windows), key=lambda item: item[0])


def build_context_ids(
    tokenizer: InfluTokenizer,
    dataset: dict[str, Any],
    window: dict[str, Any],
    subtype_token: str,
    context_strains: int,
    context_token_budget: int | None,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    context_items = window["context"][-context_strains:]
    context_ids: list[int] = []
    context_records = []
    per_strain_budget = None
    if context_token_budget is not None:
        per_strain_budget = max(12, context_token_budget // max(1, len(context_items)))

    full_context_token_count = 0
    for item in context_items:
        info = get_record_info(dataset, dataset.get("subtype", "unknown"), int(item["strain_idx"]))
        full_ids = encode_record(tokenizer, info["record"], subtype_token)
        bounded_ids = encode_record_bounded(tokenizer, info["record"], subtype_token, per_strain_budget)
        full_context_token_count += len(full_ids)
        context_ids.extend(bounded_ids)
        context_records.append(info)

    return context_ids, context_records, {
        "context_strains_used": len(context_items),
        "full_context_tokens": full_context_token_count,
        "bounded_context_tokens": len(context_ids),
        "per_strain_budget": per_strain_budget,
    }


def deduplicate_keep_target(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    result = []
    removed = 0
    for candidate in candidates:
        key = candidate["fingerprint"]
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        result.append(candidate)
    return result, removed


def build_candidate_set(
    dataset: dict[str, Any],
    subtype: str,
    window: dict[str, Any],
    month_records: dict[int, list[dict[str, Any]]],
    max_candidates: int,
    seed: int,
    deduplicate_candidates: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    target_idx = int(window["target"]["strain_idx"])
    target = get_record_info(dataset, subtype, target_idx)
    target_month_idx = target["month_index"]

    selected = [{**target, "is_target": True, "anon_id": "candidate_000"}]
    source_seen = {target_idx}
    same_month_pool = [
        record for record in month_records.get(target_month_idx, [])
        if record["source_idx"] != target_idx
    ]
    rng.shuffle(same_month_pool)

    for record in same_month_pool:
        if len(selected) >= max_candidates:
            break
        if record["source_idx"] in source_seen:
            continue
        source_seen.add(record["source_idx"])
        selected.append({**record, "is_target": False, "anon_id": f"candidate_{len(selected):03d}"})

    if len(selected) < max_candidates:
        nearby_records = []
        for idx, records in month_records.items():
            if idx == target_month_idx:
                continue
            for record in records:
                nearby_records.append((abs(idx - target_month_idx), idx, record))
        nearby_records.sort(key=lambda item: (item[0], item[1]))
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for distance, _idx, record in nearby_records:
            grouped[distance].append(record)
        for distance in sorted(grouped):
            batch = grouped[distance]
            rng.shuffle(batch)
            for record in batch:
                if len(selected) >= max_candidates:
                    break
                if record["source_idx"] in source_seen:
                    continue
                source_seen.add(record["source_idx"])
                selected.append({**record, "is_target": False, "anon_id": f"candidate_{len(selected):03d}"})
            if len(selected) >= max_candidates:
                break

    duplicate_exact_before = len(selected) - len({candidate["fingerprint"] for candidate in selected})
    duplicates_removed = 0
    if deduplicate_candidates:
        selected, duplicates_removed = deduplicate_keep_target(selected)

    for idx, candidate in enumerate(selected):
        candidate["anon_id"] = f"candidate_{idx:03d}"

    offsets = Counter(candidate["month_index"] - target_month_idx for candidate in selected)
    meta = {
        "target_month_idx": target_month_idx,
        "target_label": month_label(target["year"], target["month"]),
        "same_month_available": len(same_month_pool) + 1,
        "duplicate_exact_before": duplicate_exact_before,
        "duplicates_removed": duplicates_removed,
        "temporal_offsets": dict(sorted(offsets.items())),
    }
    return selected, meta


def score_conditional_full_forward(
    model: GPTForFluMultiTask,
    context_ids: list[int],
    continuation_ids: list[int],
    device: torch.device,
) -> dict[str, Any]:
    if not context_ids:
        raise ValueError("context_ids no puede estar vacio")
    if not continuation_ids:
        raise ValueError("continuation_ids no puede estar vacio")

    total_len = len(context_ids) + len(continuation_ids)
    if total_len > N_POSITIONS_MAX:
        raise ValueError(f"context + continuation excede n_positions: {total_len} > {N_POSITIONS_MAX}")

    with torch.inference_mode():
        full_ids = context_ids + continuation_ids
        c_len = len(context_ids)
        l_len = len(continuation_ids)
        x = torch.tensor([full_ids], dtype=torch.long, device=device)
        out = model(input_ids=x)
        logits = out["logits"]
        pred_logits = logits[:, c_len - 1 : c_len + l_len - 1, :]
        labels = torch.tensor([continuation_ids], dtype=torch.long, device=device)
        total_nll_tensor = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)).float(),
            labels.reshape(-1),
            reduction="sum",
        )
        total_nll = float(total_nll_tensor.detach().cpu())
        logits_has_nan = bool(torch.isnan(pred_logits.detach().float()).any().item())
        del x, out, logits, pred_logits, labels, total_nll_tensor

    mean_nll = total_nll / len(continuation_ids)
    return {
        "num_tokens": len(continuation_ids),
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if math.isfinite(mean_nll) else float("nan"),
        "logits_has_nan": logits_has_nan,
    }


def score_window(
    model: GPTForFluMultiTask,
    tokenizer: InfluTokenizer,
    device: torch.device,
    dataset: dict[str, Any],
    subtype: str,
    window_idx: int,
    window: dict[str, Any],
    month_records: dict[int, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    subtype_token = f"<{subtype}>"
    context_ids, context_records, context_meta = build_context_ids(
        tokenizer=tokenizer,
        dataset=dataset,
        window=window,
        subtype_token=subtype_token,
        context_strains=args.context_strains,
        context_token_budget=args.context_token_budget,
    )
    candidates, candidate_meta = build_candidate_set(
        dataset=dataset,
        subtype=subtype,
        window=window,
        month_records=month_records,
        max_candidates=args.max_candidates,
        seed=args.seed + window_idx,
        deduplicate_candidates=args.deduplicate_candidates,
    )

    context_hashes = {record["fingerprint"] for record in context_records}
    overlap_exact = sum(1 for candidate in candidates if candidate["fingerprint"] in context_hashes)
    scored_rows = []
    failures = []
    t0 = time.time()

    for candidate in candidates:
        full_ids = encode_record(tokenizer, candidate["record"], subtype_token)
        continuation_ids = full_ids[: args.max_score_tokens] if args.max_score_tokens else full_ids
        try:
            stats = score_conditional_full_forward(
                model=model,
                context_ids=context_ids,
                continuation_ids=continuation_ids,
                device=device,
            )
            scored_rows.append(
                {
                    **candidate,
                    "full_tokens": len(full_ids),
                    "scored_tokens": len(continuation_ids),
                    **stats,
                }
            )
        except Exception as exc:
            failures.append({"anon_id": candidate["anon_id"], "error": repr(exc), "is_target": candidate["is_target"]})
        finally:
            clear_device_cache(device)

    scoring_seconds = time.time() - t0
    valid_rows = [
        row for row in scored_rows
        if math.isfinite(row.get("mean_nll", float("nan"))) and not row.get("logits_has_nan", False)
    ]
    by_mean = sorted(valid_rows, key=lambda row: row["mean_nll"])
    target_rank = next((rank for rank, row in enumerate(by_mean, start=1) if row["is_target"]), None)
    num_valid = len(valid_rows)
    normalized_rank = (
        (target_rank - 1) / (num_valid - 1)
        if target_rank is not None and num_valid > 1
        else float("nan")
    )
    percentile_score = 1.0 - normalized_rank if math.isfinite(normalized_rank) else float("nan")
    reciprocal_rank = 1.0 / target_rank if target_rank else float("nan")
    target_row = next((row for row in valid_rows if row["is_target"]), None)
    top_row = by_mean[0] if by_mean else None

    return {
        "subtype": subtype,
        "window_idx": window_idx,
        "window_id": token_fingerprint([
            window_idx,
            int(window["target"]["strain_idx"]),
            int(window["target"]["year"]),
            int(window["target"]["month"]),
        ]),
        "target_month": candidate_meta["target_label"],
        "num_candidates": len(candidates),
        "num_valid": num_valid,
        "target_rank": target_rank,
        "top1": target_rank == 1,
        "top5": target_rank is not None and target_rank <= min(5, num_valid),
        "normalized_rank": normalized_rank,
        "percentile_score": percentile_score,
        "reciprocal_rank": reciprocal_rank,
        "duplicates_removed": candidate_meta["duplicates_removed"],
        "duplicate_exact_before": candidate_meta["duplicate_exact_before"],
        "overlap_exact_with_context": overlap_exact,
        "temporal_offsets": candidate_meta["temporal_offsets"],
        "same_month_available": candidate_meta["same_month_available"],
        "context_tokens": len(context_ids),
        "full_context_tokens": context_meta["full_context_tokens"],
        "context_strains_used": context_meta["context_strains_used"],
        "scored_tokens_target": target_row["scored_tokens"] if target_row else None,
        "full_tokens_target": target_row["full_tokens"] if target_row else None,
        "target_mean_nll": target_row["mean_nll"] if target_row else float("nan"),
        "top1_is_target": bool(top_row["is_target"]) if top_row else False,
        "top1_mean_nll": top_row["mean_nll"] if top_row else float("nan"),
        "top1_month_offset": (top_row["month_index"] - candidate_meta["target_month_idx"]) if top_row else None,
        "nll_min": float(np.min([row["mean_nll"] for row in valid_rows])) if valid_rows else float("nan"),
        "nll_median": float(np.median([row["mean_nll"] for row in valid_rows])) if valid_rows else float("nan"),
        "nll_max": float(np.max([row["mean_nll"] for row in valid_rows])) if valid_rows else float("nan"),
        "failures": failures,
        "nan_count": sum(1 for row in scored_rows if row.get("logits_has_nan", False)),
        "scoring_seconds": scoring_seconds,
    }


def load_prediction_model(device: torch.device) -> tuple[GPTForFluMultiTask, dict[str, Any]]:
    config = read_json(CKPT_DIR / "config.json")
    state_dict = load_state_dict(CKPT_DIR / "pytorch_model.bin")
    model = GPTForFluMultiTask(task="prediction")
    missing, unexpected = model.load_state_dict(remap_transformer_to_backbone(state_dict), strict=False)
    unexplained = [key for key in unexpected if not is_causal_mask_buffer(key)]
    del state_dict
    gc.collect()
    model.eval().to(device)
    return model, {
        "config": config,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "unexpected_explained": len(unexplained) == 0,
        "params": sum(parameter.numel() for parameter in model.parameters()),
    }


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("target_rank") is not None and row.get("num_valid", 0) > 1]
    if not valid:
        return {
            "windows_evaluated": 0,
            "top1_accuracy": float("nan"),
            "top5_accuracy": float("nan"),
            "mean_normalized_rank": float("nan"),
            "mrr": float("nan"),
        }
    return {
        "windows_evaluated": len(valid),
        "top1_accuracy": float(np.mean([row["top1"] for row in valid])),
        "top5_accuracy": float(np.mean([row["top5"] for row in valid])),
        "mean_normalized_rank": float(np.mean([row["normalized_rank"] for row in valid])),
        "mrr": float(np.mean([row["reciprocal_rank"] for row in valid])),
    }


def status_label(agg: dict[str, Any]) -> str:
    if not agg["windows_evaluated"] or not math.isfinite(agg["mean_normalized_rank"]):
        return "fallo informativo"
    if agg["top1_accuracy"] > 0 and agg["top5_accuracy"] >= 0.67 and agg["mean_normalized_rank"] < 0.5:
        return "exito fuerte"
    if agg["mean_normalized_rank"] < 0.5 or agg["top5_accuracy"] > 0:
        return "exito parcial"
    return "fallo informativo"


def build_report(
    args: argparse.Namespace,
    device: torch.device,
    model_info: dict[str, Any],
    dataset_paths: dict[str, Path],
    rows: list[dict[str, Any]],
    agg: dict[str, Any],
) -> str:
    config = model_info["config"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    window_rows = []
    for row in rows:
        window_rows.append(
            [
                row["subtype"],
                row["target_month"],
                row["num_valid"],
                row["target_rank"] if row["target_rank"] is not None else "NA",
                row["top1"],
                row["top5"],
                fmt_float(row["normalized_rank"]),
                fmt_float(row["percentile_score"]),
                row["duplicates_removed"],
                row["overlap_exact_with_context"],
                row["temporal_offsets"],
                f"{fmt_float(row['target_mean_nll'])}/{fmt_float(row['top1_mean_nll'])}",
            ]
        )

    checkpoint_rows = [
        ["checkpoint", str(CKPT_DIR.relative_to(ROOT))],
        ["device", str(device)],
        ["vocab_size", config.get("vocab_size")],
        ["n_layer/n_head/n_embd/n_positions", f"{config.get('n_layer')}/{config.get('n_head')}/{config.get('n_embd')}/{config.get('n_positions')}"],
        ["params", f"{model_info['params']:,}"],
        ["missing keys", len(model_info["missing"])],
        ["unexpected keys", len(model_info["unexpected"])],
        ["unexpected explicadas como buffers mascara causal", model_info["unexpected_explained"]],
    ]

    run_rows = [
        ["num_windows por subtipo", args.num_windows],
        ["debug_one", args.debug_one],
        ["max_candidates", args.max_candidates],
        ["context_strains", args.context_strains],
        ["context_token_budget total", args.context_token_budget],
        ["max_score_tokens por candidata", args.max_score_tokens],
        ["deduplicate_candidates", args.deduplicate_candidates],
        ["seed", args.seed],
    ]

    agg_rows = [
        ["ventanas evaluadas", agg["windows_evaluated"]],
        ["top-1 accuracy", fmt_float(agg["top1_accuracy"])],
        ["top-5 accuracy", fmt_float(agg["top5_accuracy"])],
        ["mean normalized rank", fmt_float(agg["mean_normalized_rank"])],
        ["MRR", fmt_float(agg["mrr"])],
        ["estado", status_label(agg)],
    ]

    dataset_rows = [
        [subtype, str(path.relative_to(ROOT))]
        for subtype, path in dataset_paths.items()
    ]

    lines = [
        "# Smoke test de scoring condicional para `prediction_sequence`",
        "",
        f"- Fecha local de ejecucion: `{now}`",
        f"- Script: `test_prediction_sequence_scoring.py`",
        "",
        "## Objetivo",
        "",
        "Verificar si el checkpoint `prediction_sequence/` asigna mejor likelihood condicional a candidatas reales del mes objetivo que a candidatas negativas, usando una muestra pequena y read-only. Esto no es una reproduccion completa de Figure 3A.",
        "",
        "## Restricciones cumplidas",
        "",
        "- No se entrenaron modelos.",
        "- No se modificaron pesos.",
        "- No se generaron secuencias nuevas.",
        "- No se uso generacion libre.",
        "- No se optimizaron ni mutaron secuencias.",
        "- No se imprimieron ni guardaron secuencias completas.",
        "- No se tocaron archivos `.tex`.",
        "- No se ejecuto evaluacion masiva.",
        "",
        "## Configuracion del checkpoint",
        "",
        markdown_table(["campo", "valor"], checkpoint_rows),
        "",
        "## Configuracion de ventanas y scoring",
        "",
        markdown_table(["campo", "valor"], run_rows),
        "",
        "Nota metodologica: por defecto este smoke test usa contexto historico estructurado acotado y puntua un prefijo de continuacion. Esto reduce costo y riesgo operacional; no debe interpretarse como el protocolo completo del paper.",
        "",
        "## Datasets detectados",
        "",
        markdown_table(["subtipo", "archivo"], dataset_rows),
        "",
        "## Tabla por ventana",
        "",
        markdown_table(
            [
                "subtipo",
                "mes objetivo",
                "candidatas",
                "target rank",
                "top-1",
                "top-5",
                "normalized rank",
                "percentile score",
                "dups removidos",
                "overlap contexto",
                "offsets meses",
                "mean_nll target/top1",
            ],
            window_rows,
        ),
        "",
        "## Resumen agregado",
        "",
        markdown_table(["metrica", "valor"], agg_rows),
        "",
        "## Interpretacion prudente",
        "",
        "Si el target rankea alto, el checkpoint contiene senal condicional util bajo este formato local de contexto/candidatas. Si no rankea alto, no basta para concluir que el checkpoint falla: puede deberse al contexto acotado, al prefijo puntuado, a la politica de candidatas, a la definicion local del target o a que falta el protocolo exacto de los autores.",
        "",
        "## Riesgos pendientes",
        "",
        "- No es Figure 3A completa.",
        "- No usa generacion oficial ni generacion libre.",
        "- No sabemos si el paper usa generacion, scoring, restricciones, markers adicionales o un pipeline mixto.",
        "- Falta el protocolo exacto de autores para contexto, decoding, target dominante y metrica.",
        "- El scoring esta acotado por `context_token_budget` y `max_score_tokens`; una reproduccion formal deberia cerrar primero esos detalles.",
        "- No se evaluo un horizonte estacional ni Figure 3B.",
        "",
        "## Recomendacion",
        "",
        "Si este smoke test muestra ranking alto de targets, escalar con cuidado a mas ventanas y/o mas tokens puntuados. Si no lo muestra, antes de concluir fallo conviene preguntar a autores por el protocolo exacto. La generacion condicionada puede probarse luego solo como diagnostico, no como baseline principal hasta validar estructura.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small read-only conditional scoring smoke test.")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--num-windows", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate-candidates", action="store_true")
    parser.add_argument("--debug-one", action="store_true", help="Evaluar una sola ventana total.")
    parser.add_argument("--context-strains", type=int, default=1, help="Numero de cepas historicas recientes por ventana.")
    parser.add_argument("--context-token-budget", type=int, default=512, help="Presupuesto total de tokens para contexto historico.")
    parser.add_argument("--max-score-tokens", type=int, default=256, help="Prefijo de continuacion a puntuar por candidata; 0 = candidata completa.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    print(f"Device: {device}")
    model, model_info = load_prediction_model(device)
    tokenizer = InfluTokenizer(mode="prediction")

    datasets: dict[str, dict[str, Any]] = {}
    dataset_paths: dict[str, Path] = {}
    month_maps: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for subtype in SUBTYPES:
        loaded = load_dataset(subtype)
        if loaded is None:
            print(f"Dataset no encontrado para {subtype}; se omite.")
            continue
        path, dataset = loaded
        datasets[subtype] = dataset
        dataset_paths[subtype] = path
        month_maps[subtype] = build_month_records(dataset, subtype)

    rows = []
    for subtype, dataset in datasets.items():
        selected = select_windows(
            dataset=dataset,
            num_windows=1 if args.debug_one else args.num_windows,
            seed=args.seed + sum(ord(ch) for ch in subtype),
            debug_one=args.debug_one,
        )
        for window_idx, window in selected:
            print(f"Scoring {subtype} window={window_idx} target={window['target']['year']}-{int(window['target']['month']):02d}")
            row = score_window(
                model=model,
                tokenizer=tokenizer,
                device=device,
                dataset=dataset,
                subtype=subtype,
                window_idx=window_idx,
                window=window,
                month_records=month_maps[subtype],
                args=args,
            )
            rows.append(row)
            if args.debug_one:
                break
        if args.debug_one and rows:
            break

    agg = aggregate_results(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        build_report(
            args=args,
            device=device,
            model_info=model_info,
            dataset_paths=dataset_paths,
            rows=rows,
            agg=agg,
        ),
        encoding="utf-8",
    )
    print(f"Reporte escrito: {REPORT_PATH.relative_to(ROOT)}")
    print(
        "Resumen: "
        f"windows={agg['windows_evaluated']} "
        f"top1={fmt_float(agg['top1_accuracy'])} "
        f"top5={fmt_float(agg['top5_accuracy'])} "
        f"mean_norm_rank={fmt_float(agg['mean_normalized_rank'])} "
        f"MRR={fmt_float(agg['mrr'])}"
    )


if __name__ == "__main__":
    main()
