#!/usr/bin/env python3
"""
Small sensitivity audit for conditional scoring with prediction_sequence/.

This script reuses the read-only scorer from test_prediction_sequence_scoring.py.
It never trains, generates, decodes, prints, or saves biological sequences.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from influ_tokenizer import InfluTokenizer
from test_prediction_sequence_scoring import (
    ROOT,
    SUBTYPES,
    aggregate_results,
    build_month_records,
    choose_device,
    fmt_float,
    load_dataset,
    load_prediction_model,
    markdown_table,
    score_window,
    select_windows,
)


REPORT_PATH = ROOT / "results" / "prediction_sequence_scoring_sensitivity_summary.md"


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small sensitivity audit for prediction_sequence scoring.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    parser.add_argument("--context-token-budgets", default="512,1024,2048")
    parser.add_argument("--max-score-tokens-list", default="256,512")
    parser.add_argument("--num-windows", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-strains", type=int, default=1)
    parser.add_argument("--deduplicate-candidates", dest="deduplicate_candidates", action="store_true", default=True)
    parser.add_argument("--no-deduplicate-candidates", dest="deduplicate_candidates", action="store_false")
    return parser.parse_args()


def median_normalized_rank(rows: list[dict[str, Any]]) -> float:
    values = [
        row["normalized_rank"] for row in rows
        if row.get("target_rank") is not None and math.isfinite(row.get("normalized_rank", float("nan")))
    ]
    return float(np.median(values)) if values else float("nan")


def config_summary(rows: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    agg = aggregate_results(rows)
    valid = [row for row in rows if row.get("target_rank") is not None]
    return {
        **agg,
        "median_normalized_rank": median_normalized_rank(rows),
        "mean_target_rank": float(np.mean([row["target_rank"] for row in valid])) if valid else float("nan"),
        "mean_num_candidates": float(np.mean([row["num_valid"] for row in valid])) if valid else float("nan"),
        "duplicates_removed": int(sum(row.get("duplicates_removed", 0) for row in rows)),
        "overlap_exact_with_context": int(sum(row.get("overlap_exact_with_context", 0) for row in rows)),
        "elapsed_seconds": float(elapsed_seconds),
    }


def compact_offsets(offsets: dict[Any, Any]) -> str:
    return "{" + ", ".join(f"{key}:{value}" for key, value in sorted(offsets.items())) + "}"


def interpret(config_rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    grouped_by_tokens: dict[int, list[dict[str, Any]]] = {}
    grouped_by_context: dict[int, list[dict[str, Any]]] = {}
    for row in config_rows:
        grouped_by_tokens.setdefault(row["max_score_tokens"], []).append(row)
        grouped_by_context.setdefault(row["context_token_budget"], []).append(row)

    lines.append(
        "En esta sensibilidad, `top-5` se interpreta como senal de ranking amplio; `top-1` como senal mas exigente. "
        "La lectura debe mantenerse prudente porque el contexto y la continuacion siguen acotados."
    )

    for tokens, rows in sorted(grouped_by_tokens.items()):
        ordered = sorted(rows, key=lambda row: row["context_token_budget"])
        ranks = [row["mean_normalized_rank"] for row in ordered]
        if all(math.isfinite(value) for value in ranks):
            direction = "mejora" if ranks[-1] < ranks[0] else ("empeora" if ranks[-1] > ranks[0] else "no cambia")
            lines.append(
                f"Con `max_score_tokens={tokens}`, aumentar contexto de {ordered[0]['context_token_budget']} a "
                f"{ordered[-1]['context_token_budget']} {direction} el mean normalized rank "
                f"({fmt_float(ranks[0])} -> {fmt_float(ranks[-1])})."
            )

    for context, rows in sorted(grouped_by_context.items()):
        ordered = sorted(rows, key=lambda row: row["max_score_tokens"])
        if len(ordered) >= 2:
            a, b = ordered[0], ordered[-1]
            direction = (
                "mejora" if b["mean_normalized_rank"] < a["mean_normalized_rank"]
                else ("empeora" if b["mean_normalized_rank"] > a["mean_normalized_rank"] else "no cambia")
            )
            lines.append(
                f"Con `context_token_budget={context}`, puntuar mas tokens {direction} el mean normalized rank "
                f"({fmt_float(a['mean_normalized_rank'])} -> {fmt_float(b['mean_normalized_rank'])})."
            )

    top5_values = [row["top5_accuracy"] for row in config_rows if math.isfinite(row["top5_accuracy"])]
    rank_values = [row["mean_normalized_rank"] for row in config_rows if math.isfinite(row["mean_normalized_rank"])]
    top1_values = [row["top1_accuracy"] for row in config_rows if math.isfinite(row["top1_accuracy"])]
    if top5_values:
        lines.append(
            f"`top-5` promedio sobre configuraciones: {fmt_float(float(np.mean(top5_values)))} "
            f"(rango {fmt_float(float(np.min(top5_values)))}-{fmt_float(float(np.max(top5_values)))})."
        )
    if rank_values:
        lines.append(
            f"Mean normalized rank promedio: {fmt_float(float(np.mean(rank_values)))} "
            f"(rango {fmt_float(float(np.min(rank_values)))}-{fmt_float(float(np.max(rank_values)))})."
        )
    if top1_values:
        lines.append(
            f"`top-1` promedio: {fmt_float(float(np.mean(top1_values)))}; si permanece bajo, la evidencia es de ranking parcial, no de seleccion puntual del target."
        )
    return lines


def build_report(
    args: argparse.Namespace,
    device: torch.device,
    model_info: dict[str, Any],
    dataset_paths: dict[str, Path],
    config_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config_table = []
    for row in config_rows:
        config_table.append(
            [
                row["context_token_budget"],
                row["max_score_tokens"],
                row["windows_evaluated"],
                fmt_float(row["top1_accuracy"]),
                fmt_float(row["top5_accuracy"]),
                fmt_float(row["mean_normalized_rank"]),
                fmt_float(row["median_normalized_rank"]),
                fmt_float(row["mrr"]),
                fmt_float(row["mean_target_rank"]),
                fmt_float(row["mean_num_candidates"]),
                row["duplicates_removed"],
                row["overlap_exact_with_context"],
                fmt_float(row["elapsed_seconds"], 2),
            ]
        )

    compact_window_table = []
    for row in window_rows:
        compact_window_table.append(
            [
                row["context_token_budget"],
                row["max_score_tokens"],
                row["subtype"],
                row["target_month"],
                row["num_valid"],
                row["target_rank"] if row["target_rank"] is not None else "NA",
                fmt_float(row["normalized_rank"]),
                fmt_float(row["percentile_score"]),
                row["top1"],
                row["top5"],
                row["context_tokens"],
                row["scored_tokens_target"],
            ]
        )

    dataset_table = [
        [subtype, str(path.relative_to(ROOT))]
        for subtype, path in dataset_paths.items()
    ]

    checkpoint_table = [
        ["checkpoint", "prediction_sequence"],
        ["device", str(device)],
        ["params", f"{model_info['params']:,}"],
        ["missing keys", len(model_info["missing"])],
        ["unexpected keys", len(model_info["unexpected"])],
        ["unexpected explicadas", model_info["unexpected_explained"]],
    ]

    lines = [
        "# Sensibilidad pequena del scoring condicional `prediction_sequence`",
        "",
        f"- Fecha local de ejecucion: `{now}`",
        f"- Script: `test_prediction_sequence_scoring_sensitivity.py`",
        "",
        "## Objetivo",
        "",
        "Evaluar si el smoke test de scoring condicional es estable frente a cambios moderados en tamano de contexto, tokens puntuados y numero de ventanas. No se entrena, no se genera y no se imprimen ni guardan secuencias.",
        "",
        "## Configuracion del checkpoint",
        "",
        markdown_table(["campo", "valor"], checkpoint_table),
        "",
        "## Datasets detectados",
        "",
        markdown_table(["subtipo", "archivo"], dataset_table),
        "",
        "## Tabla por configuracion",
        "",
        markdown_table(
            [
                "context_budget",
                "score_tokens",
                "ventanas",
                "top-1",
                "top-5",
                "mean norm rank",
                "median norm rank",
                "MRR",
                "mean target rank",
                "mean candidates",
                "dups removidos",
                "overlap contexto",
                "tiempo s",
            ],
            config_table,
        ),
        "",
        "## Tabla compacta por ventana",
        "",
        markdown_table(
            [
                "context_budget",
                "score_tokens",
                "subtipo",
                "mes objetivo",
                "candidatas",
                "target rank",
                "norm rank",
                "percentile",
                "top-1",
                "top-5",
                "context len",
                "tokens puntuados",
            ],
            compact_window_table,
        ),
        "",
        "## Interpretacion",
        "",
    ]
    lines.extend(f"- {line}" for line in interpret(config_rows))
    lines.extend(
        [
            "",
            "## Recomendacion",
            "",
            "- Configuracion minima razonable para una reproduccion parcial: usar al menos `context_token_budget=1024` y `max_score_tokens=512` si el tiempo lo permite, porque reduce el riesgo de depender de un prefijo demasiado corto.",
            "- Antes de escalar mucho mas, conviene preguntar a los autores por el formato exacto de contexto, continuacion, definicion de target y si el pipeline oficial usa generacion, scoring o restricciones.",
            "- Si se escala, hacerlo incrementalmente: mas ventanas primero, luego mas tokens puntuados, manteniendo la politica de candidatas fija.",
            "",
            "## Riesgos pendientes",
            "",
            "- Sigue sin ser Figure 3A completa.",
            "- Sigue sin usar generacion oficial ni generacion libre.",
            "- El ranking depende de la definicion local de target y de la politica de candidatas del mes objetivo.",
            "- El contexto usado sigue siendo una aproximacion local al protocolo de autores.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    context_budgets = parse_int_list(args.context_token_budgets)
    score_tokens_list = parse_int_list(args.max_score_tokens_list)

    device = choose_device(args.device)
    print(f"Device: {device}")
    model, model_info = load_prediction_model(device)
    tokenizer = InfluTokenizer(mode="prediction")

    datasets: dict[str, dict[str, Any]] = {}
    dataset_paths: dict[str, Path] = {}
    month_maps: dict[str, dict[int, list[dict[str, Any]]]] = {}
    selected_windows: dict[str, list[tuple[int, dict[str, Any]]]] = {}

    for subtype in SUBTYPES:
        loaded = load_dataset(subtype)
        if loaded is None:
            print(f"Dataset no encontrado para {subtype}; se omite.")
            continue
        path, dataset = loaded
        datasets[subtype] = dataset
        dataset_paths[subtype] = path
        month_maps[subtype] = build_month_records(dataset, subtype)
        selected_windows[subtype] = select_windows(
            dataset=dataset,
            num_windows=args.num_windows,
            seed=args.seed + sum(ord(ch) for ch in subtype),
            debug_one=False,
        )

    config_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for context_budget in context_budgets:
        for score_tokens in score_tokens_list:
            print(f"Config context={context_budget} score_tokens={score_tokens}")
            run_args = SimpleNamespace(
                context_strains=args.context_strains,
                context_token_budget=context_budget,
                max_score_tokens=score_tokens,
                max_candidates=args.max_candidates,
                seed=args.seed,
                deduplicate_candidates=args.deduplicate_candidates,
            )
            rows_this_config = []
            t0 = time.time()
            for subtype, dataset in datasets.items():
                for window_idx, window in selected_windows[subtype]:
                    row = score_window(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        dataset=dataset,
                        subtype=subtype,
                        window_idx=window_idx,
                        window=window,
                        month_records=month_maps[subtype],
                        args=run_args,
                    )
                    row["context_token_budget"] = context_budget
                    row["max_score_tokens"] = score_tokens
                    rows_this_config.append(row)
                    window_rows.append(row)
            elapsed = time.time() - t0
            summary = config_summary(rows_this_config, elapsed)
            summary["context_token_budget"] = context_budget
            summary["max_score_tokens"] = score_tokens
            config_rows.append(summary)
            print(
                "  "
                f"top1={fmt_float(summary['top1_accuracy'])} "
                f"top5={fmt_float(summary['top5_accuracy'])} "
                f"mean_norm={fmt_float(summary['mean_normalized_rank'])} "
                f"MRR={fmt_float(summary['mrr'])} "
                f"time={elapsed:.1f}s"
            )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        build_report(
            args=args,
            device=device,
            model_info=model_info,
            dataset_paths=dataset_paths,
            config_rows=config_rows,
            window_rows=window_rows,
        ),
        encoding="utf-8",
    )
    print(f"Reporte escrito: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
