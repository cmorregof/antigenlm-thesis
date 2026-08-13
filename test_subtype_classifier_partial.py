#!/usr/bin/env python3
"""
Partial, read-only audit of the released AntigenLM subtype classifier checkpoint.

The script uses a small balanced local sample of H1N1/H3N2 records to check
whether the checkpoint wiring, pooling choices, and a two-class local logit map
produce a coherent separation. It never prints or saves biological sequences.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from influ_tokenizer import InfluTokenizer


ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "subtype_classifier"
DATA_DIR = ROOT / "data" / "processed_gisaid"
REPORT_PATH = ROOT / "results" / "subtype_classifier_partial_summary.md"
POOLINGS = ("last", "mean", "ha", "na")
LOCAL_CLASS_A = "H1N1"
LOCAL_CLASS_B = "H3N2"


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


class SubtypeClassifierCheckpointWrapper(nn.Module):
    """Minimal wrapper matching the observed subtype_classifier checkpoint."""

    def __init__(self, config_json: dict[str, Any], num_labels: int):
        super().__init__()
        config = GPT2Config(
            vocab_size=int(config_json["vocab_size"]),
            n_layer=int(config_json["n_layer"]),
            n_embd=int(config_json["n_embd"]),
            n_head=int(config_json["n_head"]),
            n_positions=int(config_json["n_positions"]),
            n_inner=config_json.get("n_inner"),
            activation_function=config_json.get("activation_function", "gelu_new"),
            resid_pdrop=float(config_json.get("resid_pdrop", 0.1)),
            embd_pdrop=float(config_json.get("embd_pdrop", 0.1)),
            attn_pdrop=float(config_json.get("attn_pdrop", 0.1)),
            layer_norm_epsilon=float(config_json.get("layer_norm_epsilon", 1e-5)),
            initializer_range=float(config_json.get("initializer_range", 0.02)),
            scale_attn_weights=bool(config_json.get("scale_attn_weights", True)),
            use_cache=bool(config_json.get("use_cache", True)),
        )
        self.backbone = GPT2Model(config)
        self.classification_head = nn.Linear(int(config_json["n_embd"]), num_labels)

    def forward_all_poolings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ha_id: int | None,
        na_id: int | None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        pooled: dict[str, torch.Tensor] = {}

        last_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        pooled["last"] = hidden[batch_idx, last_idx]

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled["mean"] = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        for name, token_id in [("ha", ha_id), ("na", na_id)]:
            if token_id is None:
                continue
            positions = input_ids.eq(int(token_id))
            if bool(positions.any().item()):
                first_pos = positions.float().argmax(dim=1).long()
                pooled[name] = hidden[batch_idx, first_pos]

        return {name: self.classification_head(value) for name, value in pooled.items()}


def find_dataset_path(label: str) -> Path:
    exact = DATA_DIR / f"dataset_{label}.json"
    if exact.exists():
        return exact
    matches = sorted(DATA_DIR.glob(f"*{label}*.json"))
    if not matches:
        raise FileNotFoundError(f"No se encontro dataset local para {label} en {DATA_DIR}")
    return matches[0]


def load_records(label: str) -> tuple[Path, list[dict[str, Any]]]:
    path = find_dataset_path(label)
    dataset = read_json(path)
    records = dataset.get("paired_strains") or []
    usable = [
        record for record in records
        if record.get("ha_sequence") and record.get("na_sequence")
    ]
    return path, usable


def sample_records(records: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < n:
        raise ValueError(f"Muestra solicitada n={n}, pero solo hay {len(records)} registros usables")
    rng = random.Random(seed)
    selected = rng.sample(records, n)
    rng.shuffle(selected)
    return selected


def split_half(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    midpoint = len(records) // 2
    return records[:midpoint], records[midpoint:]


def encode_record(tokenizer: InfluTokenizer, record: dict[str, Any]) -> list[int]:
    return tokenizer.encode_strain(
        ha_sequence=record["ha_sequence"],
        na_sequence=record["na_sequence"],
        subtype=None,
        add_eos=True,
    )


def summarize_lengths(records_by_label: dict[str, list[dict[str, Any]]], tokenizer: InfluTokenizer) -> dict[str, Any]:
    summary = {}
    for label, records in records_by_label.items():
        ha_lengths = [len(record["ha_sequence"]) for record in records]
        na_lengths = [len(record["na_sequence"]) for record in records]
        token_lengths = [len(encode_record(tokenizer, record)) for record in records]
        summary[label] = {
            "n": len(records),
            "ha_min": int(np.min(ha_lengths)),
            "ha_median": float(np.median(ha_lengths)),
            "ha_max": int(np.max(ha_lengths)),
            "na_min": int(np.min(na_lengths)),
            "na_median": float(np.median(na_lengths)),
            "na_max": int(np.max(na_lengths)),
            "tokens_min": int(np.min(token_lengths)),
            "tokens_median": float(np.median(token_lengths)),
            "tokens_max": int(np.max(token_lengths)),
        }
    return summary


def logits_for_records(
    model: SubtypeClassifierCheckpointWrapper,
    tokenizer: InfluTokenizer,
    records: list[dict[str, Any]],
    label: str,
    split: str,
    device: torch.device,
    ha_id: int | None,
    na_id: int | None,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    logits_by_pooling: dict[str, list[np.ndarray]] = {pooling: [] for pooling in POOLINGS}
    failures = []
    nan_counts = Counter()

    model.eval()
    with torch.inference_mode():
        for idx, record in enumerate(records):
            try:
                token_ids = encode_record(tokenizer, record)
                input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                output = model.forward_all_poolings(input_ids, attention_mask, ha_id=ha_id, na_id=na_id)
                for pooling in POOLINGS:
                    if pooling not in output:
                        continue
                    logits = output[pooling].detach().float().cpu().numpy()[0]
                    if not np.all(np.isfinite(logits)):
                        nan_counts[pooling] += 1
                    else:
                        logits_by_pooling[pooling].append(logits)
                del input_ids, attention_mask, output
            except Exception as exc:
                failures.append({"label": label, "split": split, "index": idx, "error": repr(exc)})
            finally:
                if device.type == "mps":
                    torch.mps.empty_cache()

    meta = {
        "failures": failures,
        "nan_counts": dict(nan_counts),
        "n_requested": len(records),
    }
    return logits_by_pooling, meta


def macro_f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for cls in [0, 1]:
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(scores))


def evaluate_pooling(
    pooling: str,
    cal_a: list[np.ndarray],
    cal_b: list[np.ndarray],
    test_a: list[np.ndarray],
    test_b: list[np.ndarray],
    nan_count: int,
    failure_count: int,
) -> dict[str, Any]:
    if not cal_a or not cal_b or not test_a or not test_b:
        return {
            "pooling": pooling,
            "applicable": False,
            "reason": "faltan logits para calibration/test",
        }

    cal_a_arr = np.vstack(cal_a)
    cal_b_arr = np.vstack(cal_b)
    test_a_arr = np.vstack(test_a)
    test_b_arr = np.vstack(test_b)

    idx_a = int(np.argmax(cal_a_arr.mean(axis=0)))
    idx_b = int(np.argmax(cal_b_arr.mean(axis=0)))
    ambiguous = idx_a == idx_b
    test_logits = np.vstack([test_a_arr, test_b_arr])
    y_true = np.array([0] * len(test_a_arr) + [1] * len(test_b_arr))

    global_argmax = np.argmax(test_logits, axis=1)
    unknown = ~np.isin(global_argmax, [idx_a, idx_b])
    unknown_count = int(np.sum(unknown))
    unknown_pct = float(unknown_count / len(y_true)) if len(y_true) else float("nan")
    unique_global_argmax = sorted(int(x) for x in np.unique(global_argmax))
    mean_logit_std = float(np.mean(np.std(test_logits, axis=1)))

    if ambiguous:
        return {
            "pooling": pooling,
            "applicable": True,
            "idx_a": idx_a,
            "idx_b": idx_b,
            "ambiguous": True,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "confusion": [[0, 0], [0, 0]],
            "unknown_count": unknown_count,
            "unknown_pct": unknown_pct,
            "unique_global_argmax": unique_global_argmax,
            "mean_logit_std": mean_logit_std,
            "n_test": int(len(y_true)),
            "nan_count": nan_count,
            "failure_count": failure_count,
        }

    binary_scores = test_logits[:, [idx_a, idx_b]]
    y_pred = np.argmax(binary_scores, axis=1)
    accuracy = float(np.mean(y_pred == y_true))
    macro_f1 = macro_f1_binary(y_true, y_pred)
    confusion = [
        [int(np.sum((y_true == 0) & (y_pred == 0))), int(np.sum((y_true == 0) & (y_pred == 1)))],
        [int(np.sum((y_true == 1) & (y_pred == 0))), int(np.sum((y_true == 1) & (y_pred == 1)))],
    ]

    return {
        "pooling": pooling,
        "applicable": True,
        "idx_a": idx_a,
        "idx_b": idx_b,
        "ambiguous": False,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion": confusion,
        "unknown_count": unknown_count,
        "unknown_pct": unknown_pct,
        "unique_global_argmax": unique_global_argmax,
        "mean_logit_std": mean_logit_std,
        "n_test": int(len(y_true)),
        "nan_count": nan_count,
        "failure_count": failure_count,
    }


def select_best(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row for row in results
        if row.get("applicable") and not row.get("ambiguous") and math.isfinite(row.get("accuracy", float("nan")))
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (row["accuracy"], row["macro_f1"], -row["unknown_pct"]),
        reverse=True,
    )[0]


def success_label(best: dict[str, Any] | None) -> str:
    if best is None:
        return "fallo informativo"
    if best["accuracy"] >= 0.95 and best["unknown_pct"] <= 0.10:
        return "exito fuerte"
    if best["accuracy"] >= 0.70:
        return "exito parcial"
    return "fallo informativo"


def build_report(
    args: argparse.Namespace,
    device: torch.device,
    config: dict[str, Any],
    load_info: dict[str, Any],
    dataset_paths: dict[str, Path],
    length_summary: dict[str, Any],
    token_info: dict[str, Any],
    results: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_rows = []
    for row in results:
        if not row.get("applicable"):
            result_rows.append([row["pooling"], "no", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", row.get("reason", "")])
            continue
        result_rows.append(
            [
                row["pooling"],
                "si",
                row.get("idx_a", "NA"),
                row.get("idx_b", "NA"),
                row.get("ambiguous", "NA"),
                fmt_float(row.get("accuracy")),
                fmt_float(row.get("macro_f1")),
                row.get("confusion", "NA"),
                f"{row.get('unknown_count', 'NA')} ({fmt_float(row.get('unknown_pct'), 3)})",
                row.get("n_test", "NA"),
                f"NaNs={row.get('nan_count', 'NA')}; failures={row.get('failure_count', 'NA')}; std={fmt_float(row.get('mean_logit_std'))}; argmax={row.get('unique_global_argmax', 'NA')}",
            ]
        )

    length_rows = []
    for label, row in length_summary.items():
        length_rows.append(
            [
                label,
                row["n"],
                f"{row['ha_min']}/{row['ha_median']:.1f}/{row['ha_max']}",
                f"{row['na_min']}/{row['na_median']:.1f}/{row['na_max']}",
                f"{row['tokens_min']}/{row['tokens_median']:.1f}/{row['tokens_max']}",
            ]
        )

    load_rows = [
        ["checkpoint", str(CKPT_DIR.relative_to(ROOT))],
        ["device", str(device)],
        ["vocab_size config", config.get("vocab_size")],
        ["n_layer/n_head/n_embd/n_positions", f"{config.get('n_layer')}/{config.get('n_head')}/{config.get('n_embd')}/{config.get('n_positions')}"],
        ["classification_head.weight", load_info["head_weight_shape"]],
        ["classification_head.bias", load_info["head_bias_shape"]],
        ["params", f"{load_info['params']:,}"],
        ["missing keys", len(load_info["missing"])],
        ["unexpected keys", len(load_info["unexpected"])],
        ["unexpected explicadas", load_info["unexpected_explained"]],
        ["NaNs en sanity logits", load_info["sanity_has_nan"]],
    ]

    if best is None:
        best_text = "No hay pooling local no ambiguo con metrica finita."
        mapping_text = "No se infiere mapeo local usable."
    else:
        best_text = (
            f"Mejor pooling local: `{best['pooling']}` con accuracy={fmt_float(best['accuracy'])}, "
            f"macro F1={fmt_float(best['macro_f1'])}, unknown={fmt_float(best['unknown_pct'], 3)}."
        )
        mapping_text = (
            f"Mapeo local inferido solo para esta auditoria: `{LOCAL_CLASS_A}` -> logit `{best['idx_a']}`, "
            f"`{LOCAL_CLASS_B}` -> logit `{best['idx_b']}`. No es label map oficial."
        )

    lines = [
        "# Auditoria parcial del `subtype_classifier`",
        "",
        f"- Fecha local de ejecucion: `{now}`",
        f"- Script: `test_subtype_classifier_partial.py`",
        f"- Comando previsto: `--n-per-class {args.n_per_class} --seed {args.seed}`",
        "",
        "## Objetivo",
        "",
        "Verificar si el checkpoint local `subtype_classifier/` esta correctamente cableado y si separa dos clases locales conocidas con una muestra pequena y balanceada. Esta prueba no reproduce formalmente el paper.",
        "",
        "## Restricciones cumplidas",
        "",
        "- No se entrenaron modelos.",
        "- No se modificaron pesos.",
        "- No se generaron, optimizaron ni mutaron secuencias.",
        "- No se imprimieron ni guardaron secuencias completas.",
        "- No se tocaron archivos `.tex`.",
        "- Solo se uso una muestra pequena y balanceada.",
        "",
        "## Carga del checkpoint",
        "",
        markdown_table(["campo", "valor"], load_rows),
        "",
        "## Datos y muestra",
        "",
        markdown_table(
            ["clase local", "archivo detectado", "registros usados"],
            [
                [LOCAL_CLASS_A, str(dataset_paths[LOCAL_CLASS_A].relative_to(ROOT)), args.n_per_class],
                [LOCAL_CLASS_B, str(dataset_paths[LOCAL_CLASS_B].relative_to(ROOT)), args.n_per_class],
            ],
        ),
        "",
        markdown_table(["clase", "n", "HA len min/median/max", "NA len min/median/max", "tokens min/median/max"], length_rows),
        "",
        "## Formato de input usado",
        "",
        "Serializacion prudente sin token de subtipo:",
        "",
        "`<HA> HA_sequence <sep> <NA> NA_sequence <eos>`",
        "",
        markdown_table(
            ["token", "id"],
            [[name, token_id] for name, token_id in token_info.items()],
        ),
        "",
        "## Tabla por pooling",
        "",
        markdown_table(
            [
                "pooling",
                "aplicable",
                f"idx {LOCAL_CLASS_A}",
                f"idx {LOCAL_CLASS_B}",
                "ambiguo",
                "accuracy",
                "macro F1",
                "confusion [[A->A,A->B],[B->A,B->B]]",
                "unknown externo",
                "n test",
                "diagnostico",
            ],
            result_rows,
        ),
        "",
        "## Mejor pooling local",
        "",
        best_text,
        "",
        "## Mapeo local inferido",
        "",
        mapping_text,
        "",
        f"Estado segun criterios: **{success_label(best)}**.",
        "",
        "## Riesgos pendientes",
        "",
        "- Falta el label map oficial de las 12 clases.",
        "- Falta el pooling oficial usado por los autores.",
        "- Solo se evaluaron dos clases locales: H1N1 y H3N2.",
        "- No se evaluan las 12 clases y por tanto esto no reproduce la tabla completa de clasificacion del paper.",
        "- El mapeo local inferido depende de la muestra, el split 50/50 y la serializacion usada.",
        "- La prueba no valida forecasting ni `prediction_sequence/`.",
        "",
        "## Recomendacion",
        "",
        "Si el mejor pooling muestra separacion alta, el checkpoint queda cableado de forma plausible para una reproduccion parcial H1N1/H3N2, pero antes de llamarlo reproduccion formal conviene solicitar o localizar el label map y el pooling oficial. Si se necesita avanzar metodologicamente sin ese dato, el siguiente bloqueo mas relevante es `prediction_sequence` scoring condicional, porque permite evaluar forecasting sin depender del clasificador.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Partial audit for subtype_classifier checkpoint.")
    parser.add_argument("--n-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = read_json(CKPT_DIR / "config.json")
    state_dict = load_state_dict(CKPT_DIR / "pytorch_model.bin")
    head_weight = state_dict.get("classification_head.weight")
    head_bias = state_dict.get("classification_head.bias")
    if head_weight is None or head_bias is None:
        raise RuntimeError("classification_head.weight/bias no estan presentes en subtype_classifier.")

    num_labels = int(head_weight.shape[0])
    model = SubtypeClassifierCheckpointWrapper(config, num_labels=num_labels)
    missing, unexpected = model.load_state_dict(remap_transformer_to_backbone(state_dict), strict=False)
    unexplained = [
        key for key in unexpected
        if not is_causal_mask_buffer(key) and key != "lm_head.weight"
    ]
    del state_dict
    gc.collect()

    device = choose_device(args.device)
    model.eval().to(device)

    tokenizer = InfluTokenizer(mode="classification")
    ha_id = tokenizer.token_to_id("<HA>") if "<HA>" in tokenizer.vocab else None
    na_id = tokenizer.token_to_id("<NA>") if "<NA>" in tokenizer.vocab else None
    token_info = {
        "<HA>": ha_id if ha_id is not None else "no disponible",
        "<sep>": tokenizer.token_to_id("<sep>") if "<sep>" in tokenizer.vocab else "no disponible",
        "<NA>": na_id if na_id is not None else "no disponible",
        "<eos>": tokenizer.eos_token_id,
    }

    path_a, records_a_all = load_records(LOCAL_CLASS_A)
    path_b, records_b_all = load_records(LOCAL_CLASS_B)
    records_a = sample_records(records_a_all, args.n_per_class, args.seed + 101)
    records_b = sample_records(records_b_all, args.n_per_class, args.seed + 202)
    cal_a, test_a = split_half(records_a)
    cal_b, test_b = split_half(records_b)

    length_summary = summarize_lengths({LOCAL_CLASS_A: records_a, LOCAL_CLASS_B: records_b}, tokenizer)

    # Tiny real sanity forward, without saving or printing sequences.
    sanity_ids = encode_record(tokenizer, records_a[0])
    sanity_input = torch.tensor([sanity_ids], dtype=torch.long, device=device)
    sanity_mask = torch.ones_like(sanity_input)
    with torch.inference_mode():
        sanity_logits = model.forward_all_poolings(sanity_input, sanity_mask, ha_id=ha_id, na_id=na_id)["last"]
    sanity_has_nan = bool(torch.isnan(sanity_logits.detach().float()).any().item())
    del sanity_input, sanity_mask, sanity_logits

    print(f"Device: {device}")
    print(f"Evaluando muestra balanceada: {args.n_per_class}+{args.n_per_class}")

    cal_logits_a, cal_meta_a = logits_for_records(model, tokenizer, cal_a, LOCAL_CLASS_A, "calibration", device, ha_id, na_id)
    cal_logits_b, cal_meta_b = logits_for_records(model, tokenizer, cal_b, LOCAL_CLASS_B, "calibration", device, ha_id, na_id)
    test_logits_a, test_meta_a = logits_for_records(model, tokenizer, test_a, LOCAL_CLASS_A, "test", device, ha_id, na_id)
    test_logits_b, test_meta_b = logits_for_records(model, tokenizer, test_b, LOCAL_CLASS_B, "test", device, ha_id, na_id)

    results = []
    for pooling in POOLINGS:
        nan_count = (
            cal_meta_a["nan_counts"].get(pooling, 0)
            + cal_meta_b["nan_counts"].get(pooling, 0)
            + test_meta_a["nan_counts"].get(pooling, 0)
            + test_meta_b["nan_counts"].get(pooling, 0)
        )
        failure_count = (
            len(cal_meta_a["failures"])
            + len(cal_meta_b["failures"])
            + len(test_meta_a["failures"])
            + len(test_meta_b["failures"])
        )
        results.append(
            evaluate_pooling(
                pooling,
                cal_logits_a[pooling],
                cal_logits_b[pooling],
                test_logits_a[pooling],
                test_logits_b[pooling],
                nan_count=nan_count,
                failure_count=failure_count,
            )
        )

    best = select_best(results)
    load_info = {
        "head_weight_shape": tuple(head_weight.shape),
        "head_bias_shape": tuple(head_bias.shape),
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "unexpected_explained": len(unexplained) == 0,
        "sanity_has_nan": sanity_has_nan,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(
        args=args,
        device=device,
        config=config,
        load_info=load_info,
        dataset_paths={LOCAL_CLASS_A: path_a, LOCAL_CLASS_B: path_b},
        length_summary=length_summary,
        token_info=token_info,
        results=results,
        best=best,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Reporte escrito: {REPORT_PATH.relative_to(ROOT)}")
    if best:
        print(
            f"Mejor pooling local: {best['pooling']} "
            f"accuracy={best['accuracy']:.4f} macro_f1={best['macro_f1']:.4f} "
            f"unknown={best['unknown_pct']:.4f}"
        )
    else:
        print("No se encontro pooling local no ambiguo.")


if __name__ == "__main__":
    main()
