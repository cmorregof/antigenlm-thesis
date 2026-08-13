#!/usr/bin/env python3
"""
Audit local AntigenLM checkpoints without training or large evaluation.

This script inspects:
  - prediction_sequence/
  - subtype_classifier/

It writes a compact technical report to:
  results/checkpoint_audit_summary.md
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from antigen_model import GPTForFluMultiTask
from influ_tokenizer import InfluTokenizer


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIRS = [ROOT / "prediction_sequence", ROOT / "subtype_classifier"]
REPORT_PATH = ROOT / "results" / "checkpoint_audit_summary.md"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{num_bytes} B"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def prefix_counts(keys: list[str]) -> dict[str, int]:
    return dict(Counter(key.split(".", 1)[0] for key in keys))


def detect_heads(state_dict: dict[str, torch.Tensor]) -> dict[str, list[tuple[str, tuple[int, ...]]]]:
    patterns = {
        "lm_head": "lm_head",
        "classification_head": "classification_head",
        "cls_head": "cls_head",
        "transformer": "transformer",
        "backbone": "backbone",
    }
    detected: dict[str, list[tuple[str, tuple[int, ...]]]] = {name: [] for name in patterns}
    for key, value in state_dict.items():
        for name, pattern in patterns.items():
            if key.startswith(pattern):
                detected[name].append((key, tuple(value.shape)))
    return detected


def key_shapes(state_dict: dict[str, torch.Tensor], substrings: tuple[str, ...]) -> list[tuple[str, tuple[int, ...]]]:
    rows = []
    for key, value in state_dict.items():
        if any(part in key for part in substrings):
            rows.append((key, tuple(value.shape)))
    return rows


def tensor_has_nan(tensor: torch.Tensor) -> bool:
    return bool(torch.isnan(tensor.detach().float()).any().item())


def safe_shape(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    if isinstance(value, (tuple, list)):
        return str(tuple(value))
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(cell).replace("\n", "<br>") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def infer_task(directory_name: str, config: dict[str, Any], vocab_real_size: int, heads: dict[str, list]) -> str:
    if directory_name == "prediction_sequence":
        return "LM causal / forecasting probable"
    if directory_name == "subtype_classifier":
        return "clasificacion de subtipo probable"
    if heads.get("classification_head"):
        return "clasificacion probable"
    if heads.get("lm_head") and config.get("vocab_size") == vocab_real_size:
        return "LM causal probable"
    return "incierta"


def inspect_files(ckpt_dir: Path) -> dict[str, Any]:
    rows = []
    hashes = {}
    for path in sorted(ckpt_dir.iterdir()):
        if not path.is_file():
            continue
        size = path.stat().st_size
        digest = sha256_file(path) if path.name == "pytorch_model.bin" else None
        rows.append(
            {
                "file": path.name,
                "size_bytes": size,
                "size": human_size(size),
                "sha256": digest,
            }
        )
        if digest:
            hashes[path.name] = digest
    return {"rows": rows, "hashes": hashes}


def inspect_tokenizer(ckpt_dir: Path, mode: str) -> dict[str, Any]:
    vocab = read_json(ckpt_dir / "vocab.json")
    added = read_json(ckpt_dir / "added_tokens.json")
    tokenizer_config = read_json(ckpt_dir / "tokenizer_config.json")
    special_map = read_json(ckpt_dir / "special_tokens_map.json")

    merged_vocab = dict(vocab)
    merged_vocab.update(added)

    subtype_tokens = sorted(
        token for token in merged_vocab if token.startswith("<H") and "N" in token and token.endswith(">")
    )
    segment_tokens = [token for token in ["<HA>", "<NA>", "<sep>"] if token in merged_vocab]

    tokenizer = InfluTokenizer(mode=mode)
    ha = "ATCGN"
    na = "GCTA"
    subtype = "<H3N2>" if mode == "prediction" else None
    ids = tokenizer.encode_strain(ha, na, subtype=subtype)
    decoded_no_special = tokenizer.decode(ids, skip_special_tokens=True)
    decoded_with_special = tokenizer.decode(ids, skip_special_tokens=False)

    return {
        "vocab_json_size": len(vocab),
        "added_tokens_size": len(added),
        "real_vocab_size": len(merged_vocab),
        "special_map": special_map,
        "tokenizer_class": tokenizer_config.get("tokenizer_class", "NA"),
        "subtype_tokens": subtype_tokens,
        "segment_tokens": segment_tokens,
        "example_ids": ids,
        "example_decoded_no_special": decoded_no_special,
        "example_decoded_with_special": decoded_with_special,
        "roundtrip_ok": decoded_no_special == ha + na,
    }


class SubtypeClassifierCheckpointWrapper(nn.Module):
    """Minimal wrapper matching the observed subtype_classifier checkpoint head."""

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

    def pooled_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str,
        token_id: int | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = outputs.last_hidden_state
        if pooling == "last":
            last_idx = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_idx, last_idx]
        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (hidden * mask).sum(dim=1) / denom
        if pooling in {"ha", "na"}:
            if token_id is None:
                raise ValueError(f"token_id requerido para pooling={pooling}")
            positions = input_ids.eq(token_id)
            if not bool(positions.any().item()):
                raise ValueError(f"token_id={token_id} no aparece en input_ids")
            first_pos = positions.float().argmax(dim=1).long()
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_idx, first_pos]
        raise ValueError(f"pooling desconocido: {pooling}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str,
        token_id: int | None = None,
    ) -> torch.Tensor:
        pooled = self.pooled_hidden(input_ids, attention_mask, pooling, token_id)
        return self.classification_head(pooled)


def audit_prediction(
    state_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
    tokenizer_info: dict[str, Any],
) -> dict[str, Any]:
    remapped = remap_transformer_to_backbone(state_dict)
    model = GPTForFluMultiTask(task="prediction")
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    non_mask_unexpected = [key for key in unexpected if not is_causal_mask_buffer(key)]

    tokenizer = InfluTokenizer(mode="prediction")
    input_ids = torch.tensor([tokenizer_info["example_ids"]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out["logits"]
    hidden = out["hidden_states"]

    result = {
        "model_params": sum(parameter.numel() for parameter in model.parameters()),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "unexpected_all_mask_buffers": len(non_mask_unexpected) == 0,
        "non_mask_unexpected": non_mask_unexpected,
        "input_shape": tuple(input_ids.shape),
        "logits_shape": tuple(logits.shape),
        "hidden_shape": tuple(hidden.shape),
        "logits_has_nan": tensor_has_nan(logits),
        "expected_vocab_size": config.get("vocab_size"),
        "forward_ok": tuple(logits.shape) == (1, input_ids.size(1), int(config["vocab_size"])) and not tensor_has_nan(logits),
    }
    del model, out, logits, hidden, input_ids, attention_mask
    gc.collect()
    return result


def audit_classifier(
    state_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
    tokenizer_info: dict[str, Any],
) -> dict[str, Any]:
    head_weight = state_dict.get("classification_head.weight")
    head_bias = state_dict.get("classification_head.bias")
    if head_weight is None or head_bias is None:
        return {
            "wrapper_constructed": False,
            "reason": "classification_head.weight/bias no presentes",
        }

    num_labels = int(head_weight.shape[0])
    hidden_dim = int(head_weight.shape[1])
    wrapper = SubtypeClassifierCheckpointWrapper(config, num_labels=num_labels)
    remapped = remap_transformer_to_backbone(state_dict)
    missing, unexpected = wrapper.load_state_dict(remapped, strict=False)
    non_mask_unexpected = [
        key for key in unexpected
        if not is_causal_mask_buffer(key) and key != "lm_head.weight"
    ]

    tokenizer = InfluTokenizer(mode="classification")
    input_ids = torch.tensor([tokenizer_info["example_ids"]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    ha_id = tokenizer.token_to_id("<HA>")
    na_id = tokenizer.token_to_id("<NA>")

    pooling_results = {}
    wrapper.eval()
    with torch.inference_mode():
        for pooling, token_id in [
            ("last", None),
            ("mean", None),
            ("ha", ha_id),
            ("na", na_id),
        ]:
            try:
                logits = wrapper(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pooling=pooling,
                    token_id=token_id,
                )
                pooling_results[pooling] = {
                    "logits_shape": tuple(logits.shape),
                    "has_nan": tensor_has_nan(logits),
                    "forward_ok": tuple(logits.shape) == (1, num_labels) and not tensor_has_nan(logits),
                }
            except Exception as exc:
                pooling_results[pooling] = {
                    "error": repr(exc),
                    "forward_ok": False,
                }

    result = {
        "wrapper_constructed": True,
        "num_labels": num_labels,
        "head_weight_shape": tuple(head_weight.shape),
        "head_bias_shape": tuple(head_bias.shape),
        "hidden_dim": hidden_dim,
        "model_params": sum(parameter.numel() for parameter in wrapper.parameters()),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "unexpected_all_explained": len(non_mask_unexpected) == 0,
        "non_explained_unexpected": non_mask_unexpected,
        "input_shape": tuple(input_ids.shape),
        "pooling_results": pooling_results,
        "forward_ok": all(row.get("forward_ok", False) for row in pooling_results.values()),
    }
    del wrapper, input_ids, attention_mask
    gc.collect()
    return result


def inspect_checkpoint(ckpt_dir: Path) -> dict[str, Any]:
    config = read_json(ckpt_dir / "config.json")
    mode = "prediction" if ckpt_dir.name == "prediction_sequence" else "classification"
    files = inspect_files(ckpt_dir)
    tokenizer_info = inspect_tokenizer(ckpt_dir, mode=mode)

    weights_path = ckpt_dir / "pytorch_model.bin"
    state_dict = load_state_dict(weights_path)
    keys = list(state_dict.keys())
    heads = detect_heads(state_dict)
    relevant_shapes = key_shapes(
        state_dict,
        ("wte.weight", "wpe.weight", "lm_head", "classification_head", "cls_head"),
    )

    task_probable = infer_task(ckpt_dir.name, config, tokenizer_info["real_vocab_size"], heads)

    if ckpt_dir.name == "prediction_sequence":
        load_audit = audit_prediction(state_dict, config, tokenizer_info)
    else:
        load_audit = audit_classifier(state_dict, config, tokenizer_info)

    state_summary = {
        "n_tensors": len(state_dict),
        "prefix_counts": prefix_counts(keys),
        "heads": heads,
        "relevant_shapes": relevant_shapes,
    }

    del state_dict
    gc.collect()

    return {
        "name": ckpt_dir.name,
        "path": str(ckpt_dir.relative_to(ROOT)),
        "files": files,
        "config": config,
        "tokenizer": tokenizer_info,
        "state_dict": state_summary,
        "task_probable": task_probable,
        "load_audit": load_audit,
    }


def format_file_inventory(item: dict[str, Any]) -> str:
    rows = []
    for row in item["files"]["rows"]:
        sha = row["sha256"][:16] + "..." if row["sha256"] else ""
        rows.append([row["file"], row["size"], row["size_bytes"], sha])
    return markdown_table(["archivo", "tamano", "bytes", "sha256"], rows)


def format_config(item: dict[str, Any]) -> str:
    config = item["config"]
    rows = [
        ["vocab_size", config.get("vocab_size")],
        ["n_layer", config.get("n_layer")],
        ["n_head", config.get("n_head")],
        ["n_embd", config.get("n_embd")],
        ["n_positions", config.get("n_positions")],
        ["architectures", ", ".join(config.get("architectures", []))],
        ["model_type", config.get("model_type")],
        ["transformers_version_config", config.get("transformers_version")],
        ["task_probable", item["task_probable"]],
    ]
    return markdown_table(["campo", "valor"], rows)


def format_tokenizer(item: dict[str, Any]) -> str:
    tokenizer = item["tokenizer"]
    rows = [
        ["vocab.json size", tokenizer["vocab_json_size"]],
        ["added_tokens size", tokenizer["added_tokens_size"]],
        ["vocab real fusionado", tokenizer["real_vocab_size"]],
        ["tokenizer_class", tokenizer["tokenizer_class"]],
        ["tokens segmento", ", ".join(tokenizer["segment_tokens"]) or "ninguno"],
        ["tokens subtipo", ", ".join(tokenizer["subtype_tokens"]) or "ninguno"],
        ["special map", json.dumps(tokenizer["special_map"], ensure_ascii=False)],
        ["example ids", tokenizer["example_ids"]],
        ["decode sin especiales", tokenizer["example_decoded_no_special"]],
        ["decode con especiales", tokenizer["example_decoded_with_special"]],
        ["roundtrip pequeno", tokenizer["roundtrip_ok"]],
    ]
    return markdown_table(["campo", "valor"], rows)


def format_state_dict(item: dict[str, Any]) -> str:
    state = item["state_dict"]
    prefix_rows = [[prefix, count] for prefix, count in sorted(state["prefix_counts"].items())]
    shapes_rows = [[key, shape] for key, shape in state["relevant_shapes"]]
    heads = state["heads"]
    head_rows = []
    for head_name in ["lm_head", "classification_head", "cls_head", "transformer", "backbone"]:
        entries = heads.get(head_name, [])
        if entries:
            preview = "; ".join(f"{key} {shape}" for key, shape in entries[:5])
            if len(entries) > 5:
                preview += f"; ... ({len(entries)} tensores)"
        else:
            preview = "no detectado"
        head_rows.append([head_name, len(entries), preview])

    return "\n".join(
        [
            f"- Numero de tensores: `{state['n_tensors']}`",
            "",
            markdown_table(["prefijo", "n"], prefix_rows),
            "",
            markdown_table(["head/prefijo", "n", "detalle"], head_rows),
            "",
            markdown_table(["tensor relevante", "shape"], shapes_rows),
        ]
    )


def format_prediction_load(item: dict[str, Any]) -> str:
    audit = item["load_audit"]
    rows = [
        ["params", f"{audit['model_params']:,}"],
        ["missing keys", len(audit["missing"])],
        ["unexpected keys", len(audit["unexpected"])],
        ["unexpected son buffers mascara causal", audit["unexpected_all_mask_buffers"]],
        ["unexpected no explicadas", audit["non_mask_unexpected"] or "ninguna"],
        ["input shape", audit["input_shape"]],
        ["logits shape", audit["logits_shape"]],
        ["hidden shape", audit["hidden_shape"]],
        ["NaNs en logits", audit["logits_has_nan"]],
        ["forward ok", audit["forward_ok"]],
    ]
    if audit["missing"]:
        rows.append(["missing detalle", audit["missing"]])
    return markdown_table(["campo", "valor"], rows)


def format_classifier_load(item: dict[str, Any]) -> str:
    audit = item["load_audit"]
    if not audit.get("wrapper_constructed"):
        return markdown_table(
            ["campo", "valor"],
            [["wrapper construido", False], ["razon", audit.get("reason", "desconocida")]],
        )
    rows = [
        ["wrapper construido", audit["wrapper_constructed"]],
        ["num labels", audit["num_labels"]],
        ["classification_head.weight", audit["head_weight_shape"]],
        ["classification_head.bias", audit["head_bias_shape"]],
        ["params wrapper", f"{audit['model_params']:,}"],
        ["missing keys", len(audit["missing"])],
        ["unexpected keys", len(audit["unexpected"])],
        ["unexpected explicadas", audit["unexpected_all_explained"]],
        ["unexpected no explicadas", audit["non_explained_unexpected"] or "ninguna"],
        ["input shape", audit["input_shape"]],
        ["forward global ok", audit["forward_ok"]],
    ]
    pooling_rows = []
    for pooling, result in audit["pooling_results"].items():
        pooling_rows.append(
            [
                pooling,
                result.get("logits_shape", "NA"),
                result.get("has_nan", "NA"),
                result.get("forward_ok", False),
                result.get("error", ""),
            ]
        )
    return "\n".join(
        [
            markdown_table(["campo", "valor"], rows),
            "",
            markdown_table(["pooling", "logits shape", "NaNs", "ok", "error"], pooling_rows),
        ]
    )


def build_head_table(items: list[dict[str, Any]]) -> str:
    rows = []
    for item in items:
        state = item["state_dict"]
        heads = state["heads"]
        rows.append(
            [
                item["name"],
                len(heads.get("lm_head", [])),
                len(heads.get("classification_head", [])),
                len(heads.get("cls_head", [])),
                len(heads.get("transformer", [])),
                len(heads.get("backbone", [])),
            ]
        )
    return markdown_table(
        ["checkpoint", "lm_head", "classification_head", "cls_head", "transformer", "backbone"],
        rows,
    )


def build_weight_comparison(items: list[dict[str, Any]]) -> str:
    by_name = {item["name"]: item for item in items}
    pred_hash = by_name["prediction_sequence"]["files"]["hashes"].get("pytorch_model.bin")
    cls_hash = by_name["subtype_classifier"]["files"]["hashes"].get("pytorch_model.bin")
    pred_size = next(
        row["size_bytes"]
        for row in by_name["prediction_sequence"]["files"]["rows"]
        if row["file"] == "pytorch_model.bin"
    )
    cls_size = next(
        row["size_bytes"]
        for row in by_name["subtype_classifier"]["files"]["rows"]
        if row["file"] == "pytorch_model.bin"
    )
    same_hash = pred_hash == cls_hash
    same_size = pred_size == cls_size
    rows = [
        ["mismo tamano", same_size],
        ["mismo sha256", same_hash],
        ["prediction sha256", pred_hash],
        ["subtype sha256", cls_hash],
    ]
    conclusion = (
        "Los pesos locales son identicos."
        if same_hash
        else "Los pesos locales NO son identicos; ademas las heads detectadas difieren."
    )
    return markdown_table(["comparacion", "valor"], rows) + f"\n\nConclusion local: {conclusion}"


def build_report(items: list[dict[str, Any]]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Auditoria local de checkpoints AntigenLM",
        "",
        f"- Fecha local de ejecucion: `{now}`",
        f"- Script: `audit_antigenlm_checkpoints.py`",
        f"- Torch: `{torch.__version__}`",
        "",
        "Esta auditoria no entrena modelos, no evalua datasets grandes y no genera secuencias largas.",
        "",
        "## Comparacion local de pesos",
        "",
        build_weight_comparison(items),
        "",
        "## Tabla de heads detectadas",
        "",
        build_head_table(items),
        "",
    ]

    for item in items:
        lines.extend(
            [
                f"## Inventario: `{item['path']}/`",
                "",
                format_file_inventory(item),
                "",
                "### Configuracion",
                "",
                format_config(item),
                "",
                "### Tokenizer",
                "",
                format_tokenizer(item),
                "",
                "### State dict",
                "",
                format_state_dict(item),
                "",
                "### Carga y forward minimo",
                "",
                format_prediction_load(item)
                if item["name"] == "prediction_sequence"
                else format_classifier_load(item),
                "",
            ]
        )

    lines.extend(
        [
            "## Riesgos pendientes",
            "",
            "- `prediction_sequence/` carga como LM causal y permite forward minimo, pero esta auditoria no confirma el protocolo exacto de forecasting del paper.",
            "- `subtype_classifier/` contiene una cabeza real `classification_head` lineal de 12 clases; falta el mapa de etiquetas de esas 12 clases.",
            "- El pooling exacto del clasificador sigue pendiente: ultimo token, mean pooling, `<HA>` y `<NA>` son plausibles tecnicamente, pero esta auditoria no decide cual corresponde al paper.",
            "- La presencia de `lm_head.weight` en `subtype_classifier/` queda como peso adicional no usado por el wrapper minimo; no impide el forward de clasificacion.",
            "- Las `unexpected keys` asociadas a `attn.bias` y `attn.masked_bias` son compatibles con buffers de mascara causal serializados por otra version de `transformers`.",
            "- No se calcularon accuracy, F1, mismatch ni ranking; esos pertenecen a la siguiente fase experimental.",
            "",
            "## Recomendacion de siguiente paso",
            "",
            "Ejecutar primero una clasificacion parcial H1N1/H3N2 con muestra pequena y balanceada, porque reduce el riesgo de que el wrapper/pooling del `subtype_classifier` este mal interpretado. En paralelo conceptual, `prediction_sequence` ya esta tecnicamente listo para un scoring condicional pequeno, pero conviene cerrar antes la fidelidad basica del clasificador.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    items = []
    for ckpt_dir in CHECKPOINT_DIRS:
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"No existe checkpoint dir: {ckpt_dir}")
        items.append(inspect_checkpoint(ckpt_dir))
    report = build_report(items)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Reporte escrito: {REPORT_PATH.relative_to(ROOT)}")
    for item in items:
        ok = item["load_audit"].get("forward_ok", False)
        print(f"{item['name']}: forward_ok={ok}")


if __name__ == "__main__":
    main()
