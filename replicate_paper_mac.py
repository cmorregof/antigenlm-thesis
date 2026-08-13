# -*- coding: utf-8 -*-
"""
replicate_paper_mac.py
======================
Réplica fiel de la Figura 3A del paper AntigenLM (Pei et al., ICLR 2026).
Optimizado para Apple M5 con MPS y 24 GB de RAM.

Pipeline (idéntico al paper):
    Para cada ventana temporal [t-2, t-1, t] → predice t+1:
      1. Contexto: 3 cepas dominantes mensuales COMPLETAS, sin truncar
      2. Generación autorregresiva libre con KV caching (sin restricciones de vocabulario)
      3. Extracción de HA y NA de la secuencia generada
      4. Traducción DNA → aminoácidos
      5. Average Amino Acid Mismatch vs cepa real de t+1

Diferencias respecto a los scripts anteriores:
    CONTEXT_STRAINS = 3      (paper usa 3 meses; scripts anteriores usaban 1)
    Sin truncado de contexto (3 cepas ≈ 9300 tokens < n_positions=13000)
    KV caching              (~25x más rápido que recomputar el forward completo)
    Generación libre        (el modelo emite su propia estructura, como en entrenamiento)
    FP32 en MPS             (estable, sin artefactos numéricos de FP16 CUDA)

Uso:
    python replicate_paper_mac.py --test     # 10 ventanas por subtipo (~5 min)
    python replicate_paper_mac.py            # todas las ventanas (~3 h)
    caffeinate -i python replicate_paper_mac.py   # evitar modo reposo en Mac
"""

import os
import json
import time
import argparse
import math
import hashlib
import random

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from antigen_model import GPTForFluMultiTask
from influ_tokenizer import InfluTokenizer

# ---------------------------------------------------------------------------
# Configuración — fiel al paper
# ---------------------------------------------------------------------------

PROCESSED_DIR = "data/processed_gisaid"
FIGURES_DIR   = "figures/replication"
RESULTS_DIR   = "results"

CONTEXT_STRAINS      = 3        # paper: contexto de 3 meses [t-2, t-1, t]
MAX_GENERATE         = 3500     # HA (~1700) + NA (~1400) + tokens estructurales + margen
N_POSITIONS_MAX      = 13000    # límite del modelo (n_positions en config.json)
CONTEXT_TOKEN_BUDGET = 4500     # tokens totales de contexto (~1500/cepa); reduce la
                                # matriz de atención 9k×9k → 4.5k×4.5k en el primer paso
DEBUG_MAX_NEW_TOKENS = 128      # límite seguro para diagnóstico; no es réplica completa
POLICIES = {
    "same_group_same_time",
    "same_group_near_time",
    "same_group_past_only",
    "length_matched_random",
    "global_random",
}
NEAR_TIME_MONTHS = 3
LENGTH_MATCH_FRACTION = 0.03

# Modo test: evaluación rápida para verificar el pipeline antes del run completo
MAX_WINDOWS_TEST = 10

# ---------------------------------------------------------------------------
# Tabla de codones (estándar genético)
# ---------------------------------------------------------------------------

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}


def get_pyplot():
    """Importa Matplotlib solo cuando se generan figuras."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def load_model(device: torch.device) -> GPTForFluMultiTask:
    """
    Carga AntigenLM con los pesos reales del checkpoint.

    El checkpoint usa prefijo 'transformer.' para las capas del backbone
    y almacena 'lm_head.weight' sin prefijo (weight tying desactivado en guardado).
    Se remapean a 'backbone.*' para coincidir con GPTForFluMultiTask.

    Las 12 claves unexpected (backbone.h.*.attn.bias y attn.masked_bias) son
    buffers de la máscara causal de transformers 4.29.2 — completamente inofensivos.
    """
    print("Cargando modelo AntigenLM...")
    t0 = time.time()

    ckpt = torch.load(
        "prediction_sequence/pytorch_model.bin",
        map_location="cpu",
        weights_only=False,
    )

    remapped = {}
    for k, v in ckpt.items():
        if k.startswith("transformer."):
            remapped[k.replace("transformer.", "backbone.")] = v
        else:
            remapped[k] = v  # lm_head.weight no tiene prefijo

    model = GPTForFluMultiTask(task="prediction")
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    if missing:
        raise RuntimeError(f"Pesos faltantes — revisar checkpoint: {missing}")

    # Verificar que los unexpected son solo los buffers conocidos de attn
    non_buffer = [k for k in unexpected
                  if "attn.bias" not in k and "attn.masked_bias" not in k]
    if non_buffer:
        raise RuntimeError(f"Claves inesperadas desconocidas: {non_buffer}")

    model.eval()
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  OK | {n_params:,} params | dispositivo: {device} | {time.time()-t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Traducción DNA → aminoácidos
# ---------------------------------------------------------------------------

def translate_dna(seq: str) -> str:
    """Traduce secuencia de DNA a proteína. Para en el primer codón de stop."""
    seq = seq.upper().replace("N", "A")
    aa  = []
    for i in range(0, len(seq) - 2, 3):
        codon = CODON_TABLE.get(seq[i:i+3], "X")
        if codon == "*":
            break
        aa.append(codon)
    return "".join(aa)


def amino_acid_mismatch(pred_dna: str, real_dna: str):
    """
    Calcula Average Amino Acid Mismatch entre dos secuencias DNA.
    Métrica de Figura 3A del paper: número de posiciones de aminoácido
    que difieren entre la predicción y la cepa real.
    Retorna (n_mismatches, n_positions, mismatch_rate).
    """
    pred_aa = translate_dna(pred_dna)
    real_aa = translate_dna(real_dna)
    n = min(len(pred_aa), len(real_aa))
    if n == 0:
        return -1, 0, 0.0
    mm = sum(p != r for p, r in zip(pred_aa[:n], real_aa[:n]))
    return mm, n, mm / n


# ---------------------------------------------------------------------------
# Extracción de HA y NA desde tokens generados
# ---------------------------------------------------------------------------

def extract_ha_na(token_ids: list, tokenizer: InfluTokenizer):
    """
    Extrae las secuencias HA y NA de la lista de tokens generados.

    Formato esperado:
        [<subtipo>] <HA> ATCG... <sep> <NA> GCTA... <eos>

    El subtipo puede estar o no (dependiendo de si el modelo lo generó).
    """
    ha_id   = tokenizer.vocab["<HA>"]
    sep_id  = tokenizer.vocab["<sep>"]
    na_id   = tokenizer.vocab["<NA>"]
    eos_id  = tokenizer.vocab["<eos>"]
    nuc_ids = {0, 1, 2, 3, 4}  # A, C, G, T, N

    ha, na   = [], []
    segment  = None

    for tid in token_ids:
        if tid == ha_id:
            segment = "ha"
        elif tid == sep_id:
            segment = None
        elif tid == na_id:
            segment = "na"
        elif tid == eos_id:
            break
        elif tid in nuc_ids:
            if segment == "ha":
                ha.append(tokenizer.id_to_token_str(tid))
            elif segment == "na":
                na.append(tokenizer.id_to_token_str(tid))

    return "".join(ha), "".join(na)


# ---------------------------------------------------------------------------
# Utilidades de diagnóstico
# ---------------------------------------------------------------------------

def token_name(tokenizer: InfluTokenizer, token_id: int) -> str:
    """Representación legible de un token, incluidos tokens especiales."""
    return tokenizer.id_to_token_str(int(token_id))


def format_tokens(tokenizer: InfluTokenizer, token_ids: list[int], max_items: int = 40) -> str:
    shown = [token_name(tokenizer, t) for t in token_ids[:max_items]]
    if len(token_ids) > max_items:
        shown.append("...")
    return " ".join(shown)


def mps_memory(label: str, device: torch.device) -> None:
    """Imprime memoria MPS si el backend está disponible."""
    if device.type != "mps" or not hasattr(torch, "mps"):
        return

    allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
    driver = torch.mps.driver_allocated_memory() / (1024 ** 3)
    print(f"  MPS memoria {label}: allocated={allocated:.2f} GiB | driver={driver:.2f} GiB")


def clear_device_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def cache_seq_len(past_key_values) -> int | None:
    """Compatible con DynamicCache moderno y tuplas legacy de Transformers."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except TypeError:
            return int(past_key_values.get_seq_length(0))
    try:
        return int(past_key_values[0][0].shape[-2])
    except Exception:
        return None


def clone_past_key_values(past_key_values):
    """Clona KV cache para que una candidata no contamine a las demás."""
    if hasattr(past_key_values, "layers"):
        ddp_cache_data = []
        for layer in past_key_values.layers:
            if not getattr(layer, "is_initialized", False):
                ddp_cache_data.append((None, None))
                continue
            ddp_cache_data.append((layer.keys.clone(), layer.values.clone()))
        return DynamicCache(ddp_cache_data=ddp_cache_data)

    return tuple(
        tuple(t.clone() if torch.is_tensor(t) else t for t in layer)
        for layer in past_key_values
    )


def topk_tokens(logits: torch.Tensor, tokenizer: InfluTokenizer, k: int = 10) -> list[tuple[str, float]]:
    vals, ids = torch.topk(logits.detach().float().cpu(), k=min(k, logits.numel()))
    return [(token_name(tokenizer, int(tid)), float(val)) for tid, val in zip(ids, vals)]


def print_topk(label: str, logits: torch.Tensor, tokenizer: InfluTokenizer, k: int = 10) -> None:
    pairs = topk_tokens(logits, tokenizer, k=k)
    formatted = ", ".join(f"{tok}:{val:.3f}" for tok, val in pairs)
    print(f"  Top-{len(pairs)} {label}: {formatted}")


# ---------------------------------------------------------------------------
# Generación autorregresiva con KV caching
# ---------------------------------------------------------------------------

def generate_with_kv_cache(
    model: GPTForFluMultiTask,
    context_ids: list,
    tokenizer: InfluTokenizer,
    device: torch.device,
    max_new_tokens: int = MAX_GENERATE,
    inspect_topk: bool = False,
    debug_steps: int = 0,
) -> list:
    """
    Genera tokens autoregressivamente con KV caching (DynamicCache).

    Primer paso: forward completo sobre el contexto (≈9300 tokens).
    Pasos siguientes: forward sobre 1 solo token nuevo + cache acumulado.
    Esto elimina el O(L²) de atención en cada paso de generación:
    la complejidad por paso pasa de O(L²) a O(L) usando la cache.

    Retorna lista de IDs de tokens generados (sin incluir el contexto).
    """
    eos_id = tokenizer.eos_token_id

    new_tokens: list[int] = []
    stop_reason = "max_new_tokens"

    if len(context_ids) >= N_POSITIONS_MAX:
        raise ValueError(
            f"Contexto demasiado largo: {len(context_ids)} >= n_positions={N_POSITIONS_MAX}"
        )

    max_allowed = max(0, N_POSITIONS_MAX - len(context_ids) - 1)
    if max_new_tokens > max_allowed:
        print(
            f"  AVISO: max_new_tokens={max_new_tokens} excede el límite posicional; "
            f"usando {max_allowed}"
        )
        max_new_tokens = max_allowed

    with torch.inference_mode():
        # Prefill: un único forward sobre el contexto completo. Después de este
        # punto, cada paso debe recibir solo el último token y past_key_values.
        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        out = model.backbone(input_ids=x, use_cache=True, return_dict=True)
        pkv = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]
        logits = model.lm_head(hidden)[0]
        del x, out, hidden

        if inspect_topk:
            print_topk("siguiente token tras contexto", logits, tokenizer)
        print(f"  Cache: {type(pkv).__name__} | seq_len={cache_seq_len(pkv)}")

        for step in range(max_new_tokens):
            next_id = int(torch.argmax(logits).item())
            del logits

            new_tokens.append(next_id)

            if debug_steps and step < debug_steps:
                print(f"  step {step+1:02d}: {token_name(tokenizer, next_id)} ({next_id})")

            if next_id == eos_id:
                stop_reason = "eos"
                break

            # Guardia de límite de posición del modelo
            if (len(context_ids) + len(new_tokens)) >= N_POSITIONS_MAX - 1:
                stop_reason = "n_positions"
                break

            last = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = model.backbone(
                input_ids=last,
                past_key_values=pkv,
                use_cache=True,
                return_dict=True,
            )
            pkv = out.past_key_values
            hidden = out.last_hidden_state[:, -1, :]
            logits = model.lm_head(hidden)[0]
            del last, out, hidden

            if inspect_topk and step + 1 < min(debug_steps, 20):
                print_topk(f"step {step+2:02d}", logits, tokenizer)

    print(f"  Detención: {stop_reason} | generados={len(new_tokens)} | cache_seq_len={cache_seq_len(pkv)}")
    del pkv
    clear_device_cache(device)
    return new_tokens


def score_sequence_conditional(
    model: GPTForFluMultiTask,
    tokenizer: InfluTokenizer,
    context_ids: list[int],
    continuation_ids: list[int],
    device: torch.device,
) -> dict:
    """
    Calcula NLL(continuation | context) solo sobre continuation_ids.

    Implementación streaming con KV cache:
      1. Prefill del contexto completo.
      2. El logit del último token del contexto predice continuation[0].
      3. Cada token gold de la continuación alimenta el siguiente paso.

    No guarda logits históricos ni calcula pérdida sobre el contexto.
    """
    if not continuation_ids:
        return {
            "num_tokens": 0,
            "total_nll": 0.0,
            "mean_nll": float("nan"),
            "perplexity": float("nan"),
        }

    total_len = len(context_ids) + len(continuation_ids)
    if total_len > N_POSITIONS_MAX:
        raise ValueError(
            f"context + continuation excede n_positions: {total_len} > {N_POSITIONS_MAX}"
        )

    model.eval()
    total_nll = 0.0
    num_tokens = 0

    with torch.inference_mode():
        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        out = model.backbone(input_ids=x, use_cache=True, return_dict=True)
        pkv = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]
        logits = model.lm_head(hidden)[0]
        del x, out, hidden

        for idx, gold_id in enumerate(continuation_ids):
            target = torch.tensor([int(gold_id)], dtype=torch.long, device=device)
            nll = torch.nn.functional.cross_entropy(
                logits.float().unsqueeze(0),
                target,
                reduction="sum",
            )
            total_nll += float(nll.detach().cpu())
            num_tokens += 1
            del logits

            if idx == len(continuation_ids) - 1:
                del target, nll
                break

            last = target.view(1, 1)
            out = model.backbone(
                input_ids=last,
                past_key_values=pkv,
                use_cache=True,
                return_dict=True,
            )
            pkv = out.past_key_values
            hidden = out.last_hidden_state[:, -1, :]
            logits = model.lm_head(hidden)[0]
            del target, nll, last, out, hidden

    mean_nll = total_nll / num_tokens if num_tokens else float("nan")
    del pkv
    clear_device_cache(device)
    return {
        "num_tokens": num_tokens,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if math.isfinite(mean_nll) else float("nan"),
    }


def score_continuation_from_prefill(
    model: GPTForFluMultiTask,
    base_past_key_values,
    base_logits: torch.Tensor,
    continuation_ids: list[int],
    device: torch.device,
) -> dict:
    """Calcula NLL de una continuación reutilizando un prefill de contexto."""
    if not continuation_ids:
        return {
            "num_tokens": 0,
            "total_nll": 0.0,
            "mean_nll": float("nan"),
            "perplexity": float("nan"),
        }

    model.eval()
    pkv = clone_past_key_values(base_past_key_values)
    logits = base_logits.clone()
    total_nll = 0.0
    num_tokens = 0

    with torch.inference_mode():
        for idx, gold_id in enumerate(continuation_ids):
            target = torch.tensor([int(gold_id)], dtype=torch.long, device=device)
            nll = torch.nn.functional.cross_entropy(
                logits.float().unsqueeze(0),
                target,
                reduction="sum",
            )
            total_nll += float(nll.detach().cpu())
            num_tokens += 1
            del logits

            if idx == len(continuation_ids) - 1:
                del target, nll
                break

            last = target.view(1, 1)
            out = model.backbone(
                input_ids=last,
                past_key_values=pkv,
                use_cache=True,
                return_dict=True,
            )
            pkv = out.past_key_values
            hidden = out.last_hidden_state[:, -1, :]
            logits = model.lm_head(hidden)[0]
            del target, nll, last, out, hidden

    mean_nll = total_nll / num_tokens if num_tokens else float("nan")
    del pkv
    clear_device_cache(device)
    return {
        "num_tokens": num_tokens,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if math.isfinite(mean_nll) else float("nan"),
    }


def score_candidates_with_shared_context_prefill(
    model: GPTForFluMultiTask,
    tokenizer: InfluTokenizer,
    context_ids: list[int],
    candidates: list[dict],
    device: torch.device,
) -> list[dict]:
    """
    Puntúa varias candidatas compartiendo un único prefill del contexto.

    Cada candidata recibe una copia profunda de la cache base para evitar que
    los tokens de una continuación modifiquen el estado usado por otra.
    """
    del tokenizer  # La firma lo mantiene explícito para simetría con otros scorers.
    model.eval()

    with torch.inference_mode():
        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        out = model.backbone(input_ids=x, use_cache=True, return_dict=True)
        base_past_key_values = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]
        base_logits = model.lm_head(hidden)[0].detach()
        del x, out, hidden

    scored = []
    for candidate in candidates:
        stats = score_continuation_from_prefill(
            model=model,
            base_past_key_values=base_past_key_values,
            base_logits=base_logits,
            continuation_ids=candidate["ids"],
            device=device,
        )
        scored.append({**candidate, **stats})

    del base_past_key_values, base_logits
    clear_device_cache(device)
    return scored


def score_gold_continuation(
    model: GPTForFluMultiTask,
    context_ids: list[int],
    continuation_ids: list[int],
    tokenizer: InfluTokenizer,
    device: torch.device,
    max_tokens: int,
    inspect_topk: bool = False,
) -> dict:
    """Wrapper legado para el modo --debug-one --no-free-generation."""
    eval_ids = continuation_ids[:max_tokens]
    stats = score_sequence_conditional(
        model=model,
        tokenizer=tokenizer,
        context_ids=context_ids,
        continuation_ids=eval_ids,
        device=device,
    )
    return {
        "n_tokens": stats["num_tokens"],
        "total_nll": stats["total_nll"],
        "mean_nll": stats["mean_nll"],
        "perplexity": stats["perplexity"],
    }


def score_sequence_conditional_full_forward(
    model: GPTForFluMultiTask,
    context_ids: list[int],
    continuation_ids: list[int],
    device: torch.device,
) -> dict:
    """
    Referencia para tests: forward completo y CE solo sobre continuación.

    Si C=len(context_ids), L=len(continuation_ids):
      labels = full_ids[C : C + L]
      logits = logits[:, C-1 : C+L-1, :]
    """
    if not continuation_ids:
        return {
            "num_tokens": 0,
            "total_nll": 0.0,
            "mean_nll": float("nan"),
            "perplexity": float("nan"),
        }
    if not context_ids:
        raise ValueError("context_ids debe tener al menos un token")

    model.eval()
    full_ids = context_ids + continuation_ids
    c_len = len(context_ids)
    l_len = len(continuation_ids)

    with torch.inference_mode():
        x = torch.tensor([full_ids], dtype=torch.long, device=device)
        out = model.backbone(input_ids=x, use_cache=False, return_dict=True)
        logits = model.lm_head(out.last_hidden_state)
        pred_logits = logits[:, c_len - 1 : c_len + l_len - 1, :]
        labels = torch.tensor([continuation_ids], dtype=torch.long, device=device)
        total_nll_tensor = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)).float(),
            labels.reshape(-1),
            reduction="sum",
        )
        total_nll = float(total_nll_tensor.detach().cpu())
        del x, out, logits, pred_logits, labels, total_nll_tensor

    mean_nll = total_nll / l_len
    clear_device_cache(device)
    return {
        "num_tokens": l_len,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if math.isfinite(mean_nll) else float("nan"),
    }


def test_score_sequence_conditional_equivalence(
    model: GPTForFluMultiTask,
    tokenizer: InfluTokenizer,
    device: torch.device,
) -> None:
    """Compara KV-cache scoring vs forward completo en ejemplos sintéticos."""
    tol = 1e-3 if device.type == "mps" else 1e-4
    v = tokenizer.vocab
    cases = [
        {
            "case_id": "len_1",
            "context_ids": [v["<H3N2>"], v["<HA>"], v["A"], v["C"], v["G"]],
            "continuation_ids": [v["T"]],
        },
        {
            "case_id": "len_gt_1",
            "context_ids": [v["<H3N2>"], v["<HA>"], v["A"], v["C"], v["G"], v["T"]],
            "continuation_ids": [v["A"], v["C"], v["<sep>"], v["<NA>"], v["G"], v["<eos>"]],
        },
    ]

    print("\n" + "=" * 60)
    print("SCORE UNIT TEST: KV cache vs full forward")
    print("=" * 60)
    print(f"  device: {device}")
    print(f"  tolerance: {tol:g}")
    print("  case_id     C   L   total_cache  total_full   abs_diff   rel_diff")

    failures = []
    for case in cases:
        cache_stats = score_sequence_conditional(
            model=model,
            tokenizer=tokenizer,
            context_ids=case["context_ids"],
            continuation_ids=case["continuation_ids"],
            device=device,
        )
        full_stats = score_sequence_conditional_full_forward(
            model=model,
            context_ids=case["context_ids"],
            continuation_ids=case["continuation_ids"],
            device=device,
        )

        total_cache = cache_stats["total_nll"]
        total_full = full_stats["total_nll"]
        abs_diff = abs(total_cache - total_full)
        rel_diff = abs_diff / max(abs(total_full), 1e-12)

        print(
            f"  {case['case_id']:<10} "
            f"{len(case['context_ids']):>3} "
            f"{len(case['continuation_ids']):>3} "
            f"{total_cache:>13.6f} "
            f"{total_full:>11.6f} "
            f"{abs_diff:>10.6g} "
            f"{rel_diff:>10.6g}"
        )

        same_total = abs_diff <= tol
        same_mean = abs(cache_stats["mean_nll"] - full_stats["mean_nll"]) <= tol
        same_count = cache_stats["num_tokens"] == full_stats["num_tokens"]
        if not (same_total and same_mean and same_count):
            failures.append({
                "case_id": case["case_id"],
                "total_nll_cache": total_cache,
                "total_nll_full": total_full,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            })

    if failures:
        print("  status: FAIL")
        for failure in failures:
            print(
                "  failure "
                f"{failure['case_id']}: "
                f"total_nll_cache={failure['total_nll_cache']:.8f} "
                f"total_nll_full={failure['total_nll_full']:.8f} "
                f"abs_diff={failure['abs_diff']:.8g} "
                f"rel_diff={failure['rel_diff']:.8g}"
            )
        raise AssertionError("score_sequence_conditional no coincide con forward completo")

    print("  status: PASS")


def test_shared_prefill_equivalence(
    model: GPTForFluMultiTask,
    tokenizer: InfluTokenizer,
    device: torch.device,
) -> None:
    """Compara scoring repetido vs scoring con prefill compartido."""
    tol = 1e-3 if device.type == "mps" else 1e-4
    v = tokenizer.vocab
    context_ids = [v["<H3N2>"], v["<HA>"], v["A"], v["C"], v["G"], v["T"]]
    candidates = [
        {"anon_id": "candidate_000", "ids": [v["A"]], "is_target": False},
        {"anon_id": "candidate_001", "ids": [v["A"], v["C"], v["<sep>"], v["<NA>"], v["G"], v["<eos>"]], "is_target": True},
        {"anon_id": "candidate_002", "ids": [v["T"], v["G"], v["A"], v["A"]], "is_target": False},
    ]

    repeated_rows = []
    for candidate in candidates:
        stats = score_sequence_conditional(
            model=model,
            tokenizer=tokenizer,
            context_ids=context_ids,
            continuation_ids=candidate["ids"],
            device=device,
        )
        repeated_rows.append({**candidate, **stats})

    shared_rows = score_candidates_with_shared_context_prefill(
        model=model,
        tokenizer=tokenizer,
        context_ids=context_ids,
        candidates=candidates,
        device=device,
    )

    repeated_by_id = {row["anon_id"]: row for row in repeated_rows}
    shared_by_id = {row["anon_id"]: row for row in shared_rows}
    failures = []

    print("\n" + "=" * 60)
    print("SHARED PREFILL UNIT TEST")
    print("=" * 60)
    print(f"  device: {device}")
    print(f"  tolerance: {tol:g}")
    print("  anon_id        total_repeated total_shared  abs_diff   mean_diff")

    for anon_id in sorted(repeated_by_id):
        repeated = repeated_by_id[anon_id]
        shared = shared_by_id[anon_id]
        total_diff = abs(repeated["total_nll"] - shared["total_nll"])
        mean_diff = abs(repeated["mean_nll"] - shared["mean_nll"])
        print(
            f"  {anon_id:<13} "
            f"{repeated['total_nll']:>14.6f} "
            f"{shared['total_nll']:>12.6f} "
            f"{total_diff:>9.6g} "
            f"{mean_diff:>9.6g}"
        )
        if (
            repeated["num_tokens"] != shared["num_tokens"]
            or total_diff > tol
            or mean_diff > tol
        ):
            failures.append(anon_id)

    repeated_ranking = [row["anon_id"] for row in sorted(repeated_rows, key=lambda r: r["mean_nll"])]
    shared_ranking = [row["anon_id"] for row in sorted(shared_rows, key=lambda r: r["mean_nll"])]
    print(f"  ranking_repeated: {','.join(repeated_ranking)}")
    print(f"  ranking_shared:   {','.join(shared_ranking)}")
    if repeated_ranking != shared_ranking:
        failures.append("ranking")

    if failures:
        print("  status: FAIL")
        raise AssertionError(f"shared prefill no coincide: {failures}")

    print("  status: PASS")


# ---------------------------------------------------------------------------
# Evaluación de ventanas temporales
# ---------------------------------------------------------------------------

def evaluate_windows(
    model: GPTForFluMultiTask,
    tokenizer: InfluTokenizer,
    dataset: dict,
    device: torch.device,
    max_windows: int | None = None,
    max_new_tokens: int = MAX_GENERATE,
    inspect_topk: bool = False,
) -> list:
    """
    Evalúa el modelo en las ventanas temporales del dataset.

    Para cada ventana [t-2, t-1, t] → t+1:
      - Construye el contexto concatenando las 3 cepas dominantes completas
      - Genera la predicción de t+1 con KV caching
      - Calcula amino acid mismatch vs la cepa real de t+1
    """
    windows = dataset["windows"]
    strains = dataset["paired_strains"]
    subtype_token = strains[0]["subtype_token"]

    if max_windows is not None:
        windows = windows[:max_windows]

    n_total  = len(windows)
    t_global = time.time()

    print(f"\n  Subtipo: {subtype_token} | ventanas: {n_total}")
    print(f"  Contexto: {CONTEXT_STRAINS} cepas completas (~{CONTEXT_STRAINS*3100} tokens)")
    print(f"  Generación: autorregresiva libre con KV caching")
    print()

    results = []

    for w_idx, window in enumerate(windows):
        ctx = window["context"]
        tgt = window["target"]

        # -------------------------------------------------------
        # Contexto: concatenar las 3 cepas de contexto completas
        # -------------------------------------------------------
        context_ids: list[int] = []
        ok = True

        for c in ctx:
            strain = strains[c["strain_idx"]]
            if not strain.get("ha_sequence") or not strain.get("na_sequence"):
                ok = False
                break
            ids = tokenizer.encode_strain(
                ha_sequence=strain["ha_sequence"],
                na_sequence=strain["na_sequence"],
                subtype=subtype_token,
            )
            context_ids.extend(ids)

        if not ok or len(context_ids) < 200:
            continue

        # Cepa real del mes target
        target = strains[tgt["strain_idx"]]
        real_ha = target.get("ha_sequence", "")
        real_na = target.get("na_sequence", "")
        if not real_ha or not real_na:
            continue

        # Guardia posicional: depende del límite real de generación solicitado.
        budget = N_POSITIONS_MAX - max_new_tokens - 1
        if len(context_ids) > budget:
            context_ids = context_ids[-budget:]

        # -------------------------------------------------------
        # Generación con KV caching
        # -------------------------------------------------------
        t0 = time.time()
        new_tokens = generate_with_kv_cache(
            model=model,
            context_ids=context_ids,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            inspect_topk=inspect_topk and w_idx == 0,
            debug_steps=20 if inspect_topk and w_idx == 0 else 0,
        )
        t_gen = time.time() - t0

        # -------------------------------------------------------
        # Extraer HA y NA — verificar estructura generada
        # -------------------------------------------------------
        pred_ha, pred_na = extract_ha_na(new_tokens, tokenizer)

        # Diagnóstico: qué emitió el modelo como primeros tokens
        first_names = [tokenizer.id_to_token_str(t) for t in new_tokens[:6]]
        has_structure = (
            tokenizer.vocab["<HA>"] in new_tokens[:8]
            and tokenizer.vocab["<NA>"] in new_tokens
        )

        # -------------------------------------------------------
        # Calcular mismatch
        # -------------------------------------------------------
        ha_mm, ha_n, ha_rate = (-1, 0, 0.0)
        na_mm, na_n, na_rate = (-1, 0, 0.0)

        if len(pred_ha) > 100:
            ha_mm, ha_n, ha_rate = amino_acid_mismatch(pred_ha, real_ha)
        if len(pred_na) > 100:
            na_mm, na_n, na_rate = amino_acid_mismatch(pred_na, real_na)

        # -------------------------------------------------------
        # Progreso con ETA
        # -------------------------------------------------------
        n_done   = w_idx + 1
        elapsed  = time.time() - t_global
        eta_s    = (elapsed / n_done) * (n_total - n_done)

        ha_str = f"{ha_mm}aa" if ha_mm >= 0 else f"corta({len(pred_ha)}nt)"
        na_str = f"{na_mm}aa" if na_mm >= 0 else f"corta({len(pred_na)}nt)"

        print(
            f"  [{n_done:3d}/{n_total}] "
            f"{tgt['year']}-{tgt['month']:02d} | "
            f"ctx={len(context_ids)} | "
            f"gen={len(new_tokens)}tok {t_gen:.0f}s | "
            f"struct={'OK' if has_structure else 'NO'} "
            f"inicio={' '.join(first_names[:4])} | "
            f"HA={ha_str} NA={na_str} | "
            f"ETA {eta_s/60:.0f}min"
        )

        results.append({
            "window_idx":    w_idx,
            "target_year":   tgt["year"],
            "target_month":  tgt["month"],
            "context_tokens": len(context_ids),
            "n_generated":   len(new_tokens),
            "has_structure": has_structure,
            "first_tokens":  first_names,
            "pred_ha_len":   len(pred_ha),
            "pred_na_len":   len(pred_na),
            "ha_mismatch":   ha_mm,
            "ha_positions":  ha_n,
            "ha_rate":       round(ha_rate, 4),
            "na_mismatch":   na_mm,
            "na_positions":  na_n,
            "na_rate":       round(na_rate, 4),
            "gen_seconds":   round(t_gen, 2),
        })

    return results


def load_dataset(subtype: str) -> dict | None:
    fpath = os.path.join(PROCESSED_DIR, f"dataset_{subtype}.json")
    if not os.path.exists(fpath):
        return None
    with open(fpath, "r") as f:
        return json.load(f)


def month_index(year: int, month: int) -> int:
    return int(year) * 12 + int(month)


def record_group(record: dict) -> str:
    return str(record.get("subtype") or record.get("subtype_token") or "unknown")


def iter_all_records(datasets: dict[str, dict]) -> list[dict]:
    records = []
    for dataset_name, dataset in datasets.items():
        for source_idx, strain in enumerate(dataset.get("paired_strains", [])):
            if not strain.get("ha_sequence") or not strain.get("na_sequence"):
                continue
            year = strain.get("year")
            month = strain.get("month")
            if year is None or month is None:
                continue
            records.append({
                "dataset": dataset_name,
                "source_idx": source_idx,
                "record": strain,
                "group": record_group(strain),
                "year": int(year),
                "month": int(month),
                "month_index": month_index(int(year), int(month)),
            })
    return records


def get_record_by_index(dataset_name: str, dataset: dict, source_idx: int) -> dict:
    strain = dataset["paired_strains"][source_idx]
    return {
        "dataset": dataset_name,
        "source_idx": int(source_idx),
        "record": strain,
        "group": record_group(strain),
        "year": int(strain["year"]),
        "month": int(strain["month"]),
        "month_index": month_index(int(strain["year"]), int(strain["month"])),
    }


def build_window_context(
    tokenizer: InfluTokenizer,
    dataset: dict,
    window_idx: int = 0,
) -> tuple[list[int], list[int], str, dict]:
    windows = dataset["windows"]
    strains = dataset["paired_strains"]
    window = windows[window_idx]
    subtype_token = strains[0].get("subtype_token", f"<{dataset.get('subtype', 'H3N2')}>")

    context_ids: list[int] = []
    for c in window["context"]:
        strain = strains[c["strain_idx"]]
        if not strain.get("ha_sequence") or not strain.get("na_sequence"):
            raise ValueError("Ventana con cepa de contexto sin HA/NA")
        context_ids.extend(
            tokenizer.encode_strain(
                ha_sequence=strain["ha_sequence"],
                na_sequence=strain["na_sequence"],
                subtype=subtype_token,
            )
        )

    target = strains[window["target"]["strain_idx"]]
    if not target.get("ha_sequence") or not target.get("na_sequence"):
        raise ValueError("Cepa target sin HA/NA")

    target_ids = tokenizer.encode_strain(
        ha_sequence=target["ha_sequence"],
        na_sequence=target["na_sequence"],
        subtype=subtype_token,
    )
    return context_ids, target_ids, subtype_token, window


def run_debug_one(
    model: GPTForFluMultiTask,
    tokenizer: InfluTokenizer,
    device: torch.device,
    max_new_tokens: int,
    inspect_topk: bool,
    no_free_generation: bool,
) -> None:
    """Ejecuta una sola ventana corta sin escribir figuras ni JSON."""
    for subtype in ["H3N2", "H1N1"]:
        dataset = load_dataset(subtype)
        if dataset is not None and dataset.get("windows"):
            break
    else:
        raise FileNotFoundError("No se encontró dataset H3N2/H1N1 con ventanas")

    context_ids, target_ids, subtype_token, window = build_window_context(
        tokenizer=tokenizer,
        dataset=dataset,
        window_idx=0,
    )

    print("\n" + "=" * 60)
    print("MODO DIAGNÓSTICO SEGURO: una ventana")
    print("=" * 60)
    print(f"  Subtipo: {subtype_token}")
    print(f"  Target:  {window['target']['year']}-{window['target']['month']:02d}")
    print(f"  Contexto tokens: {len(context_ids)}")
    print(f"  Target tokens reales: {len(target_ids)}")
    print(f"  Primeros tokens contexto: {format_tokens(tokenizer, context_ids, 32)}")
    print(f"  Últimos tokens contexto:  {format_tokens(tokenizer, context_ids[-32:], 32)}")
    print(f"  Primeros tokens target:   {format_tokens(tokenizer, target_ids, 32)}")
    print(f"  max_new_tokens: {max_new_tokens}")
    mps_memory("antes", device)

    if no_free_generation:
        stats = score_gold_continuation(
            model=model,
            context_ids=context_ids,
            continuation_ids=target_ids,
            tokenizer=tokenizer,
            device=device,
            max_tokens=max_new_tokens,
            inspect_topk=inspect_topk,
        )
        print(
            "  Resultado no-generativo: "
            f"n={stats['n_tokens']} mean_nll={stats['mean_nll']:.3f} "
            f"ppl={stats['perplexity']:.2f}"
        )
        print("  Detención: max_new_tokens sobre continuación real")
        mps_memory("después", device)
        return

    t0 = time.time()
    new_tokens = generate_with_kv_cache(
        model=model,
        context_ids=context_ids,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        inspect_topk=inspect_topk,
        debug_steps=20,
    )
    elapsed = time.time() - t0
    pred_ha, pred_na = extract_ha_na(new_tokens, tokenizer)
    specials = {
        tok: new_tokens.count(tokenizer.vocab[tok])
        for tok in ["<HA>", "<sep>", "<NA>", "<eos>", "<H3N2>", "<H1N1>"]
        if tok in tokenizer.vocab
    }

    print(f"  Generados: {len(new_tokens)} tokens en {elapsed:.1f}s")
    print(f"  Primeros generados: {format_tokens(tokenizer, new_tokens, 80)}")
    print(f"  Tokens estructurales: {specials}")
    print(f"  HA extraída: {len(pred_ha)} nt | NA extraída: {len(pred_na)} nt")
    if tokenizer.vocab["<HA>"] not in new_tokens or tokenizer.vocab["<NA>"] not in new_tokens:
        print("  Conclusión diagnóstico: struct=NO; la generación libre no produjo HA/NA interpretable.")
    else:
        print("  Conclusión diagnóstico: aparecieron marcadores estructurales en la salida corta.")
    mps_memory("después", device)


def build_score_debug_candidates(
    tokenizer: InfluTokenizer,
    dataset: dict,
    window: dict,
    subtype_token: str,
    max_candidates: int,
) -> list[dict]:
    """Construye continuaciones candidatas con IDs anonimizados para debug técnico."""
    strains = dataset["paired_strains"]
    target_idx = int(window["target"]["strain_idx"])
    ordered_indices = [target_idx]
    ordered_indices.extend(i for i in range(len(strains)) if i != target_idx)

    candidates = []
    for strain_idx in ordered_indices:
        if len(candidates) >= max_candidates:
            break

        strain = strains[strain_idx]
        if not strain.get("ha_sequence") or not strain.get("na_sequence"):
            continue

        candidate_ids = tokenizer.encode_strain(
            ha_sequence=strain["ha_sequence"],
            na_sequence=strain["na_sequence"],
            subtype=subtype_token,
        )
        candidates.append({
            "anon_id": f"candidate_{len(candidates):03d}",
            "source_idx": strain_idx,
            "ids": candidate_ids,
            "is_target": strain_idx == target_idx,
        })

    return candidates


def ids_hash(token_ids: list[int], n: int = 12) -> str:
    payload = ",".join(str(int(t)) for t in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:n]


def record_uid(record_info: dict) -> tuple[str, int]:
    return (record_info["dataset"], int(record_info["source_idx"]))


def encode_record_ids(tokenizer: InfluTokenizer, record_info: dict) -> list[int]:
    strain = record_info["record"]
    subtype_token = strain.get("subtype_token")
    if subtype_token is None:
        subtype = strain.get("subtype")
        subtype_token = f"<{subtype}>" if subtype else None
    return tokenizer.encode_strain(
        ha_sequence=strain["ha_sequence"],
        na_sequence=strain["na_sequence"],
        subtype=subtype_token,
    )


def list_contains_subsequence(haystack: list[int], needle: list[int]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    max_start = len(haystack) - len(needle)
    for start in range(max_start + 1):
        if haystack[start] == first and haystack[start:start + len(needle)] == needle:
            return True
    return False


def build_candidate_set(
    tokenizer: InfluTokenizer,
    target_record: dict,
    all_records: list[dict],
    context_records: list[dict],
    policy: str,
    max_candidates: int,
    seed: int,
) -> tuple[list[dict], dict]:
    if policy not in POLICIES:
        raise ValueError(f"Política desconocida: {policy}")

    rng = random.Random(seed)
    target_uid = record_uid(target_record)
    context_uids = {record_uid(r) for r in context_records}
    target_ids = encode_record_ids(tokenizer, target_record)
    target_len = len(target_ids)
    target_month = target_record["month_index"]
    target_group = target_record["group"]

    def same_group(r: dict) -> bool:
        return r["group"] == target_group

    def same_time(r: dict) -> bool:
        return r["year"] == target_record["year"] and r["month"] == target_record["month"]

    if policy == "same_group_same_time":
        eligible = [r for r in all_records if same_group(r) and same_time(r)]
    elif policy == "same_group_near_time":
        eligible = [
            r for r in all_records
            if same_group(r) and abs(r["month_index"] - target_month) <= NEAR_TIME_MONTHS
        ]
    elif policy == "same_group_past_only":
        eligible = [
            r for r in all_records
            if same_group(r) and r["month_index"] < target_month
        ]
    elif policy == "length_matched_random":
        tolerance = max(1, int(round(target_len * LENGTH_MATCH_FRACTION)))
        eligible = []
        for r in all_records:
            try:
                r_len = len(encode_record_ids(tokenizer, r))
            except Exception:
                continue
            if abs(r_len - target_len) <= tolerance:
                eligible.append(r)
    else:  # global_random
        eligible = list(all_records)

    eligible_count_before_exclusions = len(eligible)
    negative_pool = [
        r for r in eligible
        if record_uid(r) != target_uid and record_uid(r) not in context_uids
    ]

    rng.shuffle(negative_pool)
    selected_records = [target_record] + negative_pool[:max(0, max_candidates - 1)]
    rng.shuffle(selected_records)

    candidates = []
    for record_info in selected_records:
        ids = encode_record_ids(tokenizer, record_info)
        candidates.append({
            "anon_id": f"candidate_{len(candidates):03d}",
            "dataset": record_info["dataset"],
            "source_idx": record_info["source_idx"],
            "group": record_info["group"],
            "year": record_info["year"],
            "month": record_info["month"],
            "month_index": record_info["month_index"],
            "ids": ids,
            "is_target": record_uid(record_info) == target_uid,
        })

    meta = {
        "policy": policy,
        "seed": seed,
        "eligible_count_before_exclusions": eligible_count_before_exclusions,
        "negative_count_after_exclusions": len(negative_pool),
        "target_uid": target_uid,
        "target_len": target_len,
        "target_year": target_record["year"],
        "target_month": target_record["month"],
        "target_group": target_group,
        "context_uids": context_uids,
    }
    return candidates, meta


def deduplicate_candidates_by_hash(candidates: list[dict]) -> tuple[list[dict], int]:
    """
    Deduplica continuaciones idénticas por hash exacto.

    Si una copia comparte hash con el target, se conserva el target y se elimina
    la copia no-target. Luego se reasignan anon_id sin dar posición especial.
    """
    if not candidates:
        return [], 0

    target_candidates = [c for c in candidates if c["is_target"]]
    target_hashes = {ids_hash(c["ids"]) for c in target_candidates}
    kept = []
    seen = set()

    for candidate in candidates:
        h = ids_hash(candidate["ids"])
        if candidate["is_target"]:
            if h in seen:
                continue
            kept.append(candidate)
            seen.add(h)
            continue
        if h in target_hashes or h in seen:
            continue
        kept.append(candidate)
        seen.add(h)

    # Si el target apareció después de una copia idéntica, reemplazar esa copia.
    if target_candidates and not any(c["is_target"] for c in kept):
        target = target_candidates[0]
        target_hash = ids_hash(target["ids"])
        kept = [c for c in kept if ids_hash(c["ids"]) != target_hash]
        kept.append(target)

    for idx, candidate in enumerate(kept):
        candidate["anon_id"] = f"candidate_{idx:03d}"

    return kept, len(candidates) - len(kept)


def harmonic_number(n: int) -> float:
    return float(sum(1.0 / k for k in range(1, n + 1))) if n > 0 else float("nan")


def audit_candidate_set(context_ids: list[int], candidates: list[dict]) -> dict:
    target_count = sum(1 for c in candidates if c["is_target"])
    context_overlap = {
        c["anon_id"]: list_contains_subsequence(context_ids, c["ids"])
        for c in candidates
    }
    hashes = {c["anon_id"]: ids_hash(c["ids"]) for c in candidates}
    hash_counts = {}
    for h in hashes.values():
        hash_counts[h] = hash_counts.get(h, 0) + 1
    duplicate_hashes = {h for h, count in hash_counts.items() if count > 1}
    duplicates = {
        c["anon_id"]: hashes[c["anon_id"]] in duplicate_hashes
        for c in candidates
    }
    lengths = [len(c["ids"]) for c in candidates]
    date_counts = {}
    group_counts = {}
    for c in candidates:
        date_key = f"{c.get('year', 'NA')}-{int(c.get('month', 0)):02d}"
        date_counts[date_key] = date_counts.get(date_key, 0) + 1
        group_key = str(c.get("group", "unknown"))
        group_counts[group_key] = group_counts.get(group_key, 0) + 1
    return {
        "target_count": target_count,
        "candidate_000_is_target": bool(candidates) and candidates[0]["is_target"],
        "context_overlap": context_overlap,
        "hashes": hashes,
        "duplicates": duplicates,
        "duplicate_hash_count": len(duplicate_hashes),
        "min_len": min(lengths) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "date_counts": date_counts,
        "group_counts": group_counts,
    }


def run_score_debug(
    model: GPTForFluMultiTask,
    tokenizer: InfluTokenizer,
    device: torch.device,
    max_candidates: int,
) -> None:
    """Debug técnico de NLL condicional; no imprime ni genera secuencias."""
    for subtype in ["H3N2", "H1N1"]:
        dataset = load_dataset(subtype)
        if dataset is not None and dataset.get("windows"):
            break
    else:
        raise FileNotFoundError("No se encontró un dataset con ventanas")

    context_ids, _target_ids, subtype_token, window = build_window_context(
        tokenizer=tokenizer,
        dataset=dataset,
        window_idx=0,
    )
    candidates = build_score_debug_candidates(
        tokenizer=tokenizer,
        dataset=dataset,
        window=window,
        subtype_token=subtype_token,
        max_candidates=max_candidates,
    )

    print("\n" + "=" * 60)
    print("SCORE DEBUG: NLL(continuation | context)")
    print("=" * 60)
    print(f"  device: {device}")
    print("  window_id: window_000")
    print(f"  context_len: {len(context_ids)}")
    print(f"  num_candidates: {len(candidates)}")

    rows = []
    for candidate in candidates:
        try:
            stats = score_sequence_conditional(
                model=model,
                tokenizer=tokenizer,
                context_ids=context_ids,
                continuation_ids=candidate["ids"],
                device=device,
            )
            rows.append({**candidate, **stats, "status": "ok"})
        except ValueError as exc:
            rows.append({
                **candidate,
                "num_tokens": len(candidate["ids"]),
                "total_nll": float("nan"),
                "mean_nll": float("nan"),
                "perplexity": float("nan"),
                "status": f"skipped:{exc}",
            })

    rows.sort(key=lambda r: r["mean_nll"] if math.isfinite(r["mean_nll"]) else float("inf"))

    print("  rank anon_id        cont_len  total_nll    mean_nll    perplexity   status")
    for rank, row in enumerate(rows, start=1):
        print(
            f"  {rank:>4} {row['anon_id']:<13} "
            f"{row['num_tokens']:>8} "
            f"{row['total_nll']:>10.3f} "
            f"{row['mean_nll']:>10.4f} "
            f"{row['perplexity']:>12.3f} "
            f"{row['status']}"
        )

    target_rank = next(
        (rank for rank, row in enumerate(rows, start=1) if row["is_target"]),
        None,
    )
    if target_rank is not None:
        print(f"  target_anon_id: candidate_000")
        print(f"  target_rank: {target_rank}")


def run_score_audit(
    model: GPTForFluMultiTask,
    tokenizer: InfluTokenizer,
    device: torch.device,
    max_candidates: int,
) -> None:
    """Audita construcción y ranking de un conjunto pequeño sin imprimir secuencias."""
    for subtype in ["H3N2", "H1N1"]:
        dataset = load_dataset(subtype)
        if dataset is not None and dataset.get("windows"):
            break
    else:
        raise FileNotFoundError("No se encontró un dataset con ventanas")

    context_ids, _target_ids, subtype_token, window = build_window_context(
        tokenizer=tokenizer,
        dataset=dataset,
        window_idx=0,
    )
    candidates = build_score_debug_candidates(
        tokenizer=tokenizer,
        dataset=dataset,
        window=window,
        subtype_token=subtype_token,
        max_candidates=max_candidates,
    )
    audit = audit_candidate_set(context_ids=context_ids, candidates=candidates)

    print("\n" + "=" * 60)
    print("SCORE AUDIT: candidate construction")
    print("=" * 60)
    print(f"  device: {device}")
    print("  window_id: window_000")
    print(f"  subtype_id: {ids_hash([tokenizer.vocab[subtype_token]], n=8)}")
    print(f"  context_len: {len(context_ids)}")
    print(f"  context_hash: {ids_hash(context_ids)}")
    print(f"  num_candidates: {len(candidates)}")
    print(f"  candidate_000_is_target: {audit['candidate_000_is_target']}")
    print(f"  target_count: {audit['target_count']}")
    print(f"  duplicate_hash_count: {audit['duplicate_hash_count']}")
    print(f"  continuation_len_min: {audit['min_len']}")
    print(f"  continuation_len_max: {audit['max_len']}")

    rows = []
    for candidate in candidates:
        stats = score_sequence_conditional(
            model=model,
            tokenizer=tokenizer,
            context_ids=context_ids,
            continuation_ids=candidate["ids"],
            device=device,
        )
        rows.append({
            **candidate,
            **stats,
            "hash": audit["hashes"][candidate["anon_id"]],
            "duplicate": audit["duplicates"][candidate["anon_id"]],
            "overlap_context": audit["context_overlap"][candidate["anon_id"]],
        })

    by_mean = sorted(rows, key=lambda r: r["mean_nll"])
    by_total = sorted(rows, key=lambda r: r["total_nll"])

    print("\n  candidates")
    print("  anon_id        hash          len   target duplicate overlap_context")
    for row in rows:
        print(
            f"  {row['anon_id']:<13} "
            f"{row['hash']:<12} "
            f"{row['num_tokens']:>5} "
            f"{str(row['is_target']):>8} "
            f"{str(row['duplicate']):>9} "
            f"{str(row['overlap_context']):>15}"
        )

    print("\n  ranking_by_mean_nll")
    print("  rank anon_id        len   mean_nll   total_nll   ppl")
    for rank, row in enumerate(by_mean, start=1):
        print(
            f"  {rank:>4} {row['anon_id']:<13} "
            f"{row['num_tokens']:>5} "
            f"{row['mean_nll']:>10.4f} "
            f"{row['total_nll']:>11.3f} "
            f"{row['perplexity']:>9.3f}"
        )

    print("\n  ranking_by_total_nll")
    print("  rank anon_id        len   total_nll   mean_nll")
    for rank, row in enumerate(by_total, start=1):
        print(
            f"  {rank:>4} {row['anon_id']:<13} "
            f"{row['num_tokens']:>5} "
            f"{row['total_nll']:>11.3f} "
            f"{row['mean_nll']:>10.4f}"
        )

    target_rank_mean = next((rank for rank, row in enumerate(by_mean, start=1) if row["is_target"]), None)
    target_rank_total = next((rank for rank, row in enumerate(by_total, start=1) if row["is_target"]), None)
    print(f"\n  target_anon_id: candidate_000")
    print(f"  target_rank_by_mean_nll: {target_rank_mean}")
    print(f"  target_rank_by_total_nll: {target_rank_total}")


def load_policy_audit_inputs(tokenizer: InfluTokenizer, window_idx: int = 0):
    datasets = {
        subtype: dataset
        for subtype in ["H3N2", "H1N1"]
        if (dataset := load_dataset(subtype)) is not None
    }
    if not datasets:
        raise FileNotFoundError("No se encontraron datasets procesados")

    dataset_name = "H3N2" if "H3N2" in datasets else next(iter(datasets))
    dataset = datasets[dataset_name]
    context_ids, _target_ids, _subtype_token, window = build_window_context(
        tokenizer=tokenizer,
        dataset=dataset,
        window_idx=window_idx,
    )

    context_records = [
        get_record_by_index(dataset_name, dataset, int(c["strain_idx"]))
        for c in window["context"]
    ]
    target_record = get_record_by_index(
        dataset_name,
        dataset,
        int(window["target"]["strain_idx"]),
    )
    all_records = iter_all_records(datasets)
    return datasets, all_records, context_records, target_record, context_ids, window, dataset_name


def print_policy_metadata_summary(datasets: dict[str, dict], all_records: list[dict]) -> None:
    record_keys = set()
    window_keys = set()
    top_keys = set()
    for dataset in datasets.values():
        top_keys.update(dataset.keys())
        for record in dataset.get("paired_strains", [])[:1000]:
            record_keys.update(record.keys())
        for window in dataset.get("windows", [])[:10]:
            window_keys.update(window.keys())

    print(f"  dataset_top_keys: {','.join(sorted(top_keys))}")
    print(f"  record_fields: {','.join(sorted(record_keys))}")
    print(f"  window_fields: {','.join(sorted(window_keys))}")
    print(f"  all_records_with_sequences: {len(all_records)}")


def run_score_policy_audit(
    model: GPTForFluMultiTask,
    tokenizer: InfluTokenizer,
    device: torch.device,
    policy: str,
    max_candidates: int,
    seed: int,
    num_windows: int | None,
    deduplicate_candidates: bool,
    profile: bool,
    shared_prefill: bool,
) -> None:
    """Audita una política de candidatos sin imprimir secuencias."""
    datasets, all_records, *_ = load_policy_audit_inputs(tokenizer, window_idx=0)

    dataset_name = "H3N2" if "H3N2" in datasets else next(iter(datasets))
    n_available = len(datasets[dataset_name].get("windows", []))
    n_windows = min(num_windows or 1, n_available)

    if n_windows <= 1:
        run_score_policy_audit_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            policy=policy,
            max_candidates=max_candidates,
            seed=seed,
            deduplicate_candidates=deduplicate_candidates,
            datasets=datasets,
            all_records=all_records,
            window_idx=0,
            profile=profile,
            shared_prefill=shared_prefill,
        )
        return

    total_t0 = time.time()
    print("\n" + "=" * 60)
    print("SCORE POLICY AUDIT: MULTI-WINDOW")
    print("=" * 60)
    print(f"  device: {device}")
    print(f"  policy: {policy}")
    print(f"  seed: {seed}")
    print(f"  deduplicate_candidates: {deduplicate_candidates}")
    print(f"  profile: {profile}")
    if profile:
        print(f"  prefill_recomputed_per_candidate: {not shared_prefill}")
        print(f"  shared_prefill: {shared_prefill}")
    print(f"  requested_windows: {num_windows}")
    print(f"  evaluated_window_limit: {n_windows}")
    print_policy_metadata_summary(datasets, all_records)
    print()
    if profile:
        print(
            "  window_id  target_month eligible final dup_removed "
            "rank_mean norm_rank pct_score rr top1 top5 top10 "
            "len_avg build_s score_s cand_s win_s"
        )
    else:
        print(
            "  window_id  target_month eligible final dup_removed "
            "rank_mean norm_rank pct_score rr top1 top5_app top5 top10_app top10 len_min len_max"
        )

    summaries = []
    skipped = 0
    for window_idx in range(n_windows):
        summary = score_policy_window(
            model=model,
            tokenizer=tokenizer,
            device=device,
            policy=policy,
            max_candidates=max_candidates,
            seed=seed,
            datasets=datasets,
            all_records=all_records,
            window_idx=window_idx,
            deduplicate_candidates=deduplicate_candidates,
            profile=profile,
            shared_prefill=shared_prefill,
        )
        if summary.get("skipped"):
            skipped += 1
        else:
            summaries.append(summary)

        if profile:
            print(
                f"  {summary['window_id']:<10} "
                f"{summary['target_month']:<12} "
                f"{summary['eligible_count']:>8} "
                f"{summary['num_candidates']:>5} "
                f"{summary['duplicates_removed']:>11} "
                f"{str(summary.get('target_rank_by_mean_nll')):>9} "
                f"{summary.get('normalized_rank', float('nan')):>9.4f} "
                f"{summary.get('percentile_score', float('nan')):>9.4f} "
                f"{summary.get('reciprocal_rank', float('nan')):>6.4f} "
                f"{str(summary.get('top1')):>4} "
                f"{str(summary.get('top5')):>4} "
                f"{str(summary.get('top10')):>5} "
                f"{summary.get('candidate_len_avg', float('nan')):>7.1f} "
                f"{summary.get('candidate_build_seconds', float('nan')):>7.2f} "
                f"{summary.get('scoring_seconds', float('nan')):>7.2f} "
                f"{summary.get('seconds_per_candidate', float('nan')):>6.2f} "
                f"{summary.get('window_seconds', float('nan')):>6.2f}",
                flush=True,
            )
        else:
            print(
                f"  {summary['window_id']:<10} "
                f"{summary['target_month']:<12} "
                f"{summary['eligible_count']:>8} "
                f"{summary['num_candidates']:>5} "
                f"{summary['duplicates_removed']:>11} "
                f"{str(summary.get('target_rank_by_mean_nll')):>9} "
                f"{summary.get('normalized_rank', float('nan')):>9.4f} "
                f"{summary.get('percentile_score', float('nan')):>9.4f} "
                f"{summary.get('reciprocal_rank', float('nan')):>6.4f} "
                f"{str(summary.get('top1')):>4} "
                f"{str(summary.get('top5_applicable')):>8} "
                f"{str(summary.get('top5')):>4} "
                f"{str(summary.get('top10_applicable')):>9} "
                f"{str(summary.get('top10')):>5} "
                f"{summary['len_min']:>7} "
                f"{summary['len_max']:>7}",
                flush=True,
            )

    ranks = [s["target_rank_by_mean_nll"] for s in summaries]
    normalized_ranks = [s["normalized_rank"] for s in summaries if math.isfinite(s["normalized_rank"])]
    percentile_scores = [s["percentile_score"] for s in summaries if math.isfinite(s["percentile_score"])]
    reciprocal_ranks = [s["reciprocal_rank"] for s in summaries if math.isfinite(s["reciprocal_rank"])]
    top5_applicable = [s for s in summaries if s["top5_applicable"]]
    top10_applicable = [s for s in summaries if s["top10_applicable"]]
    expected_mrr_random_values = [
        harmonic_number(s["num_candidates"]) / s["num_candidates"]
        for s in summaries
        if s["num_candidates"] > 0
    ]
    n_eval = len(summaries)
    print("\n  aggregate_summary")
    print(f"  windows_evaluated: {n_eval}")
    print(f"  windows_skipped_few_candidates: {skipped}")
    print(f"  avg_num_candidates: {float(np.mean([s['num_candidates'] for s in summaries])) if summaries else float('nan'):.4f}")
    print(f"  mean_rank: {float(np.mean(ranks)) if ranks else float('nan'):.4f}")
    print(f"  median_rank: {float(np.median(ranks)) if ranks else float('nan'):.4f}")
    print(f"  mean_normalized_rank: {float(np.mean(normalized_ranks)) if normalized_ranks else float('nan'):.4f}")
    print(f"  mean_percentile_score: {float(np.mean(percentile_scores)) if percentile_scores else float('nan'):.4f}")
    print(f"  mrr: {float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float('nan'):.4f}")
    print(f"  top1_accuracy: {sum(s['top1'] for s in summaries) / n_eval if n_eval else float('nan'):.4f}")
    print(f"  top5_applicable_windows: {len(top5_applicable)}")
    print(f"  top5_accuracy_applicable: {sum(s['top5'] for s in top5_applicable) / len(top5_applicable) if top5_applicable else float('nan'):.4f}")
    print(f"  top10_applicable_windows: {len(top10_applicable)}")
    print(f"  top10_accuracy_applicable: {sum(s['top10'] for s in top10_applicable) / len(top10_applicable) if top10_applicable else float('nan'):.4f}")
    print("  expected_normalized_rank_random: 0.5000")
    print(f"  expected_mrr_random: {float(np.mean(expected_mrr_random_values)) if expected_mrr_random_values else float('nan'):.4f}")
    print(f"  avg_duplicates_removed: {float(np.mean([s['duplicates_removed'] for s in summaries])) if summaries else float('nan'):.4f}")
    if profile:
        print(f"  total_seconds: {time.time() - total_t0:.2f}")
        print(f"  avg_window_seconds: {float(np.mean([s['window_seconds'] for s in summaries])) if summaries else float('nan'):.2f}")
        print(f"  avg_candidate_build_seconds: {float(np.mean([s['candidate_build_seconds'] for s in summaries])) if summaries else float('nan'):.2f}")
        print(f"  avg_scoring_seconds: {float(np.mean([s['scoring_seconds'] for s in summaries])) if summaries else float('nan'):.2f}")
        print(f"  avg_seconds_per_candidate: {float(np.mean([s['seconds_per_candidate'] for s in summaries])) if summaries else float('nan'):.2f}")
        print(f"  avg_candidate_len: {float(np.mean([s['candidate_len_avg'] for s in summaries])) if summaries else float('nan'):.2f}")


def score_policy_window(
    model: GPTForFluMultiTask,
    tokenizer: InfluTokenizer,
    device: torch.device,
    policy: str,
    max_candidates: int,
    seed: int,
    datasets: dict[str, dict],
    all_records: list[dict],
    window_idx: int,
    deduplicate_candidates: bool,
    profile: bool = False,
    shared_prefill: bool = False,
) -> dict:
    window_t0 = time.time()
    dataset_name = "H3N2" if "H3N2" in datasets else next(iter(datasets))
    dataset = datasets[dataset_name]
    context_ids, _target_ids, _subtype_token, window = build_window_context(
        tokenizer=tokenizer,
        dataset=dataset,
        window_idx=window_idx,
    )
    context_records = [
        get_record_by_index(dataset_name, dataset, int(c["strain_idx"]))
        for c in window["context"]
    ]
    target_record = get_record_by_index(
        dataset_name,
        dataset,
        int(window["target"]["strain_idx"]),
    )
    build_t0 = time.time()
    candidates, meta = build_candidate_set(
        tokenizer=tokenizer,
        target_record=target_record,
        all_records=all_records,
        context_records=context_records,
        policy=policy,
        max_candidates=max_candidates,
        seed=seed + window_idx,
    )
    duplicates_removed = 0
    if deduplicate_candidates:
        candidates, duplicates_removed = deduplicate_candidates_by_hash(candidates)

    audit = audit_candidate_set(context_ids=context_ids, candidates=candidates)
    candidate_build_seconds = time.time() - build_t0
    window_id = ids_hash([
        window_idx,
        int(window["target"]["strain_idx"]),
        int(window["target"]["year"]),
        int(window["target"]["month"]),
    ], n=10)

    base_summary = {
        "window_id": window_id,
        "target_month": f"{target_record['year']}-{target_record['month']:02d}",
        "eligible_count": meta["eligible_count_before_exclusions"],
        "num_candidates": len(candidates),
        "duplicates_removed": duplicates_removed,
        "len_min": audit["min_len"],
        "len_max": audit["max_len"],
        "candidate_len_avg": float(np.mean([len(c["ids"]) for c in candidates])) if candidates else float("nan"),
        "candidate_build_seconds": candidate_build_seconds,
        "scoring_seconds": 0.0,
        "seconds_per_candidate": float("nan"),
        "window_seconds": time.time() - window_t0,
    }

    if len(candidates) < 2 or audit["target_count"] != 1:
        return {
            **base_summary,
            "skipped": True,
            "target_rank_by_mean_nll": None,
            "target_rank_by_total_nll": None,
            "normalized_rank": float("nan"),
            "percentile_score": float("nan"),
            "reciprocal_rank": float("nan"),
            "top1": False,
            "top5_applicable": len(candidates) >= 5,
            "top5": False,
            "top10_applicable": len(candidates) >= 10,
            "top10": False,
        }

    rows = []
    scoring_t0 = time.time()
    if shared_prefill:
        rows = score_candidates_with_shared_context_prefill(
            model=model,
            tokenizer=tokenizer,
            context_ids=context_ids,
            candidates=candidates,
            device=device,
        )
    else:
        for candidate in candidates:
            stats = score_sequence_conditional(
                model=model,
                tokenizer=tokenizer,
                context_ids=context_ids,
                continuation_ids=candidate["ids"],
                device=device,
            )
            rows.append({**candidate, **stats})
    scoring_seconds = time.time() - scoring_t0
    window_seconds = time.time() - window_t0

    by_mean = sorted(rows, key=lambda r: r["mean_nll"])
    by_total = sorted(rows, key=lambda r: r["total_nll"])
    target_rank_mean = next((rank for rank, row in enumerate(by_mean, start=1) if row["is_target"]), None)
    target_rank_total = next((rank for rank, row in enumerate(by_total, start=1) if row["is_target"]), None)
    num_candidates = len(candidates)
    normalized_rank = (
        (target_rank_mean - 1) / (num_candidates - 1)
        if target_rank_mean is not None and num_candidates > 1
        else float("nan")
    )
    percentile_score = 1.0 - normalized_rank if math.isfinite(normalized_rank) else float("nan")
    reciprocal_rank = 1.0 / target_rank_mean if target_rank_mean else float("nan")
    top5_applicable = num_candidates >= 5
    top10_applicable = num_candidates >= 10

    return {
        **base_summary,
        "skipped": False,
        "target_rank_by_mean_nll": target_rank_mean,
        "target_rank_by_total_nll": target_rank_total,
        "normalized_rank": normalized_rank,
        "percentile_score": percentile_score,
        "reciprocal_rank": reciprocal_rank,
        "top1": target_rank_mean == 1,
        "top5_applicable": top5_applicable,
        "top5": top5_applicable and target_rank_mean is not None and target_rank_mean <= 5,
        "top10_applicable": top10_applicable,
        "top10": top10_applicable and target_rank_mean is not None and target_rank_mean <= 10,
        "candidate_build_seconds": candidate_build_seconds,
        "scoring_seconds": scoring_seconds,
        "seconds_per_candidate": scoring_seconds / num_candidates if num_candidates else float("nan"),
        "window_seconds": window_seconds,
        "rows": rows,
        "audit": audit,
        "meta": meta,
        "context_hash": ids_hash(context_ids),
        "target_record": target_record,
        "context_ids": context_ids,
    }


def run_score_policy_audit_one(
    model: GPTForFluMultiTask,
    tokenizer: InfluTokenizer,
    device: torch.device,
    policy: str,
    max_candidates: int,
    seed: int,
    deduplicate_candidates: bool,
    datasets: dict[str, dict],
    all_records: list[dict],
    window_idx: int,
    profile: bool = False,
    shared_prefill: bool = False,
) -> None:
    summary = score_policy_window(
        model=model,
        tokenizer=tokenizer,
        device=device,
        policy=policy,
        max_candidates=max_candidates,
        seed=seed,
        datasets=datasets,
        all_records=all_records,
        window_idx=window_idx,
        deduplicate_candidates=deduplicate_candidates,
        profile=profile,
        shared_prefill=shared_prefill,
    )
    audit = summary["audit"]
    meta = summary["meta"]
    rows = summary["rows"]
    target_record = summary["target_record"]

    print("\n" + "=" * 60)
    print("SCORE POLICY AUDIT")
    print("=" * 60)
    print(f"  device: {device}")
    print(f"  window_id: {summary['window_id']}")
    print(f"  policy: {policy}")
    print(f"  seed: {seed}")
    print(f"  deduplicate_candidates: {deduplicate_candidates}")
    print(f"  profile: {profile}")
    if profile:
        print(f"  prefill_recomputed_per_candidate: {not shared_prefill}")
        print(f"  shared_prefill: {shared_prefill}")
    print(f"  context_len: {len(summary['context_ids'])}")
    print(f"  context_hash: {summary['context_hash']}")
    print(f"  target_month_id: {target_record['year']}-{target_record['month']:02d}")
    print(f"  target_group_id: {ids_hash([tokenizer.vocab[target_record['record']['subtype_token']]], n=8)}")
    print_policy_metadata_summary(datasets, all_records)
    print(f"  eligible_count_before_exclusions: {meta['eligible_count_before_exclusions']}")
    print(f"  negative_count_after_exclusions: {meta['negative_count_after_exclusions']}")
    print(f"  requested_max_candidates: {max_candidates}")
    print(f"  num_candidates: {summary['num_candidates']}")
    print(f"  duplicates_removed: {summary['duplicates_removed']}")
    if profile:
        print(f"  candidate_build_seconds: {summary['candidate_build_seconds']:.2f}")
        print(f"  scoring_seconds: {summary['scoring_seconds']:.2f}")
        print(f"  seconds_per_candidate: {summary['seconds_per_candidate']:.2f}")
        print(f"  window_seconds: {summary['window_seconds']:.2f}")
        print(f"  candidate_len_avg: {summary['candidate_len_avg']:.2f}")
    print(f"  target_count: {audit['target_count']}")
    print(f"  candidate_000_is_target: {audit['candidate_000_is_target']}")
    print(f"  duplicate_hash_count: {audit['duplicate_hash_count']}")
    print(f"  continuation_len_min: {audit['min_len']}")
    print(f"  continuation_len_max: {audit['max_len']}")

    if summary["num_candidates"] < max_candidates:
        print("  warning: insufficient_candidates_for_requested_max")

    print("\n  date_distribution")
    for date_key, count in sorted(audit["date_counts"].items()):
        print(f"  {date_key}: {count}")

    print("\n  group_distribution")
    for group_key, count in sorted(audit["group_counts"].items()):
        print(f"  {ids_hash([sum(ord(ch) for ch in group_key)], n=8)}: {count}")

    for row in rows:
        row["hash"] = audit["hashes"][row["anon_id"]]
        row["duplicate"] = audit["duplicates"][row["anon_id"]]
        row["overlap_context"] = audit["context_overlap"][row["anon_id"]]

    by_mean = sorted(rows, key=lambda r: r["mean_nll"])
    by_total = sorted(rows, key=lambda r: r["total_nll"])
    target_anon = next((row["anon_id"] for row in rows if row["is_target"]), "missing")
    target_rank_mean = next((rank for rank, row in enumerate(by_mean, start=1) if row["is_target"]), None)
    target_rank_total = next((rank for rank, row in enumerate(by_total, start=1) if row["is_target"]), None)

    print("\n  candidates")
    print("  anon_id        hash          len   month_id  target duplicate overlap_context")
    for row in rows:
        print(
            f"  {row['anon_id']:<13} "
            f"{row['hash']:<12} "
            f"{row['num_tokens']:>5} "
            f"{row['year']}-{row['month']:02d} "
            f"{str(row['is_target']):>8} "
            f"{str(row['duplicate']):>9} "
            f"{str(row['overlap_context']):>15}"
        )

    print("\n  ranking_by_mean_nll")
    print("  rank anon_id        len   month_id  mean_nll   total_nll   ppl")
    for rank, row in enumerate(by_mean, start=1):
        print(
            f"  {rank:>4} {row['anon_id']:<13} "
            f"{row['num_tokens']:>5} "
            f"{row['year']}-{row['month']:02d} "
            f"{row['mean_nll']:>10.4f} "
            f"{row['total_nll']:>11.3f} "
            f"{row['perplexity']:>9.3f}"
        )

    print("\n  ranking_by_total_nll")
    print("  rank anon_id        len   month_id  total_nll   mean_nll")
    for rank, row in enumerate(by_total, start=1):
        print(
            f"  {rank:>4} {row['anon_id']:<13} "
            f"{row['num_tokens']:>5} "
            f"{row['year']}-{row['month']:02d} "
            f"{row['total_nll']:>11.3f} "
            f"{row['mean_nll']:>10.4f}"
        )

    print(f"\n  target_anon_id: {target_anon}")
    print(f"  target_rank_by_mean_nll: {target_rank_mean}")
    print(f"  target_rank_by_total_nll: {target_rank_total}")


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def plot_figure3a(results_h3n2: list, results_h1n1: list, suffix: str = ""):
    """
    Réplica de la Figura 3A del paper: barras de amino acid mismatch
    promedio para HA y NA full-length, con referencia del paper.
    """
    plt = get_pyplot()

    # Valores de referencia del paper (Figura 3A, lectura visual)
    paper_ref = {
        "H3N2": {"HA": 7.5, "NA": 5.5},
        "H1N1": {"HA": 5.5, "NA": 3.5},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, results, subtype in [
        (axes[0], results_h3n2, "H3N2"),
        (axes[1], results_h1n1, "H1N1"),
    ]:
        if not results:
            ax.set_title(f"{subtype}\n(sin resultados)")
            continue

        ha_v = [r["ha_mismatch"] for r in results if r["ha_mismatch"] >= 0]
        na_v = [r["na_mismatch"] for r in results if r["na_mismatch"] >= 0]

        ha_mean = np.mean(ha_v) if ha_v else 0.0
        ha_std  = np.std(ha_v)  if ha_v else 0.0
        na_mean = np.mean(na_v) if na_v else 0.0
        na_std  = np.std(na_v)  if na_v else 0.0

        x    = np.arange(2)
        bars = ax.bar(
            x, [ha_mean, na_mean],
            yerr=[ha_std, na_std],
            color=["#E63946", "#457B9D"],
            alpha=0.85, width=0.5,
            capsize=6, edgecolor="black", linewidth=0.7,
        )

        ref = paper_ref.get(subtype, {})
        ax.axhline(ref.get("HA", 0), color="#E63946", linestyle="--",
                   linewidth=1.2, alpha=0.55, label=f"Ref. paper HA ≈{ref.get('HA',0)}")
        ax.axhline(ref.get("NA", 0), color="#457B9D", linestyle="--",
                   linewidth=1.2, alpha=0.55, label=f"Ref. paper NA ≈{ref.get('NA',0)}")

        ax.set_xticks(x)
        ax.set_xticklabels(["HA full-length", "NA full-length"], fontsize=11)
        ax.set_ylabel("Average amino acid mismatch", fontsize=11)
        ax.set_title(
            f"{subtype}\n"
            f"HA: {ha_mean:.1f} ± {ha_std:.1f}    NA: {na_mean:.1f} ± {na_std:.1f}\n"
            f"({len(ha_v)} ventanas válidas de {len(results)})",
            fontsize=11,
        )
        ax.legend(fontsize=8, framealpha=0.85)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)

        for bar, val, std in zip(bars, [ha_mean, na_mean], [ha_std, na_std]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + std + 0.4,
                f"{val:.1f}",
                ha="center", fontsize=12, fontweight="bold",
            )

    plt.suptitle(
        "Réplica Figura 3A — AntigenLM (Pei et al., ICLR 2026)\n"
        "Average Amino Acid Mismatch — next-month dominant strain prediction\n"
        "Contexto: 3 cepas dominantes mensuales | Generación: autorregresiva + KV caching | "
        "Apple M5 MPS",
        fontsize=11, y=1.04,
    )
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"figure3a_replica_mac{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Figura 3A: {path}")


def plot_temporal(results: list, subtype: str, suffix: str = ""):
    """Mismatch a lo largo del tiempo — diagnóstico de cuándo falla el modelo."""
    plt = get_pyplot()

    ha_v = [r for r in results if r["ha_mismatch"] >= 0]
    if not ha_v:
        return

    times  = [r["target_year"] + r["target_month"] / 12 for r in ha_v]
    ha_mm  = [r["ha_mismatch"] for r in ha_v]
    na_v   = [r for r in results if r["na_mismatch"] >= 0]
    na_t   = [r["target_year"] + r["target_month"] / 12 for r in na_v]
    na_mm  = [r["na_mismatch"] for r in na_v]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.scatter(times, ha_mm, c="#E63946", s=14, alpha=0.7, zorder=3)
    ax1.axhline(np.mean(ha_mm), color="gray", linestyle="--",
                label=f"Media = {np.mean(ha_mm):.1f}")
    ax1.set_ylabel("HA amino acid mismatch")
    ax1.set_title(f"{subtype} — Mismatch por mes de predicción")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    if na_mm:
        ax2.scatter(na_t, na_mm, c="#457B9D", s=14, alpha=0.7, zorder=3)
        ax2.axhline(np.mean(na_mm), color="gray", linestyle="--",
                    label=f"Media = {np.mean(na_mm):.1f}")
    ax2.set_ylabel("NA amino acid mismatch")
    ax2.set_xlabel("Año")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"temporal_{subtype}{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Temporal {subtype}: {path}")


def plot_structure_diagnostics(results: list, subtype: str, suffix: str = ""):
    """
    Diagnóstico de cuántas ventanas produjeron tokens estructurales correctos
    (<HA>, <sep>, <NA>). Importante para evaluar si el modelo genera
    secuencias bien formadas sin forzar la estructura.
    """
    plt = get_pyplot()

    ok  = sum(1 for r in results if r["has_structure"])
    bad = len(results) - ok

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Pie: estructura OK vs no
    axes[0].pie(
        [ok, bad],
        labels=[f"Estructura OK\n({ok})", f"Sin estructura\n({bad})"],
        colors=["#2a9d8f", "#e76f51"],
        autopct="%1.0f%%",
        startangle=90,
    )
    axes[0].set_title(f"{subtype}\nGeneración con tokens <HA>/<sep>/<NA>")

    # Distribución de longitudes generadas
    lengths = [r["n_generated"] for r in results]
    axes[1].hist(lengths, bins=20, color="#457B9D", edgecolor="white", alpha=0.85)
    axes[1].axvline(np.mean(lengths), color="gray", linestyle="--",
                    label=f"Media={np.mean(lengths):.0f}")
    axes[1].set_xlabel("Tokens generados por ventana")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title(f"{subtype} — Longitud de generación")
    axes[1].legend()
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"structure_diag_{subtype}{suffix}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Diagnóstico estructura: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnóstico seguro / réplica experimental de AntigenLM Figura 3A en Mac"
    )
    parser.add_argument(
        "--test", action="store_true",
        help=f"Modo test acotado: {MAX_WINDOWS_TEST} ventanas por subtipo si no se usa --num-windows"
    )
    parser.add_argument(
        "--debug-one", action="store_true",
        help="Ejecuta una sola ventana corta, imprime top-k/estructura y no escribe resultados"
    )
    parser.add_argument(
        "--score-debug", action="store_true",
        help="Calcula NLL condicional en un batch pequeño sin generar ni escribir resultados"
    )
    parser.add_argument(
        "--score-audit", action="store_true",
        help="Audita candidatos, duplicados, overlap con contexto y rankings NLL"
    )
    parser.add_argument(
        "--score-policy-audit", action="store_true",
        help="Audita una política explícita de construcción de candidatos en una ventana"
    )
    parser.add_argument(
        "--score-unit-test", action="store_true",
        help="Compara scoring con KV cache contra forward completo en ejemplos sintéticos"
    )
    parser.add_argument(
        "--shared-prefill-unit-test", action="store_true",
        help="Compara scoring repetido contra scoring con prefill de contexto compartido"
    )
    parser.add_argument(
        "--num-windows", type=int, default=None,
        help="Número máximo de ventanas por subtipo; usar 1 para diagnóstico"
    )
    parser.add_argument(
        "--max-candidates", type=int, default=5,
        help="Máximo de continuaciones candidatas en modos de scoring/auditoría"
    )
    parser.add_argument(
        "--policy", choices=sorted(POLICIES), default="same_group_same_time",
        help="Política de candidatos para --score-policy-audit"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semilla para muestreo y barajado determinístico de candidatos"
    )
    parser.add_argument(
        "--deduplicate-candidates", action="store_true",
        help="Elimina continuaciones candidatas idénticas por hash exacto"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Mide tiempos de construcción y scoring en auditorías de política"
    )
    parser.add_argument(
        "--shared-prefill", action="store_true",
        help="Usa un único prefill de contexto por ventana para puntuar candidatas"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help=f"Máximo de tokens nuevos; en --debug-one/--test el default seguro es {DEBUG_MAX_NEW_TOKENS}"
    )
    parser.add_argument(
        "--device", choices=["cpu", "mps", "cuda"], default=None,
        help="Dispositivo explícito para diagnóstico"
    )
    parser.add_argument(
        "--no-free-generation", action="store_true",
        help="No genera libremente; puntúa la continuación real corta con log-likelihood"
    )
    parser.add_argument(
        "--inspect-topk", action="store_true",
        help="Imprime top-10 tokens del siguiente paso y de los primeros pasos diagnósticos"
    )
    args = parser.parse_args()

    if args.no_free_generation and not args.debug_one:
        raise SystemExit("--no-free-generation está implementado solo para --debug-one")
    if args.max_candidates < 1:
        raise SystemExit("--max-candidates debe ser >= 1")

    if args.debug_one or args.test:
        max_new_tokens = args.max_new_tokens or DEBUG_MAX_NEW_TOKENS
    else:
        max_new_tokens = args.max_new_tokens or MAX_GENERATE

    max_windows = args.num_windows
    if max_windows is None and args.test:
        max_windows = MAX_WINDOWS_TEST

    suffix = "_debug" if args.debug_one or args.score_debug or args.score_audit or args.score_policy_audit or args.score_unit_test or args.shared_prefill_unit_test else ("_test" if args.test else "_full")

    if not (args.score_debug or args.score_audit or args.score_policy_audit or args.score_unit_test or args.shared_prefill_unit_test):
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

    # Dispositivo — preferir MPS en Mac Apple Silicon, salvo override CLI.
    if args.device:
        device = torch.device(args.device)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Se pidió --device mps, pero MPS no está disponible")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Se pidió --device cuda, pero CUDA no está disponible")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("AVISO: corriendo en CPU — considera usar Mac M5 con MPS")

    print(f"\n{'='*60}")
    print(f"AntigenLM — Réplica Figura 3A")
    print(f"{'='*60}")
    print(f"Dispositivo:    {device}")
    print(f"Contexto:       {CONTEXT_STRAINS} cepas completas (~{CONTEXT_STRAINS*3105} tokens)")
    mode_name = "SHARED-PREFILL-UNIT-TEST" if args.shared_prefill_unit_test else ("SCORE-UNIT-TEST" if args.score_unit_test else ("SCORE-POLICY-AUDIT" if args.score_policy_audit else ("SCORE-AUDIT" if args.score_audit else ("SCORE-DEBUG" if args.score_debug else ("DEBUG-ONE" if args.debug_one else ("TEST" if args.test else "COMPLETO"))))))
    generation_label = "N/A: test de prefill compartido" if args.shared_prefill_unit_test else ("N/A: test sintético de scoring" if args.score_unit_test else ("N/A: auditoría de políticas de scoring" if args.score_policy_audit else ("N/A: auditoría de scoring" if args.score_audit else ("N/A: scoring condicional" if args.score_debug else ('NO libre: scoring de continuación real' if args.no_free_generation else 'autorregresiva libre + KV caching')))))
    print(f"Generación:     {generation_label}")
    print(f"Modo:           {mode_name}")
    print(f"Ventanas:       {max_windows if max_windows is not None else 'todas'}")
    print(f"max_new_tokens: {max_new_tokens}")
    print()

    model     = load_model(device)
    tokenizer = InfluTokenizer(mode="prediction")

    if args.score_unit_test:
        test_score_sequence_conditional_equivalence(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        raise SystemExit(0)

    if args.shared_prefill_unit_test:
        test_shared_prefill_equivalence(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        raise SystemExit(0)

    if args.score_audit:
        run_score_audit(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_candidates=args.max_candidates,
        )
        raise SystemExit(0)

    if args.score_policy_audit:
        run_score_policy_audit(
            model=model,
            tokenizer=tokenizer,
            device=device,
            policy=args.policy,
            max_candidates=args.max_candidates,
            seed=args.seed,
            num_windows=args.num_windows,
            deduplicate_candidates=args.deduplicate_candidates,
            profile=args.profile,
            shared_prefill=args.shared_prefill,
        )
        raise SystemExit(0)

    if args.score_debug:
        run_score_debug(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_candidates=args.max_candidates,
        )
        raise SystemExit(0)

    if args.debug_one:
        run_debug_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            inspect_topk=args.inspect_topk,
            no_free_generation=args.no_free_generation,
        )
        raise SystemExit(0)

    all_results: dict[str, list] = {}
    t_global = time.time()

    for subtype in ["H3N2", "H1N1"]:
        dataset = load_dataset(subtype)
        if dataset is None:
            print(f"\n{subtype}: dataset no encontrado — saltando")
            continue

        print(f"\n{'='*60}")
        print(f"Subtipo: {subtype}")
        print(f"{'='*60}")

        results = evaluate_windows(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            max_windows=max_windows,
            max_new_tokens=max_new_tokens,
            inspect_topk=args.inspect_topk,
        )
        all_results[subtype] = results

        # Resumen del subtipo
        ha_v = [r["ha_mismatch"] for r in results if r["ha_mismatch"] >= 0]
        na_v = [r["na_mismatch"] for r in results if r["na_mismatch"] >= 0]
        ok   = sum(1 for r in results if r["has_structure"])

        print(f"\n  ── Resumen {subtype} ──────────────────────")
        print(f"  Ventanas evaluadas:  {len(results)}")
        print(f"  Con estructura OK:   {ok}/{len(results)} "
              f"({100*ok/max(len(results),1):.0f}%)")
        if ha_v:
            print(f"  HA mismatch:         {np.mean(ha_v):.1f} ± {np.std(ha_v):.1f} aa  (n={len(ha_v)})")
        if na_v:
            print(f"  NA mismatch:         {np.mean(na_v):.1f} ± {np.std(na_v):.1f} aa  (n={len(na_v)})")

        # Figuras del subtipo
        plot_temporal(results, subtype, suffix)
        plot_structure_diagnostics(results, subtype, suffix)

    # Figura principal (Figura 3A del paper)
    plot_figure3a(
        all_results.get("H3N2", []),
        all_results.get("H1N1", []),
        suffix,
    )

    # Guardar resultados en JSON
    out_path = os.path.join(RESULTS_DIR, f"replication_results_mac{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  → Resultados JSON: {out_path}")

    # Resumen final con comparación al paper
    t_total = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"COMPLETADO en {t_total/60:.1f} minutos")
    print(f"{'='*60}")
    print("\nReferencia paper (Figura 3A, ICLR 2026):")
    print("  H3N2  HA: ~7-8 aa mismatch")
    print("  H3N2  NA: ~5-6 aa mismatch")
    print("  H1N1  HA: ~5-6 aa mismatch")
    print("  H1N1  NA: ~3-4 aa mismatch")
    print()
    print("Nuestra réplica:")
    ha_refs = {"H3N2": 7.5, "H1N1": 5.5}
    na_refs = {"H3N2": 5.5, "H1N1": 3.5}
    for subtype, results in all_results.items():
        ha_v = [r["ha_mismatch"] for r in results if r["ha_mismatch"] >= 0]
        na_v = [r["na_mismatch"] for r in results if r["na_mismatch"] >= 0]
        if ha_v:
            print(f"  {subtype}  HA: {np.mean(ha_v):.1f} ± {np.std(ha_v):.1f}  "
                  f"(ref: ~{ha_refs[subtype]})")
        if na_v:
            print(f"  {subtype}  NA: {np.mean(na_v):.1f} ± {np.std(na_v):.1f}  "
                  f"(ref: ~{na_refs[subtype]})")
    print(f"\nFiguras: {FIGURES_DIR}/")
