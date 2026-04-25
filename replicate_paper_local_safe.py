# -*- coding: utf-8 -*-
"""
replicate_paper_local_phased.py
==============================
Copia experimental de replicate_paper.py para pruebas locales con GPU pequena.

Objetivo:
    - No modificar el script original.
    - Reducir uso de VRAM.
    - Preservar formato de salida con generacion por fases.
    - Generar HA y NA con longitudes controladas para evitar bucles degenerados.

Basado en replicate_paper.py.
Replica de la Figura 3A del paper AntigenLM.

Metrica: Average Amino Acid Mismatch entre la secuencia predicha
y la secuencia real de la cepa dominante del mes siguiente.

Pipeline:
    1. Cargar modelo finetuneado con pesos reales
    2. Para cada ventana temporal [t-2, t-1, t]:
       a. Tokenizar las 3 cepas de contexto
       b. Generar autorregresivaemente la secuencia de t+1
       c. Decodificar la secuencia generada
       d. Traducir DNA -> aminoacidos
       e. Comparar con la cepa real de t+1
    3. Calcular mismatch promedio
    4. Generar figura comparable a Figure 3A del paper

Uso:
    python replicate_paper_local_phased.py

Autor: Carlos (tesis de maestria)
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from antigen_model import GPTForFluMultiTask
from influ_tokenizer import InfluTokenizer

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

PROCESSED_DIR = "data/processed_gisaid"
FIGURES_DIR   = "figures/replication"
RESULTS_DIR   = "results"

# Numero maximo de ventanas a evaluar (None = todas)
MAX_WINDOWS   = 3  # poner un numero para test rapido, ej: 10

# Longitud maxima de generacion (HA ~ 1700nt, NA ~ 1400nt + tokens especiales)
MAX_GENERATE  = 3200

# ---------------------------------------------------------------------------
# Modo local seguro para GPU pequena
# ---------------------------------------------------------------------------
# En una GPU de 4 GB, el contexto completo de 3 cepas (~9000+ tokens)
# suele reventar por CUDA OOM. Esta copia usa un contexto estructurado
# reducido y fuerza el prefijo de generacion para evitar Pred HA/NA = 0.

# Numero de cepas recientes a usar como contexto. Para 4 GB: 1.
CONTEXT_STRAINS = 1

# Limite duro de tokens que se pasan al transformer en cada forward.
# Para 4 GB prueba 2048. Si corre bien, intenta 3072.
CONTEXT_TOKEN_BUDGET = 2048

# Media precision en CUDA para ahorrar VRAM.
USE_FP16 = True

# Imprimir diagnostico de tokens generados.
DEBUG_GENERATION = True

# ---------------------------------------------------------------------------
# Generacion por fases
# ---------------------------------------------------------------------------
# El modelo local con contexto recortado puede degenerar en secuencias tipo
# C C C C... y no emitir <sep>, <NA> o <eos>. En esta version, el script
# fuerza la gramatica:
#     <subtipo> <HA> [HA nucleotidos] <sep> <NA> [NA nucleotidos] <eos>
# y solo le pide al modelo elegir nucleotidos dentro de cada segmento.

PHASED_GENERATION = True

# Muestreo restringido a nucleotidos. Si False, usa greedy restringido.
# Para evitar el colapso C C C..., conviene dejarlo en True durante pruebas.
USE_SAMPLING = True
SAMPLING_TEMPERATURE = 1.15

# Penaliza repetir demasiado los mismos nucleotidos recientes.
REPETITION_PENALTY = 1.15
REPETITION_WINDOW = 64

# Si el mismo nucleotido aparece demasiadas veces seguidas, se bloquea
# temporalmente para impedir homopolimeros absurdos.
MAX_HOMOPOLYMER_RUN = 12

# Semilla para que la prueba sea reproducible.
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def load_model(device):
    """Carga AntigenLM prediccion con pesos reales."""
    print("Cargando modelo...")
    ckpt = torch.load(
        "prediction_sequence/pytorch_model.bin",
        map_location="cpu"
    )
    remapped = {k.replace("transformer.", "backbone."): v
                for k, v in ckpt.items()}

    model = GPTForFluMultiTask(task="prediction")
    model.load_state_dict(remapped, strict=False)
    model.eval()

    # Ahorro de memoria en GPU.
    if device.type == "cuda" and USE_FP16:
        model = model.half()

    model = model.to(device)
    print(f"  Modelo cargado en {device}")

    if device.type == "cuda" and USE_FP16:
        print("  Usando FP16 para reducir memoria GPU")

    return model


# ---------------------------------------------------------------------------
# Traduccion DNA -> Aminoacidos
# ---------------------------------------------------------------------------

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L',
    'CTA':'L','CTG':'L','ATT':'I','ATC':'I','ATA':'I','ATG':'M',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V','TCT':'S','TCC':'S',
    'TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A',
    'GCA':'A','GCG':'A','TAT':'Y','TAC':'Y','TAA':'*','TAG':'*',
    'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','AAT':'N','AAC':'N',
    'AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R',
    'CGA':'R','CGG':'R','AGT':'S','AGC':'S','AGA':'R','AGG':'R',
    'GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

def translate_dna(dna_seq):
    """Traduce secuencia de DNA a aminoacidos."""
    dna_seq = dna_seq.upper().replace("N", "A")  # N -> A como aproximacion
    protein = []
    for i in range(0, len(dna_seq) - 2, 3):
        codon = dna_seq[i:i+3]
        aa = CODON_TABLE.get(codon, 'X')
        if aa == '*':
            break
        protein.append(aa)
    return "".join(protein)


def amino_acid_mismatch(pred_dna, real_dna):
    """
    Calcula el amino acid mismatch entre dos secuencias DNA.
    Retorna (mismatches, total_positions, mismatch_rate).
    """
    pred_aa = translate_dna(pred_dna)
    real_aa = translate_dna(real_dna)

    min_len = min(len(pred_aa), len(real_aa))
    if min_len == 0:
        return 0, 0, 0.0

    mismatches = sum(p != r for p, r in zip(pred_aa[:min_len], real_aa[:min_len]))
    return mismatches, min_len, mismatches / min_len


# ---------------------------------------------------------------------------
# Generacion autorrregresiva
# ---------------------------------------------------------------------------

def token_name(tokenizer, tid):
    """
    Devuelve una representacion legible de un token.
    Sirve para depurar tokens especiales y nucleotidos.
    """
    inv_vocab = {v: k for k, v in tokenizer.vocab.items()}

    if tid in inv_vocab:
        return inv_vocab[tid]

    try:
        return tokenizer.id_to_token_str(tid)
    except Exception:
        return str(tid)


def print_generation_debug(new_tokens, tokenizer, max_show=80):
    """
    Imprime los primeros tokens generados para verificar si aparecen
    <HA>, <NA>, <sep>, <eos>, etc.
    """
    preview = [token_name(tokenizer, int(t)) for t in new_tokens[:max_show]]
    print("    Primeros tokens generados:")
    print("    " + " ".join(preview))

    special_tokens = ["<HA>", "<NA>", "<sep>", "<eos>", "<H3N2>", "<H1N1>"]
    counts = {}

    for tok in special_tokens:
        if tok in tokenizer.vocab:
            tok_id = tokenizer.vocab[tok]
            counts[tok] = sum(int(t) == tok_id for t in new_tokens)

    print(f"    Conteo tokens especiales: {counts}")


def encode_strain_with_budget(tokenizer, strain, subtype_token, max_tokens):
    """
    Codifica una cepa preservando la estructura:
        <subtipo> <HA> ... <sep> <NA> ... <eos>

    Si la cepa completa no cabe, recorta HA y NA de forma estructurada,
    no cortando los tokens por la cola a ciegas.

    Nota metodologica:
        Esta funcion es para prueba local en GPU pequena. No es la replica
        final estricta del paper, porque reduce el contexto.
    """
    full_ids = tokenizer.encode_strain(
        ha_sequence=strain["ha_sequence"],
        na_sequence=strain["na_sequence"],
        subtype=subtype_token,
    )

    if len(full_ids) <= max_tokens:
        return full_ids

    # Reservamos espacio para tokens especiales.
    seq_budget = max_tokens - 10

    if seq_budget < 200:
        raise ValueError(
            f"Presupuesto demasiado pequeno para codificar una cepa: {max_tokens}"
        )

    # HA suele ser un poco mas larga que NA.
    ha_budget = int(seq_budget * 0.55)
    na_budget = seq_budget - ha_budget

    # Para prueba local usamos sufijos recientes de cada segmento.
    ha_seq = strain["ha_sequence"][-ha_budget:]
    na_seq = strain["na_sequence"][-na_budget:]

    compact_ids = tokenizer.encode_strain(
        ha_sequence=ha_seq,
        na_sequence=na_seq,
        subtype=subtype_token,
    )

    # Si aun se pasa, reducimos iterativamente.
    while len(compact_ids) > max_tokens and ha_budget > 100 and na_budget > 100:
        ha_budget = int(ha_budget * 0.9)
        na_budget = int(na_budget * 0.9)

        ha_seq = strain["ha_sequence"][-ha_budget:]
        na_seq = strain["na_sequence"][-na_budget:]

        compact_ids = tokenizer.encode_strain(
            ha_sequence=ha_seq,
            na_sequence=na_seq,
            subtype=subtype_token,
        )

    return compact_ids


def generate_sequence(model, tokenizer, context_ids, device,
                      max_new_tokens=MAX_GENERATE,
                      temperature=1.0,
                      context_token_budget=CONTEXT_TOKEN_BUDGET,
                      forced_prefix_ids=None):
    """
    Genera una secuencia autorregresivamente dado el contexto.

    Version local segura:
      - usa ventana deslizante para no reventar VRAM
      - usa inference_mode
      - usa autocast FP16 en CUDA
      - permite forzar prefijo de salida, por ejemplo <H3N2> <HA>
    """
    forced_prefix_ids = forced_prefix_ids or []

    generated = list(context_ids) + list(forced_prefix_ids)

    # Incluimos el prefijo forzado en new_tokens para que
    # extract_ha_na_from_tokens pueda ver <HA>.
    new_tokens = list(forced_prefix_ids)

    use_cuda = device.type == "cuda"

    if use_cuda:
        torch.cuda.empty_cache()

    with torch.inference_mode():
        for step in range(max_new_tokens):
            input_ids = generated[-context_token_budget:]
            x = torch.tensor([input_ids], dtype=torch.long, device=device)

            try:
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    out = model(input_ids=x)

                logits = out["logits"][0, -1, :]
                next_token = int(logits.argmax().item())

            except torch.cuda.OutOfMemoryError:
                print("\n      ERROR: CUDA OOM durante generacion.")
                print(f"      Tokens usados en este forward: {len(input_ids)}")
                print("      Baja CONTEXT_TOKEN_BUDGET a 1536 o 1024.")
                if use_cuda:
                    torch.cuda.empty_cache()
                raise

            generated.append(next_token)
            new_tokens.append(next_token)

            del x, out, logits

            if next_token == tokenizer.eos_token_id:
                break

            if use_cuda and (step + 1) % 100 == 0:
                torch.cuda.empty_cache()

            if (step + 1) % 500 == 0:
                print(f"      generados {step+1} tokens...", end="\r")

    if use_cuda:
        torch.cuda.empty_cache()

    return new_tokens


def get_nucleotide_token_ids(tokenizer):
    """Obtiene ids de nucleotidos A, C, G, T, N de forma robusta."""
    ids = []

    for tok in ["A", "C", "G", "T", "N"]:
        if tok in tokenizer.vocab:
            ids.append(tokenizer.vocab[tok])

    # Fallback para el tokenizer usado en AntigenLM: 0..4 suelen ser A,C,G,T,N.
    if len(ids) < 4:
        ids = [0, 1, 2, 3, 4]

    # Preservar orden y quitar duplicados.
    seen = set()
    clean_ids = []
    for tid in ids:
        if tid not in seen:
            clean_ids.append(int(tid))
            seen.add(tid)

    return clean_ids


def get_forbidden_homopolymer_token(segment_tokens,
                                    max_run_length=MAX_HOMOPOLYMER_RUN):
    """
    Si los ultimos max_run_length tokens son iguales, retorna ese token para
    bloquearlo temporalmente. Si no, retorna None.
    """
    if len(segment_tokens) < max_run_length:
        return None

    tail = segment_tokens[-max_run_length:]
    if all(t == tail[0] for t in tail):
        return tail[0]

    return None


def choose_next_nucleotide(logits, nucleotide_ids, segment_tokens,
                           temperature=SAMPLING_TEMPERATURE,
                           use_sampling=USE_SAMPLING,
                           repetition_penalty=REPETITION_PENALTY,
                           repetition_window=REPETITION_WINDOW):
    """
    Escoge el siguiente nucleotido restringiendo el vocabulario a A/C/G/T/N.
    Incluye penalizacion simple de repeticion para evitar colapsos tipo CCCC.
    """
    logits = logits.float().clone()

    # Mascara: todo prohibido excepto nucleotidos.
    masked_logits = torch.full_like(logits, float("-inf"))
    masked_logits[nucleotide_ids] = logits[nucleotide_ids]

    # Penalizacion por repeticion reciente.
    recent = segment_tokens[-repetition_window:]
    for tid in set(recent):
        if tid in nucleotide_ids:
            if masked_logits[tid] > 0:
                masked_logits[tid] = masked_logits[tid] / repetition_penalty
            else:
                masked_logits[tid] = masked_logits[tid] * repetition_penalty

    # Bloqueo anti-homopolimero absurdo.
    forbidden = get_forbidden_homopolymer_token(segment_tokens)
    if forbidden is not None and forbidden in nucleotide_ids:
        masked_logits[forbidden] = float("-inf")

    if not use_sampling:
        return int(torch.argmax(masked_logits).item())

    probs = torch.softmax(masked_logits / temperature, dim=-1)

    # Si por alguna razon queda degenerado, volver a greedy.
    if torch.isnan(probs).any() or probs.sum() <= 0:
        return int(torch.argmax(masked_logits).item())

    return int(torch.multinomial(probs, num_samples=1).item())


def generate_nucleotide_segment(model, tokenizer, generated, device, segment_len,
                                label,
                                context_token_budget=CONTEXT_TOKEN_BUDGET):
    """
    Genera exactamente segment_len nucleotidos, restringiendo el vocabulario.
    Actualiza y retorna generated junto con los tokens nuevos del segmento.
    """
    nucleotide_ids = get_nucleotide_token_ids(tokenizer)
    segment_tokens = []
    use_cuda = device.type == "cuda"

    print(f"      Generando {label}: {segment_len} nt objetivo")

    with torch.inference_mode():
        for step in range(segment_len):
            input_ids = generated[-context_token_budget:]
            x = torch.tensor([input_ids], dtype=torch.long, device=device)

            try:
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    out = model(input_ids=x)

                logits = out["logits"][0, -1, :]
                next_token = choose_next_nucleotide(
                    logits=logits,
                    nucleotide_ids=nucleotide_ids,
                    segment_tokens=segment_tokens,
                )

            except torch.cuda.OutOfMemoryError:
                print("\n      ERROR: CUDA OOM durante generacion por fases.")
                print(f"      Tokens usados en este forward: {len(input_ids)}")
                print("      Baja CONTEXT_TOKEN_BUDGET a 1536 o 1024.")
                if use_cuda:
                    torch.cuda.empty_cache()
                raise

            generated.append(next_token)
            segment_tokens.append(next_token)

            del x, out, logits

            if use_cuda and (step + 1) % 100 == 0:
                torch.cuda.empty_cache()

            if (step + 1) % 500 == 0:
                print(f"        {label}: {step+1}/{segment_len} nt...", end="\r")

    print(f"        {label}: {len(segment_tokens)}/{segment_len} nt generados")
    return generated, segment_tokens


def generate_sequence_phased(model, tokenizer, context_ids, device,
                             subtype_token, ha_len, na_len,
                             context_token_budget=CONTEXT_TOKEN_BUDGET):
    """
    Generacion local por fases:
        <subtipo> <HA> HA_tokens <sep> <NA> NA_tokens <eos>

    Esto evita depender de que el modelo emita por si solo <sep>, <NA> y <eos>,
    algo que falla cuando el contexto fue recortado agresivamente por limite de VRAM.
    """
    if RANDOM_SEED is not None:
        torch.manual_seed(RANDOM_SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(RANDOM_SEED)

    subtype_id = tokenizer.vocab.get(subtype_token, None)
    ha_id = tokenizer.vocab["<HA>"]
    sep_id = tokenizer.vocab["<sep>"]
    na_id = tokenizer.vocab["<NA>"]
    eos_id = tokenizer.vocab["<eos>"]

    prefix = []
    if subtype_id is not None:
        prefix.append(subtype_id)
    prefix.append(ha_id)

    generated = list(context_ids) + list(prefix)
    new_tokens = list(prefix)

    print(
        "    Prefijo forzado:",
        " ".join(token_name(tokenizer, t) for t in prefix)
    )

    generated, ha_tokens = generate_nucleotide_segment(
        model=model,
        tokenizer=tokenizer,
        generated=generated,
        device=device,
        segment_len=ha_len,
        label="HA",
        context_token_budget=context_token_budget,
    )
    new_tokens.extend(ha_tokens)

    generated.append(sep_id)
    generated.append(na_id)
    new_tokens.extend([sep_id, na_id])

    print("      Forzando separador: <sep> <NA>")

    generated, na_tokens = generate_nucleotide_segment(
        model=model,
        tokenizer=tokenizer,
        generated=generated,
        device=device,
        segment_len=na_len,
        label="NA",
        context_token_budget=context_token_budget,
    )
    new_tokens.extend(na_tokens)

    generated.append(eos_id)
    new_tokens.append(eos_id)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return new_tokens


def extract_ha_na_from_tokens(token_ids, tokenizer):
    """
    Extrae las secuencias HA y NA de una lista de tokens generados.

    Formato esperado: <subtipo> <HA> [nucleotidos] <sep> <NA> [nucleotidos] <eos>
    o sin subtipo:    <HA> [nucleotidos] <sep> <NA> [nucleotidos] <eos>
    """
    # Decodificar a string
    nucleotide_ids = {0, 1, 2, 3, 4}  # A, C, G, T, N
    ha_id  = tokenizer.vocab["<HA>"]
    na_id  = tokenizer.vocab["<NA>"]
    sep_id = tokenizer.vocab["<sep>"]
    eos_id = tokenizer.vocab["<eos>"]

    ha_seq = []
    na_seq = []
    current_segment = None

    for tid in token_ids:
        if tid == ha_id:
            current_segment = "ha"
        elif tid == sep_id:
            current_segment = None
        elif tid == na_id:
            current_segment = "na"
        elif tid == eos_id:
            break
        elif tid in nucleotide_ids and current_segment == "ha":
            ha_seq.append(tokenizer.id_to_token_str(tid))
        elif tid in nucleotide_ids and current_segment == "na":
            na_seq.append(tokenizer.id_to_token_str(tid))

    return "".join(ha_seq), "".join(na_seq)


# ---------------------------------------------------------------------------
# Evaluacion de ventanas temporales
# ---------------------------------------------------------------------------

def evaluate_windows(model, tokenizer, dataset, device, max_windows=MAX_WINDOWS):
    """
    Evalua el modelo en las ventanas temporales del dataset.

    Para cada ventana [t-2, t-1, t] -> t+1:
        1. Tokeniza las 3 cepas de contexto
        2. Genera la prediccion de t+1
        3. Compara con la cepa real de t+1
    """
    windows = dataset["windows"]
    strains = dataset["paired_strains"]

    # Verificar si las secuencias estan en el mismo archivo o separado
    if "ha_sequence" not in strains[0]:
        # Cargar secuencias del archivo separado
        subtype = dataset["subtype"]
        seq_path = os.path.join(PROCESSED_DIR, f"sequences_{subtype}.json")
        if os.path.exists(seq_path):
            print(f"  Cargando secuencias desde {seq_path}...")
            with open(seq_path, "r") as f:
                sequences = json.load(f)
            # Agregar secuencias al dataset
            for strain in strains:
                epi = strain.get("epi_isl", "")
                if epi in sequences:
                    strain["ha_sequence"] = sequences[epi]["ha"]
                    strain["na_sequence"] = sequences[epi]["na"]
        else:
            print(f"  ERROR: no se encontraron secuencias")
            return []

    if max_windows is not None:
        windows = windows[:max_windows]

    print(f"\n  Evaluando {len(windows)} ventanas temporales...")
    subtype_token = strains[0].get("subtype_token", "<H3N2>")

    results = []

    for w_idx, window in enumerate(windows):
        ctx = window["context"]
        tgt = window["target"]

        tgt_year  = tgt["year"]
        tgt_month = tgt["month"]

        print(f"\n  Ventana {w_idx+1}/{len(windows)}: "
              f"contexto -> {tgt_year}-{tgt_month:02d}")

        # Construir contexto tokenizado
        # Modo local: usamos las ultimas CONTEXT_STRAINS cepas,
        # pero preservando la estructura interna de cada cepa.
        context_ids = []

        selected_ctx = ctx[-CONTEXT_STRAINS:]

        per_strain_budget = max(
            512,
            CONTEXT_TOKEN_BUDGET // max(1, len(selected_ctx))
        )

        for c in selected_ctx:
            strain = strains[c["strain_idx"]]

            if "ha_sequence" not in strain or "na_sequence" not in strain:
                print(f"    AVISO: cepa sin secuencia, saltando ventana")
                continue

            ids = encode_strain_with_budget(
                tokenizer=tokenizer,
                strain=strain,
                subtype_token=subtype_token,
                max_tokens=per_strain_budget,
            )

            context_ids.extend(ids)

        if len(context_ids) < 100:
            print(f"    Contexto demasiado corto ({len(context_ids)} tokens)")
            continue

        if len(context_ids) > CONTEXT_TOKEN_BUDGET:
            print(
                f"    Contexto estructurado: {len(context_ids)} tokens; "
                f"recortando a {CONTEXT_TOKEN_BUDGET}"
            )
            context_ids = context_ids[-CONTEXT_TOKEN_BUDGET:]

        print(f"    Contexto: {len(context_ids)} tokens")

        # Cepa real del mes target.
        # Para la prueba local por fases usamos sus longitudes como objetivo.
        # Esto NO usa el contenido de la secuencia real para generar, solo evita
        # depender de que el modelo decida por si mismo cuando terminar HA/NA.
        target_strain = strains[tgt["strain_idx"]]
        real_ha = target_strain.get("ha_sequence", "")
        real_na = target_strain.get("na_sequence", "")

        if not real_ha or not real_na:
            print(f"    AVISO: cepa target sin secuencia")
            continue

        ha_target_len = len(real_ha)
        na_target_len = len(real_na)
        print(
            f"    Longitudes objetivo: HA={ha_target_len} nt, "
            f"NA={na_target_len} nt"
        )

        # Generar prediccion por fases:
        # <subtipo> <HA> HA... <sep> <NA> NA... <eos>
        print(f"    Generando prediccion por fases...")
        new_tokens = generate_sequence_phased(
            model=model,
            tokenizer=tokenizer,
            context_ids=context_ids,
            device=device,
            subtype_token=subtype_token,
            ha_len=ha_target_len,
            na_len=na_target_len,
        )
        print(f"    Generados: {len(new_tokens)} tokens")

        if DEBUG_GENERATION:
            print_generation_debug(new_tokens, tokenizer)

        # Extraer HA y NA de la prediccion
        pred_ha, pred_na = extract_ha_na_from_tokens(new_tokens, tokenizer)
        print(f"    Pred HA: {len(pred_ha)} nt, Pred NA: {len(pred_na)} nt")

        # Calcular amino acid mismatch
        if len(pred_ha) > 100:
            ha_mm, ha_len, ha_rate = amino_acid_mismatch(pred_ha, real_ha)
            print(f"    HA mismatch: {ha_mm}/{ha_len} aa ({ha_rate:.3f})")
        else:
            ha_mm, ha_len, ha_rate = -1, 0, 0
            print(f"    HA: prediccion demasiado corta ({len(pred_ha)} nt)")

        if len(pred_na) > 100:
            na_mm, na_len, na_rate = amino_acid_mismatch(pred_na, real_na)
            print(f"    NA mismatch: {na_mm}/{na_len} aa ({na_rate:.3f})")
        else:
            na_mm, na_len, na_rate = -1, 0, 0
            print(f"    NA: prediccion demasiado corta ({len(pred_na)} nt)")

        results.append({
            "window_idx": w_idx,
            "target_year": tgt_year,
            "target_month": tgt_month,
            "ha_mismatch": ha_mm,
            "ha_length": ha_len,
            "ha_rate": ha_rate,
            "na_mismatch": na_mm,
            "na_length": na_len,
            "na_rate": na_rate,
            "pred_ha_len": len(pred_ha),
            "pred_na_len": len(pred_na),
            "n_generated": len(new_tokens),
        })

    return results


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def plot_mismatch(results_h3n2, results_h1n1):
    """
    Genera figura comparable a Figure 3A del paper.
    Barras de amino acid mismatch promedio para HA y NA.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, results, subtype in [
        (axes[0], results_h3n2, "H3N2"),
        (axes[1], results_h1n1, "H1N1"),
    ]:
        if not results:
            ax.set_title(f"{subtype}\n(sin resultados)")
            continue

        # Filtrar resultados validos
        ha_valid = [r for r in results if r["ha_mismatch"] >= 0]
        na_valid = [r for r in results if r["na_mismatch"] >= 0]

        ha_mm = [r["ha_mismatch"] for r in ha_valid] if ha_valid else [0]
        na_mm = [r["na_mismatch"] for r in na_valid] if na_valid else [0]

        ha_mean = np.mean(ha_mm)
        ha_std  = np.std(ha_mm)
        na_mean = np.mean(na_mm)
        na_std  = np.std(na_mm)

        x = np.arange(2)
        bars = ax.bar(x, [ha_mean, na_mean],
                      yerr=[ha_std, na_std],
                      color=["#E63946", "#457B9D"],
                      alpha=0.85, width=0.5,
                      capsize=5, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(["HA full-length", "NA full-length"], fontsize=11)
        ax.set_ylabel("Average amino acid mismatch", fontsize=11)
        ax.set_title(f"{subtype}\n({len(ha_valid)} ventanas evaluadas)",
                     fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)

        # Anotar valores
        for bar, val, std in zip(bars, [ha_mean, na_mean], [ha_std, na_std]):
            ax.text(bar.get_x() + bar.get_width()/2, val + std + 0.5,
                    f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")

    plt.suptitle(
        "Replicacion de Figure 3A - AntigenLM\n"
        "Next-month amino acid mismatch prediction",
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure3a_replication_local_phased.png")
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figura guardada: {path}")


def plot_mismatch_temporal(results, subtype):
    """
    Mismatch a lo largo del tiempo — no esta en el paper pero
    es util para entender donde el modelo falla.
    """
    if not results:
        return

    valid = [r for r in results if r["ha_mismatch"] >= 0]
    if not valid:
        return

    times = [r["target_year"] + r["target_month"]/12 for r in valid]
    ha_mm = [r["ha_mismatch"] for r in valid]
    na_mm = [r["na_mismatch"] for r in valid if r["na_mismatch"] >= 0]
    na_times = [r["target_year"] + r["target_month"]/12
                for r in valid if r["na_mismatch"] >= 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.scatter(times, ha_mm, c="#E63946", s=20, alpha=0.7)
    ax1.axhline(np.mean(ha_mm), color="gray", linestyle="--",
                label=f"Media = {np.mean(ha_mm):.1f}")
    ax1.set_ylabel("HA amino acid mismatch")
    ax1.set_title(f"{subtype} - Mismatch por mes de prediccion")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    if na_mm:
        ax2.scatter(na_times, na_mm, c="#457B9D", s=20, alpha=0.7)
        ax2.axhline(np.mean(na_mm), color="gray", linestyle="--",
                    label=f"Media = {np.mean(na_mm):.1f}")
    ax2.set_ylabel("NA amino acid mismatch")
    ax2.set_xlabel("Anio")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"mismatch_temporal_local_phased_{subtype}.png")
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figura temporal guardada: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Detectar dispositivo
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model     = load_model(device)
    tokenizer = InfluTokenizer(mode="prediction")

    all_results = {}

    for subtype in ["H3N2", "H1N1"]:
        fpath = os.path.join(PROCESSED_DIR, f"dataset_{subtype}.json")
        if not os.path.exists(fpath):
            print(f"\n  {subtype}: dataset no encontrado")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluando: {subtype}")
        print(f"{'='*60}")

        with open(fpath, "r") as f:
            dataset = json.load(f)

        results = evaluate_windows(model, tokenizer, dataset, device,
                                   max_windows=MAX_WINDOWS)
        all_results[subtype] = results

        # Estadisticas
        if results:
            ha_valid = [r for r in results if r["ha_mismatch"] >= 0]
            na_valid = [r for r in results if r["na_mismatch"] >= 0]
            print(f"\n  --- Resumen {subtype} ---")
            if ha_valid:
                ha_mm = [r["ha_mismatch"] for r in ha_valid]
                print(f"  HA mismatch: {np.mean(ha_mm):.1f} +/- {np.std(ha_mm):.1f} aa"
                      f"  (n={len(ha_valid)} ventanas)")
            if na_valid:
                na_mm = [r["na_mismatch"] for r in na_valid]
                print(f"  NA mismatch: {np.mean(na_mm):.1f} +/- {np.std(na_mm):.1f} aa"
                      f"  (n={len(na_valid)} ventanas)")

        # Figura temporal
        plot_mismatch_temporal(results, subtype)

    # Figura comparativa (Figure 3A)
    plot_mismatch(
        all_results.get("H3N2", []),
        all_results.get("H1N1", []),
    )

    # Guardar resultados en JSON
    results_path = os.path.join(RESULTS_DIR, "replication_results_local_phased.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Resultados guardados: {results_path}")

    # Comparacion con paper
    print("\n" + "=" * 60)
    print("COMPARACION CON PAPER (Figure 3A)")
    print("=" * 60)
    print("\nPaper reporta para next-month prediction:")
    print("  H3N2 HA full-length: ~7-8 aa mismatch")
    print("  H3N2 NA full-length: ~5-6 aa mismatch")
    print("  H1N1 HA full-length: ~5-6 aa mismatch")
    print("  H1N1 NA full-length: ~3-4 aa mismatch")

    for subtype, results in all_results.items():
        if results:
            ha_valid = [r["ha_mismatch"] for r in results if r["ha_mismatch"] >= 0]
            na_valid = [r["na_mismatch"] for r in results if r["na_mismatch"] >= 0]
            if ha_valid:
                print(f"\nNuestra replica {subtype}:")
                print(f"  HA: {np.mean(ha_valid):.1f} +/- {np.std(ha_valid):.1f}")
            if na_valid:
                print(f"  NA: {np.mean(na_valid):.1f} +/- {np.std(na_valid):.1f}")

    print("\n" + "=" * 60)
    print("Replicacion completada.")
    print(f"Figuras: {FIGURES_DIR}/")
    print("=" * 60)
