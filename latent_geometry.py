# -*- coding: utf-8 -*-
"""
latent_geometry.py  (v3 - auditado y corregido)
================================================
Analisis de geometria del espacio latente de AntigenLM.

Primer resultado original de la tesis: verificar empiricamente
que el espacio latente z_t in R^384 tiene estructura geometrica
suficiente para soportar una SDE.

Correcciones respecto a v2:
    - Secuencias completas (no truncadas a 512)
    - Spearman calculado POR SUBTIPO (no mezclado)
    - Interpolacion mide variacion real entre pasos
    - TwoNN con normalizacion previa
    - Encoding UTF-8 correcto

Autor: Carlos (tesis de maestria)
"""

import os
import json
import random
import argparse
import pickle
import hashlib
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("AVISO: umap-learn no instalado. Ejecuta: pip install umap-learn")

try:
    import skdim
    SKDIM_AVAILABLE = True
except ImportError:
    SKDIM_AVAILABLE = False
    print("AVISO: scikit-dimension no instalado. Ejecuta: pip install scikit-dimension")

from antigen_model import GPTForFluMultiTask
from influ_tokenizer import InfluTokenizer

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

PROCESSED_DIR = "data/processed_gisaid"
FIGURES_DIR   = "figures/gisaid"
RESULTS_DIR   = "results"
CHECKPOINT_PATH = "prediction_sequence/pytorch_model.bin"

MAX_STRAINS_PER_SUBTYPE = 2000
MAX_SEQ_LENGTH          = 4000  # secuencias completas HA+NA (~3100 tokens)
RANDOM_SEED             = 42
SAMPLING_STRATEGIES     = ("first_n", "random", "stratified_by_year", "all")
DISTANCE_METRICS        = ("temporal", "hamming_nt", "hamming_ha", "hamming_ha_na")
INTRINSIC_METHODS       = ("twonn",)
INTRINSIC_NORMALIZATIONS = ("none", "standard", "l2", "standard_l2")
HAMMING_LENGTH_TOLERANCE = 0.05
PCA_VARIANCE_THRESHOLDS = (0.80, 0.90, 0.95, 0.99)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def load_model(device):
    """Carga AntigenLM con pesos reales y remapeo de claves."""
    print("Cargando modelo AntigenLM...")
    ckpt = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )
    remapped = {k.replace("transformer.", "backbone."): v
                for k, v in ckpt.items()}

    model = GPTForFluMultiTask(task="prediction")
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    assert len(missing) == 0, f"Pesos faltantes en el checkpoint: {missing}"
    # Las claves unexpected son buffers de máscara causal de transformers 4.29.2
    # (backbone.h.*.attn.bias / attn.masked_bias). Versiones modernas los
    # recalculan on-the-fly — no afectan ningún parámetro entrenable.

    model.eval()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Modelo cargado en {device} - {n_params:,} params")
    return model


def checkpoint_metadata(path=CHECKPOINT_PATH):
    """Metadatos livianos del checkpoint para trazabilidad del cache."""
    if not os.path.exists(path):
        return {"path": path, "exists": False}

    stat = os.stat(path)
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return {
        "path": path,
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": sha256.hexdigest(),
    }


# ---------------------------------------------------------------------------
# Extraccion de representaciones latentes
# ---------------------------------------------------------------------------

def extract_latent(model, tokenizer, strain, device):
    """
    Extrae z_t in R^384 para una cepa.

    Usa la secuencia COMPLETA (HA + NA), no truncada.
    Representacion: mean pooling de todos los hidden states
    (mas robusto que tomar solo el ultimo token).
    """
    try:
        ids = tokenizer.encode_strain(
            ha_sequence=strain["ha_sequence"],
            na_sequence=strain["na_sequence"],
            subtype=strain["subtype_token"],
        )
        if len(ids) < 10:
            return None

        # Limitar a MAX_SEQ_LENGTH tokens (no a 512)
        # HA (~1700) + NA (~1400) + tokens especiales = ~3100
        ids = ids[:MAX_SEQ_LENGTH]

        input_ids = torch.tensor([ids], dtype=torch.long).to(device)
        attn_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask)

        hidden = out["hidden_states"]  # [1, seq_len, 384]

        # Mean pooling: promedio de todos los hidden states
        # Mas robusto que tomar solo el ultimo token
        z = hidden[0].mean(dim=0).cpu().numpy()  # [384]

        # Verificar que no hay NaN
        if np.isnan(z).any() or np.isinf(z).any():
            return None

        return z

    except Exception as e:
        return None


def extract_latent_batch(model, tokenizer, strains, device, batch_size=1):
    """
    Extrae embeddings por lotes sin imprimir ni guardar secuencias.

    Devuelve una lista alineada con `strains`, con `None` para registros que
    no pudieron codificarse o generaron NaN/Inf. Si un lote falla por memoria,
    se divide recursivamente hasta llegar a procesamiento individual.
    """
    if batch_size <= 1 or len(strains) <= 1:
        return [extract_latent(model, tokenizer, strain, device) for strain in strains]

    encoded = []
    valid_positions = []
    for pos, strain in enumerate(strains):
        try:
            ids = tokenizer.encode_strain(
                ha_sequence=strain["ha_sequence"],
                na_sequence=strain["na_sequence"],
                subtype=strain["subtype_token"],
            )
            if len(ids) < 10:
                continue
            encoded.append(ids[:MAX_SEQ_LENGTH])
            valid_positions.append(pos)
        except Exception:
            continue

    outputs = [None] * len(strains)
    if not encoded:
        return outputs

    max_len = max(len(ids) for ids in encoded)
    input_ids = torch.full(
        (len(encoded), max_len),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(encoded):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out["hidden_states"]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * mask).sum(dim=1) / denom
        pooled_np = pooled.detach().cpu().numpy()

        for row, pos in enumerate(valid_positions):
            z = pooled_np[row]
            if np.isfinite(z).all():
                outputs[pos] = z
        return outputs
    except RuntimeError as exc:
        # En full-data, MPS/CUDA puede agotar memoria para lotes largos. Dividir
        # el lote conserva el progreso sin relajar las restricciones metodologicas.
        if "out of memory" in str(exc).lower() or "mps" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = max(1, len(strains) // 2)
            return (
                extract_latent_batch(model, tokenizer, strains[:mid], device, batch_size=max(1, batch_size // 2))
                + extract_latent_batch(model, tokenizer, strains[mid:], device, batch_size=max(1, batch_size // 2))
            )
        raise


def sample_strains(strains, max_per_subtype, strategy, seed):
    """Selecciona cepas de forma configurable y reproducible."""
    if strategy == "all" or max_per_subtype is None or max_per_subtype < 0:
        return list(strains)

    if len(strains) <= max_per_subtype:
        return list(strains)

    if strategy == "first_n":
        return list(strains[:max_per_subtype])

    rng = random.Random(seed)
    if strategy == "random":
        return rng.sample(strains, max_per_subtype)

    if strategy == "stratified_by_year":
        by_year = defaultdict(list)
        for strain in strains:
            by_year[int(strain["year"])].append(strain)

        years_sorted = sorted(by_year)
        selected = []
        remaining_slots = max_per_subtype
        remaining_years = len(years_sorted)

        for year in years_sorted:
            bucket = list(by_year[year])
            rng.shuffle(bucket)
            quota = max(1, remaining_slots // max(remaining_years, 1))
            take = min(len(bucket), quota)
            selected.extend(bucket[:take])
            remaining_slots -= take
            remaining_years -= 1

        if len(selected) < max_per_subtype:
            selected_ids = {id(s) for s in selected}
            leftovers = [s for s in strains if id(s) not in selected_ids]
            rng.shuffle(leftovers)
            selected.extend(leftovers[:max_per_subtype - len(selected)])

        rng.shuffle(selected)
        return selected[:max_per_subtype]

    raise ValueError(f"Estrategia desconocida: {strategy}")


def summarize_sampling_distribution(years, types):
    summary = {}
    for subtype in sorted(set(types)):
        mask = types == subtype
        counts = Counter(int(y) for y in years[mask])
        summary[subtype] = dict(sorted(counts.items()))
    return summary


def record_resume_key(record):
    """Clave estable para reanudar sin imprimir ni persistir secuencias crudas nuevas."""
    ha = record.get("ha_sequence", "")
    na = record.get("na_sequence", "")
    sequence_hash = hashlib.sha1(f"{ha}|{na}".encode("utf-8")).hexdigest()
    stable_id = record.get("epi_isl") or record.get("strain_name") or "no_id"
    return "|".join(
        [
            str(record.get("subtype", "")),
            str(record.get("year", "")),
            str(record.get("month", "")),
            str(stable_id),
            sequence_hash,
        ]
    )


def validate_embedding_alignment(Z, years, months, types, records):
    lengths = {
        "embeddings": len(Z),
        "years": len(years),
        "months": len(months),
        "types": len(types),
        "records": len(records),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Cache/embeddings desalineados: {lengths}")


def save_embeddings_cache(path, Z, years, months, types, records, args):
    """Guarda embeddings y registros alineados sin recalcular el modelo."""
    validate_embedding_alignment(Z, years, months, types, records)
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    payload = {
        "embeddings": Z,
        "years": years,
        "months": months,
        "types": types,
        "records": records,
        "metadata": {
            "sampling_strategy": args.sampling_strategy,
            "seed": args.seed,
            "max_per_subtype": args.max_per_subtype,
            "max_seq_length": MAX_SEQ_LENGTH,
            "embedding_batch_size": getattr(args, "embedding_batch_size", 1),
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "checkpoint": checkpoint_metadata(),
        },
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nCache de embeddings guardado: {path}")
    print(f"  embeddings={Z.shape} records={len(records)}")


def load_embeddings_cache(path):
    """Carga embeddings cacheados preservando la alineacion con records."""
    print(f"\nCargando cache de embeddings: {path}")
    with open(path, "rb") as f:
        payload = pickle.load(f)

    required = ("embeddings", "years", "months", "types", "records")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Cache invalido, faltan claves: {missing}")

    Z = np.asarray(payload["embeddings"])
    years = np.asarray(payload["years"])
    months = np.asarray(payload["months"])
    types = np.asarray(payload["types"])
    records = payload["records"]
    metadata = payload.get("metadata", {})

    validate_embedding_alignment(Z, years, months, types, records)
    print(f"  Cache cargado: embeddings={Z.shape} records={len(records)}")
    print(
        "  Metadata: "
        f"sampling={metadata.get('sampling_strategy')} "
        f"seed={metadata.get('seed')} "
        f"max_per_subtype={metadata.get('max_per_subtype')}"
    )
    ckpt = metadata.get("checkpoint", {})
    if ckpt:
        print(f"  Checkpoint cacheado: {ckpt.get('path')} size={ckpt.get('size_bytes')}")
    return Z, years, months, types, records, metadata


def collect_embeddings(model, tokenizer, device,
                       max_per_subtype=MAX_STRAINS_PER_SUBTYPE,
                       sampling_strategy="random",
                       seed=RANDOM_SEED,
                       embedding_batch_size=1,
                       resume_cache_path=None,
                       cache_every=0,
                       args_for_cache=None):
    """Recolecta embeddings de cepas de ambos subtipos."""
    print(f"\nExtrayendo representaciones latentes...")
    max_text = "todos" if sampling_strategy == "all" or max_per_subtype is None or max_per_subtype < 0 else max_per_subtype
    print(f"  Maximo {max_text} cepas por subtipo")
    print(f"  Muestreo: {sampling_strategy} | seed={seed}")
    print(f"  Secuencia maxima: {MAX_SEQ_LENGTH} tokens")
    print(f"  Batch size embeddings: {embedding_batch_size}")

    all_z      = []
    all_years  = []
    all_months = []
    all_types  = []
    all_records = []
    resume_keys = set()

    if resume_cache_path and os.path.exists(resume_cache_path):
        print(f"  Reanudacion activada desde cache existente: {resume_cache_path}")
        Z0, y0, m0, t0, r0, _ = load_embeddings_cache(resume_cache_path)
        all_z = [np.asarray(row) for row in Z0]
        all_years = list(y0)
        all_months = list(m0)
        all_types = list(t0)
        all_records = list(r0)
        resume_keys = {record_resume_key(record) for record in all_records}
        print(f"  Registros ya cacheados que se omitiran: {len(resume_keys):,}")

    for subtype in ["H3N2", "H1N1"]:
        fpath = os.path.join(PROCESSED_DIR, f"dataset_{subtype}.json")
        if not os.path.exists(fpath):
            print(f"  AVISO: {fpath} no encontrado")
            continue

        with open(fpath, "r") as f:
            dataset = json.load(f)

        strains = dataset["paired_strains"]
        strains = [s for s in strains if s.get("year") is not None]
        strains = sample_strains(strains, max_per_subtype, sampling_strategy, seed)
        if resume_keys:
            before_resume = len(strains)
            strains = [s for s in strains if record_resume_key(s) not in resume_keys]
            print(
                f"  {subtype}: {before_resume - len(strains):,} ya estaban en cache; "
                f"faltan {len(strains):,}"
            )

        print(f"\n  {subtype}: procesando {len(strains)} cepas...")
        ok = 0
        failed = 0
        for i in range(0, len(strains), max(1, embedding_batch_size)):
            if i % 100 == 0:
                print(f"    {i}/{len(strains)}...", end="\r")

            batch = strains[i:i + max(1, embedding_batch_size)]
            batch_z = extract_latent_batch(
                model, tokenizer, batch, device,
                batch_size=max(1, embedding_batch_size),
            )
            for strain, z in zip(batch, batch_z):
                if z is not None:
                    all_z.append(z)
                    all_years.append(strain["year"])
                    all_months.append(strain.get("month", 6))
                    all_types.append(subtype)
                    all_records.append(strain)
                    ok += 1
                else:
                    failed += 1

            if (
                cache_every
                and resume_cache_path
                and args_for_cache is not None
                and ok > 0
                and ok % cache_every < len(batch)
            ):
                save_embeddings_cache(
                    resume_cache_path,
                    np.array(all_z),
                    np.array(all_years),
                    np.array(all_months),
                    np.array(all_types),
                    all_records,
                    args_for_cache,
                )

        print(f"    {ok} OK, {failed} fallidos de {len(strains)} total")

    if len(all_z) == 0:
        print("ERROR: no se extrajo ningun embedding")
        return None, None, None, None, None

    Z      = np.array(all_z)
    years  = np.array(all_years)
    months = np.array(all_months)
    types  = np.array(all_types)

    print(f"\n  Total embeddings: {len(Z)}")
    print(f"  Forma: {Z.shape}")
    print(f"  Rango: {years.min()}-{years.max()}")
    print(f"  Norma media: {np.linalg.norm(Z, axis=1).mean():.2f}")
    print(f"  NaN count: {np.isnan(Z).sum()}")
    return Z, years, months, types, all_records


# ---------------------------------------------------------------------------
# Experimento 1: UMAP
# ---------------------------------------------------------------------------

def plot_umap(Z, years, types, skip=False):
    """UMAP 2D: coloreado por anio y por subtipo."""
    if skip:
        print("\n[Experimento 1] UMAP omitido por --skip-umap")
        return None

    if not UMAP_AVAILABLE:
        print("\n  UMAP no disponible")
        return None

    print("\n[Experimento 1] UMAP del espacio latente...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_SEED,
    )
    Z_2d = reducer.fit_transform(Z)
    print(f"  UMAP completado: {Z_2d.shape}")

    # Figura 1A: por anio
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c=years, cmap="plasma",
                    s=8, alpha=0.7, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Anio de coleccion")
    ax.set_title("Espacio latente de AntigenLM - coloreado por anio\n"
                 r"$z_t \in \mathbb{R}^{384}$ proyectado con UMAP", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "umap_by_year.png")
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")

    # Figura 1B: por subtipo
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {"H3N2": "#E63946", "H1N1": "#457B9D"}
    for subtype, color in colors.items():
        mask = types == subtype
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=color, label=subtype,
                   s=8, alpha=0.7, linewidths=0)
    ax.legend(markerscale=3, framealpha=0.9)
    ax.set_title("Espacio latente de AntigenLM - coloreado por subtipo\n"
                 r"$z_t \in \mathbb{R}^{384}$ proyectado con UMAP", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "umap_by_subtype.png")
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")

    return Z_2d


# ---------------------------------------------------------------------------
# Experimento 2: Spearman POR SUBTIPO (corregido)
# ---------------------------------------------------------------------------

def normalized_hamming(seq_a, seq_b, tolerance=HAMMING_LENGTH_TOLERANCE):
    """Distancia de Hamming normalizada, omitiendo longitudes incompatibles."""
    if not seq_a or not seq_b:
        return None

    len_a = len(seq_a)
    len_b = len(seq_b)
    max_len = max(len_a, len_b)
    min_len = min(len_a, len_b)
    if max_len == 0:
        return None
    if (max_len - min_len) / max_len > tolerance:
        return None

    mismatches = sum(a != b for a, b in zip(seq_a[:min_len], seq_b[:min_len]))
    return mismatches / min_len


def biological_distance(record_a, record_b, metric):
    if metric == "hamming_ha":
        return normalized_hamming(
            record_a.get("ha_sequence", ""),
            record_b.get("ha_sequence", ""),
        )

    if metric == "hamming_nt":
        seq_a = record_a.get("ha_sequence", "") + record_a.get("na_sequence", "")
        seq_b = record_b.get("ha_sequence", "") + record_b.get("na_sequence", "")
        return normalized_hamming(seq_a, seq_b)

    if metric == "hamming_ha_na":
        ha_a = record_a.get("ha_sequence", "")
        ha_b = record_b.get("ha_sequence", "")
        na_a = record_a.get("na_sequence", "")
        na_b = record_b.get("na_sequence", "")

        d_ha = normalized_hamming(ha_a, ha_b)
        d_na = normalized_hamming(na_a, na_b)
        if d_ha is None or d_na is None:
            return None

        aligned_ha = min(len(ha_a), len(ha_b))
        aligned_na = min(len(na_a), len(na_b))
        denom = aligned_ha + aligned_na
        if denom == 0:
            return None
        return (d_ha * aligned_ha + d_na * aligned_na) / denom

    raise ValueError(f"Metrica biologica desconocida: {metric}")


def spearman_analysis(Z, years, months, types, records,
                      n_pairs=5000, distance_metric="temporal",
                      write_figures=True):
    """
    Correlacion de Spearman entre distancia latente y temporal.

    CORRECCION CRITICA: se calcula POR SUBTIPO, no mezclando ambos.
    Mezclar subtipos inflaba la correlacion artificialmente porque la
    distancia inter-subtipo (~29) domina sobre la variacion temporal.
    """
    print("\n[Experimento 2] Correlacion de Spearman (por subtipo)...")
    print(f"  Metrica de distancia: {distance_metric}")
    if distance_metric == "temporal":
        print("  AVISO: usa distancia temporal en meses como proxy; no es Hamming ni distancia biologica.")
    else:
        print("  Hamming normalizado: mismatches / aligned_length; no se imprimen secuencias.")
        print(f"  Tolerancia de longitud relativa: {HAMMING_LENGTH_TOLERANCE:.2%}")

    results_per_subtype = {}
    stats_per_subtype = {}

    for subtype in ["H3N2", "H1N1"]:
        mask = types == subtype
        Z_sub = Z[mask]
        y_sub = years[mask]
        m_sub = months[mask]
        rec_sub = [r for r, keep in zip(records, mask) if keep]
        N = len(Z_sub)

        if N < 50:
            print(f"  {subtype}: insuficientes datos ({N})")
            continue

        # Muestrear pares dentro del mismo subtipo
        actual_pairs = min(n_pairs, N * (N - 1) // 2)
        pairs = set()
        while len(pairs) < actual_pairs:
            i, j = random.sample(range(N), 2)
            if (i, j) not in pairs and (j, i) not in pairs:
                pairs.add((i, j))

        d_latente  = []
        d_metric = []
        omitted = 0
        for i, j in pairs:
            dl = np.linalg.norm(Z_sub[i] - Z_sub[j])
            if distance_metric == "temporal":
                dt = abs(y_sub[i] * 12 + m_sub[i] - y_sub[j] * 12 - m_sub[j])
            else:
                dt = biological_distance(rec_sub[i], rec_sub[j], distance_metric)
                if dt is None:
                    omitted += 1
                    continue
            d_latente.append(dl)
            d_metric.append(dt)

        d_latente  = np.array(d_latente)
        d_metric = np.array(d_metric)

        if len(d_metric) < 10:
            print(f"  {subtype}: pares validos insuficientes ({len(d_metric)}), omitidos={omitted}")
            continue

        rho, pval = spearmanr(d_latente, d_metric)
        results_per_subtype[subtype] = (rho, pval, len(d_metric))
        stats_per_subtype[subtype] = {
            "valid_pairs": len(d_metric),
            "omitted_pairs": omitted,
            "requested_pairs": len(pairs),
        }

        if rho > 0.5:
            interp = "FUERTE"
        elif rho > 0.3:
            interp = "MODERADA"
        elif rho > 0.1:
            interp = "DEBIL"
        else:
            interp = "AUSENTE"
        print(f"  {subtype}: rho={rho:.4f}  p={pval:.2e}  "
              f"validos={len(d_metric):,} omitidos={omitted:,}  [{interp}]")

    if not write_figures:
        rhos = [v[0] for v in results_per_subtype.values()]
        pvals = [v[1] for v in results_per_subtype.values()]
        return np.mean(rhos), np.mean(pvals), stats_per_subtype

    # Figura: un subplot por subtipo
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, subtype in enumerate(["H3N2", "H1N1"]):
        ax = axes[ax_idx]
        mask = types == subtype
        Z_sub = Z[mask]
        y_sub = years[mask]
        m_sub = months[mask]
        rec_sub = [r for r, keep in zip(records, mask) if keep]
        N = len(Z_sub)

        # Recalcular pares para la figura
        pairs_fig = set()
        while len(pairs_fig) < min(3000, N * (N-1) // 2):
            i, j = random.sample(range(N), 2)
            if (i, j) not in pairs_fig:
                pairs_fig.add((i, j))

        dl = []
        dt = []
        for i, j in pairs_fig:
            if distance_metric == "temporal":
                d = abs(y_sub[i]*12 + m_sub[i] - y_sub[j]*12 - m_sub[j])
            else:
                d = biological_distance(rec_sub[i], rec_sub[j], distance_metric)
                if d is None:
                    continue
            dl.append(np.linalg.norm(Z_sub[i] - Z_sub[j]))
            dt.append(d)

        rho_val = results_per_subtype[subtype][0]
        pval_val = results_per_subtype[subtype][1]

        hb = ax.hexbin(dt, dl, gridsize=35, cmap="YlOrRd", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Pares")
        ax.set_xlabel("Distancia temporal (meses)" if distance_metric == "temporal" else distance_metric)
        ax.set_ylabel(r"$\|z_i - z_j\|_2$")
        ax.set_title(f"{subtype}\nSpearman rho = {rho_val:.3f}  "
                     f"(p = {pval_val:.1e})")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"Correlacion distancia latente vs {distance_metric} (por subtipo)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "spearman_correlation.png")
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")

    # Retornar rho promedio
    rhos = [v[0] for v in results_per_subtype.values()]
    pvals = [v[1] for v in results_per_subtype.values()]
    return np.mean(rhos), np.mean(pvals), stats_per_subtype


# ---------------------------------------------------------------------------
# Experimento 3: Interpolacion lineal (corregido)
# ---------------------------------------------------------------------------

def interpolation_analysis(Z, years, types, n_pairs=5, skip=False, write_figures=True):
    """
    Interpolacion lineal entre cepas en el espacio latente.

    CORRECCION: mide la UNIFORMIDAD del paso entre interpolaciones,
    no solo la norma. Un espacio suave produce pasos uniformes.
    Un espacio con discontinuidades produce pasos desiguales.

    Tambien mide si la norma de los puntos intermedios se mantiene
    en un rango razonable (no colapsa a 0 ni explota).
    """
    if skip:
        print("\n[Experimento 3] Interpolacion omitida por --skip-interpolation")
        return None

    print("\n[Experimento 3] Interpolacion lineal...")
    print("  AVISO: esta prueba mide pasos lineales dentro del mismo espacio latente;")
    print("         un CV cercano a 0 puede ser una propiedad tautologica de la interpolacion lineal.")

    # Solo H3N2 para consistencia
    h3n2_mask = types == "H3N2"
    h3n2_idx  = np.where(h3n2_mask)[0]
    h3n2_years = years[h3n2_idx]

    # Ordenar por anio
    order = np.argsort(h3n2_years)
    sorted_idx = h3n2_idx[order]

    # Seleccionar pares separados por >5 anios
    pairs = []
    used = set()
    for k in range(0, len(sorted_idx) - 1):
        i = sorted_idx[k]
        for offset in range(50, min(200, len(sorted_idx) - k)):
            j = sorted_idx[min(k + offset, len(sorted_idx) - 1)]
            if abs(years[i] - years[j]) >= 5 and i not in used and j not in used:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break
        if len(pairs) >= n_pairs:
            break

    if len(pairs) == 0:
        print("  No se encontraron pares con >5 anios de separacion")
        return None

    lambdas = np.linspace(0, 1, 21)  # 21 puntos para mas resolucion

    if write_figures:
        fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 3 * len(pairs)))
        if len(pairs) == 1:
            axes = [axes]

    all_cv = []
    all_ratio = []

    for pair_idx, (i, j) in enumerate(pairs):
        z_a = Z[i]
        z_b = Z[j]
        dist_ab = np.linalg.norm(z_a - z_b)

        # Interpolacion lineal
        z_interp = np.array([(1 - l) * z_a + l * z_b for l in lambdas])

        # Metrica 1: uniformidad del paso
        # En un espacio suave, ||z(lambda_k+1) - z(lambda_k)|| es constante
        step_sizes = np.linalg.norm(np.diff(z_interp, axis=0), axis=1)
        cv = np.std(step_sizes) / (np.mean(step_sizes) + 1e-10)
        all_cv.append(cv)

        # Metrica 2: ratio norma minima / norma maxima
        # En un espacio bien condicionado, no hay colapso ni explosion
        norms = np.linalg.norm(z_interp, axis=1)
        ratio = norms.min() / (norms.max() + 1e-10)
        all_ratio.append(ratio)

        if write_figures:
            # Subfigura izquierda: norma a lo largo de la interpolacion
            ax_left = axes[pair_idx][0] if len(pairs) > 1 else axes[0]
            ax_left.plot(lambdas, norms, "o-", color="#457B9D",
                         linewidth=2, markersize=4)
            ax_left.set_ylabel(r"$\|z(\lambda)\|_2$")
            ax_left.set_title(
                f"Par {pair_idx+1}: {years[i]} -> {years[j]}  |  "
                f"ratio min/max = {ratio:.4f}")
            ax_left.spines[["top", "right"]].set_visible(False)

            # Subfigura derecha: tamanio de paso
            ax_right = axes[pair_idx][1] if len(pairs) > 1 else axes[1]
            ax_right.bar(range(len(step_sizes)), step_sizes,
                         color="#E63946", alpha=0.7)
            ax_right.axhline(np.mean(step_sizes), color="gray",
                             linestyle="--", linewidth=1)
            ax_right.set_ylabel("Tamanio del paso")
            ax_right.set_title(f"CV = {cv:.6f}  |  dist = {dist_ab:.2f}")
            ax_right.spines[["top", "right"]].set_visible(False)

            if pair_idx == len(pairs) - 1:
                ax_left.set_xlabel("lambda (0=cepa A, 1=cepa B)")
                ax_right.set_xlabel("Paso de interpolacion")

    mean_cv = np.mean(all_cv)
    mean_ratio = np.mean(all_ratio)
    path = None
    if write_figures:
        fig.suptitle(
            f"Interpolacion lineal en espacio latente\n"
            f"CV promedio = {mean_cv:.6f}  |  "
            f"Ratio min/max promedio = {mean_ratio:.4f}",
            fontsize=13, y=1.02)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "interpolation.png")
        if os.path.exists(path):
            os.remove(path)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    if mean_cv < 0.01:
        interp = "espacio localmente euclidiano (pasos perfectamente uniformes)"
    elif mean_cv < 0.3:
        interp = "espacio localmente suave"
    else:
        interp = "variacion irregular"

    print(f"  CV promedio: {mean_cv:.6f}")
    print(f"  Ratio min/max promedio: {mean_ratio:.4f}")
    print(f"  Interpretacion: {interp}")
    if path:
        print(f"  Guardado: {path}")

    return mean_cv


# ---------------------------------------------------------------------------
# Experimento 4: Dimension intrinseca
# ---------------------------------------------------------------------------

def sequence_key(record):
    ha = record.get("ha_sequence", "") if record else ""
    na = record.get("na_sequence", "") if record else ""
    subtype = record.get("subtype_token", record.get("subtype", "")) if record else ""
    text = f"{subtype}|{ha}|{na}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def duplicate_frequency_summary(counts, max_items=5):
    freq = Counter(counts)
    parts = [f"{size}x:{freq[size]}" for size in sorted(freq)[:max_items]]
    if len(freq) > max_items:
        parts.append("...")
    return ", ".join(parts)


def audit_intrinsic_duplicates(embeddings, records=None):
    """Audita duplicados exactos sin imprimir secuencias."""
    print("  Auditoria de duplicados exactos:")
    n = len(embeddings)

    if records is not None:
        seq_counts = Counter(sequence_key(record) for record in records)
        duplicate_groups = [count for count in seq_counts.values() if count > 1]
        duplicate_points = sum(count - 1 for count in duplicate_groups)
        print(
            f"    HA+NA: unicos={len(seq_counts):,}/{n:,} "
            f"duplicados_extra={duplicate_points:,} "
            f"grupos_duplicados={len(duplicate_groups):,}"
        )
        print(
            f"    HA+NA freq grupos: "
            f"{duplicate_frequency_summary(seq_counts.values())}"
        )
    else:
        print("    HA+NA: records no disponibles")

    Z = np.asarray(embeddings)
    finite = np.isfinite(Z).all(axis=1)
    if not finite.all():
        print(f"    embeddings: {np.sum(~finite):,} filas no finitas omitidas para unique")
    Z_finite = Z[finite]
    if len(Z_finite) == 0:
        print("    embeddings: no hay filas finitas")
        return

    _, emb_counts = np.unique(Z_finite, axis=0, return_counts=True)
    duplicate_groups = [int(count) for count in emb_counts if count > 1]
    duplicate_points = sum(count - 1 for count in duplicate_groups)
    print(
        f"    embeddings exactos: unicos={len(emb_counts):,}/{len(Z_finite):,} "
        f"duplicados_extra={duplicate_points:,} "
        f"grupos_duplicados={len(duplicate_groups):,}"
    )
    print(
        f"    embeddings freq grupos: "
        f"{duplicate_frequency_summary(emb_counts)}"
    )


def deduplicate_by_sequence(embeddings, types, records):
    """Deduplica por hash exacto de subtipo+HA+NA, conservando primera ocurrencia."""
    if records is None:
        print("  AVISO: no hay records; no se puede deduplicar por secuencia")
        return embeddings, types, records

    seen = set()
    keep = []
    for idx, record in enumerate(records):
        key = sequence_key(record)
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)

    keep = np.array(keep, dtype=int)
    removed = len(records) - len(keep)
    print(
        f"  Deduplicacion por HA+NA: {len(records):,} -> {len(keep):,} "
        f"(removidos={removed:,})"
    )
    dedup_records = [records[i] for i in keep]
    dedup_types = types[keep] if types is not None else None
    return embeddings[keep], dedup_types, dedup_records


def normalize_embeddings_for_intrinsic(embeddings, mode="none", eps=1e-12):
    X = np.asarray(embeddings, dtype=np.float64)
    zero_var_dims = 0

    if mode in ("standard", "standard_l2"):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        zero_var_dims = int(np.sum(std <= eps))
        std[std <= eps] = 1.0
        X = (X - mean) / std

    if mode in ("l2", "standard_l2"):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        zero_norm_rows = norms[:, 0] <= eps
        norms[zero_norm_rows] = 1.0
        X = X / norms
    else:
        zero_norm_rows = np.zeros(len(X), dtype=bool)

    return X, {
        "normalization": mode,
        "zero_var_dims": zero_var_dims,
        "zero_norm_rows": int(np.sum(zero_norm_rows)),
    }


def nearest_neighbor_quantiles(embeddings, eps=1e-12):
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError("scikit-learn es necesario para auditoria NN") from exc

    X = np.asarray(embeddings, dtype=np.float64)
    if len(X) < 3:
        return None
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    qs = [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
    return {
        "r1_zero": int(np.sum(r1 <= eps)),
        "r2_zero": int(np.sum(r2 <= eps)),
        "r1_quantiles": dict(zip(qs, np.quantile(r1, qs))),
        "r2_quantiles": dict(zip(qs, np.quantile(r2, qs))),
    }


def format_quantiles(qdict):
    return ", ".join(f"q{q:g}={value:.4g}" for q, value in qdict.items())


def estimate_intrinsic_dimension_twonn(embeddings, eps=1e-12,
                                       normalize="none",
                                       trim_quantile=0.0):
    """
    Estima dimension intrinseca con TwoNN sin matriz par-a-par completa.

    TwoNN usa mu = r2 / r1, donde r1 y r2 son las distancias al primer
    y segundo vecino mas cercano. Para datos localmente uniformes,
    -log(1 - F(mu)) es aproximadamente lineal en log(mu), con pendiente d.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError("scikit-learn es necesario para TwoNN propio") from exc

    X = np.asarray(embeddings, dtype=np.float64)
    finite_mask = np.isfinite(X).all(axis=1)
    X = X[finite_mask]
    n_input = len(embeddings)
    n_nonfinite = n_input - len(X)
    trim_quantile = max(0.0, min(float(trim_quantile), 0.49))

    if len(X) < 3:
        return {
            "dimension": None,
            "n_input": n_input,
            "n_used": 0,
            "n_omitted": n_input,
            "n_zero_omitted": 0,
            "n_nonfinite_omitted": n_nonfinite,
            "warnings": ["insuficientes puntos finitos para TwoNN"],
            "normalization": normalize,
        }

    X_norm, norm_stats = normalize_embeddings_for_intrinsic(X, mode=normalize, eps=eps)

    nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nn.fit(X_norm)
    distances, _ = nn.kneighbors(X_norm)

    r1 = distances[:, 1]
    r2 = distances[:, 2]
    nn_quantiles = nearest_neighbor_quantiles(X_norm, eps=eps)
    zero_mask = (r1 <= eps) | (r2 <= eps)
    finite_dist_mask = np.isfinite(r1) & np.isfinite(r2)
    ordered_mask = r2 > r1
    valid_mask = finite_dist_mask & ordered_mask & (~zero_mask)

    mu = r2[valid_mask] / r1[valid_mask]
    mu = mu[np.isfinite(mu) & (mu > 1.0 + eps)]

    n_used = len(mu)
    n_zero_omitted = int(np.sum(zero_mask))
    n_omitted = n_input - n_used
    warnings = []

    if n_nonfinite:
        warnings.append(f"{n_nonfinite} puntos omitidos por NaN/Inf")
    if n_zero_omitted:
        warnings.append(f"{n_zero_omitted} puntos omitidos por distancias cero")
    if n_used < 50:
        warnings.append("muestra valida pequena para TwoNN")

    if n_used < 3:
        return {
            "dimension": None,
            "n_input": n_input,
            "n_used": n_used,
            "n_omitted": n_omitted,
            "n_zero_omitted": n_zero_omitted,
            "n_nonfinite_omitted": n_nonfinite,
            "zero_var_dims": norm_stats["zero_var_dims"],
            "zero_norm_rows": norm_stats["zero_norm_rows"],
            "normalization": normalize,
            "trim_quantile": trim_quantile,
            "warnings": warnings + ["insuficientes razones mu validas"],
        }

    n_before_trim = len(mu)
    if trim_quantile > 0:
        lo = np.quantile(mu, trim_quantile)
        hi = np.quantile(mu, 1.0 - trim_quantile)
        mu = mu[(mu >= lo) & (mu <= hi)]
    n_trimmed = n_before_trim - len(mu)

    if len(mu) < 3:
        return {
            "dimension": None,
            "n_input": n_input,
            "n_used": len(mu),
            "n_omitted": n_input - len(mu),
            "n_zero_omitted": n_zero_omitted,
            "n_nonfinite_omitted": n_nonfinite,
            "zero_var_dims": norm_stats["zero_var_dims"],
            "zero_norm_rows": norm_stats["zero_norm_rows"],
            "normalization": normalize,
            "trim_quantile": trim_quantile,
            "n_trimmed": n_trimmed,
            "nn_quantiles": nn_quantiles,
            "warnings": warnings + ["insuficientes razones mu validas despues de trimming"],
        }

    n_used = len(mu)
    n_omitted = n_input - n_used
    mu_sorted = np.sort(mu)
    x = np.log(mu_sorted)
    # Posiciones empiricas con correccion de rango para evitar F=1.
    F = (np.arange(1, n_used + 1) - 0.5) / n_used
    y = -np.log1p(-F)

    A = np.column_stack([x, np.ones_like(x)])
    if float(np.var(x)) <= eps:
        dimension = None
        intercept = np.nan
        r2_score = np.nan
        warnings.append("varianza insuficiente en log(mu)")
    else:
        dimension, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        dimension = float(dimension)
        intercept = float(intercept)
        y_hat = dimension * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_score = 1.0 - ss_res / ss_tot if ss_tot > eps else np.nan
        if np.isfinite(r2_score) and r2_score < 0.95:
            warnings.append(f"ajuste lineal TwoNN potencialmente inestable (R2={r2_score:.3f})")

    if dimension is not None and dimension > X.shape[1]:
        warnings.append("dimension estimada mayor que la dimension ambiente")

    return {
        "dimension": dimension,
        "n_input": n_input,
        "n_used": n_used,
        "n_omitted": n_omitted,
        "n_zero_omitted": n_zero_omitted,
        "n_nonfinite_omitted": n_nonfinite,
        "zero_var_dims": norm_stats["zero_var_dims"],
        "zero_norm_rows": norm_stats["zero_norm_rows"],
        "normalization": normalize,
        "trim_quantile": trim_quantile,
        "n_trimmed": n_trimmed,
        "intercept": intercept,
        "r2_score": r2_score,
        "mu_min": float(np.min(mu)),
        "mu_median": float(np.median(mu)),
        "mu_max": float(np.max(mu)),
        "nn_quantiles": nn_quantiles,
        "warnings": warnings,
    }


def sample_embeddings_for_intrinsic(Z, sample_size, seed):
    if sample_size is None or sample_size <= 0 or len(Z) <= sample_size:
        return Z
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=sample_size, replace=False)
    return Z[idx]


def print_twonn_result(label, result):
    if result["dimension"] is None:
        dim_text = "N/A"
    else:
        dim_text = f"{result['dimension']:.2f}"

    print(
        f"  {label}: d={dim_text} | usados={result['n_used']:,} "
        f"| omitidos={result['n_omitted']:,} "
        f"| dist_cero={result['n_zero_omitted']:,} "
        f"| norm={result.get('normalization', 'N/A')} "
        f"| trim={result.get('trim_quantile', 0.0):.2f}"
    )
    if result.get("r2_score") is not None and np.isfinite(result.get("r2_score", np.nan)):
        print(
            f"    diagnostico: R2={result['r2_score']:.4f} "
            f"intercept={result.get('intercept', np.nan):.4f} "
            f"mu_mediana={result['mu_median']:.4f} "
            f"mu_rango=[{result['mu_min']:.4f}, {result['mu_max']:.4f}]"
        )
    if result.get("n_trimmed"):
        print(f"    trimming: {result['n_trimmed']:,} razones mu removidas")
    nnq = result.get("nn_quantiles")
    if nnq:
        print(
            f"    NN r1 ceros={nnq['r1_zero']:,} | "
            f"cuantiles r1: {format_quantiles(nnq['r1_quantiles'])}"
        )
        print(
            f"    NN r2 ceros={nnq['r2_zero']:,} | "
            f"cuantiles r2: {format_quantiles(nnq['r2_quantiles'])}"
        )
    if result.get("zero_var_dims"):
        print(f"    aviso: {result['zero_var_dims']} dimensiones con varianza ~0 antes de normalizar")
    if result.get("zero_norm_rows"):
        print(f"    aviso: {result['zero_norm_rows']} filas con norma ~0 antes de L2")
    for warning in result.get("warnings", []):
        print(f"    AVISO: {warning}")


def intrinsic_dimension(Z, types=None, skip=False, method="twonn",
                        sample_size=500, by_subtype=False, seed=RANDOM_SEED,
                        records=None, deduplicate=False,
                        normalization="none", trim_quantile=0.0):
    """Ejecuta estimacion configurable de dimension intrinseca."""
    if skip:
        print("\n[Experimento 4] Dimension intrinseca omitida por --skip-intrinsic-dim")
        return None

    print("\n[Experimento 4] Dimension intrinseca...")
    print(f"  Metodo: {method}")
    print(f"  Sample size maximo: {sample_size if sample_size else 'sin submuestreo'}")
    print(f"  Deduplicar por HA+NA: {'si' if deduplicate else 'no'}")
    print(f"  Normalizacion final: {normalization}")
    print(f"  TwoNN trim quantile: {trim_quantile:.2f}")

    if method != "twonn":
        raise ValueError(f"Metodo de dimension intrinseca no soportado: {method}")

    audit_intrinsic_duplicates(Z, records)

    Z_work = Z
    types_work = types
    records_work = records
    if deduplicate:
        Z_work, types_work, records_work = deduplicate_by_sequence(Z, types, records)
        audit_intrinsic_duplicates(Z_work, records_work)

    print("  Comparacion global TwoNN:")
    Z_raw_sample = sample_embeddings_for_intrinsic(Z, sample_size, seed)
    result_raw = estimate_intrinsic_dimension_twonn(
        Z_raw_sample,
        normalize="none",
        trim_quantile=trim_quantile,
    )
    print_twonn_result("global sin dedup/sin normalizar", result_raw)

    if deduplicate:
        Z_final_sample = sample_embeddings_for_intrinsic(Z_work, sample_size, seed)
        result_dedup = estimate_intrinsic_dimension_twonn(
            Z_final_sample,
            normalize="none",
            trim_quantile=trim_quantile,
        )
        print_twonn_result("global dedup/sin normalizar", result_dedup)
    else:
        Z_final_sample = Z_raw_sample
        result_dedup = result_raw

    if normalization != "none":
        result_global = estimate_intrinsic_dimension_twonn(
            Z_final_sample,
            normalize=normalization,
            trim_quantile=trim_quantile,
        )
        print_twonn_result(f"global final/{normalization}", result_global)
    else:
        result_global = result_dedup

    subtype_results = {}
    if by_subtype:
        if types_work is None:
            print("  AVISO: no se recibieron subtipos; se omite estimacion por subtipo")
        else:
            print("  Estimacion final por subtipo:")
            for subtype in sorted(set(types_work)):
                mask = types_work == subtype
                Z_sub = sample_embeddings_for_intrinsic(Z_work[mask], sample_size, seed)
                result = estimate_intrinsic_dimension_twonn(
                    Z_sub,
                    normalize=normalization,
                    trim_quantile=trim_quantile,
                )
                subtype_results[subtype] = result
                print_twonn_result(subtype, result)

    if subtype_results:
        dims = [
            result["dimension"]
            for result in subtype_results.values()
            if result["dimension"] is not None
        ]
        if dims:
            print(f"  Dimension media por subtipo: {np.mean(dims):.2f}")

    print("  Nota: TwoNN es sensible a muestreo, duplicados, metrica y normalizacion.")
    print("        Este resultado diagnostica plausibilidad geometrica; no justifica por si solo una SDE.")

    return result_global["dimension"]


# ---------------------------------------------------------------------------
# Experimento 5: Dimension efectiva espectral via PCA
# ---------------------------------------------------------------------------

def pca_spectrum_summary(embeddings, max_components=100, eps=1e-12):
    """
    Calcula espectro PCA exacto via matriz de covarianza.

    La dimension ambiente es pequena (384), asi que esta ruta evita cargar
    AntigenLM y calcula todo el espectro sin matriz par-a-par entre cepas.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    finite_mask = np.isfinite(X).all(axis=1)
    X = X[finite_mask]
    n_input = len(embeddings)
    n_samples = len(X)

    if n_samples < 2:
        return {
            "n_input": n_input,
            "n_samples": n_samples,
            "n_omitted_nonfinite": n_input - n_samples,
            "embedding_dim": int(np.asarray(embeddings).shape[1]) if np.asarray(embeddings).ndim == 2 else 0,
            "error": "insuficientes puntos finitos para PCA",
        }

    embedding_dim = X.shape[1]
    max_components = max(1, min(int(max_components), embedding_dim))
    X = X - X.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(n_samples - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.clip(eigvals, 0.0, None)
    positive = eigvals[eigvals > eps]

    if len(positive) == 0:
        return {
            "n_input": n_input,
            "n_samples": n_samples,
            "n_omitted_nonfinite": n_input - n_samples,
            "embedding_dim": embedding_dim,
            "error": "todas las varianzas PCA son cero",
        }

    total_var = float(np.sum(positive))
    ratios = positive / total_var
    cumulative = np.cumsum(ratios)
    threshold_components = {}
    for threshold in PCA_VARIANCE_THRESHOLDS:
        threshold_components[threshold] = int(np.searchsorted(cumulative, threshold) + 1)

    participation_ratio = float((np.sum(positive) ** 2) / np.sum(positive ** 2))
    display_count = min(max_components, len(ratios))

    return {
        "n_input": n_input,
        "n_samples": n_samples,
        "n_omitted_nonfinite": n_input - n_samples,
        "embedding_dim": embedding_dim,
        "n_positive_components": int(len(positive)),
        "max_components_reported": display_count,
        "explained_variance": positive,
        "explained_variance_ratio": ratios,
        "cumulative_variance_ratio": cumulative,
        "n_components_by_threshold": threshold_components,
        "participation_ratio": participation_ratio,
    }


def format_top_ratios(ratios, n=10):
    top = ratios[:min(n, len(ratios))]
    return ", ".join(f"{value:.4f}" for value in top)


def print_pca_summary(label, summary):
    print(f"  {label}:")
    if summary.get("error"):
        print(f"    ERROR: {summary['error']}")
        return

    thresholds = summary["n_components_by_threshold"]
    print(
        f"    n_samples={summary['n_samples']:,} "
        f"| embedding_dim={summary['embedding_dim']} "
        f"| componentes_positivas={summary['n_positive_components']}"
    )
    if summary["n_omitted_nonfinite"]:
        print(f"    omitidos por NaN/Inf={summary['n_omitted_nonfinite']:,}")
    print(
        f"    n80={thresholds[0.80]} | n90={thresholds[0.90]} "
        f"| n95={thresholds[0.95]} | n99={thresholds[0.99]}"
    )
    print(f"    participation_ratio={summary['participation_ratio']:.2f}")
    print(
        "    top10 explained_variance_ratio="
        f"[{format_top_ratios(summary['explained_variance_ratio'], n=10)}]"
    )


def save_pca_figures(results, max_components=100):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    global_summary = results.get("global")
    if global_summary and not global_summary.get("error"):
        ratios = global_summary["explained_variance_ratio"]
        cumulative = global_summary["cumulative_variance_ratio"]
        n_plot = min(max_components, len(ratios))
        xs = np.arange(1, n_plot + 1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(xs, ratios[:n_plot], marker="o", markersize=3, linewidth=1.5)
        ax.set_xlabel("Componente PCA")
        ax.set_ylabel("Varianza explicada")
        ax.set_title("PCA global - scree plot")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "pca_scree_global.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Guardado: {path}")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(xs, cumulative[:n_plot], marker="o", markersize=3, linewidth=1.5)
        for threshold in PCA_VARIANCE_THRESHOLDS:
            ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Componentes PCA")
        ax.set_ylabel("Varianza acumulada")
        ax.set_ylim(0, 1.02)
        ax.set_title("PCA global - varianza acumulada")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "pca_cumulative_global.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Guardado: {path}")

    subtype_results = {
        label: summary
        for label, summary in results.items()
        if label != "global" and not summary.get("error")
    }
    if subtype_results:
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, summary in subtype_results.items():
            ratios = summary["explained_variance_ratio"]
            n_plot = min(max_components, len(ratios))
            ax.plot(
                np.arange(1, n_plot + 1), ratios[:n_plot],
                marker="o", markersize=3, linewidth=1.5, label=label
            )
        ax.set_xlabel("Componente PCA")
        ax.set_ylabel("Varianza explicada")
        ax.set_title("PCA por subtipo - scree plot")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "pca_scree_by_subtype.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Guardado: {path}")

        fig, ax = plt.subplots(figsize=(9, 5))
        for label, summary in subtype_results.items():
            cumulative = summary["cumulative_variance_ratio"]
            n_plot = min(max_components, len(cumulative))
            ax.plot(
                np.arange(1, n_plot + 1), cumulative[:n_plot],
                marker="o", markersize=3, linewidth=1.5, label=label
            )
        for threshold in PCA_VARIANCE_THRESHOLDS:
            ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Componentes PCA")
        ax.set_ylabel("Varianza acumulada")
        ax.set_ylim(0, 1.02)
        ax.set_title("PCA por subtipo - varianza acumulada")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "pca_cumulative_by_subtype.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Guardado: {path}")


def pca_effective_dimension(Z, types=None, enabled=False, max_components=100,
                            by_subtype=False, save_figures=False):
    if not enabled:
        return None

    print("\n[Experimento 5] Dimension efectiva espectral via PCA...")
    print("  PCA desde embeddings en memoria/cache; no carga AntigenLM.")
    print(f"  Componentes maximas para reporte/figuras: {max_components}")
    print("  Participation ratio usa el espectro completo de la covarianza.")

    results = {}
    results["global"] = pca_spectrum_summary(Z, max_components=max_components)
    print_pca_summary("global", results["global"])

    if by_subtype:
        if types is None:
            print("  AVISO: no se recibieron subtipos; se omite PCA por subtipo")
        else:
            for subtype in sorted(set(types)):
                mask = types == subtype
                results[subtype] = pca_spectrum_summary(
                    Z[mask],
                    max_components=max_components,
                )
                print_pca_summary(subtype, results[subtype])

    if save_figures:
        print("  Guardando figuras PCA por solicitud explicita --save-pca-figures")
        save_pca_figures(results, max_components=max_components)
    else:
        print("  Figuras PCA no escritas (usa --save-pca-figures para guardarlas).")

    print("  Nota: PCA mide dimension efectiva lineal; si discrepa de TwoNN,")
    print("        puede indicar curvatura/manifold no lineal o anisotropia global.")
    return results


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def write_report(rho, pval, smoothness, intrinsic_dim, sampling_summary,
                 distance_metric, distance_stats, write_to_disk=True):
    path = os.path.join(RESULTS_DIR, "geometry_report.txt")
    if write_to_disk:
        os.makedirs(RESULTS_DIR, exist_ok=True)

    lines = [
        "=" * 60,
        "REPORTE DE GEOMETRIA DEL ESPACIO LATENTE - AntigenLM",
        "=" * 60,
        "",
        "EXPERIMENTO 1: UMAP",
        "  Ver figures/gisaid/umap_by_year.png si fue ejecutado",
        "  Ver figures/gisaid/umap_by_subtype.png si fue ejecutado",
        "",
        "MUESTREO USADO:",
    ]

    for subtype, counts in sampling_summary.items():
        total = sum(counts.values())
        compact = ", ".join(f"{year}:{count}" for year, count in counts.items())
        lines.append(f"  {subtype} total={total} | {compact}")

    lines.extend([
        "",
        "EXPERIMENTO 2: CORRELACION DE SPEARMAN (por subtipo)",
        f"  Metrica = {distance_metric}",
        f"  rho promedio = {rho:.4f}" if rho is not None else "  No ejecutado",
        f"  p promedio   = {pval:.2e}" if pval is not None else "  No ejecutado",
    ])

    if distance_metric == "temporal":
        lines.extend([
        "  ADVERTENCIA: Spearman usa distancia temporal en meses como proxy;",
        "  no mide Hamming, distancia antigenica ni distancia biologica real.",
        ])
    else:
        lines.extend([
            "  Hamming normalizado = mismatches / aligned_length.",
            f"  Tolerancia relativa de longitud = {HAMMING_LENGTH_TOLERANCE:.2%}.",
        ])
        for subtype, stats in distance_stats.items():
            lines.append(
                f"  {subtype}: validos={stats['valid_pairs']} "
                f"omitidos={stats['omitted_pairs']} "
                f"muestreados={stats['requested_pairs']}"
            )

    lines.extend([
        "",
        "EXPERIMENTO 3: INTERPOLACION LINEAL",
        f"  CV promedio = {smoothness:.6f}" if smoothness is not None else "  No ejecutado",
        "  ADVERTENCIA: la prueba interpola linealmente en el propio espacio latente;",
        "  un CV cercano a cero puede ser tautologico y no prueba dinamica biologica.",
        "",
        "EXPERIMENTO 4: DIMENSION INTRINSECA",
        f"  Estimacion: {intrinsic_dim:.1f} dims efectivas (de 384)" if intrinsic_dim else "  No ejecutado",
        "",
        "CONCLUSION:",
        "  Los resultados preliminares no descartan el uso del espacio latente",
        "  para dinamicas continuas, pero aun son insuficientes para justificar",
        "  plenamente una SDE. Se requieren pruebas adicionales de metrica",
        "  biologica, dimension intrinseca y estabilidad bajo muestreo.",
    ])
    lines.append("")

    report = "\n".join(lines)
    print("\n" + report)
    if write_to_disk:
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Reporte: {path}")
    else:
        print("Reporte no escrito en disco.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auditoria configurable de geometria latente AntigenLM"
    )
    parser.add_argument("--max-per-subtype", type=int,
                        default=MAX_STRAINS_PER_SUBTYPE)
    parser.add_argument("--pair-samples", type=int, default=5000)
    parser.add_argument("--sampling-strategy", choices=SAMPLING_STRATEGIES,
                        default="random")
    parser.add_argument("--distance-metric", choices=DISTANCE_METRICS,
                        default="temporal")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--skip-interpolation", action="store_true")
    parser.add_argument("--skip-intrinsic-dim", action="store_true")
    parser.add_argument("--intrinsic-method", choices=INTRINSIC_METHODS,
                        default="twonn")
    parser.add_argument("--intrinsic-sample-size", type=int, default=500)
    parser.add_argument("--intrinsic-by-subtype", action="store_true")
    parser.add_argument("--deduplicate-for-intrinsic", action="store_true")
    parser.add_argument("--normalize-embeddings-for-intrinsic",
                        choices=INTRINSIC_NORMALIZATIONS, default="none")
    parser.add_argument("--twonn-trim-quantile", type=float, default=0.0)
    parser.add_argument("--save-embeddings-cache", action="store_true",
                        help="Guarda embeddings y records alineados en --cache-path")
    parser.add_argument("--load-embeddings-cache", action="store_true",
                        help="Carga embeddings desde --cache-path sin ejecutar el modelo")
    parser.add_argument("--cache-path", default=os.path.join(RESULTS_DIR, "embeddings_cache.pkl"))
    parser.add_argument("--pca-effective-dim", action="store_true",
                        help="Calcula dimension efectiva espectral desde embeddings en memoria/cache")
    parser.add_argument("--pca-max-components", type=int, default=100,
                        help="Numero maximo de componentes PCA para reporte y figuras")
    parser.add_argument("--pca-by-subtype", action="store_true",
                        help="Calcula PCA global y separado por subtipo")
    parser.add_argument("--save-pca-figures", action="store_true",
                        help="Guarda figuras PCA en figures/gisaid; no se guardan por defecto")
    parser.add_argument("--embedding-batch-size", type=int, default=1,
                        help="Batch size para extraccion de embeddings; 1 reproduce la ruta historica")
    parser.add_argument("--resume-cache", action="store_true",
                        help="Si --cache-path existe, reanuda y omite registros ya cacheados")
    parser.add_argument("--checkpoint-cache-every", type=int, default=0,
                        help="Guarda cache parcial cada N embeddings exitosos durante extraccion")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    write_figures = not args.skip_umap and not args.skip_interpolation and not args.skip_intrinsic_dim
    if write_figures:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.load_embeddings_cache:
        Z, years, months, types, records, cache_metadata = load_embeddings_cache(args.cache_path)
    else:
        # Detectar dispositivo solo cuando hay que ejecutar el modelo.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model     = load_model(device)
        tokenizer = InfluTokenizer(mode="prediction")

        Z, years, months, types, records = collect_embeddings(
            model, tokenizer, device,
            max_per_subtype=args.max_per_subtype,
            sampling_strategy=args.sampling_strategy,
            seed=args.seed,
            embedding_batch_size=args.embedding_batch_size,
            resume_cache_path=args.cache_path if args.resume_cache else None,
            cache_every=args.checkpoint_cache_every,
            args_for_cache=args,
        )

        if Z is not None and args.save_embeddings_cache:
            save_embeddings_cache(args.cache_path, Z, years, months, types, records, args)

    if Z is None:
        print("\nERROR: sin embeddings.")
        exit(1)

    sampling_summary = summarize_sampling_distribution(years, types)

    # Experimentos
    Z_2d       = plot_umap(Z, years, types, skip=args.skip_umap)
    rho, pval, distance_stats = spearman_analysis(
        Z, years, months, types, records,
        n_pairs=args.pair_samples,
        distance_metric=args.distance_metric,
        write_figures=write_figures,
    )
    smoothness = interpolation_analysis(
        Z, years, types,
        skip=args.skip_interpolation,
        write_figures=write_figures,
    )
    id_est     = intrinsic_dimension(
        Z, types=types,
        skip=args.skip_intrinsic_dim,
        method=args.intrinsic_method,
        sample_size=args.intrinsic_sample_size,
        by_subtype=args.intrinsic_by_subtype,
        seed=args.seed,
        records=records,
        deduplicate=args.deduplicate_for_intrinsic,
        normalization=args.normalize_embeddings_for_intrinsic,
        trim_quantile=args.twonn_trim_quantile,
    )
    pca_effective_dimension(
        Z, types=types,
        enabled=args.pca_effective_dim,
        max_components=args.pca_max_components,
        by_subtype=args.pca_by_subtype,
        save_figures=args.save_pca_figures,
    )

    write_report(
        rho, pval, smoothness, id_est, sampling_summary,
        args.distance_metric, distance_stats,
        write_to_disk=write_figures,
    )

    print("\n" + "=" * 60)
    print("Analisis completado.")
    if write_figures:
        print(f"Figuras: {FIGURES_DIR}/")
        print(f"Reporte: {RESULTS_DIR}/geometry_report.txt")
    else:
        print("Figuras: no escritas en esta corrida")
        print("Reporte: no escrito en esta corrida")
    print("=" * 60)
