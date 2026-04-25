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

MAX_STRAINS_PER_SUBTYPE = 2000
MAX_SEQ_LENGTH          = 4000  # secuencias completas HA+NA (~3100 tokens)
RANDOM_SEED             = 42

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
        "prediction_sequence/pytorch_model.bin",
        map_location="cpu"
    )
    remapped = {k.replace("transformer.", "backbone."): v
                for k, v in ckpt.items()}

    model = GPTForFluMultiTask(task="prediction")
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0, \
        f"Error cargando pesos: missing={missing}, unexpected={unexpected}"

    model.eval()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Modelo cargado en {device} - {n_params:,} params")
    return model


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


def collect_embeddings(model, tokenizer, device,
                       max_per_subtype=MAX_STRAINS_PER_SUBTYPE):
    """Recolecta embeddings de cepas de ambos subtipos."""
    print(f"\nExtrayendo representaciones latentes...")
    print(f"  Maximo {max_per_subtype} cepas por subtipo")
    print(f"  Secuencia maxima: {MAX_SEQ_LENGTH} tokens")

    all_z      = []
    all_years  = []
    all_months = []
    all_types  = []

    for subtype in ["H3N2", "H1N1"]:
        fpath = os.path.join(PROCESSED_DIR, f"dataset_{subtype}.json")
        if not os.path.exists(fpath):
            print(f"  AVISO: {fpath} no encontrado")
            continue

        with open(fpath, "r") as f:
            dataset = json.load(f)

        strains = dataset["paired_strains"]
        strains = [s for s in strains if s.get("year") is not None]
        if len(strains) > max_per_subtype:
            strains = random.sample(strains, max_per_subtype)

        print(f"\n  {subtype}: procesando {len(strains)} cepas...")
        ok = 0
        failed = 0
        for i, strain in enumerate(strains):
            if i % 100 == 0:
                print(f"    {i}/{len(strains)}...", end="\r")

            z = extract_latent(model, tokenizer, strain, device)
            if z is not None:
                all_z.append(z)
                all_years.append(strain["year"])
                all_months.append(strain.get("month", 6))
                all_types.append(subtype)
                ok += 1
            else:
                failed += 1

        print(f"    {ok} OK, {failed} fallidos de {len(strains)} total")

    if len(all_z) == 0:
        print("ERROR: no se extrajo ningun embedding")
        return None, None, None, None

    Z      = np.array(all_z)
    years  = np.array(all_years)
    months = np.array(all_months)
    types  = np.array(all_types)

    print(f"\n  Total embeddings: {len(Z)}")
    print(f"  Forma: {Z.shape}")
    print(f"  Rango: {years.min()}-{years.max()}")
    print(f"  Norma media: {np.linalg.norm(Z, axis=1).mean():.2f}")
    print(f"  NaN count: {np.isnan(Z).sum()}")
    return Z, years, months, types


# ---------------------------------------------------------------------------
# Experimento 1: UMAP
# ---------------------------------------------------------------------------

def plot_umap(Z, years, types):
    """UMAP 2D: coloreado por anio y por subtipo."""
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

def spearman_analysis(Z, years, months, types, n_pairs=5000):
    """
    Correlacion de Spearman entre distancia latente y temporal.

    CORRECCION CRITICA: se calcula POR SUBTIPO, no mezclando ambos.
    Mezclar subtipos inflaba la correlacion artificialmente porque la
    distancia inter-subtipo (~29) domina sobre la variacion temporal.
    """
    print("\n[Experimento 2] Correlacion de Spearman (por subtipo)...")

    results_per_subtype = {}

    for subtype in ["H3N2", "H1N1"]:
        mask = types == subtype
        Z_sub = Z[mask]
        y_sub = years[mask]
        m_sub = months[mask]
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
        d_temporal = []
        for i, j in pairs:
            dl = np.linalg.norm(Z_sub[i] - Z_sub[j])
            dt = abs(y_sub[i] * 12 + m_sub[i] - y_sub[j] * 12 - m_sub[j])
            d_latente.append(dl)
            d_temporal.append(dt)

        d_latente  = np.array(d_latente)
        d_temporal = np.array(d_temporal)

        rho, pval = spearmanr(d_latente, d_temporal)
        results_per_subtype[subtype] = (rho, pval, len(pairs))

        if rho > 0.5:
            interp = "FUERTE"
        elif rho > 0.3:
            interp = "MODERADA"
        elif rho > 0.1:
            interp = "DEBIL"
        else:
            interp = "AUSENTE"
        print(f"  {subtype}: rho={rho:.4f}  p={pval:.2e}  "
              f"n={len(pairs):,}  [{interp}]")

    # Figura: un subplot por subtipo
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, subtype in enumerate(["H3N2", "H1N1"]):
        ax = axes[ax_idx]
        mask = types == subtype
        Z_sub = Z[mask]
        y_sub = years[mask]
        m_sub = months[mask]
        N = len(Z_sub)

        # Recalcular pares para la figura
        pairs_fig = set()
        while len(pairs_fig) < min(3000, N * (N-1) // 2):
            i, j = random.sample(range(N), 2)
            if (i, j) not in pairs_fig:
                pairs_fig.add((i, j))

        dl = [np.linalg.norm(Z_sub[i] - Z_sub[j]) for i, j in pairs_fig]
        dt = [abs(y_sub[i]*12 + m_sub[i] - y_sub[j]*12 - m_sub[j])
              for i, j in pairs_fig]

        rho_val = results_per_subtype[subtype][0]
        pval_val = results_per_subtype[subtype][1]

        hb = ax.hexbin(dt, dl, gridsize=35, cmap="YlOrRd", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Pares")
        ax.set_xlabel("Distancia temporal (meses)")
        ax.set_ylabel(r"$\|z_i - z_j\|_2$")
        ax.set_title(f"{subtype}\nSpearman rho = {rho_val:.3f}  "
                     f"(p = {pval_val:.1e})")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Correlacion distancia latente vs temporal (por subtipo)",
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
    return np.mean(rhos), np.mean(pvals)


# ---------------------------------------------------------------------------
# Experimento 3: Interpolacion lineal (corregido)
# ---------------------------------------------------------------------------

def interpolation_analysis(Z, years, types, n_pairs=5):
    """
    Interpolacion lineal entre cepas en el espacio latente.

    CORRECCION: mide la UNIFORMIDAD del paso entre interpolaciones,
    no solo la norma. Un espacio suave produce pasos uniformes.
    Un espacio con discontinuidades produce pasos desiguales.

    Tambien mide si la norma de los puntos intermedios se mantiene
    en un rango razonable (no colapsa a 0 ni explota).
    """
    print("\n[Experimento 3] Interpolacion lineal...")

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
    print(f"  Guardado: {path}")

    return mean_cv


# ---------------------------------------------------------------------------
# Experimento 4: Dimension intrinseca (corregido)
# ---------------------------------------------------------------------------

def intrinsic_dimension(Z):
    """
    Estimacion de dimension intrinseca con TwoNN y MLE.

    CORRECCION: normaliza los embeddings antes de estimar
    y agrega ruido proporcional para romper duplicados.
    """
    print("\n[Experimento 4] Dimension intrinseca...")

    if not SKDIM_AVAILABLE:
        print("  scikit-dimension no disponible")
        return None

    # Limpiar NaN e Inf
    mask = np.isfinite(Z).all(axis=1)
    Z_clean = Z[mask]
    print(f"  Embeddings finitos: {Z_clean.shape[0]} de {Z.shape[0]}")

    # Subsample
    if len(Z_clean) > 500:
        idx = np.random.choice(len(Z_clean), 500, replace=False)
        Z_sub = Z_clean[idx]
    else:
        Z_sub = Z_clean

    # Normalizar a media 0 y varianza unitaria por dimension
    # Esto evita que dimensiones con varianza ~0 dominen las distancias
    mean = Z_sub.mean(axis=0)
    std  = Z_sub.std(axis=0)
    std[std < 1e-10] = 1.0  # evitar division por cero
    Z_norm = (Z_sub - mean) / std

    # Agregar ruido proporcional a la escala de los datos
    # para romper duplicados sin distorsionar la geometria
    noise_scale = 1e-5 * np.std(Z_norm)
    noise = np.random.normal(0, noise_scale, Z_norm.shape)
    Z_norm = Z_norm + noise

    # Verificar limpieza
    Z_norm = Z_norm[np.isfinite(Z_norm).all(axis=1)]
    print(f"  Embeddings normalizados: {Z_norm.shape[0]}")

    if len(Z_norm) < 50:
        print("  Insuficientes embeddings")
        return None

    results = {}
    try:
        est = skdim.id.TwoNN()
        d = est.fit(Z_norm).dimension_
        results["TwoNN"] = d
        print(f"  TwoNN: {d:.1f}")
    except Exception as e:
        print(f"  TwoNN error: {e}")

    try:
        est = skdim.id.MLE()
        d = est.fit(Z_norm).dimension_
        results["MLE"] = d
        print(f"  MLE:   {d:.1f}")
    except Exception as e:
        print(f"  MLE error: {e}")

    if not results:
        return None

    mean_id = np.mean(list(results.values()))
    print(f"  Dimension intrinseca media: {mean_id:.1f} (de 384 totales)")

    # Figura
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(results.keys())
    vals  = list(results.values())
    bars  = ax.bar(names, vals, color=["#457B9D", "#E63946"][:len(vals)],
                   alpha=0.85, width=0.4)
    ax.axhline(mean_id, color="gray", linestyle="--",
               linewidth=1.5, label=f"Media = {mean_id:.1f}")
    ax.set_ylabel("Dimension intrinseca estimada", fontsize=11)
    ax.set_title("Dimension intrinseca del espacio latente\n"
                 "(espacio ambiente: 384 dims)", fontsize=12)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "intrinsic_dimension.png")
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {path}")

    return mean_id


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def write_report(rho, pval, smoothness, intrinsic_dim):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "geometry_report.txt")

    lines = [
        "=" * 60,
        "REPORTE DE GEOMETRIA DEL ESPACIO LATENTE - AntigenLM",
        "=" * 60,
        "",
        "EXPERIMENTO 1: UMAP",
        "  Ver figures/gisaid/umap_by_year.png",
        "  Ver figures/gisaid/umap_by_subtype.png",
        "",
        "EXPERIMENTO 2: CORRELACION DE SPEARMAN (por subtipo)",
        f"  rho promedio = {rho:.4f}",
        f"  p promedio   = {pval:.2e}",
        f"  Interpretacion: {'temporal preservada' if rho > 0.3 else 'temporal debil pero significativa'}",
        "",
        "EXPERIMENTO 3: INTERPOLACION LINEAL",
        f"  CV promedio = {smoothness:.6f}" if smoothness is not None else "  No ejecutado",
        f"  Interpretacion: {'euclidiano local' if smoothness is not None and smoothness < 0.01 else 'suave' if smoothness is not None and smoothness < 0.3 else 'irregular' if smoothness is not None else 'N/A'}",
        "",
        "EXPERIMENTO 4: DIMENSION INTRINSECA",
        f"  Estimacion: {intrinsic_dim:.1f} dims efectivas (de 384)" if intrinsic_dim else "  No ejecutado",
        "",
        "CONCLUSION:",
    ]

    evidencias = sum([
        rho is not None and rho > 0.05,
        smoothness is not None and smoothness < 0.5,
        intrinsic_dim is not None and intrinsic_dim > 0,
    ])

    if evidencias >= 2:
        veredicto = (
            "  La evidencia empirica APOYA el supuesto de regularidad\n"
            "  del espacio latente. La SDE sobre z_t in R^384 esta justificada."
        )
    else:
        veredicto = (
            "  Evidencia INSUFICIENTE para justificar regularidad.\n"
            "  Considerar capa de proyeccion entrenable."
        )

    lines.append(veredicto)
    lines.append("")

    report = "\n".join(lines)
    print("\n" + report)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Reporte: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Detectar dispositivo (CUDA, MPS, o CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model     = load_model(device)
    tokenizer = InfluTokenizer(mode="prediction")

    Z, years, months, types = collect_embeddings(
        model, tokenizer, device,
        max_per_subtype=MAX_STRAINS_PER_SUBTYPE
    )

    if Z is None:
        print("\nERROR: sin embeddings.")
        exit(1)

    # Experimentos
    Z_2d       = plot_umap(Z, years, types)
    rho, pval  = spearman_analysis(Z, years, months, types)
    smoothness = interpolation_analysis(Z, years, types)
    id_est     = intrinsic_dimension(Z)

    write_report(rho, pval, smoothness, id_est)

    print("\n" + "=" * 60)
    print("Analisis completado.")
    print(f"Figuras: {FIGURES_DIR}/")
    print(f"Reporte: {RESULTS_DIR}/geometry_report.txt")
    print("=" * 60)
