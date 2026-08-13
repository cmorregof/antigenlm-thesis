"""
twonn_all_sequences.py
TwoNN sobre TODAS las secuencias HA del cache corregido.
Usa búsqueda de vecinos por chunks — nunca materializa la matriz N×N.
"""
import pickle, numpy as np
from scipy.spatial.distance import cdist

def twonn_from_r1r2(r1, r2, trim=0.05):
    valid = r1 > 1e-10
    mu = np.where(valid, r2 / np.maximum(r1, 1e-10), 1.0)
    mu = np.sort(mu)
    n = len(mu); t = int(trim * n)
    mu = mu[t:n-t]
    emp = np.arange(1, len(mu)+1) / len(mu)
    mask = (mu > 1.001) & (emp < 0.9999)
    if mask.sum() < 10: return np.nan
    lm = np.log(mu[mask]); lc = -np.log(1 - emp[mask])
    ok = np.isfinite(lm) & np.isfinite(lc)
    return np.polyfit(lm[ok], lc[ok], 1)[0] if ok.sum() > 5 else np.nan

def seqs_to_array(seqs, L=400):
    AA = "ACDEFGHIKLMNPQRSTVWY-X"
    v2i = {c: i for i, c in enumerate(AA)}
    arr = np.zeros((len(seqs), L), dtype=np.uint8)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:L]):
            arr[i, j] = v2i.get(c.upper(), 21)
    return arr

def find_nn_chunked(arr, chunk_size=500):
    """r1, r2 para todos los puntos sin materializar la matriz N×N."""
    n = arr.shape[0]
    r1 = np.full(n, np.inf, dtype=np.float32)
    r2 = np.full(n, np.inf, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        D = cdist(arr[start:end], arr, metric='hamming').astype(np.float32)
        for local_i in range(end - start):
            D[local_i, start + local_i] = np.inf  # excluir self
        top2 = np.partition(D, 2, axis=1)[:, :2]
        r1[start:end] = top2[:, 0]
        r2[start:end] = top2[:, 1]
        if start % 10000 == 0:
            pct = 100 * start / n
            print(f"  {start:>6}/{n}  ({pct:.0f}%)")
    return r1, r2

# ── Carga ──────────────────────────────────────────────────────────────────
print("Cargando cache...")
with open("results/embeddings_cache_full_FIXED.pkl", "rb") as f:
    cache = pickle.load(f)

records  = cache["records"]
types    = np.array(cache["types"])
embeddings = cache["embeddings"]

# ── Filtrar secuencias HA válidas ──────────────────────────────────────────
ha_seqs, subtypes, global_idx = [], [], []
for i, (rec, t) in enumerate(zip(records, types)):
    if t in ("H3N2", "H1N1"):
        ha = rec.get("ha_sequence", "")
        if len(ha) > 300:
            ha_seqs.append(ha)
            subtypes.append(t)
            global_idx.append(i)

subtypes   = np.array(subtypes)
global_idx = np.array(global_idx)
n_h3 = (subtypes == "H3N2").sum()
n_h1 = (subtypes == "H1N1").sum()
print(f"Secuencias válidas: {len(ha_seqs)}  (H3N2={n_h3}, H1N1={n_h1})")

# ── Convertir a uint8 ──────────────────────────────────────────────────────
print("Convirtiendo a array uint8 (L=400)...")
arr = seqs_to_array(ha_seqs, L=400)
print(f"Array shape: {arr.shape}  ({arr.nbytes/1e6:.0f} MB)")

# ── Vecinos más cercanos (chunked) ─────────────────────────────────────────
print("Buscando vecinos (chunked, sin matriz N×N)...")
r1, r2 = find_nn_chunked(arr, chunk_size=500)

# ── TwoNN ──────────────────────────────────────────────────────────────────
print("\n=== RESULTADOS TwoNN — Hamming HA ===")
print(f"{'':10}  {'trim=0.01':>10}  {'trim=0.05':>10}")
for label, mask in [
    ("Global",  np.ones(len(subtypes), bool)),
    ("H3N2",    subtypes == "H3N2"),
    ("H1N1",    subtypes == "H1N1"),
]:
    d01 = twonn_from_r1r2(r1[mask], r2[mask], trim=0.01)
    d05 = twonn_from_r1r2(r1[mask], r2[mask], trim=0.05)
    print(f"  {label:8}  {d01:>10.2f}  {d05:>10.2f}")

# ── También AntigenLM en el mismo conjunto ─────────────────────────────────
print("\n=== AntigenLM (mismas secuencias, chunked) ===")
Z = embeddings[global_idx]
r1_ag = np.full(len(Z), np.inf, dtype=np.float32)
r2_ag = np.full(len(Z), np.inf, dtype=np.float32)
CHUNK = 500
for start in range(0, len(Z), CHUNK):
    end = min(start + CHUNK, len(Z))
    D = cdist(Z[start:end], Z, metric='cosine').astype(np.float32)
    for li in range(end - start):
        D[li, start + li] = np.inf
    top2 = np.partition(D, 2, axis=1)[:, :2]
    r1_ag[start:end] = top2[:, 0]
    r2_ag[start:end] = top2[:, 1]
    if start % 20000 == 0:
        print(f"  AntigenLM {start}/{len(Z)}...")

print(f"\n{'':10}  {'trim=0.01':>10}  {'trim=0.05':>10}")
for label, mask in [
    ("Global",  np.ones(len(subtypes), bool)),
    ("H3N2",    subtypes == "H3N2"),
    ("H1N1",    subtypes == "H1N1"),
]:
    d01 = twonn_from_r1r2(r1_ag[mask], r2_ag[mask], trim=0.01)
    d05 = twonn_from_r1r2(r1_ag[mask], r2_ag[mask], trim=0.05)
    print(f"  {label:8}  {d01:>10.2f}  {d05:>10.2f}")

print("\nListo.")