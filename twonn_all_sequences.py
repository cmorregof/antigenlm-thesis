import pickle, numpy as np
from scipy.spatial.distance import cdist

def twonn_from_r1r2(r1, r2, trim=0.05):
    valid = r1 > 1e-10
    mu = np.where(valid, r2 / np.maximum(r1, 1e-10), 1.0)
    mu = np.sort(mu); n = len(mu); t = int(trim * n)
    mu = mu[t:n-t]; emp = np.arange(1, len(mu)+1) / len(mu)
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
    n = arr.shape[0]
    r1 = np.full(n, np.inf, dtype=np.float32)
    r2 = np.full(n, np.inf, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        D = cdist(arr[start:end], arr, metric='hamming').astype(np.float32)
        for li in range(end - start):
            D[li, start + li] = np.inf
        top2 = np.partition(D, 2, axis=1)[:, :2]
        r1[start:end] = top2[:, 0]
        r2[start:end] = top2[:, 1]
        if start % 10000 == 0:
            print(f"  {start:>6}/{n} ({100*start/n:.0f}%)", flush=True)
    return r1, r2

print("Cargando cache...")
with open("results/embeddings_cache_full_FIXED.pkl", "rb") as f:
    cache = pickle.load(f)
records = cache["records"]; types = np.array(cache["types"])

ha_seqs, subtypes = [], []
for i, (rec, t) in enumerate(zip(records, types)):
    if t in ("H3N2", "H1N1"):
        ha = rec.get("ha_sequence", "")
        if len(ha) > 300:
            ha_seqs.append(ha); subtypes.append(t)
subtypes = np.array(subtypes)
print(f"Secuencias: {len(ha_seqs)}  H3N2={( subtypes=='H3N2').sum()}  H1N1={(subtypes=='H1N1').sum()}")

print("Convirtiendo..."); arr = seqs_to_array(ha_seqs, L=400)
print(f"Array: {arr.shape}  {arr.nbytes/1e6:.0f} MB")
print("Buscando vecinos (chunked)...")
r1, r2 = find_nn_chunked(arr)

print("\n=== TwoNN Hamming HA ===")
print(f"{'':8}  trim=0.01  trim=0.05")
for label, mask in [("Global", np.ones(len(subtypes),bool)),("H3N2",subtypes=="H3N2"),("H1N1",subtypes=="H1N1")]:
    d01 = twonn_from_r1r2(r1[mask], r2[mask], 0.01)
    d05 = twonn_from_r1r2(r1[mask], r2[mask], 0.05)
    print(f"  {label:6}    {d01:.2f}       {d05:.2f}")
