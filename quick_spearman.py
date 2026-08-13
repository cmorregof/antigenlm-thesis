import pickle, random
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine

CACHE = "results/embeddings_cache_full_FIXED.pkl"
N_PAIRS = 50000
SEED = 42

print("Cargando cache...")
with open(CACHE, 'rb') as f:
    cache = pickle.load(f)

embeddings = cache['embeddings']          # (111756, 384)
types      = np.array(cache['types'])     # subtipos
records    = cache['records']             # lista de dicts con secuencias

# Inspeccionar un record
print("Claves de records[0]:", list(records[0].keys()) if isinstance(records[0], dict) else type(records[0]))

# Separar índices por subtipo
idx_h3n2 = np.where(np.array([str(t).upper() for t in types]) == 'H3N2')[0]
idx_h1n1 = np.where(np.array([str(t).upper() for t in types]) == 'H1N1')[0]
print(f"H3N2: {len(idx_h3n2)} | H1N1: {len(idx_h1n1)}")

def hamming_norm(s1, s2):
    if not s1 or not s2: return np.nan
    n = min(len(s1), len(s2))
    return sum(c1 != c2 for c1, c2 in zip(s1[:n], s2[:n])) / n

def get_seqs(rec):
    if isinstance(rec, dict):
        ha = rec.get('ha_sequence', '')
        na = rec.get('na_sequence', '')
        return ha, na
    return '', ''

def spearman_for(idx_pool, label):
    rng = random.Random(SEED)
    pool = list(idx_pool)
    pairs = [rng.sample(pool, 2) for _ in range(min(N_PAIRS, len(pool)*(len(pool)-1)//2))]
    lat, ham_ha, ham_hana = [], [], []
    for k, (a, b) in enumerate(pairs):
        ea, eb = embeddings[a], embeddings[b]
        lat.append(cosine(ea, eb))
        ha_a, na_a = get_seqs(records[a])
        ha_b, na_b = get_seqs(records[b])
        ham_ha.append(hamming_norm(ha_a, ha_b))
        ham_hana.append(hamming_norm(ha_a + na_a, ha_b + na_b))
        if (k+1) % 10000 == 0:
            print(f"  {label}: {k+1}/{len(pairs)}")
    # filtrar NaN
    mask = ~np.isnan(ham_ha)
    rho_ha,   _ = spearmanr(np.array(lat)[mask],   np.array(ham_ha)[mask])
    mask2 = ~np.isnan(ham_hana)
    rho_hana, _ = spearmanr(np.array(lat)[mask2], np.array(ham_hana)[mask2])
    return rho_ha, rho_hana

for label, idx in [('H3N2', idx_h3n2), ('H1N1', idx_h1n1)]:
    print(f"\n{label}: calculando Spearman...")
    rho_ha, rho_hana = spearman_for(idx, label)
    print(f"  rho(latente, Hamming HA)     = {rho_ha:.4f}")
    print(f"  rho(latente, Hamming HA+NA)  = {rho_hana:.4f}")