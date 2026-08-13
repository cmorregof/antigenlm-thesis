"""
regenerate_cache_fixed.py
=========================
Regenera el cache completo de embeddings de AntigenLM con la carga CORRECTA
(sin el bug de weight tying). Drop-in: mismo formato que
results/embeddings_cache_full_all_available.pkl.

- Escribe a un archivo NUEVO (no pisa el cache con bug).
- Checkpointing cada CHUNK cepas: si se interrumpe, reanuda donde iba.
- Orden identico al cache original: H3N2 primero (65631), luego H1N1 (46125).
- Usa MPS si esta disponible (Mac Apple Silicon), si no CPU.

Ejecutar desde la raiz del repo:
    python regenerate_cache_fixed.py
Reanudar tras interrupcion: volver a ejecutar el mismo comando.
"""
import os, json, time, pickle
import numpy as np
import torch
from antigenlm_loader_fixed import load_prediction_model, InfluTok

CKPT   = "prediction_sequence/pytorch_model.bin"
DATA   = "data/processed_gisaid"
OUT    = "results/embeddings_cache_full_FIXED.pkl"
PROG   = "results/embeddings_cache_full_FIXED.progress.npz"
MAX_SEQ = 4000
CHUNK   = 500          # guarda progreso cada 500 cepas
ORDER   = ["H3N2", "H1N1"]   # mismo orden que el cache original


def stream_records(path):
    """Lee dataset_{sub}.json sin cargar los 200MB de golpe (parser de llaves)."""
    buf, depth, n = [], 0, 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if depth == 0 and s == "{" and line.startswith("    "):
                depth, buf = 1, [line]; continue
            if depth:
                buf.append(line)
                depth += line.count("{") - line.count("}")
                if depth == 0:
                    yield json.loads("".join(buf).rstrip().rstrip(","))
                    buf = []


def device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():          return torch.device("cuda")
    return torch.device("cpu")


def main():
    dev = device()
    print(f"dispositivo: {dev}")
    backbone, lm_head = load_prediction_model(CKPT, device=dev, dtype=torch.float32)
    tok = InfluTok()

    # construir la lista maestra de cepas en el orden canonico
    strains = []
    for sub in ORDER:
        p = os.path.join(DATA, f"dataset_{sub}.json")
        cnt = 0
        for r in stream_records(p):
            if r.get("year") is None:
                continue
            strains.append(r); cnt += 1
        print(f"  {sub}: {cnt} cepas")
    N = len(strains)
    print(f"total: {N} cepas")

    # reanudar si hay progreso
    Z = np.zeros((N, 384), dtype=np.float32)
    start = 0
    if os.path.exists(PROG):
        d = np.load(PROG)
        done = int(d["done"]); Z[:done] = d["Z"][:done]
        start = done
        print(f"reanudando desde la cepa {start}")

    t0 = time.time()
    with torch.no_grad():
        for i in range(start, N):
            s = strains[i]
            ids = tok.encode(s["ha_sequence"], s["na_sequence"],
                             s["subtype_token"], max_len=MAX_SEQ)
            x = torch.tensor([ids], dtype=torch.long, device=dev)
            h = backbone(input_ids=x).last_hidden_state[0]
            Z[i] = h.mean(0).float().cpu().numpy()
            if (i + 1) % 100 == 0:
                rate = (i + 1 - start) / (time.time() - t0 + 1e-9)
                eta = (N - i - 1) / rate / 60
                print(f"  {i+1}/{N}  {rate:.1f} cepas/s  ETA {eta:.0f} min", end="\r")
            if (i + 1) % CHUNK == 0:
                np.savez(PROG, Z=Z, done=i + 1)
    np.savez(PROG, Z=Z, done=N)
    print(f"\nembeddings listos en {(time.time()-t0)/60:.1f} min")

    out = {
        "embeddings": Z,
        "years":  np.array([s["year"] for s in strains], dtype=np.int64),
        "months": np.array([s.get("month", 6) for s in strains], dtype=np.int64),
        "types":  np.array([s["subtype"] for s in strains], dtype="<U4"),
        "records": strains,
        "metadata": {
            "sampling_strategy": "all", "seed": 42, "max_per_subtype": -1,
            "max_seq_length": MAX_SEQ, "embedding_batch_size": 1,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint": {"path": CKPT, "sha256":
                "6b942a0e2d6af0528a7307ff5754438ad55fdb97390297e9c0f11ffc9803dbff"},
            "fix": "weight_tying_removed; wte and lm_head loaded independently",
        },
    }
    with open(OUT, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"guardado: {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")
    if os.path.exists(PROG):
        os.remove(PROG)


if __name__ == "__main__":
    main()
