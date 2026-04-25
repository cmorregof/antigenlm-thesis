"""
diagnostic_bug.py — Test quirúrgico para el bug C-C-C-C
NO modifica archivos. NO genera secuencias largas. SEGURO.
"""
import torch
import numpy as np

# ================================================================
# PASO 1: Inspección del checkpoint (ataca H3)
# ================================================================
print("=" * 60)
print("PASO 1: Claves del checkpoint")
print("=" * 60)

ckpt = torch.load("prediction_sequence/pytorch_model.bin", map_location="cpu")

print(f"Total claves: {len(ckpt)}")
print(f"\nPrimeras 10 claves:")
for i, k in enumerate(sorted(ckpt.keys())):
    if i < 10:
        print(f"  {k:50s} shape={list(ckpt[k].shape)}")
print("...")
print(f"Últimas 5 claves:")
for k in sorted(ckpt.keys())[-5:]:
    print(f"  {k:50s} shape={list(ckpt[k].shape)}")

# ¿Existe lm_head.weight en el checkpoint?
has_lm_head = any("lm_head" in k for k in ckpt.keys())
print(f"\n¿Contiene 'lm_head'? → {has_lm_head}")

# ¿Las claves empiezan con 'transformer.'?
transformer_keys = [k for k in ckpt.keys() if k.startswith("transformer.")]
other_keys = [k for k in ckpt.keys() if not k.startswith("transformer.")]
print(f"Claves con 'transformer.': {len(transformer_keys)}")
print(f"Claves sin 'transformer.': {len(other_keys)}")
if other_keys:
    print(f"  Claves sin prefijo: {other_keys}")

# ================================================================
# PASO 2: Carga de pesos con strict=True (ataca H3)
# ================================================================
print("\n" + "=" * 60)
print("PASO 2: Carga de pesos — strict mode")
print("=" * 60)

from antigen_model import GPTForFluMultiTask

model = GPTForFluMultiTask(task="prediction")
remapped = {k.replace("transformer.", "backbone."): v for k, v in ckpt.items()}

missing, unexpected = model.load_state_dict(remapped, strict=False)
print(f"Missing keys ({len(missing)}):    {missing}")
print(f"Unexpected keys ({len(unexpected)}): {unexpected}")

if len(missing) == 0 and len(unexpected) == 0:
    print("✅ Carga perfecta — todos los pesos coinciden")
else:
    print("⚠️ HAY DISCREPANCIAS — revisar")

# ================================================================
# PASO 3: Test FP32 vs FP16 con input mínimo (ataca H2)
# ================================================================
print("\n" + "=" * 60)
print("PASO 3: Comparación FP32 vs FP16")
print("=" * 60)

model.eval()

# Input mínimo: <H3N2> <HA> A T C G (6 tokens)
test_input = torch.tensor([[11, 8, 0, 3, 1, 2]], dtype=torch.long)

with torch.no_grad():
    # FP32
    out_fp32 = model(input_ids=test_input)
    logits_fp32 = out_fp32["logits"][0, -1, :]  # logits del último token
    top5_fp32 = torch.topk(logits_fp32, 5)

    print("FP32 — Top-5 tokens predichos después de '<H3N2> <HA> A T C G':")
    for val, idx in zip(top5_fp32.values, top5_fp32.indices):
        print(f"  token_id={idx.item():2d}  logit={val.item():+.4f}")
    print(f"  argmax → id={logits_fp32.argmax().item()}")
    print(f"  Logit rango: [{logits_fp32.min().item():.4f}, {logits_fp32.max().item():.4f}]")
    print(f"  Logit std:   {logits_fp32.std().item():.4f}")

    # FP16
    model_fp16 = model.half()
    test_fp16 = test_input  # long no necesita .half()
    out_fp16 = model_fp16(input_ids=test_fp16)
    logits_fp16 = out_fp16["logits"][0, -1, :]

    top5_fp16 = torch.topk(logits_fp16.float(), 5)
    print(f"\nFP16 — Top-5 tokens predichos:")
    for val, idx in zip(top5_fp16.values, top5_fp16.indices):
        print(f"  token_id={idx.item():2d}  logit={val.item():+.4f}")
    print(f"  argmax → id={logits_fp16.float().argmax().item()}")
    print(f"  Logit rango: [{logits_fp16.float().min().item():.4f}, {logits_fp16.float().max().item():.4f}]")
    print(f"  Logit std:   {logits_fp16.float().std().item():.4f}")

    # Divergencia
    diff = (logits_fp32 - logits_fp16.float()).abs()
    print(f"\n  Divergencia FP32-FP16:")
    print(f"    Max: {diff.max().item():.6f}")
    print(f"    Mean: {diff.mean().item():.6f}")
    print(f"    ¿Mismo argmax? {logits_fp32.argmax().item() == logits_fp16.float().argmax().item()}")

# ================================================================
# PASO 4: Test de truncado de contexto (ataca H4)
# ================================================================
print("\n" + "=" * 60)
print("PASO 4: Efecto del truncado de contexto")
print("=" * 60)

from influ_tokenizer import InfluTokenizer
tok = InfluTokenizer(mode="prediction")

# Simular una cepa realista (HA ~1700nt, NA ~1400nt)
fake_ha = "ATCGATCG" * 212   # 1696 nt
fake_na = "GCTAGCTA" * 175   # 1400 nt
ids = tok.encode_strain(fake_ha, fake_na, subtype="<H3N2>")

print(f"Longitud total de una cepa: {len(ids)} tokens")
print(f"Primeros 5 tokens: {ids[:5]} → {[tok.id_to_token_str(i) for i in ids[:5]]}")

# Simular truncado a 2048
truncated = ids[-2048:]
print(f"\nDespués de truncar a 2048 (context_ids[-2048:]):")
print(f"  Primeros 5 tokens: {truncated[:5]} → {[tok.id_to_token_str(i) for i in truncated[:5]]}")
print(f"  ¿Token de subtipo presente? {'<H3N2>' in [tok.id_to_token_str(i) for i in truncated]}")
print(f"  ¿Token <HA> presente? {'<HA>' in [tok.id_to_token_str(i) for i in truncated]}")
print(f"  Tokens perdidos: {len(ids) - 2048}")

print("\n" + "=" * 60)
print("DIAGNÓSTICO COMPLETO")
print("=" * 60)
