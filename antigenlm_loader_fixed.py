"""
antigenlm_loader_fixed.py
=========================
Cargador CORREGIDO del checkpoint de prediccion de AntigenLM.

Bug corregido respecto a antigen_model.py:142:
    El modelo original hacia  self.lm_head.weight = self.backbone.wte.weight
    (weight tying). Pero el checkpoint tiene lm_head.weight y transformer.wte.weight
    como matrices DISTINTAS (corr global -0.002). Al atar y luego cargar con
    strict=False, load_state_dict escribia AMBAS claves sobre el mismo tensor y
    la tabla de embeddings de entrada quedaba reemplazada por la de salida.
    Resultado: generacion 'C C C C', ppl ~1000, y un cache de embeddings
    calculado sobre una tabla de entrada equivocada.

Este cargador:
  - NO ata los pesos.
  - Carga cada matriz por separado.
  - Verifica explicitamente que wte != lm_head tras cargar.
  - Verifica que solo faltan los 6 bias de c_attn (que el checkpoint no trae;
    GPT-2 los inicializa en 0, comportamiento correcto) y que no hay claves
    inesperadas salvo los buffers de mascara causal.

Uso:
    from antigenlm_loader_fixed import load_prediction_model, InfluTok
    backbone, lm_head = load_prediction_model("prediction_sequence/pytorch_model.bin")
"""
import torch, torch.nn as nn
from transformers import GPT2Config, GPT2Model

BACKBONE_CONFIG = dict(
    n_layer=6, n_embd=384, n_head=6, n_positions=13000, n_inner=None,
    attn_pdrop=0.1, embd_pdrop=0.1, resid_pdrop=0.1,
    activation_function="gelu_new", layer_norm_epsilon=1e-5,
    initializer_range=0.02, scale_attn_weights=True, use_cache=False,
)

_SUBTYPES = ["<H1N1>","<H3N2>","<H5N6>","<H7N9>","<H9N2>","<H5N1>","<H10N3>",
             "<H1N2>","<H3N8>","<H6N1>","<H2N2>","<H10N8>","<H10N5>","<H5N8>","<H7N4>"]


class InfluTok:
    """Tokenizer character-level identico al reconstruido, modo prediction."""
    def __init__(self):
        self.vocab = {"A":0,"C":1,"G":2,"T":3,"N":4,"<pad>":5,"<sep>":6,
                      "<eos>":7,"<HA>":8,"<NA>":9}
        for i,t in enumerate(_SUBTYPES):
            self.vocab[t] = 10+i
        self.id2tok = {v:k for k,v in self.vocab.items()}
        self.pad_id, self.eos_id = 5, 7

    def encode(self, ha, na, subtype_token, max_len=None):
        ids = [self.vocab[subtype_token], self.vocab["<HA>"]]
        ids += [self.vocab.get(c,4) for c in ha.upper()]
        ids += [self.vocab["<sep>"], self.vocab["<NA>"]]
        ids += [self.vocab.get(c,4) for c in na.upper()]
        ids += [self.vocab["<eos>"]]
        return ids[:max_len] if max_len else ids


def load_prediction_model(bin_path, device="cpu", dtype=torch.float32, verbose=True):
    """Devuelve (backbone GPT2Model, lm_head nn.Linear) con pesos correctamente cargados."""
    ckpt = torch.load(bin_path, map_location="cpu")

    # separar backbone vs cabeza; remapear 'transformer.' -> '' para GPT2Model
    bb_sd, head_w = {}, None
    for k, v in ckpt.items():
        if k.endswith(("attn.bias", "attn.masked_bias")):
            continue                      # buffers de mascara causal, no persistentes
        if k == "lm_head.weight":
            head_w = v
        elif k.startswith("transformer."):
            bb_sd[k[len("transformer."):]] = v

    cfg = GPT2Config(vocab_size=25, **BACKBONE_CONFIG)
    backbone = GPT2Model(cfg)
    missing, unexpected = backbone.load_state_dict(bb_sd, strict=False)

    # los unicos missing legitimos son los 6 bias de c_attn (no estan en el ckpt)
    bad_missing = [m for m in missing if not m.endswith("attn.c_attn.bias")]
    assert not bad_missing, f"claves faltantes inesperadas: {bad_missing}"
    assert not unexpected, f"claves inesperadas: {unexpected}"

    lm_head = nn.Linear(384, 25, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(head_w)

    # LA verificacion que nunca se hizo:
    assert not torch.equal(backbone.wte.weight, lm_head.weight), \
        "wte == lm_head: el weight tying sigue activo, el bug NO esta corregido"

    backbone = backbone.to(device=device, dtype=dtype).eval()
    lm_head  = lm_head.to(device=device, dtype=dtype).eval()

    if verbose:
        import numpy as np
        w = backbone.wte.weight.detach().float().cpu().numpy()
        l = lm_head.weight.detach().float().cpu().numpy()
        corr = float(np.corrcoef(w.ravel(), l.ravel())[0,1])
        print(f"[loader] carga OK | missing(bias c_attn)={len(missing)} unexpected={len(unexpected)}")
        print(f"[loader] wte != lm_head: True | corr(wte,lm_head)={corr:+.4f} (debe ser ~0)")
        print(f"[loader] ||wte||={np.linalg.norm(w):.3f}  ||lm_head||={np.linalg.norm(l):.3f}")
    return backbone, lm_head


@torch.no_grad()
def mean_pool_embedding(backbone, ids, device="cpu"):
    """Embedding z in R^384 = media de hidden states (attention mask completa)."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    h = backbone(input_ids=x).last_hidden_state[0]
    return h.mean(0).float().cpu().numpy()
