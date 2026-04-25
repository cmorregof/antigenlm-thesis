"""
antigen_model.py
================
Reimplementación de GPTForFluMultiTask para AntigenLM.

Esta clase fue reconstruida a partir de:
  - prediction_sequence/config.json  (arquitectura del backbone)
  - subtype_classifier/config.json   (misma arquitectura, distinto vocab)
  - El paper: Pei et al., ICLR 2026. arXiv:2602.09067

Idea central del paper:
    Un único backbone GPT-style (6 capas, 384 dims, 6 heads, 13k posiciones)
    se comparte entre dos tareas:
        1. Predicción de secuencia antigénica  → cabeza LM (generativa)
        2. Clasificación de subtipo            → cabeza de clasificación

    El backbone se preentrenan en 54,512 genomas completos de influenza A
    (~600M nucleótidos) usando modelado de lenguaje autorregresivo estándar.
    Luego se afina para cada tarea por separado.

"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Constantes derivadas de config.json
# ---------------------------------------------------------------------------

# Hiperparámetros del backbone — idénticos en ambos config.json
BACKBONE_CONFIG = dict(
    n_layer       = 6,       # número de capas Transformer
    n_embd        = 384,     # dimensión de embeddings y estados ocultos
    n_head        = 6,       # cabezas de atención (head_dim = 384/6 = 64)
    n_positions   = 13000,   # contexto máximo en tokens (~genoma completo)
    n_inner       = None,    # FFN inner dim; None → usa 4 * n_embd = 1536
    attn_pdrop    = 0.1,     # dropout en atención
    embd_pdrop    = 0.1,     # dropout en embeddings
    resid_pdrop   = 0.1,     # dropout en residuales
    activation_function = "gelu_new",
    layer_norm_epsilon  = 1e-5,
    initializer_range   = 0.02,
    scale_attn_weights  = True,
    use_cache           = True,
)

# Número de subtipos para clasificación (derivado de added_tokens.json)
NUM_SUBTYPES = 15

# Vocabularios (según config.json de cada módulo)
VOCAB_SIZE_PREDICTION    = 25   # prediction_sequence
VOCAB_SIZE_CLASSIFICATION = 10  # subtype_classifier


# ---------------------------------------------------------------------------
# Modelo principal
# ---------------------------------------------------------------------------

class GPTForFluMultiTask(nn.Module):
    """
    Modelo GPT-style para influenza con dos cabezas de tarea.

    Arquitectura (paper Figura 1B):

        [Entrada tokenizada]
               ↓
        [Embedding de tokens + posiciones]      ← compartido
               ↓
        [6x Transformer Block]                  ← compartido (backbone)
        [  Self-Attention (6 heads, dim=384)  ]
        [  LayerNorm + Residual              ]
        [  FFN (384 → 1536 → 384)           ]
        [  LayerNorm + Residual              ]
               ↓
        ┌──────┴──────┐
        ↓             ↓
    [LM Head]    [CLS Head]
    (generativa) (clasificación)
    vocab×384    384→NUM_SUBTYPES

    El backbone es GPT2 de Hugging Face con vocab_size ajustado.
    Las cabezas son módulos adicionales entrenados durante el finetuning.

    Args:
        task:     'prediction' o 'classification'
        vocab_size: tamaño del vocabulario (25 para predicción, 10 para clasificación)
        num_subtypes: número de clases para clasificación (15 según added_tokens.json)
    """

    def __init__(
        self,
        task: str = "prediction",
        vocab_size: Optional[int] = None,
        num_subtypes: int = NUM_SUBTYPES,
    ):
        super().__init__()

        assert task in ("prediction", "classification"), \
            "task debe ser 'prediction' o 'classification'"

        self.task = task

        # Resolver vocab_size según la tarea si no se especifica
        if vocab_size is None:
            vocab_size = (
                VOCAB_SIZE_PREDICTION
                if task == "prediction"
                else VOCAB_SIZE_CLASSIFICATION
            )
        self.vocab_size = vocab_size

        # -------------------------------------------------------------------
        # Backbone: GPT2 con hiperparámetros del paper
        # -------------------------------------------------------------------
        # GPT2Config extiende la arquitectura estándar de GPT-2 pero con
        # los parámetros específicos de AntigenLM (mucho más pequeño que GPT-2:
        # 117M params original vs ~6M params aquí)
        config = GPT2Config(
            vocab_size   = vocab_size,
            **BACKBONE_CONFIG,
        )
        self.backbone = GPT2Model(config)

        # -------------------------------------------------------------------
        # Cabeza de tarea 1: Predicción de secuencia (LM Head)
        # -------------------------------------------------------------------
        # Proyección lineal sin bias: hidden_state → logits sobre vocabulario
        # Igual que GPT-2 estándar. Se usa para next-token prediction.
        # Durante finetuning: predice token t+1 dada la historia hasta t.
        if task == "prediction":
            self.lm_head = nn.Linear(
                BACKBONE_CONFIG["n_embd"],  # 384
                vocab_size,                 # 25
                bias=False,
            )
            # Weight tying: los pesos del LM head se comparten con el
            # embedding de entrada. Práctica estándar en modelos de lenguaje.
            # Reduce parámetros y estabiliza el entrenamiento.
            self.lm_head.weight = self.backbone.wte.weight

        # -------------------------------------------------------------------
        # Cabeza de tarea 2: Clasificación de subtipo (CLS Head)
        # -------------------------------------------------------------------
        # summary_type: "cls_index" (según config.json)
        # Usa el último token no-padding para extraer representación de la
        # secuencia completa, luego proyecta a num_subtypes clases.
        elif task == "classification":
            self.cls_head = nn.Sequential(
                nn.Linear(BACKBONE_CONFIG["n_embd"], BACKBONE_CONFIG["n_embd"]),
                nn.Tanh(),          # activación estándar para pooling en GPT-2
                nn.Dropout(0.1),    # summary_first_dropout del config
                nn.Linear(BACKBONE_CONFIG["n_embd"], num_subtypes),
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Paso forward del modelo.

        Args:
            input_ids:      [batch, seq_len] — IDs de tokens de entrada
            attention_mask: [batch, seq_len] — 1 para tokens reales, 0 para padding
            labels:         Para predicción: [batch, seq_len] shifted input_ids
                            Para clasificación: [batch] con índice de subtipo

        Returns:
            Dict con claves:
                'loss'    → escalar si labels fue provisto, None si no
                'logits'  → [batch, seq_len, vocab_size] (predicción)
                            o [batch, num_subtypes] (clasificación)
                'hidden_states' → últimos estados ocultos del backbone
        """
        # Paso por el backbone compartido
        # hidden_states: [batch, seq_len, n_embd=384]
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        # ---------------------------------------------------------------
        # Rama 1: Predicción de secuencia
        # ---------------------------------------------------------------
        if self.task == "prediction":
            # logits: [batch, seq_len, vocab_size=25]
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                # Desplazamiento estándar para LM causal:
                # predice token[i+1] dado token[i]
                # shift_logits: [batch, seq_len-1, vocab_size]
                # shift_labels: [batch, seq_len-1]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss_fn = nn.CrossEntropyLoss(ignore_index=5)  # ignora <pad>=5
                loss = loss_fn(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                )

            return {
                "loss":          loss,
                "logits":        logits,
                "hidden_states": hidden_states,
            }

        # ---------------------------------------------------------------
        # Rama 2: Clasificación de subtipo
        # ---------------------------------------------------------------
        elif self.task == "classification":
            # Extraer representación del último token real (cls_index)
            # Para GPT-2 estilo, el último token no-padding actúa como [CLS]
            if attention_mask is not None:
                # Índice del último token real para cada elemento del batch
                # [batch]
                last_token_idx = attention_mask.sum(dim=1) - 1
                batch_size = hidden_states.size(0)
                # Seleccionar estado oculto del último token real
                # pooled: [batch, n_embd=384]
                pooled = hidden_states[
                    torch.arange(batch_size, device=hidden_states.device),
                    last_token_idx,
                ]
            else:
                # Si no hay máscara, tomar el último token
                pooled = hidden_states[:, -1, :]

            # logits: [batch, num_subtypes=15]
            logits = self.cls_head(pooled)

            loss = None
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)

            return {
                "loss":          loss,
                "logits":        logits,
                "hidden_states": hidden_states,
            }

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict:
        """Cuenta parámetros totales y entrenables."""
        total      = sum(p.numel() for p in self.parameters())
        trainable  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_p = sum(p.numel() for p in self.backbone.parameters())
        return {
            "total":      total,
            "trainable":  trainable,
            "backbone":   backbone_p,
            "task_head":  total - backbone_p,
        }

    def freeze_backbone(self):
        """
        Congela los pesos del backbone.
        Útil para finetuning donde solo se entrena la cabeza de tarea,
        especialmente cuando se parte de pesos preentrenados.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Descongela el backbone para finetuning completo."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"GPTForFluMultiTask(\n"
            f"  task       = '{self.task}'\n"
            f"  vocab_size = {self.vocab_size}\n"
            f"  n_layer    = {BACKBONE_CONFIG['n_layer']}\n"
            f"  n_embd     = {BACKBONE_CONFIG['n_embd']}\n"
            f"  n_head     = {BACKBONE_CONFIG['n_head']}\n"
            f"  n_positions= {BACKBONE_CONFIG['n_positions']}\n"
            f"  params     = {params['total']:,} total, "
            f"{params['trainable']:,} trainable\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Bloque de prueba — ejecutar con: python antigen_model.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: GPTForFluMultiTask")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    # -------------------------------------------------------------------
    # Test 1: Modelo de predicción de secuencia
    # -------------------------------------------------------------------
    print("\n[1] Modelo de predicción de secuencia")
    model_pred = GPTForFluMultiTask(task="prediction").to(device)
    print(model_pred)

    params = model_pred.count_parameters()
    print(f"\n  Parámetros del backbone:  {params['backbone']:>10,}")
    print(f"  Parámetros de la cabeza:  {params['task_head']:>10,}")
    print(f"  Total entrenables:        {params['trainable']:>10,}")

    # Forward pass con datos ficticios
    batch_size = 2
    seq_len    = 50  # secuencia corta para el test
    input_ids  = torch.randint(0, 25, (batch_size, seq_len)).to(device)
    attn_mask  = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)

    with torch.no_grad():
        out = model_pred(input_ids=input_ids, attention_mask=attn_mask)

    print(f"\n  input_ids shape:  {input_ids.shape}")
    print(f"  logits shape:     {out['logits'].shape}")
    expected_logits = (batch_size, seq_len, VOCAB_SIZE_PREDICTION)
    assert out["logits"].shape == expected_logits, \
        f"Error: se esperaba {expected_logits}, se obtuvo {out['logits'].shape}"
    print(f"  ✅ Forma de logits correcta: {expected_logits}")

    # Forward con labels (calcula loss)
    labels = input_ids.clone()
    out_with_loss = model_pred(
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=labels,
    )
    print(f"  Loss (predicción): {out_with_loss['loss'].item():.4f}")
    assert out_with_loss["loss"] is not None
    print(f"  ✅ Loss calculada correctamente")

    # -------------------------------------------------------------------
    # Test 2: Modelo de clasificación de subtipo
    # -------------------------------------------------------------------
    print("\n[2] Modelo de clasificación de subtipo")
    model_cls = GPTForFluMultiTask(task="classification").to(device)
    print(model_cls)

    input_ids_cls = torch.randint(0, 10, (batch_size, seq_len)).to(device)
    attn_mask_cls = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
    subtype_labels = torch.randint(0, NUM_SUBTYPES, (batch_size,)).to(device)

    with torch.no_grad():
        out_cls = model_cls(
            input_ids=input_ids_cls,
            attention_mask=attn_mask_cls,
            labels=subtype_labels,
        )

    print(f"\n  logits shape:    {out_cls['logits'].shape}")
    expected_cls = (batch_size, NUM_SUBTYPES)
    assert out_cls["logits"].shape == expected_cls, \
        f"Error: se esperaba {expected_cls}, se obtuvo {out_cls['logits'].shape}"
    print(f"  ✅ Forma de logits correcta: {expected_cls}")
    print(f"  Loss (clasificación): {out_cls['loss'].item():.4f}")
    print(f"  ✅ Loss calculada correctamente")

    # -------------------------------------------------------------------
    # Test 3: Verificar weight tying en predicción
    # -------------------------------------------------------------------
    print("\n[3] Weight tying: LM head comparte pesos con embedding")
    assert model_pred.lm_head.weight.data_ptr() == \
           model_pred.backbone.wte.weight.data_ptr(), \
           "Error: weight tying no está activo"
    print("  ✅ Weight tying confirmado")

    # -------------------------------------------------------------------
    # Test 4: Freeze/unfreeze backbone
    # -------------------------------------------------------------------
    print("\n[4] Freeze y unfreeze del backbone")
    model_pred.freeze_backbone()
    frozen = sum(1 for p in model_pred.backbone.parameters() if not p.requires_grad)
    print(f"  Parámetros congelados en backbone: {frozen}")
    assert frozen > 0, "Error: freeze_backbone no funcionó"

    model_pred.unfreeze_backbone()
    unfrozen = sum(1 for p in model_pred.backbone.parameters() if not p.requires_grad)
    assert unfrozen == 0, "Error: unfreeze_backbone no funcionó"
    print("  ✅ Freeze/unfreeze funcionan correctamente")

    print("\n" + "=" * 60)
    print("✅ Todos los tests pasaron. GPTForFluMultiTask funciona correctamente.")
    print("=" * 60)
