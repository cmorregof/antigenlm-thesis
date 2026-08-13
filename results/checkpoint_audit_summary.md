# Auditoria local de checkpoints AntigenLM

- Fecha local de ejecucion: `2026-04-27 22:33:28`
- Script: `audit_antigenlm_checkpoints.py`
- Torch: `2.11.0`

Esta auditoria no entrena modelos, no evalua datasets grandes y no genera secuencias largas.

## Comparacion local de pesos

| comparacion | valor |
| --- | --- |
| mismo tamano | False |
| mismo sha256 | False |
| prediction sha256 | 6b942a0e2d6af0528a7307ff5754438ad55fdb97390297e9c0f11ffc9803dbff |
| subtype sha256 | a43fc134c933210e49752f9510f875860b745f022ebf0d0ab7d1e90b0d3783fe |

Conclusion local: Los pesos locales NO son identicos; ademas las heads detectadas difieren.

## Tabla de heads detectadas

| checkpoint | lm_head | classification_head | cls_head | transformer | backbone |
| --- | --- | --- | --- | --- | --- |
| prediction_sequence | 1 | 0 | 0 | 88 | 0 |
| subtype_classifier | 1 | 2 | 0 | 88 | 0 |

## Inventario: `prediction_sequence/`

| archivo | tamano | bytes | sha256 |
| --- | --- | --- | --- |
| added_tokens.json | 246 B | 246 |  |
| config.json | 750 B | 750 |  |
| pytorch_model.bin | 1.0 GB | 1076663612 | 6b942a0e2d6af052... |
| special_tokens_map.json | 156 B | 156 |  |
| tokenizer_config.json | 290 B | 290 |  |
| vocab.json | 120 B | 120 |  |

### Configuracion

| campo | valor |
| --- | --- |
| vocab_size | 25 |
| n_layer | 6 |
| n_head | 6 |
| n_embd | 384 |
| n_positions | 13000 |
| architectures | GPTForFluMultiTask |
| model_type | gpt2 |
| transformers_version_config | 4.29.2 |
| task_probable | LM causal / forecasting probable |

### Tokenizer

| campo | valor |
| --- | --- |
| vocab.json size | 10 |
| added_tokens size | 15 |
| vocab real fusionado | 25 |
| tokenizer_class | InfluTokenizer |
| tokens segmento | <HA>, <NA>, <sep> |
| tokens subtipo | <H10N3>, <H10N5>, <H10N8>, <H1N1>, <H1N2>, <H2N2>, <H3N2>, <H3N8>, <H5N1>, <H5N6>, <H5N8>, <H6N1>, <H7N4>, <H7N9>, <H9N2> |
| special map | {"additional_special_tokens": ["<HA>", "<NA>"], "eos_token": "<eos>", "pad_token": "<pad>", "sep_token": "<sep>", "unk_token": "N"} |
| example ids | [11, 8, 0, 3, 1, 2, 4, 6, 9, 2, 1, 3, 0, 7] |
| decode sin especiales | ATCGNGCTA |
| decode con especiales | <H3N2><HA>ATCGN<sep><NA>GCTA<eos> |
| roundtrip pequeno | True |

### State dict

- Numero de tensores: `89`

| prefijo | n |
| --- | --- |
| lm_head | 1 |
| transformer | 88 |

| head/prefijo | n | detalle |
| --- | --- | --- |
| lm_head | 1 | lm_head.weight (25, 384) |
| classification_head | 0 | no detectado |
| cls_head | 0 | no detectado |
| transformer | 88 | transformer.wte.weight (25, 384); transformer.wpe.weight (13000, 384); transformer.h.0.ln_1.weight (384,); transformer.h.0.ln_1.bias (384,); transformer.h.0.attn.bias (1, 1, 13000, 13000); ... (88 tensores) |
| backbone | 0 | no detectado |

| tensor relevante | shape |
| --- | --- |
| transformer.wte.weight | (25, 384) |
| transformer.wpe.weight | (13000, 384) |
| lm_head.weight | (25, 384) |

### Carga y forward minimo

| campo | valor |
| --- | --- |
| params | 15,649,152 |
| missing keys | 0 |
| unexpected keys | 12 |
| unexpected son buffers mascara causal | True |
| unexpected no explicadas | ninguna |
| input shape | (1, 14) |
| logits shape | (1, 14, 25) |
| hidden shape | (1, 14, 384) |
| NaNs en logits | False |
| forward ok | True |

## Inventario: `subtype_classifier/`

| archivo | tamano | bytes | sha256 |
| --- | --- | --- | --- |
| config.json | 750 B | 750 |  |
| pytorch_model.bin | 1.0 GB | 1076636610 | a43fc134c933210e... |
| special_tokens_map.json | 156 B | 156 |  |
| tokenizer_config.json | 290 B | 290 |  |
| vocab.json | 120 B | 120 |  |

### Configuracion

| campo | valor |
| --- | --- |
| vocab_size | 10 |
| n_layer | 6 |
| n_head | 6 |
| n_embd | 384 |
| n_positions | 13000 |
| architectures | GPTForFluMultiTask |
| model_type | gpt2 |
| transformers_version_config | 4.29.2 |
| task_probable | clasificacion de subtipo probable |

### Tokenizer

| campo | valor |
| --- | --- |
| vocab.json size | 10 |
| added_tokens size | 0 |
| vocab real fusionado | 10 |
| tokenizer_class | InfluTokenizer |
| tokens segmento | <HA>, <NA>, <sep> |
| tokens subtipo | ninguno |
| special map | {"additional_special_tokens": ["<HA>", "<NA>"], "eos_token": "<eos>", "pad_token": "<pad>", "sep_token": "<sep>", "unk_token": "N"} |
| example ids | [8, 0, 3, 1, 2, 4, 6, 9, 2, 1, 3, 0, 7] |
| decode sin especiales | ATCGNGCTA |
| decode con especiales | <HA>ATCGN<sep><NA>GCTA<eos> |
| roundtrip pequeno | True |

### State dict

- Numero de tensores: `91`

| prefijo | n |
| --- | --- |
| classification_head | 2 |
| lm_head | 1 |
| transformer | 88 |

| head/prefijo | n | detalle |
| --- | --- | --- |
| lm_head | 1 | lm_head.weight (10, 384) |
| classification_head | 2 | classification_head.weight (12, 384); classification_head.bias (12,) |
| cls_head | 0 | no detectado |
| transformer | 88 | transformer.wte.weight (10, 384); transformer.wpe.weight (13000, 384); transformer.h.0.ln_1.weight (384,); transformer.h.0.ln_1.bias (384,); transformer.h.0.attn.bias (1, 1, 13000, 13000); ... (88 tensores) |
| backbone | 0 | no detectado |

| tensor relevante | shape |
| --- | --- |
| transformer.wte.weight | (10, 384) |
| transformer.wpe.weight | (13000, 384) |
| lm_head.weight | (10, 384) |
| classification_head.weight | (12, 384) |
| classification_head.bias | (12,) |

### Carga y forward minimo

| campo | valor |
| --- | --- |
| wrapper construido | True |
| num labels | 12 |
| classification_head.weight | (12, 384) |
| classification_head.bias | (12,) |
| params wrapper | 15,648,012 |
| missing keys | 0 |
| unexpected keys | 13 |
| unexpected explicadas | True |
| unexpected no explicadas | ninguna |
| input shape | (1, 13) |
| forward global ok | True |

| pooling | logits shape | NaNs | ok | error |
| --- | --- | --- | --- | --- |
| last | (1, 12) | False | True |  |
| mean | (1, 12) | False | True |  |
| ha | (1, 12) | False | True |  |
| na | (1, 12) | False | True |  |

## Riesgos pendientes

- `prediction_sequence/` carga como LM causal y permite forward minimo, pero esta auditoria no confirma el protocolo exacto de forecasting del paper.
- `subtype_classifier/` contiene una cabeza real `classification_head` lineal de 12 clases; falta el mapa de etiquetas de esas 12 clases.
- El pooling exacto del clasificador sigue pendiente: ultimo token, mean pooling, `<HA>` y `<NA>` son plausibles tecnicamente, pero esta auditoria no decide cual corresponde al paper.
- La presencia de `lm_head.weight` en `subtype_classifier/` queda como peso adicional no usado por el wrapper minimo; no impide el forward de clasificacion.
- Las `unexpected keys` asociadas a `attn.bias` y `attn.masked_bias` son compatibles con buffers de mascara causal serializados por otra version de `transformers`.
- No se calcularon accuracy, F1, mismatch ni ranking; esos pertenecen a la siguiente fase experimental.

## Recomendacion de siguiente paso

Ejecutar primero una clasificacion parcial H1N1/H3N2 con muestra pequena y balanceada, porque reduce el riesgo de que el wrapper/pooling del `subtype_classifier` este mal interpretado. En paralelo conceptual, `prediction_sequence` ya esta tecnicamente listo para un scoring condicional pequeno, pero conviene cerrar antes la fidelidad basica del clasificador.
