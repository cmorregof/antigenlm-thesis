# Auditoria parcial del `subtype_classifier`

- Fecha local de ejecucion: `2026-04-27 22:55:25`
- Script: `test_subtype_classifier_partial.py`
- Comando previsto: `--n-per-class 200 --seed 42`

## Objetivo

Verificar si el checkpoint local `subtype_classifier/` esta correctamente cableado y si separa dos clases locales conocidas con una muestra pequena y balanceada. Esta prueba no reproduce formalmente el paper.

## Restricciones cumplidas

- No se entrenaron modelos.
- No se modificaron pesos.
- No se generaron, optimizaron ni mutaron secuencias.
- No se imprimieron ni guardaron secuencias completas.
- No se tocaron archivos `.tex`.
- Solo se uso una muestra pequena y balanceada.

## Carga del checkpoint

| campo | valor |
| --- | --- |
| checkpoint | subtype_classifier |
| device | mps |
| vocab_size config | 10 |
| n_layer/n_head/n_embd/n_positions | 6/6/384/13000 |
| classification_head.weight | (12, 384) |
| classification_head.bias | (12,) |
| params | 15,648,012 |
| missing keys | 0 |
| unexpected keys | 13 |
| unexpected explicadas | True |
| NaNs en sanity logits | False |

## Datos y muestra

| clase local | archivo detectado | registros usados |
| --- | --- | --- |
| H1N1 | data/processed_gisaid/dataset_H1N1.json | 200 |
| H3N2 | data/processed_gisaid/dataset_H3N2.json | 200 |

| clase | n | HA len min/median/max | NA len min/median/max | tokens min/median/max |
| --- | --- | --- | --- | --- |
| H1N1 | 200 | 1695/1734.0/1777 | 1410/1420.0/1462 | 3109/3158.0/3241 |
| H3N2 | 200 | 1701/1731.0/1762 | 1410/1434.5/1468 | 3115/3160.5/3234 |

## Formato de input usado

Serializacion prudente sin token de subtipo:

`<HA> HA_sequence <sep> <NA> NA_sequence <eos>`

| token | id |
| --- | --- |
| <HA> | 8 |
| <sep> | 6 |
| <NA> | 9 |
| <eos> | 7 |

## Tabla por pooling

| pooling | aplicable | idx H1N1 | idx H3N2 | ambiguo | accuracy | macro F1 | confusion [[A->A,A->B],[B->A,B->B]] | unknown externo | n test | diagnostico |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| last | si | 2 | 5 | False | 0.8500 | 0.8496 | [[80, 20], [10, 90]] | 47 (0.235) | 200 | NaNs=0; failures=0; std=2.7269; argmax=[2, 3, 4, 5, 6, 8, 9, 10, 11] |
| mean | si | 2 | 5 | False | 0.8750 | 0.8743 | [[80, 20], [5, 95]] | 41 (0.205) | 200 | NaNs=0; failures=0; std=2.2219; argmax=[2, 3, 5, 8, 9, 10, 11] |
| ha | si | 10 | 10 | True | NA | NA | [[0, 0], [0, 0]] | 0 (0.000) | 200 | NaNs=0; failures=0; std=2.0412; argmax=[10] |
| na | si | 2 | 5 | False | 0.8600 | 0.8595 | [[80, 20], [8, 92]] | 41 (0.205) | 200 | NaNs=0; failures=0; std=2.7000; argmax=[2, 3, 5, 8, 9, 10, 11] |

## Mejor pooling local

Mejor pooling local: `mean` con accuracy=0.8750, macro F1=0.8743, unknown=0.205.

## Mapeo local inferido

Mapeo local inferido solo para esta auditoria: `H1N1` -> logit `2`, `H3N2` -> logit `5`. No es label map oficial.

Estado segun criterios: **exito parcial**.

## Riesgos pendientes

- Falta el label map oficial de las 12 clases.
- Falta el pooling oficial usado por los autores.
- Solo se evaluaron dos clases locales: H1N1 y H3N2.
- No se evaluan las 12 clases y por tanto esto no reproduce la tabla completa de clasificacion del paper.
- El mapeo local inferido depende de la muestra, el split 50/50 y la serializacion usada.
- La prueba no valida forecasting ni `prediction_sequence/`.

## Recomendacion

Si el mejor pooling muestra separacion alta, el checkpoint queda cableado de forma plausible para una reproduccion parcial H1N1/H3N2, pero antes de llamarlo reproduccion formal conviene solicitar o localizar el label map y el pooling oficial. Si se necesita avanzar metodologicamente sin ese dato, el siguiente bloqueo mas relevante es `prediction_sequence` scoring condicional, porque permite evaluar forecasting sin depender del clasificador.
