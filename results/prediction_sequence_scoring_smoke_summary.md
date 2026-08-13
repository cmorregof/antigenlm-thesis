# Smoke test de scoring condicional para `prediction_sequence`

- Fecha local de ejecucion: `2026-04-27 23:17:48`
- Script: `test_prediction_sequence_scoring.py`

## Objetivo

Verificar si el checkpoint `prediction_sequence/` asigna mejor likelihood condicional a candidatas reales del mes objetivo que a candidatas negativas, usando una muestra pequena y read-only. Esto no es una reproduccion completa de Figure 3A.

## Restricciones cumplidas

- No se entrenaron modelos.
- No se modificaron pesos.
- No se generaron secuencias nuevas.
- No se uso generacion libre.
- No se optimizaron ni mutaron secuencias.
- No se imprimieron ni guardaron secuencias completas.
- No se tocaron archivos `.tex`.
- No se ejecuto evaluacion masiva.

## Configuracion del checkpoint

| campo | valor |
| --- | --- |
| checkpoint | prediction_sequence |
| device | cpu |
| vocab_size | 25 |
| n_layer/n_head/n_embd/n_positions | 6/6/384/13000 |
| params | 15,649,152 |
| missing keys | 0 |
| unexpected keys | 12 |
| unexpected explicadas como buffers mascara causal | True |

## Configuracion de ventanas y scoring

| campo | valor |
| --- | --- |
| num_windows por subtipo | 3 |
| debug_one | False |
| max_candidates | 10 |
| context_strains | 1 |
| context_token_budget total | 512 |
| max_score_tokens por candidata | 256 |
| deduplicate_candidates | True |
| seed | 42 |

Nota metodologica: por defecto este smoke test usa contexto historico estructurado acotado y puntua un prefijo de continuacion. Esto reduce costo y riesgo operacional; no debe interpretarse como el protocolo completo del paper.

## Datasets detectados

| subtipo | archivo |
| --- | --- |
| H1N1 | data/processed_gisaid/dataset_H1N1.json |
| H3N2 | data/processed_gisaid/dataset_H3N2.json |

## Tabla por ventana

| subtipo | mes objetivo | candidatas | target rank | top-1 | top-5 | normalized rank | percentile score | dups removidos | overlap contexto | offsets meses | mean_nll target/top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1N1 | 2012-11 | 10 | 4 | False | True | 0.3333 | 0.6667 | 0 | 0 | {0: 10} | 7.5299/7.5092 |
| H1N1 | 2016-02 | 10 | 4 | False | True | 0.3333 | 0.6667 | 0 | 0 | {0: 10} | 7.7934/7.6123 |
| H1N1 | 2017-03 | 10 | 5 | False | True | 0.4444 | 0.5556 | 0 | 0 | {0: 10} | 7.8903/7.5887 |
| H3N2 | 2003-12 | 10 | 3 | False | True | 0.2222 | 0.7778 | 0 | 0 | {0: 10} | 7.3919/7.2736 |
| H3N2 | 2012-02 | 10 | 3 | False | True | 0.2222 | 0.7778 | 0 | 0 | {0: 10} | 7.4381/7.4095 |
| H3N2 | 2012-04 | 10 | 4 | False | True | 0.3333 | 0.6667 | 0 | 0 | {0: 10} | 7.4727/7.4303 |

## Resumen agregado

| metrica | valor |
| --- | --- |
| ventanas evaluadas | 6 |
| top-1 accuracy | 0.0000 |
| top-5 accuracy | 1.0000 |
| mean normalized rank | 0.3148 |
| MRR | 0.2694 |
| estado | exito parcial |

## Interpretacion prudente

Si el target rankea alto, el checkpoint contiene senal condicional util bajo este formato local de contexto/candidatas. Si no rankea alto, no basta para concluir que el checkpoint falla: puede deberse al contexto acotado, al prefijo puntuado, a la politica de candidatas, a la definicion local del target o a que falta el protocolo exacto de los autores.

## Riesgos pendientes

- No es Figure 3A completa.
- No usa generacion oficial ni generacion libre.
- No sabemos si el paper usa generacion, scoring, restricciones, markers adicionales o un pipeline mixto.
- Falta el protocolo exacto de autores para contexto, decoding, target dominante y metrica.
- El scoring esta acotado por `context_token_budget` y `max_score_tokens`; una reproduccion formal deberia cerrar primero esos detalles.
- No se evaluo un horizonte estacional ni Figure 3B.

## Recomendacion

Si este smoke test muestra ranking alto de targets, escalar con cuidado a mas ventanas y/o mas tokens puntuados. Si no lo muestra, antes de concluir fallo conviene preguntar a autores por el protocolo exacto. La generacion condicionada puede probarse luego solo como diagnostico, no como baseline principal hasta validar estructura.
