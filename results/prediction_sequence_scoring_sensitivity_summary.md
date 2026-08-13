# Sensibilidad pequena del scoring condicional `prediction_sequence`

- Fecha local de ejecucion: `2026-04-27 23:24:10`
- Script: `test_prediction_sequence_scoring_sensitivity.py`

## Objetivo

Evaluar si el smoke test de scoring condicional es estable frente a cambios moderados en tamano de contexto, tokens puntuados y numero de ventanas. No se entrena, no se genera y no se imprimen ni guardan secuencias.

## Configuracion del checkpoint

| campo | valor |
| --- | --- |
| checkpoint | prediction_sequence |
| device | cpu |
| params | 15,649,152 |
| missing keys | 0 |
| unexpected keys | 12 |
| unexpected explicadas | True |

## Datasets detectados

| subtipo | archivo |
| --- | --- |
| H1N1 | data/processed_gisaid/dataset_H1N1.json |
| H3N2 | data/processed_gisaid/dataset_H3N2.json |

## Tabla por configuracion

| context_budget | score_tokens | ventanas | top-1 | top-5 | mean norm rank | median norm rank | MRR | mean target rank | mean candidates | dups removidos | overlap contexto | tiempo s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 256 | 10 | 0.0000 | 0.8000 | 0.3847 | 0.3333 | 0.2717 | 4.4000 | 9.9000 | 1 | 0 | 3.16 |
| 512 | 512 | 10 | 0.1000 | 0.7000 | 0.4597 | 0.4097 | 0.2921 | 5.1000 | 9.9000 | 1 | 0 | 4.40 |
| 1024 | 256 | 10 | 0.1000 | 0.4000 | 0.4625 | 0.5903 | 0.3087 | 5.1000 | 9.9000 | 1 | 0 | 5.65 |
| 1024 | 512 | 10 | 0.2000 | 0.6000 | 0.4361 | 0.3333 | 0.3669 | 4.9000 | 9.9000 | 1 | 0 | 7.06 |
| 2048 | 256 | 10 | 0.2000 | 0.4000 | 0.4847 | 0.5903 | 0.3493 | 5.3000 | 9.9000 | 1 | 0 | 11.32 |
| 2048 | 512 | 10 | 0.1000 | 0.6000 | 0.5042 | 0.4444 | 0.2671 | 5.5000 | 9.9000 | 1 | 0 | 13.30 |

## Tabla compacta por ventana

| context_budget | score_tokens | subtipo | mes objetivo | candidatas | target rank | norm rank | percentile | top-1 | top-5 | context len | tokens puntuados |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 256 | H1N1 | 2001-01 | 10 | 3 | 0.2222 | 0.7778 | False | True | 512 | 256 |
| 512 | 256 | H1N1 | 2012-11 | 10 | 4 | 0.3333 | 0.6667 | False | True | 512 | 256 |
| 512 | 256 | H1N1 | 2016-02 | 10 | 4 | 0.3333 | 0.6667 | False | True | 512 | 256 |
| 512 | 256 | H1N1 | 2017-03 | 10 | 5 | 0.4444 | 0.5556 | False | True | 512 | 256 |
| 512 | 256 | H1N1 | 2018-04 | 10 | 10 | 1.0000 | 0.0000 | False | False | 512 | 256 |
| 512 | 256 | H3N2 | 2003-12 | 10 | 3 | 0.2222 | 0.7778 | False | True | 512 | 256 |
| 512 | 256 | H3N2 | 2006-12 | 10 | 2 | 0.1111 | 0.8889 | False | True | 512 | 256 |
| 512 | 256 | H3N2 | 2012-02 | 10 | 3 | 0.2222 | 0.7778 | False | True | 512 | 256 |
| 512 | 256 | H3N2 | 2012-04 | 10 | 4 | 0.3333 | 0.6667 | False | True | 512 | 256 |
| 512 | 256 | H3N2 | 2012-10 | 9 | 6 | 0.6250 | 0.3750 | False | False | 512 | 256 |
| 512 | 512 | H1N1 | 2001-01 | 10 | 3 | 0.2222 | 0.7778 | False | True | 512 | 512 |
| 512 | 512 | H1N1 | 2012-11 | 10 | 3 | 0.2222 | 0.7778 | False | True | 512 | 512 |
| 512 | 512 | H1N1 | 2016-02 | 10 | 4 | 0.3333 | 0.6667 | False | True | 512 | 512 |
| 512 | 512 | H1N1 | 2017-03 | 10 | 9 | 0.8889 | 0.1111 | False | False | 512 | 512 |
| 512 | 512 | H1N1 | 2018-04 | 10 | 5 | 0.4444 | 0.5556 | False | True | 512 | 512 |
| 512 | 512 | H3N2 | 2003-12 | 10 | 1 | 0.0000 | 1.0000 | True | True | 512 | 512 |
| 512 | 512 | H3N2 | 2006-12 | 10 | 5 | 0.4444 | 0.5556 | False | True | 512 | 512 |
| 512 | 512 | H3N2 | 2012-02 | 10 | 10 | 1.0000 | 0.0000 | False | False | 512 | 512 |
| 512 | 512 | H3N2 | 2012-04 | 10 | 7 | 0.6667 | 0.3333 | False | False | 512 | 512 |
| 512 | 512 | H3N2 | 2012-10 | 9 | 4 | 0.3750 | 0.6250 | False | True | 512 | 512 |
| 1024 | 256 | H1N1 | 2001-01 | 10 | 8 | 0.7778 | 0.2222 | False | False | 1024 | 256 |
| 1024 | 256 | H1N1 | 2012-11 | 10 | 7 | 0.6667 | 0.3333 | False | False | 1024 | 256 |
| 1024 | 256 | H1N1 | 2016-02 | 10 | 7 | 0.6667 | 0.3333 | False | False | 1024 | 256 |
| 1024 | 256 | H1N1 | 2017-03 | 10 | 5 | 0.4444 | 0.5556 | False | True | 1024 | 256 |
| 1024 | 256 | H1N1 | 2018-04 | 10 | 7 | 0.6667 | 0.3333 | False | False | 1024 | 256 |
| 1024 | 256 | H3N2 | 2003-12 | 10 | 6 | 0.5556 | 0.4444 | False | False | 1024 | 256 |
| 1024 | 256 | H3N2 | 2006-12 | 10 | 1 | 0.0000 | 1.0000 | True | True | 1024 | 256 |
| 1024 | 256 | H3N2 | 2012-02 | 10 | 2 | 0.1111 | 0.8889 | False | True | 1024 | 256 |
| 1024 | 256 | H3N2 | 2012-04 | 10 | 2 | 0.1111 | 0.8889 | False | True | 1024 | 256 |
| 1024 | 256 | H3N2 | 2012-10 | 9 | 6 | 0.6250 | 0.3750 | False | False | 1024 | 256 |
| 1024 | 512 | H1N1 | 2001-01 | 10 | 8 | 0.7778 | 0.2222 | False | False | 1024 | 512 |
| 1024 | 512 | H1N1 | 2012-11 | 10 | 9 | 0.8889 | 0.1111 | False | False | 1024 | 512 |
| 1024 | 512 | H1N1 | 2016-02 | 10 | 10 | 1.0000 | 0.0000 | False | False | 1024 | 512 |
| 1024 | 512 | H1N1 | 2017-03 | 10 | 6 | 0.5556 | 0.4444 | False | False | 1024 | 512 |
| 1024 | 512 | H1N1 | 2018-04 | 10 | 4 | 0.3333 | 0.6667 | False | True | 1024 | 512 |
| 1024 | 512 | H3N2 | 2003-12 | 10 | 1 | 0.0000 | 1.0000 | True | True | 1024 | 512 |
| 1024 | 512 | H3N2 | 2006-12 | 10 | 1 | 0.0000 | 1.0000 | True | True | 1024 | 512 |
| 1024 | 512 | H3N2 | 2012-02 | 10 | 4 | 0.3333 | 0.6667 | False | True | 1024 | 512 |
| 1024 | 512 | H3N2 | 2012-04 | 10 | 3 | 0.2222 | 0.7778 | False | True | 1024 | 512 |
| 1024 | 512 | H3N2 | 2012-10 | 9 | 3 | 0.2500 | 0.7500 | False | True | 1024 | 512 |
| 2048 | 256 | H1N1 | 2001-01 | 10 | 7 | 0.6667 | 0.3333 | False | False | 2048 | 256 |
| 2048 | 256 | H1N1 | 2012-11 | 10 | 8 | 0.7778 | 0.2222 | False | False | 2048 | 256 |
| 2048 | 256 | H1N1 | 2016-02 | 10 | 10 | 1.0000 | 0.0000 | False | False | 2048 | 256 |
| 2048 | 256 | H1N1 | 2017-03 | 10 | 1 | 0.0000 | 1.0000 | True | True | 2048 | 256 |
| 2048 | 256 | H1N1 | 2018-04 | 10 | 8 | 0.7778 | 0.2222 | False | False | 2048 | 256 |
| 2048 | 256 | H3N2 | 2003-12 | 10 | 6 | 0.5556 | 0.4444 | False | False | 2048 | 256 |
| 2048 | 256 | H3N2 | 2006-12 | 10 | 1 | 0.0000 | 1.0000 | True | True | 2048 | 256 |
| 2048 | 256 | H3N2 | 2012-02 | 10 | 3 | 0.2222 | 0.7778 | False | True | 2048 | 256 |
| 2048 | 256 | H3N2 | 2012-04 | 10 | 3 | 0.2222 | 0.7778 | False | True | 2048 | 256 |
| 2048 | 256 | H3N2 | 2012-10 | 9 | 6 | 0.6250 | 0.3750 | False | False | 2048 | 256 |
| 2048 | 512 | H1N1 | 2001-01 | 10 | 7 | 0.6667 | 0.3333 | False | False | 2048 | 512 |
| 2048 | 512 | H1N1 | 2012-11 | 10 | 6 | 0.5556 | 0.4444 | False | False | 2048 | 512 |
| 2048 | 512 | H1N1 | 2016-02 | 10 | 10 | 1.0000 | 0.0000 | False | False | 2048 | 512 |
| 2048 | 512 | H1N1 | 2017-03 | 10 | 5 | 0.4444 | 0.5556 | False | True | 2048 | 512 |
| 2048 | 512 | H1N1 | 2018-04 | 10 | 5 | 0.4444 | 0.5556 | False | True | 2048 | 512 |
| 2048 | 512 | H3N2 | 2003-12 | 10 | 4 | 0.3333 | 0.6667 | False | True | 2048 | 512 |
| 2048 | 512 | H3N2 | 2006-12 | 10 | 1 | 0.0000 | 1.0000 | True | True | 2048 | 512 |
| 2048 | 512 | H3N2 | 2012-02 | 10 | 9 | 0.8889 | 0.1111 | False | False | 2048 | 512 |
| 2048 | 512 | H3N2 | 2012-04 | 10 | 4 | 0.3333 | 0.6667 | False | True | 2048 | 512 |
| 2048 | 512 | H3N2 | 2012-10 | 9 | 4 | 0.3750 | 0.6250 | False | True | 2048 | 512 |

## Interpretacion

- En esta sensibilidad, `top-5` se interpreta como senal de ranking amplio; `top-1` como senal mas exigente. La lectura debe mantenerse prudente porque el contexto y la continuacion siguen acotados.
- Con `max_score_tokens=256`, aumentar contexto de 512 a 2048 empeora el mean normalized rank (0.3847 -> 0.4847).
- Con `max_score_tokens=512`, aumentar contexto de 512 a 2048 empeora el mean normalized rank (0.4597 -> 0.5042).
- Con `context_token_budget=512`, puntuar mas tokens empeora el mean normalized rank (0.3847 -> 0.4597).
- Con `context_token_budget=1024`, puntuar mas tokens mejora el mean normalized rank (0.4625 -> 0.4361).
- Con `context_token_budget=2048`, puntuar mas tokens empeora el mean normalized rank (0.4847 -> 0.5042).
- `top-5` promedio sobre configuraciones: 0.5833 (rango 0.4000-0.8000).
- Mean normalized rank promedio: 0.4553 (rango 0.3847-0.5042).
- `top-1` promedio: 0.1167; si permanece bajo, la evidencia es de ranking parcial, no de seleccion puntual del target.

## Recomendacion

- Configuracion minima razonable para una reproduccion parcial: usar al menos `context_token_budget=1024` y `max_score_tokens=512` si el tiempo lo permite, porque reduce el riesgo de depender de un prefijo demasiado corto.
- Antes de escalar mucho mas, conviene preguntar a los autores por el formato exacto de contexto, continuacion, definicion de target y si el pipeline oficial usa generacion, scoring o restricciones.
- Si se escala, hacerlo incrementalmente: mas ventanas primero, luego mas tokens puntuados, manteniendo la politica de candidatas fija.

## Riesgos pendientes

- Sigue sin ser Figure 3A completa.
- Sigue sin usar generacion oficial ni generacion libre.
- El ranking depende de la definicion local de target y de la politica de candidatas del mes objetivo.
- El contexto usado sigue siendo una aproximacion local al protocolo de autores.
