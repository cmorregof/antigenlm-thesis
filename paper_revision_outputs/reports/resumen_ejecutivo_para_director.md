# Resumen ejecutivo para revisión con director

## 1. Pregunta central del manuscrito

El manuscrito pregunta si las representaciones latentes locales de AntigenLM para secuencias HA/NA de Influenza A tienen una geometría útil como posible espacio de estados: es decir, si organizan registros molecularmente similares, si muestran estructura temporal local y si presentan baja dimensión efectiva bajo diagnósticos geométricos.

La pregunta no es si AntigenLM predice vacunas, pronostica la evolución futura o mide antigenicidad. La pregunta es más estrecha: si el espacio latente merece ser tratado como un candidato razonable para modelamiento posterior.

## 2. Qué datos se analizaron

Se analizaron 111,756 embeddings HA/NA almacenados en caché, derivados de registros de Influenza A H1N1 y H3N2 procesados localmente.

El conjunto incluye:

- 46,125 registros H1N1.
- 65,631 registros H3N2.
- Registros recolectados entre 2000 y 2022.
- Metadatos de subtipo, año y mes.
- Secuencias HA y NA emparejadas por identificador de aislado durante el preprocesamiento local.

Las secuencias completas no se redistribuyen en el manuscrito.

## 3. Qué es AntigenLM en este trabajo

En este manuscrito, AntigenLM se usa como un codificador local que produce embeddings de dimensión 384 para pares HA/NA:

`z_i = Enc_theta(HA_i, NA_i) in R^384`

El análisis audita la geometría de esos embeddings ya calculados. No reproduce el pipeline completo de pronóstico de AntigenLM, no genera secuencias y no evalúa predicción prospectiva.

## 4. Principales resultados cuantitativos

Las correlaciones de Spearman entre distancia euclidiana latente y proxies moleculares son altas dentro de subtipo:

- Hamming HA+NA: 0.8538 en H1N1 y 0.6678 en H3N2.
- Hamming HA-only: 0.8280 en H1N1 y 0.6082 en H3N2.

Las correlaciones globales con separación temporal absoluta son débiles:

- 0.0612 en H1N1.
- 0.1540 en H3N2.

PCA muestra fuerte concentración de varianza:

- Globalmente, 3 componentes explican 95% de la varianza.
- La razón de participación global es 1.91.

TwoNN, después de deduplicación exacta HA+NA, estima dimensiones bajas pero sensibles al recorte:

- Aproximadamente 3.85--3.90 con recorte 0.01.
- Aproximadamente 5.40--5.50 con recorte 0.05.

Los vecindarios latentes locales son mucho más cercanos temporalmente que pares aleatorios emparejados por subtipo:

- Mediana de vecinos latentes: 2 meses.
- Línea base aleatoria H1N1: 42--43 meses.
- Línea base aleatoria H3N2: 35 meses.

## 5. Qué controles de robustez se hicieron

Se hicieron controles focalizados, no exhaustivos:

- Deduplicación exacta HA+NA reteniendo el primer representante en el orden de la caché.
- Análisis separados por subtipo H1N1 y H3N2.
- Muestreo de 200,000 pares por subtipo y por semilla.
- Tres semillas aleatorias: 42, 7 y 123.
- Distancias HA, NA y HA+NA de nucleótidos.
- Proxy simple HA+NA de aminoácidos traducidos como análisis de sensibilidad.
- Permutación dentro de subtipo de etiquetas de mes de colección, manteniendo fijo el grafo de vecinos latentes.
- Valores de k = 5, 10 y 20 para vecinos temporales locales.

## 6. Qué NO se está afirmando

El manuscrito no afirma:

- Validación antigénica.
- Equivalencia entre distancia molecular y distancia antigénica.
- Validación por HI, neutralización o cartografía antigénica.
- Validación filogenética.
- Reproducción del pronóstico completo de AntigenLM.
- Pronóstico prospectivo.
- Generación confiable de secuencias.
- Relevancia para selección de cepas vacunales.
- Recomendaciones clínicas o de salud pública.
- Dimensión intrínseca exacta de la biología viral.

La afirmación central es más conservadora: el espacio latente auditado está organizado molecularmente bajo los proxies evaluados, tiene estructura efectiva baja bajo PCA/TwoNN y muestra coherencia temporal local.

## 7. Limitaciones principales

Las limitaciones principales son:

- No hay alineamiento múltiple de secuencias curado.
- No hay datos de HI, neutralización ni cartografía antigénica.
- No hay árboles filogenéticos, clados ni distancias filogenéticas.
- No hay validación prospectiva de pronóstico.
- No hay intervalos de confianza por bootstrap; solo desviaciones estándar entre semillas.
- No hay controles con embeddings aleatorios, embeddings aleatorios con espectro emparejado, embeddings blanqueados por PCA ni checkpoint no entrenado.
- El proxy de aminoácidos es una traducción simple en marco 0, no una extracción proteica curada.
- Puede haber sesgos de vigilancia por año, geografía, linaje, huésped e intensidad de muestreo.

## 8. Posibles siguientes análisis

Los análisis más útiles antes de envío serían:

- Control negativo con embeddings aleatorios o proyecciones aleatorias.
- Control con embeddings aleatorios con espectro emparejado.
- Intervalos de confianza por bootstrap o un marco de incertidumbre más formal.
- Análisis de recuperación kNN de similitud molecular.
- Trustworthiness/continuity o métricas de preservación local de rangos.
- Comparación con distancias filogenéticas si se construyen árboles confiables.
- Validación con clados Nextstrain o linajes si están disponibles.
- Comparación con datos antigénicos si se consiguen HI, neutralización o coordenadas de cartografía antigénica.
- Análisis estratificado por año, linaje o intensidad de muestreo.

## 9. Posibles revistas objetivo

La opción más razonable actualmente parece ser BMC Bioinformatics, enmarcando el trabajo como una auditoría computacional reproducible de geometría de embeddings para secuencias virales.

Otras opciones:

- Scientific Reports: viable si se enfatiza la relevancia biológica amplia y la reproducibilidad, aunque la novedad conceptual sea moderada.
- Algorithms for Molecular Biology: viable si se presenta como marco metodológico de auditoría de representaciones.
- Virus Evolution: requeriría más validación evolutiva, idealmente clados, filogenia o dinámica de linajes.
- Bioinformatics: requeriría una contribución metodológica o de software más generalizable.
- PLOS Computational Biology: requeriría una contribución biológica o conceptual más fuerte y validaciones adicionales.

## 10. Preguntas concretas para discutir con el director

1. ¿El manuscrito debe presentarse como artículo metodológico, análisis computacional de datos o capítulo/resultado de tesis?
2. ¿Conviene enviar con el alcance actual a BMC Bioinformatics después de una revisión visual y de procedencia de datos?
3. ¿Qué control negativo sería más convincente y factible: embeddings aleatorios, proyecciones aleatorias o espectro emparejado?
4. ¿Es prioritario construir una validación filogenética antes del envío?
5. ¿Tenemos acceso realista a datos antigénicos para una validación adicional, o debe quedar claramente como trabajo futuro?
6. ¿Cómo debe redactarse la disponibilidad de datos considerando términos de GISAID y restricciones sobre secuencias completas?
7. ¿Qué afiliaciones, ORCID y autor de correspondencia deben aparecer en la versión de envío?
8. ¿El título debe enfatizar “auditoría geométrica”, “representaciones latentes” o “Influenza A HA/NA” según la revista objetivo?
