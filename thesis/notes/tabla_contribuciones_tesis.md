# Tabla de Contribuciones Originales

**Proyecto:** Modelado estocástico de la deriva antigénica de Influenza A  
**Fecha:** 26 de abril de 2026  

---

## Resumen de Contribuciones

La tesis contiene siete contribuciones potenciales, organizadas en tres niveles de originalidad. No todas las contribuciones tienen el mismo peso ni el mismo riesgo. Algunas son defendibles independientemente de los resultados empíricos; otras dependen críticamente de que los experimentos confirmen las hipótesis.

**Nivel 1 — Contribuciones estructurales** (existen independientemente de los resultados):
- C1: Formulación de SDE sobre espacio latente viral.
- C2: Auditoría geométrica del espacio latente de AntigenLM.

**Nivel 2 — Contribuciones condicionales** (requieren resultados empíricos positivos):
- C3: Descomposición interpretable del drift en viabilidad y escape.
- C4: Predicción distribucional con incertidumbre calibrada.
- C5: Extensión del horizonte predictivo a 3–12 meses.

**Nivel 3 — Contribuciones de validación** (requieren ejecución experimental completa):
- C6: Validación prospectiva genuina 2022–2026.
- C7: Cuantificación del ratio α/β como observable biológico.

---

## C1: SDE sobre espacio latente de un modelo de lenguaje viral

### Qué problema resuelve

Los modelos actuales de predicción antigénica operan en espacio discreto de secuencias (next-token prediction). No existe un marco que modele la evolución viral como dinámica continua en un espacio latente aprendido, permitiendo integración temporal arbitraria.

### Paper relacionado

**PRESCIENT** — Yeo, Saksena & Gifford (2021). *Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions.* Nature Communications, 12:3222.

### Diferencia con el paper relacionado

PRESCIENT usa SDEs sobre el espacio latente de un autoencoder para modelar trayectorias de diferenciación celular. Las diferencias son:

- **Dominio biológico diferente.** PRESCIENT modela diferenciación celular (un proceso determinista con ruido); esta tesis modela evolución viral (un proceso estocástico con presión selectiva cambiante). La dinámica evolutiva tiene un componente adversarial (el virus "compite" contra el sistema inmune) que no existe en la diferenciación celular.
- **Tipo de encoder.** PRESCIENT usa un PCA o autoencoder variacional, que por construcción tiene buenas propiedades geométricas (regularización KL). AntigenLM es un GPT autoregresivo, que no tiene estas garantías. Esto hace la propuesta más arriesgada pero también más general: si funciona con un GPT, el enfoque se extiende a cualquier modelo de lenguaje biológico.
- **Estructura del drift.** PRESCIENT usa un drift aprendido genérico (red neuronal). Esta tesis propone un drift con estructura biológica explícita: viabilidad + escape. Esto es simultáneamente una restricción (menos flexible) y una ventaja (más interpretable).
- **Horizonte temporal.** PRESCIENT predice trayectorias celulares a escala de horas/días. Esta tesis predice trayectorias virales a escala de meses/años, con datos mucho más dispersos temporalmente.

### Evidencia experimental necesaria

- Que la SDE sea entrenable de forma estable sobre embeddings de AntigenLM (EXP-8).
- Que las trayectorias generadas permanezcan en la región válida del espacio latente.
- Comparación directa: la SDE en espacio latente viral vs. un baseline autorregresivo en espacio de secuencias (AntigenLM).

### Riesgo

**Medio-alto.** La novedad es genuina, pero la viabilidad depende de H1 (geometría latente). Si el espacio latente de AntigenLM no soporta la dinámica, la contribución se reduce a una propuesta teórica sin validación empírica.

### Cómo defenderla ante un jurado

"La contribución no es solo que la SDE funcione empíricamente; es haber identificado las condiciones bajo las cuales un espacio latente soporta dinámicas estocásticas continuas, y haberlas verificado (o refutado) para un modelo de lenguaje viral concreto. PRESCIENT asume estas condiciones porque trabaja con un VAE; nosotros las sometemos a verificación explícita porque trabajamos con un GPT."

Si la SDE no funciona: "La formulación matemática es correcta y las condiciones que identificamos son generalizables. El problema no es el marco sino el espacio latente específico de AntigenLM, que no fue diseñado para soportar dinámicas continuas. Esto es en sí mismo un resultado de valor: identifica un requisito de diseño para futuros modelos de lenguaje biológico."

---

## C2: Auditoría geométrica del espacio latente de AntigenLM

### Qué problema resuelve

No existe una evaluación sistemática de las propiedades geométricas de los espacios latentes de modelos de lenguaje virales. Los papers que usan embeddings de PLMs (protein language models) asumen implícitamente que la geometría latente es razonable, pero nadie ha verificado esto para un modelo autoregresivo entrenado sobre genomas de Influenza.

### Paper relacionado

**Latent Space Oddity** — Arvanitidis, Hansen & Hauberg (2018). *Latent Space Oddity: on the Curvature of Deep Generative Models.* ICLR 2018.

### Diferencia con el paper relacionado

Arvanitidis et al. estudian la geometría Riemanniana de espacios latentes de VAEs sobre imágenes (MNIST, CelebA). Las diferencias son:

- **Tipo de modelo.** Ellos analizan VAEs; esta tesis analiza un GPT autoregresivo. Los VAEs tienen regularización KL que promueve suavidad; los GPTs no. Esto significa que los problemas geométricos que esta tesis podría encontrar son potencialmente más severos.
- **Tipo de dato.** Ellos trabajan con imágenes (continuas, alta resolución); esta tesis trabaja con secuencias biológicas (discretas, longitud fija, con estructura evolutiva). Las propiedades geométricas relevantes son diferentes: para imágenes importa la continuidad visual; para virus importa la preservación de distancia biológica.
- **Objetivo del análisis.** Arvanitidis et al. caracterizan la curvatura como fenómeno teórico. Esta tesis evalúa si propiedades geométricas específicas (métrica, suavidad, dimensión) se satisfacen lo suficiente como para soportar una aplicación concreta (la SDE). Es un análisis orientado a uso, no a descripción.
- **Metodología.** Ellos computan la métrica pullback del decoder analíticamente. Esta tesis usa un enfoque empírico (Spearman, interpolación, TwoNN, PCA) que es más accesible computacionalmente y más directamente informativo para la pregunta de si una SDE es viable.

### Evidencia experimental necesaria

- EXP-1: Normalización de embeddings.
- EXP-2: Spearman por subtipo con Hamming en HA.
- EXP-3: Comparación de métricas (euclidiana, coseno, correlación).
- EXP-4: PCA y varianza explicada.
- EXP-5: Dimensión intrínseca (TwoNN, MLE).
- EXP-6: Interpolación con decodificación.

### Riesgo

**Bajo.** Esta contribución es robusta porque los resultados son informativos independientemente de su signo. Un resultado positivo ("la geometría soporta la SDE") habilita C1. Un resultado negativo ("la geometría no soporta la SDE") es igualmente publicable como primera auditoría de su tipo.

### Cómo defenderla ante un jurado

"Esta es la primera evaluación sistemática de las propiedades geométricas del espacio latente de un modelo de lenguaje viral con el objetivo específico de determinar si soporta dinámicas continuas. Nuestros resultados muestran que [ρ = X.XX, d = X, monotonicidad = X%], lo que implica [conclusión]. Independientemente de si la SDE funciona, estos resultados orientan el diseño de futuros modelos que pretendan soportar dinámicas en espacio latente."

Si los resultados son negativos: "Nuestro estudio demuestra que los modelos de lenguaje autoregresivos, sin regularización geométrica explícita, no producen espacios latentes adecuados para dinámicas continuas. Este es un hallazgo que no existía en la literatura y que tiene implicaciones directas para toda la línea de investigación que usa embeddings de PLMs como espacio de trabajo para modelado dinámico."

---

## C3: Descomposición del drift en viabilidad biológica y escape inmunológico

### Qué problema resuelve

Los modelos de predicción antigénica existentes tratan la evolución viral como una caja negra. No descomponen las fuerzas que dirigen la evolución en componentes interpretables. Esto limita la utilidad del modelo para comprender por qué el virus evoluciona en una dirección particular, no solo cuál.

### Paper relacionado

**Łuksza & Lässig (2014).** *A predictive fitness model for influenza.* Nature, 507:57–61.

### Diferencia con el paper relacionado

Łuksza y Lässig proponen un modelo de fitness que combina escape inmunológico y pérdida de fitness intrínseco, pero operan en espacio de secuencias discretas y usando distancias inmunológicas derivadas de ensayos HI (inhibición de hemaglutinación). Las diferencias son:

- **Espacio de trabajo.** Łuksza-Lässig operan en espacio de secuencias; esta tesis opera en espacio latente continuo. Esto permite usar gradientes (continuos) en lugar de diferencias finitas (discretas), y habilita la formulación como SDE.
- **Medida de escape.** Łuksza-Lässig usan distancias inmunológicas experimentales (datos de HI assays). Esta tesis propone un kernel gaussiano repulsivo sobre distancias latentes, que es un proxy que no requiere datos experimentales adicionales. Esto es una fortaleza (no depende de datos caros y escasos) y una debilidad (el proxy podría no capturar la presión inmune real).
- **Medida de viabilidad.** Łuksza-Lässig usan un modelo de fitness basado en frecuencias de mutaciones observadas. Esta tesis propone la log-verosimilitud del decoder como proxy de viabilidad, lo que aprovecha el conocimiento implícito del modelo de lenguaje.
- **Parámetros interpretables.** El ratio α/β cuantifica la importancia relativa de viabilidad vs. escape. Łuksza-Lässig tienen un parámetro análogo, pero en su caso está estimado con datos HI, mientras que en esta tesis se aprende directamente de las trayectorias observadas.
- **Output.** Łuksza-Lässig producen un ranking de fitness de cepas existentes. Esta tesis produce una distribución sobre trayectorias futuras. Son outputs fundamentalmente diferentes.

### Evidencia experimental necesaria

- EXP-8: Verificar que ∇F_viab apunta hacia secuencias biológicamente plausibles.
- EXP-8: Verificar que ∇F_escape apunta en la dirección de la evolución observada.
- EXP-9: Ablación α=0 (sin viabilidad) y ablación β=0 (sin escape).
- La ablación α=0 debería generar trayectorias que escapan la inmunidad pero producen secuencias no funcionales.
- La ablación β=0 debería generar trayectorias que mantienen funcionalidad pero no escapan la inmunidad (predicen estasis en lugar de evolución).

### Riesgo

**Alto.** Esta contribución depende de tres cosas simultáneas: (1) que el espacio latente tenga geometría adecuada (H1/S1.1), (2) que los funcionales sean operacionalmente correctos, y (3) que las ablaciones produzcan los efectos esperados. Si cualquiera falla, la descomposición pierde validación.

### Cómo defenderla ante un jurado

"La descomposición del drift en viabilidad y escape no es solo un recurso computacional: es una formalización de la teoría biológica existente. Łuksza y Lässig (2014) demostraron que el fitness de Influenza puede descomponerse en estas dos presiones. Nuestra contribución es trasladar esta descomposición al espacio latente de un modelo de lenguaje, donde puede operarse con gradientes continuos."

Si las ablaciones no son claras: "Los valores aprendidos de α y β, y sus intervalos de confianza, son en sí mismos un resultado: cuantifican la contribución relativa de cada presión selectiva tal como es capturada por el modelo de lenguaje. Incluso si la ablación no produce un colapso catastrófico, el ratio α/β es interpretable como un observable biológico derivado de datos genómicos."

---

## C4: Predicción distribucional con incertidumbre calibrada

### Qué problema resuelve

Los modelos actuales de predicción antigénica (incluido AntigenLM) producen una única cepa predicha sin cuantificación de incertidumbre. Esto impide: (a) estimar la confiabilidad de la predicción, (b) identificar periodos de alta impredecibilidad evolutiva, (c) comunicar riesgo a tomadores de decisiones en salud pública.

### Paper relacionado

**AntigenLM** — Pei, Chi & Kang (2026). *AntigenLM: Structure-Aware DNA Language Modeling for Influenza.* ICLR 2026.

### Diferencia con el paper relacionado

AntigenLM produce la secuencia más probable para el mes siguiente (argmax de la distribución del modelo de lenguaje). Las diferencias son:

- **Tipo de output.** AntigenLM produce una secuencia determinista. Esta tesis produce una distribución empírica (N muestras de trayectorias) sobre secuencias posibles. No solo "cuál" sino "cuáles y con qué probabilidad".
- **Cuantificación de incertidumbre.** AntigenLM no tiene mecanismo para decir "esta predicción es más incierta que aquella". La SDE lo tiene naturalmente: la varianza de la distribución de endpoints refleja la incertidumbre. Meses con alta varianza son meses donde el modelo "no sabe" hacia dónde va el virus.
- **Calibración.** Se evalúa explícitamente si los intervalos de credibilidad contienen la cepa real con la frecuencia esperada. Esto no es posible con predicciones puntuales.

### Evidencia experimental necesaria

- EXP-9: Coverage@90 ∈ [0.80, 0.95] (calibración).
- EXP-9: CRPS (Continuous Ranked Probability Score) menor que el de baselines.
- Correlación entre la varianza predicha y el error real de AntigenLM (si la SDE predice alta incertidumbre en los mismos meses donde AntigenLM falla, la incertidumbre es informativa incluso si la predicción puntual no mejora).

### Riesgo

**Medio.** La distribución sobre trayectorias existe por construcción (la SDE genera muestras). El riesgo es que la distribución esté mal calibrada: demasiado estrecha (sobre-confiada) o demasiado amplia (inútil). La calibración depende de que el drift y la difusión estén bien estimados.

### Cómo defenderla ante un jurado

"La contribución principal no es que nuestra predicción puntual mediana supere a AntigenLM — es que producimos una distribución completa sobre trayectorias posibles. Para la selección de cepas vacunales, saber que hay un 80% de probabilidad de que la cepa dominante esté en esta región del espacio antigénico es más útil que un solo punto predicho. La OMS no necesita la cepa exacta; necesita una región antigénica con alta probabilidad de cobertura."

Si la predicción puntual no supera a AntigenLM: "Nuestro marco no está diseñado para competir en predicción puntual — para eso el modelo de lenguaje es difícil de superar. La contribución es la información adicional: la incertidumbre. Mostramos que la varianza predicha por la SDE correlaciona con el error de AntigenLM (ρ = X.XX), lo que significa que nuestro modelo identifica correctamente cuándo la predicción es confiable y cuándo no."

---

## C5: Extensión del horizonte predictivo a 3–12 meses

### Qué problema resuelve

AntigenLM predice un mes adelante. La industria de vacunas necesita predicciones a 6–12 meses (el lead time de producción). No existe un modelo que extienda naturalmente el horizonte predictivo sin reentrenamiento.

### Paper relacionado

**Neher, Russell & Shraiman (2014).** *Predicting evolution from the shape of genealogical trees.* eLife, 3:e03568.

### Diferencia con el paper relacionado

LBI (Local Branching Index) de Neher et al. predice qué linaje se expande usando la forma del árbol filogenético. Las diferencias son:

- **Horizonte.** LBI produce un ranking de fitness de linajes existentes, sin horizonte temporal explícito. La SDE produce trayectorias a horizonte arbitrario: el parámetro de integración es continuo.
- **Mecanismo.** LBI es un indicador heurístico basado en la topología del árbol. La SDE es un modelo generativo que produce secuencias futuras, no solo rankings.
- **Extensibilidad.** LBI no puede predecir secuencias que no existen aún. La SDE sí, porque genera puntos nuevos en espacio latente que pueden decodificarse.

### Evidencia experimental necesaria

- EXP-9: Comparación de AAM a horizontes 1, 3, 6, 12 meses vs. baselines (persistencia, AntigenLM, LBI).
- La contribución se demuestra si AAM(SDE, k=6) < AAM(persistencia, k=6). No necesita superar a AntigenLM a k=1, solo demostrar ventaja a horizontes donde AntigenLM no opera.

### Riesgo

**Alto.** La acumulación de errores en integraciones largas puede hacer que las predicciones a 12 meses sean puro ruido. Si AAM(SDE, k=6) ≥ AAM(caminata aleatoria, k=6), la SDE no añade información a horizontes largos.

### Cómo defenderla ante un jurado

"La extensión a horizontes largos no es un truco; es una propiedad intrínseca de las SDEs. Integrar la ecuación por más pasos es matemáticamente natural. La pregunta empírica es si la acumulación de errores degrada la predicción más rápido que un baseline naive. Mostramos que a horizonte 6 meses, la SDE retiene un XX% de la ventaja que tiene a 1 mes, mientras que AntigenLM no tiene mecanismo para hacer esta predicción."

Si las predicciones a horizonte largo son malas: "Identificamos el horizonte máximo útil del modelo: a k=X meses, la SDE supera a persistencia; a k=Y meses, se vuelve indistinguible. Este resultado es en sí mismo informativo: cuantifica el horizonte predictivo de la dinámica evolutiva de Influenza A bajo nuestro modelo."

---

## C6: Validación prospectiva genuina 2022–2026

### Qué problema resuelve

La gran mayoría de los trabajos de predicción antigénica (incluido AntigenLM) se validan retrospectivamente: entrenan con datos hasta 2019 y "predicen" 2020, pero los datos de 2020 existían cuando el modelo fue diseñado. Esto introduce riesgo de overfitting metodológico: decisiones de diseño informadas (consciente o inconscientemente) por los datos de evaluación.

### Paper relacionado

Ningún paper de predicción antigénica ha realizado una validación prospectiva genuina con datos que no existían en el momento de diseño del modelo base.

### Diferencia con el paper relacionado

Esta es una contribución sin precedente directo en el campo. Los datos de 2022–2026 no existían cuando AntigenLM fue publicado. El modelo se entrena con datos ≤ 2021, y se evalúa sobre un periodo que ni el autor de esta tesis ni los autores de AntigenLM podían haber visto al tomar decisiones de diseño.

La diferencia con las validaciones retrospectivas convencionales es metodológica, no técnica: no hay posibilidad de leakage temporal porque los datos literalmente no existían. Esto elimina una de las objeciones más comunes en revisión por pares.

### Evidencia experimental necesaria

- EXP-9: Todas las métricas (AAM, CRPS, Coverage@90, Top-5) calculadas sobre el periodo 2022–2026.
- Registro fechado de decisiones metodológicas tomadas antes de acceder a datos prospectivos (decision_log.md).
- Comparación del rendimiento retrospectivo (2019–2021) vs. prospectivo (2022–2026) para evaluar estabilidad.

### Riesgo

**Medio.** El riesgo no es metodológico (la validación prospectiva es sólida por diseño) sino empírico: el periodo 2022–2026 podría tener eventos atípicos (pandemia, salto antigénico mayor, cobertura de secuenciación sesgada) que perjudiquen al modelo. Esto no sería culpa del modelo, pero debilitaría el argumento de que "funciona en el futuro".

### Cómo defenderla ante un jurado

"Nuestra validación prospectiva es inusual en el campo y constituye el estándar más exigente de evaluación predictiva. A diferencia de las validaciones retrospectivas, donde el investigador conoce los datos de evaluación durante el diseño, nosotros congelamos todas las decisiones metodológicas antes de acceder al periodo 2022–2026. Este protocolo es análogo al pre-registro de ensayos clínicos, y elimina la posibilidad de overfitting metodológico."

Si los resultados prospectivos son peores que los retrospectivos: "La degradación del rendimiento en el periodo prospectivo es esperable y en sí misma informativa. Documentamos que el rendimiento se degrada en un XX% respecto al retrospectivo, lo que cuantifica el sesgo de overfitting metodológico implícito en las evaluaciones retrospectivas estándar. Esto tiene implicaciones para cómo el campo debería evaluar sus modelos."

---

## C7: Cuantificación del ratio α/β como observable biológico

### Qué problema resuelve

No existe una cuantificación directa, derivada de datos genómicos, de la importancia relativa de la presión de viabilidad biológica vs. la presión de escape inmunológico en la evolución antigénica de Influenza A. Los modelos existentes o no descomponen estas presiones, o las estiman con datos experimentales caros (ensayos HI).

### Paper relacionado

**Hie, Yang & Kim (2022).** *Evolutionary velocity with protein language models predicts evolutionary dynamics of diverse proteins.* Cell Systems, 13(4):274–285.

### Diferencia con el paper relacionado

Hie et al. definen "velocidad evolutiva" en el espacio latente de un PLM, mostrando que la dirección del embedding cambia de forma consistente con la evolución observada. Las diferencias son:

- **Descomposición de la velocidad.** Hie et al. miden la velocidad total sin descomponerla en componentes. Esta tesis descompone la velocidad (el drift) en dos componentes interpretables. Esto permite preguntar no solo "¿a qué velocidad evoluciona?" sino "¿cuánto de esa evolución es por viabilidad y cuánto por escape?"
- **Modelado dinámico.** Hie et al. calculan velocidades pero no proponen una ecuación dinámica predictiva. Esta tesis cierra el ciclo: la velocidad evolutiva es el drift de una SDE que puede integrarse hacia adelante para generar predicciones.
- **Interpretabilidad biológica.** El ratio α/β es un número que un virólogo puede interpretar: "en este periodo, el 70% de la presión evolutiva fue escape inmunológico y el 30% fue mantener funcionalidad." Hie et al. no producen un observable comparable.

### Evidencia experimental necesaria

- EXP-9: Valores de α y β aprendidos y sus intervalos de confianza.
- Estabilidad de α/β entre diferentes splits de train/validation.
- Variación temporal de α/β: ¿el ratio cambia entre temporadas? ¿Es diferente para H3N2 vs. H1N1?
- Correlación entre β/α y la tasa de cambio antigénico observada (si β domina, el virus debería cambiar más rápido).

### Riesgo

**Alto.** Los valores de α y β solo son interpretables si F_viab y F_escape están bien calibrados y en escalas comparables. Si los funcionales tienen escalas muy diferentes, α y β absorben la diferencia de escala y pierden interpretabilidad. Además, si el espacio latente no preserva métrica biológica (H1 falla), el ratio carece de significado biológico.

### Cómo defenderla ante un jurado

"El ratio α/β cuantifica, por primera vez y de forma directa desde datos genómicos, el balance entre las dos presiones selectivas principales sobre Influenza A. Mostramos que α/β = X.XX para H3N2, con una variación temporal que correlaciona con [evento biológico observable]. Esto sugiere que el escape inmunológico explica el XX% de la velocidad evolutiva observada."

Si α/β no es estable o interpretable: "Los valores estimados de α y β tienen alta varianza entre splits, lo que indica que los datos disponibles no son suficientes para estimar de forma robusta la contribución relativa de cada presión. Sin embargo, la formulación del drift como suma ponderada permanece como un marco válido que puede beneficiarse de datos adicionales (ensayos HI, datos de deep mutational scanning) para calibrar mejor los funcionales."

---

## Tabla Resumen

| # | Contribución | Problema | Paper clave | Diferencia central | Evidencia | Riesgo | Defensa |
|---|---|---|---|---|---|---|---|
| C1 | SDE latente viral | No hay dinámicas continuas sobre LMs virales | PRESCIENT (Yeo 2021) | Encoder GPT (no VAE), drift biológico estructurado, dominio viral | EXP-8: SDE entrenable, trayectorias en región válida | Medio-alto | Marco general + condiciones de viabilidad |
| C2 | Auditoría geométrica | Nadie ha evaluado geometría de LMs virales para dinámicas | Arvanitidis 2018 | GPT vs. VAE, datos biológicos, análisis orientado a uso | EXP-1 a EXP-6: ρ, d, monotonicidad | **Bajo** | Resultado valioso en cualquier dirección |
| C3 | Drift viabilidad + escape | Modelos de caja negra sin descomposición | Łuksza & Lässig 2014 | Espacio continuo, no requiere datos HI, gradientes | EXP-8 + EXP-9: ablaciones α=0, β=0 | Alto | Formalización de teoría biológica existente |
| C4 | Predicción distribucional | Predicciones puntuales sin incertidumbre | AntigenLM (Pei 2026) | Distribución sobre trayectorias vs. argmax | EXP-9: Coverage@90, CRPS | Medio | Incertidumbre informativa incluso sin mejorar punto |
| C5 | Horizonte extendido | Solo predicen 1 mes; vacunas necesitan 6–12 | Neher et al. 2014 (LBI) | SDE integrable a horizonte arbitrario | EXP-9: AAM a k=1,3,6,12 | Alto | Cuantificar horizonte máximo útil |
| C6 | Validación prospectiva | Todas las validaciones son retrospectivas | Ninguno | Datos 2022–2026 no existían al diseñar modelo | EXP-9 + decision_log.md | Medio | Estándar más exigente = más creíble |
| C7 | Ratio α/β | No hay cuantificación genómica del balance viabilidad/escape | Hie et al. 2022 | Velocidad descompuesta, no solo medida | EXP-9: α, β aprendidos, variación temporal | Alto | Observable biológico nuevo |

---

## Priorización Estratégica

### Si tienes que elegir tres contribuciones para la defensa de la tesis

1. **C2 (Auditoría geométrica)** — Es la contribución más segura. Tiene riesgo bajo y produce resultados publicables independientemente de su signo. Debería ocupar un capítulo completo de la tesis.

2. **C4 (Predicción distribucional)** — Es la contribución conceptual más fuerte. Cambia el paradigma de predicción puntual a distribución. Incluso si la predicción mediana no supera a AntigenLM, la información de incertidumbre tiene valor. Es la Narrativa 3 recomendada en la auditoría.

3. **C6 (Validación prospectiva)** — Es la contribución más difícil de objetar. Si los resultados son positivos, es el argumento de cierre más fuerte ante cualquier jurado. Si son negativos, la honestidad del protocolo es en sí misma un estándar metodológico.

### Si solo puedes demostrar una cosa

**C2.** La auditoría geométrica es autocontenida, producible en las primeras 6 semanas, y constituye una tesis mínima defendible. Todo lo demás se construye sobre ella.

---

*Tabla de contribuciones generada el 26 de abril de 2026. Las contribuciones están priorizadas por riesgo y por independencia respecto a resultados empíricos.*
