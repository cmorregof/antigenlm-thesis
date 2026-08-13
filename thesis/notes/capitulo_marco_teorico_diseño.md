# Diseño del Capítulo de Marco Teórico

**Proyecto:** Modelado estocástico de la deriva antigénica de Influenza A  
**Fecha:** 26 de abril de 2026  

---

## Principios de Diseño del Capítulo

Antes de detallar cada sección, tres reglas que deben gobernar la escritura:

**Regla 1: Cada párrafo debe justificar una decisión de la tesis.** El marco teórico no es una enciclopedia. Es un argumento que construye, pieza por pieza, la necesidad lógica de lo que propones. Si un párrafo no conecta con una decisión metodológica concreta, sobra.

**Regla 2: Profundidad inversamente proporcional a la familiaridad del jurado.** Tu jurado es de Matemática Aplicada. Las SDEs las conocen — no expliques qué es un proceso de Wiener. La biología de Influenza no la conocen tan bien — ahí sí necesitas pedagogía. Los modelos de lenguaje están en un punto intermedio.

**Regla 3: Cada sección termina con una "bisagra" que conecta con la siguiente.** El lector nunca debería preguntarse "¿por qué me está contando esto ahora?" La última oración de cada sección debe hacer necesaria la primera oración de la siguiente.

---

## Estructura Propuesta

```
2. Marco Teórico
├── 2.1 Influenza A y deriva antigénica
├── 2.2 El problema de predicción antigénica
├── 2.3 Modelos de lenguaje biológicos
│   └── 2.3.1 AntigenLM
├── 2.4 Espacios latentes y su geometría
│   ├── 2.4.1 Representaciones aprendidas
│   └── 2.4.2 Propiedades geométricas necesarias
├── 2.5 Ecuaciones diferenciales estocásticas
│   ├── 2.5.1 Formulación y existencia
│   └── 2.5.2 SDEs en biología computacional
├── 2.6 Incertidumbre predictiva y evaluación
│   └── 2.6.1 Validación retrospectiva vs. prospectiva
```

Extensión estimada: 25–35 páginas. No más. Un marco teórico de 60 páginas indica falta de control sobre el material.

---

## 2.1 Influenza A y Deriva Antigénica

### Objetivo

Dar al lector de matemática aplicada el contexto biológico mínimo necesario para entender por qué la predicción antigénica es un problema difícil y por qué las soluciones actuales son insuficientes. No es una clase de virología; es una motivación del problema matemático.

### Papers clave

- **Webster et al. (1992).** *Evolution and ecology of influenza A viruses.* Microbiological Reviews. — Referencia clásica que establece la biología básica. Citar pero no reseñar extensamente.
- **Smith et al. (2004).** *Mapping the Antigenic and Genetic Evolution of Influenza Virus.* Science, 305:371–376. — Introduce el concepto de "cartografía antigénica" y los clusters antigénicos. Esencial porque establece que la evolución antigénica tiene estructura geométrica en un espacio de baja dimensión, lo que conecta directamente con tu propuesta de trabajar en un espacio latente.
- **Bedford et al. (2014).** *Integrating influenza antigenic dynamics with molecular evolution.* eLife. — Conecta la evolución molecular con la evolución antigénica. Útil para justificar que la secuencia genómica contiene información sobre la antigénicidad.

### Conceptos necesarios

- Hemaglutinina (HA) y neuraminidasa (NA) como proteínas de superficie, y por qué HA es el blanco principal de anticuerpos neutralizantes.
- Deriva antigénica vs. salto antigénico: mutaciones graduales vs. recombinación. La tesis modela solo la deriva (proceso continuo), no el salto (evento discreto catastrófico).
- Subtipos H3N2 y H1N1: por qué son los dos subtipos de interés para vacunas estacionales, y por qué H3N2 evoluciona más rápido que H1N1.
- El ciclo de selección de cepas vacunales de la OMS: la reunión de febrero/septiembre, el lead time de 6–8 meses, y por qué eso crea la necesidad de predicción.

### Qué no explicar demasiado

- No explicar la biología molecular de la replicación viral (transcripción, traducción, ensamblaje). El lector no necesita saber cómo el virus se replica para entender la tesis.
- No explicar el sistema inmune adaptativo en detalle (células B, células T, maduración de afinidad). Basta con decir que los anticuerpos neutralizantes reconocen la forma de HA, y que las mutaciones que cambian esa forma permiten al virus "escapar" la inmunidad existente.
- No dedicar más de un párrafo a la historia de las pandemias de Influenza. Es contexto cultural, no técnico.

### Bisagra hacia 2.2

> La deriva antigénica genera un problema práctico: la cepa dominante de la próxima temporada no es la misma que la de la temporada actual. Predecir hacia dónde evoluciona el virus es el problema que abordamos en la siguiente sección.

---

## 2.2 El Problema de Predicción Antigénica

### Objetivo

Formalizar qué significa "predecir la evolución antigénica", revisar los enfoques existentes, e identificar sus dos limitaciones principales (predicción puntual, horizonte fijo) que tu tesis propone resolver.

### Papers clave

- **Łuksza & Lässig (2014).** *A predictive fitness model for influenza.* Nature, 507:57–61. — El modelo de fitness que descompone la aptitud viral en escape inmunológico y costo de fitness. Es tu antecedente más directo para la descomposición del drift. Explicar su formulación con suficiente detalle para que el lector vea la conexión con tu μ = α∇F_viab + β∇F_escape.
- **Neher, Russell & Shraiman (2014).** *Predicting evolution from the shape of genealogical trees.* eLife. — LBI (Local Branching Index). Explicar brevemente como baseline heurístico.
- **Pei, Chi & Kang (2026).** *AntigenLM.* ICLR. — Estado del arte actual. Se detalla en 2.3.1, pero introducirlo aquí como el modelo que tu tesis extiende.

### Conceptos necesarios

- Formalización del problema: dado un conjunto de cepas observadas hasta el tiempo t, predecir la cepa dominante (o la distribución de cepas) en el tiempo t+k.
- Taxonomía de enfoques: filogenéticos (LBI), basados en fitness (Łuksza-Lässig), basados en aprendizaje profundo (AntigenLM). Tres párrafos, no tres páginas.
- Las dos limitaciones compartidas: (1) producen una predicción puntual (una cepa, un ranking, una secuencia) sin cuantificación de incertidumbre; (2) tienen un horizonte temporal fijo (un mes, una temporada) sin extensibilidad natural.

### Qué no explicar demasiado

- No hacer una revisión exhaustiva de todos los modelos de predicción antigénica. Citar los tres enfoques principales con un paper representativo de cada uno, y en una tabla comparativa resumir 5–6 trabajos adicionales con una columna de "limitaciones" que muestre el patrón (todos: puntual, horizonte fijo).
- No explicar en detalle cómo funciona LBI (basta con decir que mide la "velocidad de expansión" de un linaje en el árbol filogenético).
- No dedicar más de un párrafo a los ensayos de inhibición de hemaglutinación (HI assays). Mencionar que existen como referencia experimental, pero que tu modelo no los requiere.

### Bisagra hacia 2.3

> Los modelos basados en aprendizaje profundo, y en particular los modelos de lenguaje biológico, ofrecen una representación alternativa de las cepas virales: en lugar de operar sobre secuencias discretas o árboles filogenéticos, aprenden representaciones continuas que codifican la información evolutiva. Esto abre la posibilidad de formular la predicción antigénica como una dinámica en un espacio continuo.

---

## 2.3 Modelos de Lenguaje Biológicos

### Objetivo

Explicar qué son los modelos de lenguaje de proteínas/genomas, qué representaciones producen, y por qué esas representaciones son candidatas para soportar dinámicas continuas. No es una revisión de la arquitectura Transformer; es una explicación de por qué la representación interna de un modelo de lenguaje es un objeto matemático sobre el cual se puede operar.

### Papers clave

- **Rives et al. (2021).** *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.* PNAS. — ESM. Establece que los PLMs aprenden representaciones con significado biológico.
- **Lin et al. (2023).** *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science. — ESMFold. Muestra que las representaciones internas contienen información estructural 3D.
- **Hie, Yang & Kim (2022).** *Evolutionary velocity with protein language models.* Cell Systems. — Directamente relevante: las direcciones en el espacio latente de un PLM corresponden a direcciones evolutivas. Es el puente entre "modelo de lenguaje" y "dinámica evolutiva".

### Conceptos necesarios

- Analogía secuencia biológica ↔ texto natural: los aminoácidos son "palabras" cuyo "significado" depende del contexto.
- Representaciones internas (embeddings): la capa oculta produce un vector z ∈ ℝ^d para cada posición. Este vector codifica información contextual aprendida.
- Pooling: mean pooling vs. CLS token vs. último token. Justificar la elección de mean pooling (captura información global de toda la secuencia).
- Diferencia entre modelos tipo BERT (bidireccionales) y GPT (autoregresivos). AntigenLM es GPT — no tiene la regularización implícita de un VAE.

### Qué no explicar demasiado

- No explicar la arquitectura Transformer en detalle (atención, multi-head, layer norm). El jurado la conoce o puede consultarla. Un párrafo + referencia a Vaswani et al. (2017) basta.
- No explicar backpropagation ni SGD.
- No hacer una revisión de todos los PLMs existentes. Citar 2–3 como evidencia general y pasar a AntigenLM.

### 2.3.1 AntigenLM

**Objetivo:** Describir AntigenLM con detalle suficiente para entender (a) qué input recibe, (b) qué output produce, (c) qué representación interna genera, y (d) qué limitaciones tiene.

**Conceptos necesarios:**
- Arquitectura: GPT-2 con 6 capas, 384 dimensiones, 6 cabezas. Input: HA+NA concatenada en nucleótidos, tokenización character-level (25 tokens). Output: next-token prediction + clasificación de cepa dominante.
- Representación interna: z_t = Enc(HA_t, NA_t) ∈ ℝ^384. Se usa mean pooling sobre última capa oculta.
- Resultados reportados: AAM como métrica principal. Superioridad sobre LBI y otros baselines a horizonte de 1 mes.
- Limitaciones: predicción puntual, horizonte fijo, sin incertidumbre.

**Qué no explicar demasiado:**
- No reproducir las tablas del paper. Basta con reportar el resultado principal.
- No detallar el proceso de fine-tuning.

**Bisagra hacia 2.4:**

> AntigenLM produce una representación z_t ∈ ℝ^384 para cada cepa. Para que una dinámica continua sobre este espacio tenga sentido, es necesario que z_t no sea un vector arbitrario, sino un punto en una variedad con propiedades geométricas específicas. La siguiente sección examina qué propiedades son necesarias y cómo verificarlas.

---

## 2.4 Espacios Latentes y su Geometría

### Objetivo

La sección técnicamente más densa y la más importante. Debe: (1) definir qué es un espacio latente, (2) explicar qué propiedades geométricas necesita para soportar una SDE, y (3) describir los métodos para verificarlas.

### 2.4.1 Representaciones Aprendidas

**Papers clave:**
- **Bengio, Courville & Vincent (2013).** *Representation Learning: A Review.* IEEE TPAMI. — Definición formal de espacio latente.
- **Kingma & Welling (2014).** *Auto-Encoding Variational Bayes.* — VAEs. Explicar brevemente por qué tienen propiedades geométricas más controladas que los GPTs (regularización KL).

**Conceptos necesarios:**
- Definición formal: espacio latente = imagen de un encoder f: X → Z ⊂ ℝ^d.
- Distinción entre espacio latente "por diseño" (VAE, con KL) y "emergente" (GPT, sin regularización). AntigenLM cae en la segunda categoría — esto justifica la auditoría geométrica.

**Qué no explicar demasiado:**
- No derivar el ELBO. El contraste GPT vs. VAE es el punto, no la derivación.

### 2.4.2 Propiedades Geométricas Necesarias

**Papers clave:**
- **Arvanitidis, Hansen & Hauberg (2018).** *Latent Space Oddity.* ICLR. — Geometría Riemanniana de espacios latentes. Explicar métrica pullback y geodésicas intuitivamente.
- **Palma et al. (2025).** *FlatVI.* ICML. — Sin regularización, la interpolación lineal no es geodésica. Precedente de que "la geometría importa".
- **Facco et al. (2017).** *TwoNN.* Scientific Reports. — Estimador de dimensión intrínseca. Explicar con detalle matemático: ratio de distancias a vecinos, distribución Pareto, estimación del parámetro. La derivación es corta y elegante — incluirla le da solidez ante un jurado de matemática.
- **Levina & Bickel (2004).** *MLE for intrinsic dimension.* NeurIPS. — Punto de comparación para TwoNN.

**Conceptos necesarios (los tres de tu documento de tesis):**

1. **Continuidad local (Lipschitz).** Definir formalmente. Conectar con interpolación lineal como test empírico.

2. **Preservación de métrica biológica.** d_Z(z_i, z_j) ∝ d_bio(i, j). Spearman como medida. Por qué Spearman y no Pearson: la relación solo necesita ser monótona, no lineal.

3. **Dimensión intrínseca finita.** Los datos viven en M ⊂ ℝ^384 con dim(M) = d ≪ 384. Incluir la derivación breve de TwoNN (el jurado la apreciará): si los datos viven en una variedad de dimensión d, el ratio μ = r₂/r₁ sigue Pareto(d), luego d = n/Σlog(μᵢ).

**Qué no explicar demasiado:**
- No derivar geodésicas en variedades Riemannianas. Definición intuitiva + cita.
- No explicar UMAP ni t-SNE en detalle. Una frase aclarando que no preservan distancias globales.
- No discutir topología algebraica a menos que la uses.

**Bisagra hacia 2.5:**

> Si el espacio latente satisface estas propiedades — continuidad local, preservación de métrica y dimensión finita — entonces es posible definir sobre él una ecuación diferencial que modele la evolución temporal.

---

## 2.5 Ecuaciones Diferenciales Estocásticas

### 2.5.1 Formulación y Existencia

**Papers clave:**
- **Øksendal (2003).** *Stochastic Differential Equations.* Springer. — Referencia estándar.
- **Kidger et al. (2021).** *Efficient and Accurate Gradients for Neural SDEs.* NeurIPS.
- **Li et al. (2020).** *Scalable Gradients for Stochastic Differential Equations.* AISTATS. — torchsde.

**Conceptos necesarios:**
- Ecuación de Itô: dz_t = μ(z_t, t) dt + Σ(z_t) dW_t.
- Condiciones de existencia y unicidad (Lipschitz local, crecimiento lineal). Enunciar el teorema, no demostrar.
- Solución fuerte vs. débil. Tu modelo busca solución fuerte.
- Euler-Maruyama. La discretización que usarás.
- Neural SDEs: μ y/o Σ parametrizados por redes neuronales. Entrenamiento por adjoint method (citar, no derivar).
- Conexión Langevin-Boltzmann: si μ = -∇V y Σ = √(2/β)·I, la distribución estacionaria es p(z) ∝ exp(-βV(z)). Motiva la interpretación del drift como gradiente de potencial.

**Qué no explicar demasiado:**
- No demostrar existencia/unicidad.
- No explicar Itô vs. Stratonovich (nota al pie basta).
- No explicar cálculo de Malliavin ni Girsanov.
- No derivar adjoint methods.

### 2.5.2 SDEs en Biología Computacional

**Papers clave:**
- **Yeo, Saksena & Gifford (2021).** *PRESCIENT.* Nature Communications. — Antecedente más directo. Describir con detalle suficiente para que el lector vea similitudes y diferencias.
- **Hie, Yang & Kim (2022).** *Evolutionary velocity.* Cell Systems. — No usa SDEs, pero define velocidades en espacio latente que son conceptualmente el drift.
- **Łuksza & Lässig (2014).** Retomar para conectar su descomposición de fitness con tu descomposición del drift.

**Conceptos necesarios:**
- Analogía Langevin ↔ evolución biológica: fitness landscape = potencial, mutaciones = ruido, selección = drift. Con la advertencia de que las mutaciones reales son discretas (campo medio).
- Tabla breve PRESCIENT vs. tu propuesta (tipo de encoder, tipo de drift, dominio, horizonte).
- Diferencia con Hie et al.: ellos miden velocidad post-hoc; tú propones un modelo generativo.

**Qué no explicar demasiado:**
- No revisar todos los usos de SDEs en biología. Mencionar la tradición (2–3 oraciones) y focalizar en los dos antecedentes directos.
- No explicar Fokker-Planck (no la usas).

**Bisagra hacia 2.6:**

> La SDE genera una distribución sobre trayectorias, no una predicción puntual. Esto plantea la pregunta de cómo evaluar la calidad de una distribución predictiva.

---

## 2.6 Incertidumbre Predictiva y Evaluación

### Objetivo

Presentar las herramientas para evaluar predicciones distribucionales y el protocolo de validación prospectiva. Sección más corta: instrumental, no conceptual.

### Papers clave

- **Gneiting & Raftery (2007).** *Strictly Proper Scoring Rules.* JASA. — Define CRPS y calibración.
- **Gneiting, Balabdaoui & Raftery (2007).** *Probabilistic forecasts, calibration and sharpness.* JRSS-B. — "Maximally sharp subject to calibration."

### Conceptos necesarios

- Predicción puntual vs. distribucional. AAM evalúa el punto; CRPS evalúa la distribución.
- Calibración: los intervalos al α% contienen la observación α% de las veces. Definir Coverage@α.
- Sharpness: calibrada y lo más concentrada posible. Una distribución uniforme está calibrada pero es inútil.
- CRPS: CRPS = E|X - y| - ½E|X - X'|. Generaliza el error absoluto a distribuciones.
- Validación retrospectiva vs. prospectiva. Definir ambas. Explicar por qué la prospectiva es más exigente. Explicar que tu proyecto puede hacerla (datos 2022–2026 no existían al diseñar el modelo).

### Qué no explicar demasiado

- No derivar propiedades de CRPS. Enunciar y citar.
- No discutir otras scoring rules (log score, Brier). Nota al pie.
- No explicar bootstrap.

### Bisagra hacia metodología

> Con los elementos presentados — la biología de la deriva antigénica, las representaciones de AntigenLM, las propiedades geométricas del espacio latente, las SDEs como marco dinámico, y las herramientas de evaluación distribucional — es posible formular la propuesta de esta tesis.

---

## Mapa de Conexiones

Cada sección justifica una decisión de la metodología:

| Sección | Concepto presentado | Decisión que justifica |
|---|---|---|
| 2.1 | HA muta continuamente | Modelar evolución como proceso continuo |
| 2.1 | Lead time de 6–8 meses | Necesidad de horizontes > 1 mes |
| 2.2 | Modelos existentes: puntuales, horizonte fijo | Motivación de la propuesta |
| 2.2 | Łuksza-Lässig: fitness = escape + costo | Estructura del drift: μ = α∇F_viab + β∇F_escape |
| 2.3 | PLMs codifican estructura biológica | Usar AntigenLM como encoder |
| 2.3.1 | AntigenLM: GPT, 384 dim | Variable de estado z_t ∈ ℝ^384 |
| 2.3.1 | AntigenLM: sin regularización geométrica | Necesidad de auditoría geométrica |
| 2.4.1 | GPT vs. VAE: sin KL | Justificación de por qué la geometría no es obvia |
| 2.4.2 | Preservación de métrica, Lipschitz, dim. intrínseca | Los experimentos de geometría |
| 2.5.1 | SDE de Itô, Euler-Maruyama | Formulación e integración numérica |
| 2.5.1 | Langevin-Boltzmann | Drift como gradiente de potencial |
| 2.5.2 | PRESCIENT | Antecedente metodológico directo |
| 2.5.2 | Hie et al. | Antecedente conceptual directo |
| 2.6 | Calibración, CRPS | Métricas de evaluación |
| 2.6.1 | Retrospectiva vs. prospectiva | Protocolo con datos 2022–2026 |

---

## Errores Comunes a Evitar

1. **El marco teórico enciclopédico.** No intentes cubrir todo lo que leíste. Si un concepto no conecta con una decisión metodológica, no pertenece aquí.

2. **Explicar lo que el jurado ya sabe.** Las SDEs, PCA, Spearman, y procesos estocásticos. Fijar notación y pasar.

3. **No explicar lo que el jurado no sabe.** Hemaglutinina, modelos de lenguaje, el ciclo de la OMS. Ahí sí necesitas pedagogía.

4. **Secciones desconectadas.** Cada sección termina con una "bisagra" hacia la siguiente.

5. **Demasiado largo.** 25–30 páginas. Si pasa de 35, hay repetición o material de apéndice.

6. **Citar sin posicionar.** No digas "Hie et al. propusieron velocidades evolutivas." Di "Hie et al. mostraron que las direcciones en el espacio latente corresponden a direcciones evolutivas, lo que sugiere que una dinámica en este espacio puede tener correspondencia biológica. Sin embargo, su enfoque mide velocidades post-hoc sin proponer un modelo generativo. Nuestra propuesta cierra este gap."

7. **Omitir los supuestos.** Cada vez que importes un resultado de la literatura, di explícitamente qué supuesto haces al aplicarlo a tu caso, y señala dónde se verifica empíricamente.

---

## Orden de Escritura Recomendado

No escribas el capítulo de principio a fin. Escríbelo en este orden:

1. **2.4.2 (propiedades geométricas).** La sección más técnica y más importante. Ya tienes los conceptos claros.
2. **2.5.1 (SDEs).** La segunda más técnica. Formalizarla temprano te obliga a definir tu ecuación antes de implementarla.
3. **2.1 y 2.2 (biología y problema).** Las más narrativas. Más fáciles de escribir.
4. **2.3 y 2.3.1 (PLMs y AntigenLM).** Requieren releer el paper con cuidado.
5. **2.5.2 y 2.6 (SDEs en biología y evaluación).** Conectan todo. Más fáciles cuando el resto existe.

---

*Diseño de capítulo generado el 26 de abril de 2026. Estructura pensada para una tesis de Maestría en Matemática Aplicada con un jurado que conoce las SDEs pero no la biología de Influenza.*
