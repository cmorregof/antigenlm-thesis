# Auditoría Internacional — Tesis de Maestría

**Proyecto:** Modelado estocástico de la deriva antigénica de Influenza A en el espacio latente de AntigenLM  
**Autor:** Carlos Manuel Orrego Franco  
**Programa:** Maestría en Matemática Aplicada, Universidad Nacional de Colombia  
**Fecha de evaluación:** Abril 2026  

---

## 1. Resumen Ejecutivo

### 1.1 Tesis central (en palabras del comité)

El proyecto propone reemplazar la predicción puntual y discreta de cepas dominantes de Influenza A (tal como la realiza AntigenLM) por una dinámica estocástica continua en el espacio latente del mismo modelo. La idea es que la evolución antigénica del virus puede modelarse como una ecuación diferencial estocástica (SDE) donde el drift tiene dos componentes interpretables —viabilidad biológica y escape inmunológico— y la difusión captura la estocasticidad mutacional. El resultado no es una cepa predicha, sino una distribución sobre trayectorias evolutivas posibles, lo que permite cuantificar incertidumbre y extender el horizonte predictivo más allá de un mes.

### 1.2 Problema científico que intenta resolver

Dos limitaciones concretas de los modelos actuales de predicción antigénica: (1) producen predicciones puntuales sin cuantificación de incertidumbre, y (2) están atados a un horizonte fijo (típicamente un mes), cuando la industria de vacunas necesita horizontes de 6–12 meses.

### 1.3 Contribución original si funciona

Triple contribución:
- Un marco probabilístico para predicción antigénica basado en SDEs sobre espacios latentes de modelos de lenguaje biológicos.
- Una descomposición interpretable de las presiones selectivas (viabilidad vs. escape) como términos del drift.
- La primera auditoría geométrica sistemática del espacio latente de un modelo de lenguaje viral para determinar si soporta dinámicas continuas.

### 1.4 Coherencia conceptual

La formulación es conceptualmente coherente y tiene una lógica interna sólida. Sin embargo, hay **tres saltos argumentales** que necesitan explicitación:

1. **Salto 1: De secuencia a embedding.** Se asume que congelar AntigenLM y usar sus embeddings como variable de estado preserva información suficiente para la dinámica evolutiva. Esto no es obvio: AntigenLM fue entrenado para next-token prediction, no para producir un espacio latente con buenas propiedades geométricas. El ρ ≈ 0.13 observado sugiere que este salto es problemático.

2. **Salto 2: De embedding a gradiente.** Se propone usar ∇F_viab y ∇F_escape como campos vectoriales en ℝ^384. Esto requiere que F_viab y F_escape sean diferenciables y que sus gradientes sean informativos. Pero si el espacio latente tiene una métrica degenerada (como sugiere CV = 0.000), los gradientes euclidianos podrían no apuntar en direcciones biológicamente significativas.

3. **Salto 3: De trayectoria latente a secuencia.** Se asume que las trayectorias generadas por la SDE pueden decodificarse a secuencias biológicamente válidas. Esto depende completamente de la calidad del decoder de AntigenLM, que no fue diseñado para este uso.

---

## 2. Tabla de Calificaciones

| Dimensión | Nota | Justificación resumida |
|---|---|---|
| Originalidad científica | 8/10 | La combinación SDE + espacio latente viral + descomposición viabilidad/escape es genuinamente nueva. |
| Rigor matemático | 6/10 | La formulación es correcta pero incompleta: faltan condiciones de existencia, análisis de Σ, y definiciones operacionales de los funcionales. |
| Viabilidad computacional | 7/10 | torchsde existe, los datos existen, la infraestructura es factible. El riesgo está en la estabilidad del entrenamiento. |
| Viabilidad experimental | 5/10 | El ρ ≈ 0.13 es una señal de alarma seria. Si el espacio latente no preserva métrica biológica, toda la cadena se debilita. |
| Claridad metodológica | 7/10 | El documento está bien escrito y las fases son claras. Falta operacionalizar los funcionales y especificar métricas de evaluación. |
| Potencial de publicación | 6/10 | Publicable en workshops con certeza; en journals depende críticamente de los resultados de geometría y de la comparación con baselines. |
| Riesgo técnico | 7/10 | Alto. Hay al menos tres puntos de falla independientes: geometría latente, estabilidad SDE, decodificación. |
| Alineación con estándares internacionales | 7/10 | Bien alineado con la literatura (Hie et al., Łuksza & Lässig, PRESCIENT). Falta citar trabajo reciente en Neural SDEs para biología. |
| Valor como tesis de maestría | 8/10 | Excelente alcance para maestría. Incluso el escenario C (auditoría geométrica sola) constituiría una tesis defendible. |
| Potencial doctoral/paper | 7/10 | Claro potencial. Si funciona, es un paper fuerte. Si no funciona pero la auditoría es rigurosa, sigue siendo publicable. |

### Justificaciones detalladas y caminos de mejora

**Originalidad científica (8/10).** La idea de modelar evolución viral como SDE en espacio latente es original. Existen precedentes parciales: PRESCIENT (Yeo et al., 2021) usa SDEs para trayectorias celulares, Hie et al. (2022) usan velocidades evolutivas en espacios latentes de proteínas, y Łuksza & Lässig (2014) modelan fitness de influenza. Pero nadie ha combinado estas tres ideas para predicción antigénica. Para llegar a 9: formalizar explícitamente qué se aporta respecto a cada precedente y mostrar que la combinación no es trivial.

**Rigor matemático (6/10).** La SDE está formulada correctamente a nivel notacional, pero faltan piezas críticas: (a) condiciones sobre μ y Σ para existencia y unicidad de solución (Lipschitz local, crecimiento lineal), (b) definición operacional de F_viab y F_escape, (c) análisis de si los gradientes de estos funcionales están bien definidos en todo el espacio latente, (d) tratamiento del condicionamiento en H_t. Para llegar a 8: incluir un teorema (o al menos una proposición) que establezca condiciones suficientes bajo las cuales la SDE tiene solución fuerte, y verificar empíricamente que los funcionales propuestos satisfacen esas condiciones.

**Viabilidad computacional (7/10).** La infraestructura computacional existe: torchsde, los pesos de AntigenLM, los datos de GISAID. El riesgo principal es que el entrenamiento de Neural SDEs es notoriamente inestable, especialmente con adjoint methods. Para llegar a 8: implementar primero una versión determinista (ODE) para verificar que el drift aprende algo antes de añadir difusión, y tener un plan B con Euler-Maruyama simple si torchsde falla.

**Viabilidad experimental (5/10).** Esta es la nota más baja y la más importante. El ρ ≈ 0.13 es problemático. Significa que la distancia euclidiana en el espacio latente explica menos del 2% de la varianza en distancia biológica. Si esto no mejora sustancialmente al calcular por subtipo con distancia de Hamming real (no temporal), toda la premisa de usar gradientes euclidianos para definir escape inmunológico se debilita. El CV = 0.000 es igualmente preocupante: podría significar que la norma de los embeddings es constante (están en una esfera), lo que haría la interpolación trivialmente suave sin que esto implique regularidad geométrica real. Para llegar a 7: calcular inmediatamente el Spearman por subtipo con Hamming real, verificar si los embeddings están normalizados (¿viven en una esfera?), y calcular Spearman en distancia coseno además de euclidiana.

**Claridad metodológica (7/10).** El documento está bien estructurado, las fases son razonables, y la tabla comparativa con AntigenLM es efectiva. Falta: (a) especificar las métricas exactas de evaluación (¿amino acid mismatch promedio? ¿top-k accuracy? ¿log-likelihood de la cepa observada bajo la distribución predicha?), (b) definir operacionalmente qué significa "la SDE gana" vs. "no gana", (c) especificar cómo se seleccionarán α, β (validación cruzada, grid search, optimización bayesiana). Para llegar a 8: incluir una sección de métricas formales con definiciones matemáticas.

**Potencial de publicación (6/10).** El escenario ideal es un paper en Bioinformatics o PLOS Comp Bio, pero esto requiere que la SDE supere a AntigenLM en al menos un horizonte. Si no supera, la auditoría geométrica sola podría ir a un workshop de NeurIPS/ICML. Para llegar a 7: diseñar la tesis de modo que la auditoría geométrica sea un resultado autocontenido y publicable independientemente de la SDE.

**Riesgo técnico (7/10 = riesgo alto).** Tres puntos de falla independientes: (1) el espacio latente no soporta la dinámica, (2) la SDE no es entrenable establemente, (3) la decodificación produce basura. La probabilidad de que al menos uno falle es alta. Para mitigar: diseñar el proyecto de modo que cada fase sea un resultado en sí mismo, no un prerrequisito cuyo fallo invalida todo.

**Alineación internacional (7/10).** La propuesta está bien anclada en la literatura relevante. Falta citar: (a) Tong et al. (2024), Conditional Flow Matching, que es una alternativa a SDEs para dinámicas en espacios latentes, (b) trabajos recientes de score-based generative modeling que usan SDEs de forma inversa, (c) Melnyk et al. o trabajos recientes de protein design con dinámicas latentes. Para llegar a 8: hacer una revisión más exhaustiva de la intersección Neural SDE + biología computacional post-2023.

**Valor como tesis de maestría (8/10).** Incluso con resultados parciales, este proyecto tiene suficiente sustancia para una tesis sólida. La estructura en escenarios (A, B, C) es estratégicamente inteligente. Para llegar a 9: asegurar que el escenario C (auditoría geométrica sola) esté tan bien desarrollado que sea indistinguible de un resultado intencional, no de un plan de contingencia.

**Potencial doctoral/paper (7/10).** Si la SDE funciona y la validación prospectiva 2022–2026 confirma, esto es un paper completo en un journal de primer nivel. Si no funciona, la auditoría geométrica + análisis negativo sigue siendo publicable. Para llegar a 8: incluir una contribución teórica (aunque sea modesta) sobre condiciones bajo las cuales un espacio latente soporta dinámicas estocásticas.

---

## 3. Auditoría Técnica Detallada

### 3.1 Auditoría Metodológica

#### La elección de trabajar en el espacio latente de AntigenLM

**Fortaleza:** Es pragmático. AntigenLM ya codifica información evolutiva aprendida de datos reales. Aprovechar un encoder pre-entrenado evita reentrenar desde cero.

**Debilidad crítica:** AntigenLM fue entrenado con next-token prediction, no con un objetivo que promueva regularidad geométrica del espacio latente. Los modelos GPT no tienen la presión regularizadora de un VAE (KL divergence) ni de un autoencoder con cuello de botella geométrico. Esto significa que el espacio latente podría ser altamente irregular, con regiones de alta densidad separadas por valles degenerados.

**Supuesto fuerte:** Que las representaciones internas de un GPT entrenado para predicción de tokens capturan implícitamente la estructura geométrica necesaria para definir gradientes de funcionales biológicos. Este supuesto no tiene respaldo teórico fuerte y la evidencia empírica (ρ ≈ 0.13) es débil.

**Recomendación:** Considerar seriamente la capa de proyección (escenario B) como plan principal, no como plan B. Una capa MLP entrenable de ℝ^384 → ℝ^d (con d = dimensión intrínseca estimada) con una pérdida que combine reconstrucción + regularización métrica podría transformar un espacio latente irregular en uno que soporte la SDE.

#### El uso de una SDE tipo Langevin

**Fortaleza:** Langevin es la elección natural para modelar una dinámica con drift dirigido + ruido. Es bien entendida matemáticamente, tiene conexión con distribuciones de Boltzmann en equilibrio, y es implementable con herramientas existentes.

**Debilidad:** La dinámica de Langevin asume que el sistema eventualmente alcanza equilibrio estacionario. La evolución viral no tiene equilibrio: es un proceso no estacionario por definición (el paisaje inmunológico cambia con cada ola epidémica). Esto no invalida el modelo, pero significa que H_t (la historia) debe actualizarse dinámicamente, lo que complica tanto la formulación como la inferencia.

**Supuesto no justificado:** Que la estocasticidad mutacional es bien aproximada por ruido browniano aditivo. En realidad, las mutaciones son discretas, raras, y con efectos epistáticos no lineales. El ruido browniano es una aproximación de campo medio que podría ser inadecuada a escalas temporales cortas.

**Recomendación:** Justificar explícitamente la aproximación browniana citando la literatura de genética de poblaciones (ecuación de Kimura, difusión de frecuencias alélicas). Esto le daría soporte teórico a la elección.

#### La definición del drift como combinación viabilidad + escape

**Fortaleza conceptual:** Esta descomposición es elegante e interpretable. Captura las dos presiones selectivas principales sobre Influenza: mantener funcionalidad biológica (viabilidad) y evadir inmunidad preexistente (escape). El ratio α/β es un resultado interpretable por sí mismo.

**Debilidad:** Falta una tercera presión selectiva crucial: la **transmisibilidad**. Una cepa puede ser viable y escapar la inmunidad pero tener baja transmisibilidad, y no se convertiría en dominante. En la formulación de Łuksza & Lässig (2014), el fitness incluye tanto escape como transmisibilidad.

**Riesgo matemático:** ∇F_viab y ∇F_escape podrían apuntar en direcciones opuestas en regiones del espacio donde viabilidad y escape son antagónicos (y lo son frecuentemente: las mutaciones de escape suelen reducir fitness intrínseco). Esto podría crear puntos silla en el paisaje resultante, haciendo la dinámica sensible a los valores de α y β.

**Recomendación:** (a) Reconocer la omisión de transmisibilidad y justificar por qué es aceptable como primera aproximación. (b) Analizar el campo vectorial αF_viab + βF_escape antes de correr la SDE, para detectar puntos silla y regiones degeneradas.

#### Gradientes de funcionales en el espacio latente

**Fortaleza:** Si F_viab se define como log-verosimilitud del decoder, el gradiente es computable mediante backpropagation a través de AntigenLM congelado.

**Debilidad seria:** Los gradientes de una red neuronal profunda en su espacio interno pueden ser extremadamente ruidosos, con magnitudes que varían en órdenes de magnitud entre regiones. No hay garantía de que ∇F_viab sea Lipschitz, que es condición necesaria para existencia de solución de la SDE.

**Riesgo:** Exploiting gradients, es decir, que el gradiente apunte hacia regiones de alta log-verosimilitud que no corresponden a secuencias biológicas reales sino a artefactos del modelo (adversarial examples en espacio latente).

**Recomendación:** (a) Verificar empíricamente la norma de ∇F_viab en una muestra de puntos y reportar su distribución. Si la varianza es alta, considerar gradient clipping o normalización. (b) Verificar que maximizar F_viab por gradient ascent produce secuencias biológicamente plausibles, no adversarial examples.

#### α y β como parámetros interpretables

**Fortaleza:** La idea de que α/β cuantifica la fuerza relativa de viabilidad vs. escape es potencialmente un resultado biológico significativo.

**Debilidad:** α y β solo son interpretables si F_viab y F_escape están en escalas comparables. Si F_viab ∈ [-1000, 0] y F_escape ∈ [0, 1], entonces α y β absorben la diferencia de escala y pierden interpretabilidad.

**Recomendación:** Normalizar ambos funcionales antes de combinarlos, o reportar α·Var(∇F_viab) y β·Var(∇F_escape) como las presiones efectivas.

#### La decisión de congelar AntigenLM

**Fortaleza:** Reduce el número de parámetros entrenables, simplifica la implementación, y permite atribuir resultados a la SDE (no a un re-entrenamiento del encoder).

**Debilidad:** El encoder congelado produce un espacio latente fijo cuya geometría no se puede mejorar. Si ρ ≈ 0.13 se confirma como intrínsecamente bajo (no un artefacto de la métrica), el proyecto queda atrapado en un espacio latente inadecuado.

**Recomendación:** Mantener AntigenLM congelado como diseño principal, pero tener lista la capa de proyección como extensión inmediata. Definir un criterio numérico claro (por ejemplo, ρ < 0.3 por subtipo con Hamming real) que active automáticamente el plan B.

#### Decodificación de trayectorias latentes a secuencias

**Riesgo alto.** AntigenLM es un modelo autoregresivo, no un autoencoder con decoder explícito. La "decodificación" desde z_t requiere usar z_t como contexto para generar una secuencia token por token. Esto introduce varias complicaciones:

1. El proceso de decodificación es estocástico (sampling con temperatura).
2. No hay garantía de que z_t arbitrario produzca una secuencia válida; solo los z_t que corresponden a cepas reales del entrenamiento están "en distribución".
3. Las trayectorias de la SDE pasan por regiones del espacio latente que podrían estar fuera de la distribución de entrenamiento (OOD).

**Recomendación:** (a) Definir una métrica de validez de decodificación (por ejemplo, fracción de secuencias decodificadas que pasan un test de proteína funcional). (b) Monitorear esta métrica durante la integración de la SDE para detectar si las trayectorias salen de la región válida. (c) Considerar una constricción que penalice trayectorias que se alejen de la variedad de datos.

#### Relación entre distancia euclidiana y distancia biológica

**Este es el punto más crítico del proyecto.** Con ρ ≈ 0.13 (y pendiente de recalcular por subtipo con Hamming real), la premisa de que distancia euclidiana ≈ distancia biológica es débil. Sin esta correspondencia, F_escape (que depende de distancias euclidianas a cepas históricas) pierde su interpretación inmunológica.

**Posibilidades:**
- El ρ ≈ 0.13 está calculado mezclando subtipos → al separar podría subir.
- El ρ ≈ 0.13 usa distancia temporal como proxy de distancia biológica → con Hamming real podría ser diferente.
- El ρ ≈ 0.13 es intrínsecamente bajo → el espacio latente de AntigenLM no preserva métrica biológica.

**Recomendación:** Este análisis debería ser la **prioridad absoluta** de las próximas dos semanas. Los resultados de geometría determinan si el proyecto va por el escenario A, B o C.

#### Regularidad del espacio latente

El CV = 0.000 en interpolación es sospechoso. Hay tres posibles explicaciones:

1. **Los embeddings están L2-normalizados** (viven en una esfera S^383). En ese caso, la norma es constante por construcción y CV = 0 es trivial, no informativo.
2. **El espacio es genuinamente liso** pero la interpolación se realizó entre puntos demasiado cercanos.
3. **Error en la implementación** (por ejemplo, truncación de secuencias que produce embeddings similares).

**Recomendación:** (a) Verificar inmediatamente si ||z_i||_2 es constante para todas las cepas. (b) Si sí, usar distancia coseno en lugar de euclidiana. (c) Repetir la interpolación con pares de cepas distantes (por ejemplo, cepa de 2005 vs. cepa de 2020).

### 3.2 Auditoría de la Geometría Latente

#### Pruebas propuestas y su evaluación

**Interpolación lineal — Evaluación: DÉBIL en su forma actual.**

La prueba de CV ≈ 0 es necesaria pero no suficiente. Una esfera tiene CV = 0 para la norma pero es un espacio con curvatura no nula donde la interpolación lineal sale de la variedad. La prueba necesita complementarse con: (a) verificar que las secuencias decodificadas a lo largo de la interpolación son biológicamente sensatas (no solo que las normas son suaves), y (b) verificar que los puntos interpolados están dentro de la distribución de datos (midiendo, por ejemplo, la distancia al vecino más cercano en el conjunto de entrenamiento).

**Correlación Spearman — Evaluación: FUERTE si se hace correctamente.**

Es la prueba más directa y más informativa. Pero tiene requisitos:
- Debe calcularse por subtipo (H3N2 y H1N1 separados).
- Debe usar distancia de Hamming real en aminoácidos de HA, no distancia temporal.
- Debe incluir un análisis de la monotonicidad (¿la relación es monótona o tiene inversiones?).
- Debe reportarse no solo el ρ global sino la distribución de ρ locales (por ejemplo, ρ dentro de cada año).

Resultados que validarían la tesis: ρ > 0.5 por subtipo con Hamming real.
Resultados que la debilitarían: ρ < 0.2 por subtipo con Hamming real.

**UMAP por subtipo/año — Evaluación: DÉBIL como prueba formal, ÚTIL como visualización.**

UMAP no preserva distancias globales y depende fuertemente de hiperparámetros (n_neighbors, min_dist). No puede usarse como evidencia de regularidad geométrica. Es útil para comunicar resultados cualitativamente, pero no para tomar decisiones metodológicas.

**Dimensión intrínseca (TwoNN y MLE) — Evaluación: FUERTE.**

Esta es una prueba importante y bien elegida. Si d_intrínseca ≈ 20–40, la SDE es factible. Si d_intrínseca ≈ 200+, necesitaría reducción dimensional previa. Nota: TwoNN y MLE pueden dar resultados diferentes si la distribución no es localmente uniforme, lo que ocurriría si el espacio latente tiene clusters. Recomendación: reportar ambos estimadores y discutir las discrepancias.

**PCA complementario — Evaluación: MODERADA.**

PCA muestra varianza explicada pero asume linealidad. Es útil como sanity check: si los primeros 20 componentes principales explican >90% de la varianza, esto es consistente con baja dimensión intrínseca. Pero PCA no detecta estructura no lineal.

**Distancia de Hamming en aminoácidos — Evaluación: FUERTE y ESENCIAL.**

Esta es la distancia biológica más directa y computacionalmente barata. Es la línea base contra la cual Spearman debe calcularse. Asegurarse de que se calcula sobre HA solamente, no sobre HA+NA concatenados (la presión inmunológica actúa principalmente sobre HA).

**Separar H3N2 y H1N1 — Evaluación: ESENCIAL.**

No es una prueba en sí misma, sino un requisito metodológico para todas las demás pruebas. Mezclar subtipos infla artificialmente las correlaciones porque la distancia inter-subtipo domina.

#### Pruebas adicionales recomendadas

1. **Test de normalización:** Calcular ||z_i||_2 para todas las cepas y verificar si es constante. Si lo es, cambiar a distancia coseno en todo el análisis.

2. **Geodésicas vs. líneas rectas:** Estimar geodésicas locales usando el pullback metric del decoder (Arvanitidis et al., 2018) y comparar con interpolaciones lineales. Si difieren significativamente, el espacio tiene curvatura no despreciable y la SDE euclidiana es inadecuada.

3. **Persistent homology / Topology:** Usar persistent homology para detectar si el espacio latente tiene agujeros topológicos (que serían obstáculos para trayectorias continuas).

4. **Kernel alignment:** Comparar la matriz de kernel gaussiano en espacio latente con la matriz de kernel basada en Hamming, usando Centered Kernel Alignment (CKA). Esto generaliza la correlación Spearman a una comparación de estructuras completas.

5. **Decodificación de puntos aleatorios:** Muestrear puntos uniformemente en una bola alrededor de un embedding real y decodificarlos. ¿Qué fracción produce secuencias válidas? Esto caracteriza el "radio de validez" del decoder.

6. **Análisis de Jacobiano local:** Calcular la norma del Jacobiano del decoder en varios puntos para detectar regiones donde la decodificación es sensible (altas normas) vs. insensible (bajas normas).

### 3.3 Auditoría Experimental

#### Réplica de AntigenLM

**Evaluación: Bien planteada.** La réplica con verificación de amino acid mismatch es el punto de partida correcto. Preguntas que deben responderse: ¿Los resultados replican dentro de ±5% del paper? ¿La réplica usa exactamente los mismos splits de train/test?

**Riesgo:** Si la réplica difiere significativamente, no se sabrá si el problema es la reimplementación o los datos. Mitigación: documentar exactamente qué checkpoints se usan y verificar con los autores originales si es posible.

#### Validación retrospectiva 2019–2022

**Evaluación: Correcta pero insuficiente por sí sola.** La validación retrospectiva es el estándar mínimo. Pero tiene un problema: los datos de 2019–2022 existían cuando el proyecto fue diseñado, así que hay riesgo (involuntario) de sobre-ajuste por decisiones metodológicas informadas por los datos.

#### Validación prospectiva 2022–2026

**Evaluación: Excelente.** Esta es la ventaja competitiva real del proyecto. Pocos trabajos pueden hacer validación prospectiva genuina. Recomendaciones:
- Documentar el timestamp de cada decisión metodológica para demostrar que se tomó antes de ver los datos prospectivos.
- Pre-registrar los análisis prospectivos (aunque sea informalmente en un log fechado) antes de ejecutarlos.
- Reportar todos los resultados prospectivos, incluyendo los negativos.

#### Comparación con baselines

**AntigenLM y LBI son necesarios pero insuficientes.** Faltan:
- **Baseline naive:** La cepa más reciente persiste (predicción = última cepa observada). Sorprendentemente difícil de superar a horizonte de 1 mes.
- **Baseline de consenso:** Secuencia consenso de las cepas de los últimos 3 meses.
- **Baseline estadístico:** Modelo autorregresivo simple (VAR) sobre los embeddings, sin SDE.
- **Baseline ODE:** La SDE con σ = 0 ya está propuesta como ablación, pero debería tratarse como un baseline completo, no solo como ablación.

#### Ablaciones

**α = 0 (sin viabilidad), β = 0 (sin escape), σ = 0 (sin estocasticidad):** Bien diseñadas. Falta:
- **Ablación de H_t:** ¿Qué pasa si H_t se reemplaza por un historial truncado (últimos 6 meses vs. últimos 5 años)? Esto testea la memoria relevante del sistema inmune.
- **Ablación de Σ:** ¿Σ isotrópica vs. diagonal vs. aprendida? El costo de cada opción vs. su beneficio.

#### Métricas de evaluación

**No están especificadas en el documento.** Esto es una omisión importante. Métricas recomendadas:

- **Amino acid mismatch (AAM):** Distancia de Hamming entre la secuencia predicha y la cepa dominante observada. Es la métrica usada por AntigenLM.
- **Top-k accuracy:** Fracción de veces que la cepa observada está entre las k más probables de la distribución predicha.
- **Log-likelihood calibration:** La log-probabilidad asignada a la cepa observada bajo la distribución predicha. Mide calibración.
- **Coverage de intervalos de confianza:** ¿El intervalo de credibilidad al 90% contiene la cepa observada el 90% de las veces?
- **Continuous Ranked Probability Score (CRPS):** Métrica estándar para distribuciones predictivas que penaliza tanto falta de calibración como falta de nitidez.
- **Distancia de Wasserstein:** Entre la distribución predicha y la distribución empírica observada de cepas.

#### Cómo evitar cherry-picking

- Pre-registrar las métricas principales (AAM y CRPS como primarias).
- Reportar todas las métricas en todos los horizontes, no solo las favorables.
- Incluir intervalos de confianza (bootstrap) para todas las métricas.
- No seleccionar el horizonte temporal "que mejor funciona" como resultado principal.

#### Experimento mínimo de valor

Si la SDE completa no gana: demostrar que la distribución sobre trayectorias tiene mejor calibración que la predicción puntual de AntigenLM. Es decir, que la contribución de incertidumbre tiene valor predictivo real, aunque la predicción puntual mediana no mejore.

---

## 4. Riesgos y Mitigaciones

| Riesgo | Prob. | Impacto | Señal temprana | Mitigación | Plan B |
|---|---|---|---|---|---|
| AntigenLM no replica resultados del paper | Media | Alto | AAM de réplica difiere >10% del reportado | Verificar checkpoints, contactar autores, usar pesos exactos del repositorio | Documentar discrepancias y proceder con los pesos disponibles; la tesis no depende de replicar exactamente el paper |
| Espacio latente no preserva métrica biológica | **Alta** | **Alto** | ρ(Spearman) < 0.2 por subtipo con Hamming real | Probar distancia coseno, probar embeddings de capas intermedias, probar mean pooling vs. último token | Añadir capa de proyección entrenable con pérdida de preservación métrica (escenario B) |
| La SDE no mejora frente a baselines | Media | Alto | En validación retrospectiva, AAM ≥ AAM de AntigenLM en horizonte 1 mes | Verificar que el drift converge, ajustar learning rate, probar ODE primero | Pivotar narrativa: contribución = auditoría geométrica + marco probabilístico como framework, no como mejora puntual |
| Decodificación produce secuencias inválidas | Media | Medio | >30% de secuencias decodificadas tienen aminoácidos imposibles o stop codons tempranos | Monitorear validez de decodificación durante integración; restringir trayectorias a ε-vecindad de datos reales | Evaluar la SDE solo en espacio latente (con métricas de distancia a cepa observada) sin decodificar |
| Entrenamiento SDE inestable | Media | Medio | Loss diverge, NaN en gradientes, trayectorias explotan | Empezar con ODE, usar gradient clipping, step size adaptativo, Euler-Maruyama en lugar de adjoint | Simplificar a drift lineal + difusión constante (Ornstein-Uhlenbeck) |
| Proyecto excede tiempo de maestría | **Alta** | Alto | Fase 2 toma >4 semanas; fase 3 toma >6 semanas | Priorizar escenario C (auditoría geométrica sola) como tesis mínima; empezar escritura en paralelo desde fase 2 | Tesis = auditoría geométrica + formulación teórica de la SDE + resultados preliminares. SDE completa = trabajo futuro |
| Sesgo temporal/geográfico en datos | Media | Medio | Distribución de cepas fuertemente sesgada hacia ciertos países o temporadas | Analizar distribución geográfica y temporal al inicio de fase 1; ponderar o estratificar si hay sesgo extremo | Documentar el sesgo como limitación; enfocar la evaluación en regiones/periodos bien representados |
| Validación prospectiva 2022–2026 no favorece el modelo | Media | Medio | AAM prospectivo > AAM retrospectivo significativamente | Verificar que no haya shift distribucional extremo (por ejemplo, una pandemia de un subtipo nuevo) | Reportar honestamente; analizar por qué falló (¿el periodo fue atípico? ¿hubo un salto antigénico mayor?) |
| Dimensión intrínseca demasiado alta (>100) | Baja | Alto | TwoNN y MLE reportan d > 100 | Verificar con PCA; si se confirma, considerar reducción dimensional previa | Proyectar a ℝ^d con d = varianza explicada 95% de PCA, luego aplicar SDE en el subespacio |

---

## 5. Mejoras Priorizadas

### A. Mejoras esenciales para que la tesis sea sólida

**A1. Calcular Spearman por subtipo con Hamming real en HA.**
- Problema que resuelve: Determina si la premisa fundamental del proyecto es válida.
- Esfuerzo: 1–2 días de implementación.
- Impacto: Máximo. Determina si el proyecto va por escenario A, B o C.
- Fase: Inmediatamente (esta semana).

**A2. Verificar si los embeddings están normalizados (esfera).**
- Problema que resuelve: Explica el CV = 0.000 y determina si se debe usar distancia coseno.
- Esfuerzo: 30 minutos.
- Impacto: Alto. Cambia la interpretación de toda la geometría.
- Fase: Inmediatamente (hoy).

**A3. Definir operacionalmente F_viab y F_escape.**
- Problema que resuelve: Sin definiciones concretas, la SDE es solo una fórmula bonita.
- Esfuerzo: 3–5 días.
- Impacto: Alto. Sin esto, la fase 3 no puede empezar.
- Fase: Final de fase 2.

**A4. Implementar baselines naive y consenso.**
- Problema que resuelve: Sin baselines simples, no se puede evaluar si la SDE aporta valor.
- Esfuerzo: 1–2 días.
- Impacto: Alto. Un reviewer siempre preguntará "¿por qué no comparó con persistencia?".
- Fase: Inicio de fase 4.

**A5. Definir métricas de evaluación formalmente.**
- Problema que resuelve: Sin métricas pre-especificadas, los resultados no son defendibles.
- Esfuerzo: 1 día.
- Impacto: Alto.
- Fase: Antes de fase 4.

**A6. Empezar escritura desde fase 2.**
- Problema que resuelve: La escritura siempre toma más de lo planeado; empezar temprano evita crisis.
- Esfuerzo: Continuo.
- Impacto: Alto para viabilidad temporal.
- Fase: Desde ahora.

### B. Mejoras deseables si hay tiempo

**B1. Implementar ODE antes de SDE.**
- Problema que resuelve: Verifica que el drift aprende algo antes de añadir difusión.
- Esfuerzo: 2–3 días (la ODE es un caso especial de la SDE).
- Impacto: Medio-alto. Reduce riesgo de fase 3.
- Fase: Inicio de fase 3.

**B2. Capa de proyección entrenable.**
- Problema que resuelve: Mejora la geometría del espacio latente si ρ es bajo.
- Esfuerzo: 1 semana.
- Impacto: Alto si ρ < 0.3, bajo si ρ > 0.5.
- Fase: Final de fase 2 / inicio de fase 3.

**B3. Ablación de horizonte de H_t.**
- Problema que resuelve: Determina la ventana temporal relevante del sistema inmune.
- Esfuerzo: 1–2 días.
- Impacto: Medio. Resultado interpretable biológicamente.
- Fase: Fase 4.

**B4. Coverage de intervalos de confianza.**
- Problema que resuelve: Evalúa la calibración de la incertidumbre predicha.
- Esfuerzo: 2–3 días.
- Impacto: Medio-alto. Diferenciador clave frente a predicciones puntuales.
- Fase: Fase 4.

### C. Mejoras ambiciosas para convertirla en paper

**C1. Contribución teórica: condiciones para que un espacio latente soporte SDEs.**
- Problema que resuelve: Generaliza los hallazgos de la auditoría geométrica.
- Esfuerzo: 2–4 semanas.
- Impacto: Alto para publicabilidad. Eleva de resultado empírico a contribución metodológica.
- Fase: Escritura (si hay tiempo).

**C2. Validación en un segundo sistema (por ejemplo, SARS-CoV-2).**
- Problema que resuelve: Generalización del marco.
- Esfuerzo: 3–4 semanas (requiere nuevo encoder y datos).
- Impacto: Muy alto para publicabilidad, pero probablemente inviable en 6 meses.
- Fase: Trabajo futuro.

**C3. Comparación con Conditional Flow Matching.**
- Problema que resuelve: Posiciona la SDE frente a la alternativa generativa más moderna.
- Esfuerzo: 2–3 semanas.
- Impacto: Alto para venues ML (NeurIPS, ICML).
- Fase: Trabajo futuro o extensión post-tesis.

### D. Ideas atractivas que deberían evitarse por ahora

**D1. Fine-tuning de AntigenLM con pérdida geométrica.**
- Por qué evitarla: Requiere reentrenar un modelo GPT completo, cambia la variable de estado, y difumina la contribución original.

**D2. Difusión anisótropa aprendida Σ(z_t) como red neuronal.**
- Por qué evitarla: Añade complejidad sin garantía de mejora; empezar con σ·I y solo complicar si hay evidencia empírica de que es necesario.

**D3. Modelar epistatsis explícitamente en el drift.**
- Por qué evitarla: Atractivo biológicamente, pero la epistasis en espacio latente es un problema abierto de investigación que excede el alcance de una maestría.

**D4. Usar modelos de fundación de proteínas (ESM-2) en lugar de AntigenLM.**
- Por qué evitarla: Cambia fundamentalmente el proyecto. ESM-2 no está entrenado específicamente en Influenza, y el espacio latente sería diferente. Podría ser un proyecto doctoral separado.

---

## 6. Narrativa Recomendada

### Narrativa 1: "Usamos SDEs para mejorar la predicción antigénica de Influenza A"

- **Fuerza científica:** 6/10. Es directa pero frágil: si la SDE no supera a AntigenLM, la tesis "fracasa".
- **Riesgo:** Alto. Depende de un resultado empírico incierto.
- **Defendibilidad:** Baja si los resultados no acompañan.
- **Potencial publicable:** Alto si funciona, bajo si no.
- **Veredicto:** Demasiado arriesgada como narrativa principal.

### Narrativa 2: "Auditamos geométricamente el espacio latente de AntigenLM y evaluamos si soporta dinámicas evolutivas continuas"

- **Fuerza científica:** 7/10. Es una contribución de infraestructura: nadie ha hecho esta auditoría y sus resultados son valiosos independientemente de la SDE.
- **Riesgo:** Bajo. Los resultados geométricos son siempre informativos (positivos o negativos).
- **Defendibilidad:** Alta. No depende de que "algo funcione".
- **Potencial publicable:** Medio. Suficiente para workshops, posiblemente para un journal si es el primer estudio de este tipo.
- **Veredicto:** Excelente como narrativa de seguridad, pero podría percibirse como insuficientemente ambiciosa para una maestría en Matemática Aplicada.

### Narrativa 3: "Construimos un marco probabilístico para generar distribuciones de trayectorias antigénicas en lugar de predicciones puntuales"

- **Fuerza científica:** 8/10. Captura la contribución conceptual más profunda: el cambio de paradigma de predicción puntual a distribución sobre trayectorias.
- **Riesgo:** Medio. Incluso si la SDE no supera en AAM, la distribución sobre trayectorias puede tener mejor calibración.
- **Defendibilidad:** Alta. El marco es una contribución metodológica incluso si los resultados empíricos son modestos.
- **Potencial publicable:** Alto. La narrativa de "incertidumbre cuantificada" es valorada en la comunidad.
- **Veredicto:** La narrativa más fuerte.

### Recomendación

**Narrativa principal: Narrativa 3.**

"Proponemos el primer marco probabilístico para modelar la evolución antigénica de Influenza A como una distribución sobre trayectorias continuas, en lugar de una predicción puntual. Para ello, formulamos una SDE sobre el espacio latente de un modelo de lenguaje viral, cuyo drift descompone las presiones selectivas en viabilidad biológica y escape inmunológico."

**Narrativa secundaria: Narrativa 2.**

"Como prerequisito metodológico, realizamos la primera auditoría geométrica sistemática del espacio latente de AntigenLM, evaluando si sus propiedades métricas, topológicas y dimensionales soportan dinámicas estocásticas continuas."

**Narrativa terciaria (si los resultados lo permiten): Narrativa 1.**

"Empíricamente, demostramos que este marco mejora la predicción a horizontes de 3–12 meses y ofrece incertidumbre calibrada."

---

## 7. Plan de Seis Meses Corregido

### Tesis segura (escenario C garantizado)

**Semanas 1–2: Diagnóstico urgente de geometría.**
- ¿Los embeddings están normalizados? (30 min)
- Spearman por subtipo con Hamming real en HA. (2 días)
- Corregir TwoNN. (1 día)
- PCA: varianza explicada acumulada. (1 día)
- Decisión go/no-go para SDE euclidiana.

**Semanas 3–4: Réplica de AntigenLM.**
- Reproducir AAM del paper.
- Tabla de comparación réplica vs. paper.
- Implementar baselines naive y consenso.

**Semanas 5–6: Auditoría geométrica completa.**
- Análisis publicable con 4 figuras.
- Reporte técnico.
- Si ρ < 0.3: implementar capa de proyección.
- Decisión: escenario A o B.
- **Comenzar escritura de capítulo de metodología.**

**Semanas 7–10: SDE.**
- Semana 7: Implementar ODE como versión simplificada.
- Semana 8: Si ODE funciona, añadir difusión (Euler-Maruyama simple, σ constante).
- Semanas 9–10: Entrenar drift paramétrico. Si inestable, simplificar a Ornstein-Uhlenbeck.

**Semanas 11–16: Evaluación.**
- Retrospectiva 2019–2022. (2 semanas)
- Prospectiva 2022–2026. (2 semanas)
- Ablaciones y comparaciones. (2 semanas)
- Todas las métricas pre-especificadas.

**Semanas 17–24: Escritura.**
- Capítulo de resultados. (3 semanas)
- Conclusiones y trabajo futuro. (1 semana)
- Revisión completa y defensa. (4 semanas)

### Tesis ambiciosa (escenario A)

Todo lo anterior, más:
- Capa de proyección con regularización geométrica (semanas 5–6).
- Difusión diagonal aprendida (semana 10).
- Coverage de intervalos de confianza y CRPS (semana 14).
- Contribución teórica sobre condiciones para SDEs en espacios latentes (semanas 17–18).
- Comparación con un tercer baseline (Łuksza & Lässig fitness model, semana 15).

### Núcleo indispensable

1. Réplica de AntigenLM verificada.
2. Auditoría geométrica completa con 4+ figuras.
3. Formulación matemática rigurosa de la SDE.
4. Al menos una versión de la SDE implementada y evaluada.
5. Comparación con al menos 2 baselines (AntigenLM + persistencia).
6. Validación retrospectiva con métricas pre-especificadas.

### Extensión fuerte

7. Validación prospectiva 2022–2026.
8. Ablaciones completas.
9. Coverage de incertidumbre.

### Extensión opcional

10. Capa de proyección.
11. Comparación con LBI y Łuksza-Lässig.
12. Análisis de sensibilidad a α/β.

### Trabajo futuro

13. Generalización a SARS-CoV-2.
14. Comparación con Flow Matching.
15. Contribución teórica sobre espacios latentes y SDEs.

---

## 8. Revisión del Marco Matemático

### ¿Está bien planteado?

La ecuación dz_t = μ(z_t, t, H_t) dt + Σ(z_t) dW_t es formalmente correcta como SDE de Itô en ℝ^384. Sin embargo, hay varias cuestiones pendientes.

### Condiciones necesarias

Para que la SDE tenga **solución fuerte única**:

1. **Lipschitz local en z para μ y Σ:** ∃ L > 0 tal que ||μ(z,t,H) - μ(z',t,H)|| ≤ L||z - z'|| y ||Σ(z) - Σ(z')|| ≤ L||z - z'|| para z, z' en compactos. Esto es problemático si μ usa gradientes de redes neuronales, que no son Lipschitz en general.

2. **Crecimiento lineal:** ||μ(z,t,H)|| ≤ C(1 + ||z||) y ||Σ(z)|| ≤ C(1 + ||z||). Necesario para evitar explosión en tiempo finito.

3. **Mensurabilidad progresiva** respecto a la filtración generada por W_t. H_t debe ser adaptado.

**Recomendación práctica:** No es necesario demostrar estas condiciones formalmente en la tesis, pero sí discutirlas y verificar empíricamente que (a) los gradientes de los funcionales están acotados, y (b) las trayectorias numéricas no explotan.

### Definición operacional de F_viab

**Propuesta del documento:** Log-verosimilitud del decoder.

**Evaluación:** Es la opción más natural y computacionalmente conveniente. F_viab(z) = log p_decoder(x|z), donde x es la secuencia generada. Su gradiente ∇_z F_viab es computable por backpropagation a través del decoder congelado.

**Problema:** La log-verosimilitud del decoder puede estar mal calibrada. Puntos fuera de distribución pueden tener alta log-verosimilitud (conocido problema de OOD detection en modelos autoregresivos).

**Alternativa más robusta:** Usar una versión regularizada, por ejemplo F_viab(z) = log p_decoder(x|z) - λ·||z - z_nearest||², donde z_nearest es el embedding más cercano en el conjunto de entrenamiento. Esto penaliza alejarse de regiones con datos reales.

### Definición operacional de F_escape

**Propuesta del documento:** Kernel gaussiano repulsivo respecto a cepas históricas.

**Formalización sugerida:**

F_escape(z, H_t) = Σ_{z_h ∈ H_t} w(t - t_h) · exp(-||z - z_h||² / 2σ²_k)

donde w(t - t_h) da más peso a cepas recientes (modelando la decaimiento de la inmunidad) y σ_k controla el rango de la presión inmune.

El gradiente ∇_z F_escape apunta hacia fuera de las cepas históricas, que es la dirección de escape inmunológico deseada.

**Problema:** ¿Qué cepas incluir en H_t? ¿Todas las cepas observadas o solo las dominantes? ¿Cuántas? Si |H_t| = 100,000, el cálculo del gradiente es costoso. Si |H_t| = 10 (solo dominantes), puede ser ruidoso.

**Recomendación:** Empezar con H_t = centroides mensuales de las cepas observadas. Esto reduce |H_t| a ~240 puntos (20 años × 12 meses) y captura la tendencia sin el ruido de cepas individuales.

### ¿La difusión debería ser isotrópica, diagonal o aprendida?

**Recomendación estratificada:**

- **Versión 0 (mínima, empezar aquí):** Σ = σ·I con σ escalar aprendible. Un solo parámetro de ruido. Suficiente para demostrar el concepto.
- **Versión 1 (si versión 0 funciona):** Σ = diag(σ₁, ..., σ_d) con d = dimensión intrínseca estimada. Permite ruido diferente por dirección.
- **Versión 2 (ambiciosa, evitar en tesis):** Σ(z) parametrizada como red neuronal. Demasiados parámetros para los datos disponibles.

### Formulación mínima defendible

La versión más simple que vale la pena implementar:

dz_t = [α · clip(∇F_viab(z_t)) + β · clip(∇F_escape(z_t, H_t))] dt + σ dW_t

con:
- F_viab = log-verosimilitud del decoder
- F_escape = suma ponderada de kernels gaussianos repulsivos sobre centroides mensuales
- σ escalar (no dependiente de z)
- Gradient clipping para estabilidad
- α, β, σ aprendidos por maximización de verosimilitud sobre trayectorias observadas

Esta versión tiene 3 hiperparámetros aprendibles (α, β, σ) más los parámetros del kernel de escape (σ_k, peso temporal). Es suficientemente expresiva para capturar la intuición biológica y suficientemente simple para ser entrenable con ~240 transiciones mensuales.

---

## 9. Publicabilidad por Venue

### Bioinformatics (Oxford)

- **Realismo:** 5/10
- **Exigencia:** Resultados empíricos fuertes. La SDE debe superar a AntigenLM en al menos una métrica clave.
- **Resultados mínimos:** Mejora significativa en predicción a horizonte ≥3 meses, validación prospectiva convincente.
- **Debilidades fatales:** Si la SDE no mejora la predicción empíricamente, rechazo probable. Bioinformatics valora resultados sobre metodología.

### PLOS Computational Biology

- **Realismo:** 5/10
- **Exigencia:** Similar a Bioinformatics pero con más peso en la contribución biológica. Los revisores serán biólogos computacionales, no solo ML.
- **Resultados mínimos:** Demostrar que la descomposición viabilidad/escape produce insights biológicos (por ejemplo, que α/β varía entre temporadas de formas consistentes con la epidemiología conocida).
- **Debilidades fatales:** Si la contribución biológica no es clara (es decir, si parece "solo" un modelo ML), rechazo probable.

### Journal of Computational Biology

- **Realismo:** 6/10
- **Exigencia:** Menor que Bioinformatics. Acepta contribuciones metodológicas con validación razonable.
- **Resultados mínimos:** Auditoría geométrica rigurosa + SDE funcional + comparación honesta con baselines.
- **Debilidades fatales:** Si la comparación con baselines no es exhaustiva.

### NeurIPS / ICML Workshops (ML4Health, Learning Meaningful Representations of Life)

- **Realismo:** 7/10
- **Exigencia:** Idea nueva + resultados preliminares prometedores. No necesita superar baselines, pero sí mostrar que el enfoque tiene potencial.
- **Resultados mínimos:** Auditoría geométrica + formulación SDE + resultados preliminares en un subtipo.
- **Debilidades fatales:** Si la formulación no es novedosa para la audiencia ML (las SDEs latentes ya son conocidas).

### ICLR / ML4Science Workshops

- **Realismo:** 6/10
- **Exigencia:** Contribución en la intersección ML-ciencia. Valora aplicaciones novedosas de métodos ML.
- **Resultados mínimos:** Similar a NeurIPS workshops.
- **Debilidades fatales:** Competencia fuerte; necesita un resultado claro, no solo una formulación.

### Otros venues recomendados

- **RECOMB / ISMB:** Conferencias de bioinformática computacional. Realismo 5/10. Exigen contribución algorítmica clara.
- **Physical Review E / Journal of Statistical Mechanics:** Si la contribución teórica sobre SDEs en espacios latentes se desarrolla. Realismo 4/10. Audiencia diferente, pero la perspectiva de física estadística podría ser valorada.
- **bioRxiv preprint:** Realismo 10/10. Publicar un preprint inmediatamente después de la defensa para establecer prioridad.

---

## 10. Veredicto como Comité Internacional

### ¿Es viable como tesis de maestría?

**Sí, claramente.** Incluso el escenario más conservador (auditoría geométrica + formulación teórica + resultados preliminares) constituye una tesis de maestría en Matemática Aplicada de calidad superior al promedio. La estructura en escenarios (A, B, C) es estratégicamente inteligente y protege al autor.

### ¿Es demasiado ambicioso?

**El escenario A (SDE completa que supera a AntigenLM) es ambicioso para 6 meses. Pero la tesis no depende de alcanzar el escenario A.** El riesgo real no es la ambición del objetivo, sino la posibilidad de invertir demasiado tiempo en la SDE cuando la geometría latente no la soporta. La recomendación clave es hacer el diagnóstico geométrico primero y tomar una decisión informada antes de invertir en la SDE.

### ¿Cuál es su mayor fortaleza?

**La estructura conceptual.** La descomposición del drift en viabilidad + escape, la conexión con la teoría de Łuksza-Lässig, el cambio de predicción puntual a distribución sobre trayectorias, y los datos prospectivos 2022–2026. Estas son fortalezas reales que distinguen el proyecto de un ejercicio técnico.

### ¿Cuál es su mayor debilidad?

**La evidencia empírica de que el espacio latente de AntigenLM soporta la dinámica propuesta es, por ahora, insuficiente.** El ρ ≈ 0.13 y el CV = 0.000 son señales de alarma que deben resolverse antes de proceder. Sin una geometría latente razonable, la SDE opera sobre un espacio donde las distancias euclidianas no tienen significado biológico, lo que invalida la interpretación del drift.

### ¿Cuál debería ser el foco de las próximas tres semanas?

1. **Esta semana (urgente):** Verificar normalización de embeddings. Calcular Spearman por subtipo con Hamming real. Corregir TwoNN. PCA varianza explicada. Estos cuatro análisis determinan el rumbo del proyecto.

2. **Semana 2:** Si ρ > 0.3 → proceder con escenario A. Si ρ < 0.3 → implementar capa de proyección y repetir análisis (escenario B). Si ρ < 0.1 incluso con coseno y por subtipo → considerar seriamente escenario C.

3. **Semana 3:** Completar réplica de AntigenLM. Tabla réplica vs. paper. Comenzar escritura del capítulo de geometría latente.

### ¿Qué resultado mínimo ya justificaría una tesis sólida?

Una auditoría geométrica completa y rigurosa del espacio latente de AntigenLM, con la formulación matemática de la SDE como contribución teórica, y resultados preliminares (aunque sea con una ODE o SDE simplificada) que demuestren que el marco es funcional. Esto sería el escenario B o C bien ejecutado.

### ¿Qué resultado lo convertiría en un trabajo internacionalmente competitivo?

La SDE superando a AntigenLM en predicción a horizontes de 3+ meses, con incertidumbre bien calibrada, validada prospectivamente en 2022–2026, y con la auditoría geométrica como fundamento riguroso. Esto sería escenario A con validación prospectiva positiva: un paper completo para Bioinformatics o PLOS Comp Bio.

### ¿Qué decisión estratégica debería tomar ahora el autor?

**Resolver la geometría antes de todo lo demás.** Los resultados de las pruebas geométricas (Spearman por subtipo, normalización, dimensión intrínseca) determinan si el proyecto va por escenario A, B o C. Cualquier hora invertida en la SDE antes de tener claridad sobre la geometría es hora potencialmente desperdiciada. El autor tiene los datos, tiene el modelo, y tiene la implementación parcial. Lo que necesita es ejecutar tres análisis esta semana, mirar los números con honestidad, y tomar una decisión informada.

**Segunda decisión estratégica:** Adoptar la Narrativa 3 como principal. "Un marco probabilístico para distribuciones de trayectorias antigénicas" es defendible independientemente de los resultados empíricos, mientras que "mejoramos la predicción con SDEs" solo es defendible si efectivamente mejora.

---

*Evaluación realizada el 26 de abril de 2026. Los juicios expresados representan la evaluación técnica de un comité simulado y no sustituyen la evaluación formal de un comité de tesis real. Se recomienda al autor discutir estos puntos con su director de tesis.*
