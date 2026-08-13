# Mapa Argumental Completo de la Tesis

**Proyecto:** Modelado estocástico de la deriva antigénica de Influenza A  
**Última actualización:** 26 de abril de 2026  

---

## 1. Tesis Central

**Formulación:** La evolución antigénica de Influenza A puede modelarse como una dinámica estocástica continua sobre el espacio latente de AntigenLM, produciendo distribuciones sobre trayectorias evolutivas posibles, con un drift que descompone las presiones selectivas en viabilidad biológica y escape inmunológico.

**En lenguaje llano:** En lugar de predecir "cuál será la cepa dominante del mes que viene" (una predicción puntual), modelamos "cómo se mueve el virus en el espacio de representaciones a lo largo del tiempo" (una distribución sobre caminos posibles).

**Condición necesaria para que la tesis se sostenga:** Las cuatro hipótesis siguientes deben cumplirse simultáneamente. Si H1 falla, la SDE no tiene soporte geométrico. Si H2 falla, el drift no tiene interpretación biológica. Si H3 falla, no hay contribución predictiva. Si H4 falla, las trayectorias no producen secuencias biológicas.

**Dependencia crítica:** H1 es la raíz de todo. Sin geometría latente adecuada, H2, H3 y H4 carecen de sentido. Por eso H1 se evalúa primero.

---

## 2. Hipótesis H1: El espacio latente tiene geometría suficiente

### Formulación

Las representaciones internas de AntigenLM codifican la estructura biológica de las cepas virales de forma geométricamente regular: cepas biológicamente similares están cerca, la variedad de datos tiene dimensión intrínseca manejable, y la decodificación es localmente suave.

### Supuestos

**S1.1 — Preservación de métrica biológica**

La distancia euclidiana (o coseno) entre embeddings refleja la distancia biológica entre las cepas correspondientes, medida por distancia de Hamming en aminoácidos de HA.

- Tipo de supuesto: Empírico, verificable.
- Justificación parcial: Hie et al. (2022) muestran que los espacios latentes de modelos de lenguaje de proteínas codifican direcciones evolutivas interpretables. Pero AntigenLM fue entrenado con next-token prediction, no con un objetivo de preservación métrica.
- Estado actual: ρ ≈ 0.13 (preliminar, sin separar subtipos, con distancia temporal como proxy). **No verificado correctamente aún.**
- Evidencia necesaria: ρ(Spearman) entre distancia latente y Hamming en HA, calculada por subtipo.
- Experimento: EXP-2 (Spearman por subtipo con Hamming real).
- Valida: ρ ≥ 0.30 por subtipo, con IC 95% que no cruce 0.15.
- Refuta: ρ < 0.10 por subtipo, o ρ < 0.15 para ambos subtipos.
- Zona gris: ρ ∈ [0.15, 0.30]. Justifica capa de proyección (escenario B).

**S1.2 — Suavidad local (Lipschitz)**

Pequeños desplazamientos en el espacio latente producen pequeños cambios en la secuencia decodificada. Formalmente, el decoder es localmente Lipschitz.

- Tipo de supuesto: Empírico, verificable parcialmente.
- Justificación parcial: Los espacios latentes de redes profundas suelen ser localmente suaves, pero pueden tener discontinuidades en fronteras entre clusters.
- Estado actual: CV = 0.000 en interpolación. **Sospechoso de ser trivial** (posiblemente embeddings normalizados).
- Evidencia necesaria: (a) Que los embeddings no estén L2-normalizados. (b) Que la interpolación lineal produzca secuencias que varíen monótonamente en Hamming.
- Experimentos: EXP-1 (normalización), EXP-6 (interpolación con decodificación).
- Valida: Interpolaciones monótonas en ≥ 70% de los pares, con secuencias biológicamente plausibles en los puntos intermedios.
- Refuta: Interpolaciones con saltos abruptos en > 50% de los pares, o secuencias intermedias masivamente inválidas.

**S1.3 — Dimensión intrínseca finita y manejable**

Los embeddings viven en (o cerca de) una variedad de dimensión mucho menor que 384.

- Tipo de supuesto: Empírico, verificable.
- Justificación: Los datos biológicos tienen estructura de baja dimensión (el número de grados de libertad evolutivos es finito y probablemente del orden de decenas, no cientos).
- Estado actual: TwoNN tiene un error técnico pendiente. No verificado.
- Evidencia necesaria: Estimación de dimensión intrínseca consistente entre dos estimadores.
- Experimentos: EXP-4 (PCA), EXP-5 (TwoNN + MLE).
- Valida: d_intrínseca ∈ [10, 50] con TwoNN y MLE consistentes (diferencia < 30%).
- Refuta: d_intrínseca > 150, o TwoNN y MLE muy discrepantes (diferencia > 100%), lo que indicaría que la distribución de embeddings no es localmente uniforme.
- Implicación para la SDE: Si d ≤ 50, la SDE opera en un espacio de dimensión manejable. Si d > 100, se necesita proyección PCA previa.

### Diagrama de dependencia de H1

```
H1 (Geometría suficiente)
├── S1.1 (Métrica) ← EXP-1 (normalización) + EXP-2 (Spearman) + EXP-3 (métricas)
├── S1.2 (Suavidad) ← EXP-1 (normalización) + EXP-6 (interpolación)
└── S1.3 (Dimensión) ← EXP-4 (PCA) + EXP-5 (TwoNN/MLE)
```

### Si H1 es refutada: redacción académica

> Nuestros resultados muestran que el espacio latente de AntigenLM, entrenado con un objetivo de next-token prediction sobre secuencias genómicas, no preserva la estructura métrica biológica en grado suficiente para soportar directamente una dinámica estocástica euclidiana. La correlación de Spearman entre distancia latente y distancia de Hamming en aminoácidos de HA es ρ = X.XX para H3N2 y ρ = X.XX para H1N1, indicando que la geometría euclidiana del espacio latente no es un proxy adecuado de la distancia inmunológica.
>
> Este resultado tiene implicaciones más allá de este proyecto: sugiere que los modelos de lenguaje biológico entrenados con objetivos autorregresivos no producen, sin regularización geométrica adicional, espacios latentes con las propiedades necesarias para dinámicas continuas. Esto contrasta con los autoencoders variacionales, donde la regularización KL induce mayor suavidad, y con enfoques recientes como FlatVI (Palma et al., 2025), que imponen explícitamente geometría euclidiana mediante pérdidas adicionales.
>
> Proponemos que futuros modelos de lenguaje viral destinados a soportar dinámicas en espacio latente incorporen una de las siguientes estrategias: (i) pérdidas auxiliares de preservación de rango basadas en distancias biológicas; (ii) una capa de proyección post-hoc entrenada con triplet loss sobre distancias de Hamming; o (iii) regularización geométrica tipo métrica pullback.

---

## 3. Hipótesis H2: El drift captura las presiones selectivas reales

### Formulación

La descomposición μ = α∇F_viab + β∇F_escape modela adecuadamente las dos presiones selectivas principales sobre Influenza A: mantener funcionalidad biológica y evadir inmunidad preexistente.

### Supuestos

**S2.1 — F_viab captura viabilidad biológica**

La log-verosimilitud del decoder de AntigenLM, evaluada en un punto z del espacio latente, es un proxy válido de la viabilidad biológica de la cepa correspondiente.

- Tipo de supuesto: Fuerte, parcialmente verificable.
- Justificación: Un modelo de lenguaje entrenado sobre secuencias virales reales asigna mayor verosimilitud a secuencias que se parecen a secuencias reales. Esto es un proxy razonable (aunque imperfecto) de funcionalidad biológica.
- Riesgo principal: Adversarial examples. Puntos que maximizan la log-verosimilitud del decoder podrían no corresponder a proteínas funcionales sino a artefactos del modelo.
- Evidencia necesaria: (a) Las secuencias con alta log-verosimilitud de decoder son biológicamente plausibles. (b) Las secuencias con baja log-verosimilitud incluyen mutaciones deletéreas conocidas.
- Experimento: EXP-8 (evaluación del landscape de F_viab).
- Valida: Las cepas reales tienen F_viab significativamente mayor que secuencias aleatorias, y las cepas que circularon ampliamente tienen F_viab mayor que cepas raras o defectuosas.
- Refuta: No hay correlación entre F_viab y circulación/fitness observado, o la maximización de F_viab produce secuencias claramente no biológicas.

**S2.2 — F_escape modela presión inmunológica**

Un kernel gaussiano repulsivo centrado en cepas históricas recientes es un proxy válido de la presión de escape inmunológico.

- Tipo de supuesto: Fuerte, parcialmente verificable.
- Justificación: La presión inmune empuja al virus lejos de las cepas a las cuales la población tiene anticuerpos. Esto es conceptualmente consistente con un campo repulsivo en el espacio de representaciones. Łuksza y Lässig (2014) usan una formulación similar en espacio de secuencias.
- Riesgos: (a) La presión inmune real actúa sobre epítopos específicos, no sobre la secuencia completa. (b) La composición de H_t (qué cepas incluir, qué pesos temporales) es arbitraria. (c) El ancho del kernel σ_k determina el rango de la presión inmune y es un hiperparámetro sensible.
- Evidencia necesaria: Las cepas que efectivamente escaparon la inmunidad (medido por datos serológicos o HI assays) están en la dirección de ∇F_escape desde sus predecesoras.
- Experimento: Parte de EXP-8 + análisis retrospectivo del campo vectorial.
- Valida: La dirección de ∇F_escape en cepas del año t predice cualitativamente la dirección de la evolución observada en t+1.
- Refuta: ∇F_escape apunta en direcciones ortogonales a la evolución observada, o no hay correlación entre la magnitud de ∇F_escape y la tasa de cambio antigénico observada.

**S2.3 — Los gradientes de F_viab y F_escape son informativos en el espacio latente**

Los gradientes euclidianos de los funcionales, calculados en el espacio latente de AntigenLM, apuntan en direcciones biológicamente significativas.

- Tipo de supuesto: Fuerte, depende de H1 (S1.1 especificamente).
- Justificación: Si S1.1 es verdadero (distancia euclidiana ≈ distancia biológica), entonces los gradientes euclidianos tienen correspondencia biológica. Si S1.1 es falso, los gradientes podrían ser informativos computacionalmente pero sin interpretación biológica.
- Riesgo principal: Los gradientes de redes neuronales profundas son ruidosos y su magnitud varía en órdenes de magnitud entre regiones del espacio.
- Evidencia necesaria: (a) La norma de ∇F_viab y ∇F_escape está acotada razonablemente (sin explosiones). (b) La dirección del gradiente es estable bajo pequeñas perturbaciones del punto de evaluación. (c) El gradiente de un par de pasos consecutivos (z_t → z_{t+1} observado vs. dirección de drift) es positivo.
- Experimento: Parte de EXP-8.
- Valida: Correlación positiva entre la dirección del drift predicho y el desplazamiento observado en > 60% de las transiciones mensuales.
- Refuta: Correlación nula o negativa, o normas de gradiente con varianza > 2 órdenes de magnitud.

### Dependencia crítica

S2.3 depende de S1.1. Si la métrica euclidiana no preserva distancia biológica, los gradientes euclidianos no tienen interpretación biológica. Este es el canal principal por el cual H1 condiciona H2.

### Si H2 es refutada: redacción académica

> La descomposición del drift en componentes de viabilidad y escape inmunológico, si bien conceptualmente motivada por la biología evolutiva de Influenza, no produce campos vectoriales que correlacionen significativamente con la evolución antigénica observada. El gradiente de F_viab (definido como la log-verosimilitud del decoder) exhibe [alta varianza / direcciones ruidosas / atracción hacia artefactos del modelo], lo que sugiere que la log-verosimilitud autoregresiva es un proxy insuficiente de la viabilidad biológica cuando se evalúa como campo vectorial.
>
> Estos hallazgos motivan la exploración de funcionales de viabilidad alternativos, tales como: (i) viabilidad aprendida a partir de datos experimentales de fitness viral (DMS data); (ii) predictores de estabilidad de proteínas como Rosetta o ESMFold; o (iii) funcionales basados en conservación filogenética en lugar de verosimilitud de secuencia.
>
> No obstante, la formulación matemática del drift como suma ponderada de presiones selectivas permanece válida como marco general. El problema reside en la operacionalización de los funcionales, no en la estructura de la ecuación.

---

## 4. Hipótesis H3: La predicción distribucional tiene valor

### Formulación

Modelar la evolución antigénica como una distribución sobre trayectorias produce predicciones con mejor calibración, incertidumbre informativa y extensibilidad a horizontes largos, comparado con las predicciones puntuales de AntigenLM.

### Supuestos

**S3.1 — La distribución predicha es calibrada**

Los intervalos de credibilidad derivados de la distribución sobre trayectorias contienen la cepa observada con la frecuencia esperada. Es decir, el intervalo al 90% contiene la cepa real en ~90% de los meses.

- Tipo de supuesto: Empírico, verificable ex post.
- Justificación: La SDE genera múltiples trayectorias; la distribución empírica de sus endpoints constituye una distribución predictiva. Si la SDE está bien especificada, esta distribución debería estar calibrada.
- Riesgo principal: La SDE podría estar sobre-confiada (intervalos demasiado estrechos) o sub-confiada (intervalos tan amplios que son inútiles).
- Evidencia necesaria: Coverage@90 ∈ [0.80, 0.95] en validación retrospectiva.
- Experimento: EXP-9 (validación retrospectiva y prospectiva).
- Valida: Coverage@90 ∈ [0.80, 0.95] en ambas validaciones.
- Refuta: Coverage@90 < 0.50 (sub-calibración severa) o Coverage@90 > 0.99 con intervalos enormes (la predicción no discrimina).

**S3.2 — La SDE se extiende naturalmente a horizontes largos**

Integrar la SDE por k pasos en lugar de 1 produce predicciones razonables a horizontes de 3, 6 y 12 meses, degradándose gradualmente (no catastróficamente).

- Tipo de supuesto: Empírico, verificable.
- Justificación: Las ODEs y SDEs son integrables a horizontes arbitrarios por construcción matemática. Pero la calidad de la predicción depende de que el drift sea estable y de que los errores no se acumulen catastróficamente.
- Riesgo principal: Los errores del drift se acumulan multiplicativamente. A horizonte 12 meses, la predicción podría ser puro ruido.
- Evidencia necesaria: AAM(horizonte k) crece sub-linealmente con k (o al menos más lento que una caminata aleatoria).
- Experimento: EXP-9 con horizontes 1, 3, 6, 12.
- Valida: AAM(SDE, k=6) < AAM(persistencia, k=6) y AAM(SDE, k=12) < AAM(caminata aleatoria, k=12).
- Refuta: AAM(SDE, k=3) > AAM(persistencia, k=3). Si la SDE pierde contra persistencia a horizonte 3 meses, no tiene valor predictivo a horizontes extendidos.

### Dependencia crítica

H3 depende de H1 (la geometría debe ser adecuada para que las trayectorias tengan sentido) y de H2 (el drift debe ser informativo para que la distribución tenga estructura, no sea solo ruido).

### Si H3 es refutada: redacción académica

> La distribución sobre trayectorias generada por la SDE no mejora las predicciones puntuales de AntigenLM en ninguno de los horizontes evaluados. El AAM mediano de la SDE es X.XX ± X.XX comparado con X.XX ± X.XX de AntigenLM a horizonte de 1 mes, y la diferencia no es estadísticamente significativa (p = X.XX, test de Wilcoxon pareado).
>
> Sin embargo, el análisis de calibración revela que la incertidumbre predicha por la SDE es [informativa/no informativa]: el coverage al 90% es X.XX%, lo que indica que [la distribución captura la variabilidad real / la distribución está severamente descalibrada].
>
> Estos resultados sugieren que, para la predicción antigénica de Influenza A, la ganancia principal del enfoque distribucional no reside en mejorar la predicción puntual mediana sino en [cuantificar la incertidumbre / identificar bifurcaciones evolutivas / alertar sobre periodos de alta impredecibilidad]. La SDE identifica correctamente los meses donde la evolución es más impredecible (σ_predicho alto correlaciona con error de AntigenLM, ρ = X.XX), lo que constituye una contribución de valor práctico para la vigilancia epidemiológica.

**Nota:** Esta redacción pivota de "mejoramos la predicción" a "aportamos información complementaria sobre incertidumbre". Es una retirada honesta pero constructiva.

---

## 5. Hipótesis H4: La decodificación produce secuencias válidas

### Formulación

Los puntos del espacio latente visitados por las trayectorias de la SDE pueden decodificarse a secuencias de aminoácidos biológicamente plausibles usando el decoder de AntigenLM.

### Supuestos

**S4.1 — El decoder es robusto a perturbaciones pequeñas**

Puntos cercanos a embeddings reales (dentro de un ε-vecindario) producen secuencias válidas y similares a la cepa original.

- Tipo de supuesto: Empírico, verificable.
- Justificación: Las redes neuronales son generalmente continuas, pero la decodificación autoregresiva puede amplificar perturbaciones (un token erróneo cambia todo el contexto para los tokens siguientes).
- Estado actual: No verificado.
- Evidencia necesaria: Fracción de secuencias válidas vs. magnitud de perturbación ε.
- Experimento: EXP-7 (decoder OOD).
- Valida: ≥ 80% de secuencias válidas a ε = 0.5 (escala de la norma del paso de la SDE).
- Refuta: < 50% de secuencias válidas a ε = 0.1 (el decoder es demasiado frágil para soportar cualquier trayectoria continua).
- Zona gris: 50-80% válidas. La SDE necesita constricción para no salir del dominio de validez.

### Dependencia

H4 es relativamente independiente de H1-H3. El decoder es una propiedad de AntigenLM que puede evaluarse directamente. Sin embargo, si H4 falla, las hipótesis H2 y H3 pierden su output observable: no se pueden evaluar predicciones en espacio de secuencias.

### Si H4 es refutada: redacción académica

> Las trayectorias generadas por la SDE, al decodificarse mediante el modelo autoregresivo de AntigenLM, producen secuencias biológicamente inválidas en un X.XX% de los pasos de integración. El análisis de robustez del decoder muestra que perturbaciones de magnitud ε ≥ X.XX producen secuencias con stop codons prematuros, aminoácidos fuera del vocabulario biológico, o longitudes inconsistentes con las proteínas de superficie de Influenza A.
>
> Este resultado no invalida el marco dinámico propuesto, sino que identifica una limitación del modelo base: AntigenLM fue diseñado para generar secuencias a partir de un prompt discreto, no para decodificar puntos arbitrarios de su espacio latente continuo.
>
> Proponemos dos direcciones para resolver esta limitación: (i) evaluar la SDE exclusivamente en espacio latente, usando métricas de distancia a la cepa observada sin decodificación explícita (lo que preserva la contribución de modelado dinámico); (ii) entrenar un decoder auxiliar con pérdida de reconstrucción sobre pares (z, secuencia), lo que aportaría robustez a la decodificación de puntos fuera de distribución.
>
> Mientras tanto, reportamos las métricas en espacio latente como resultados primarios, con las secuencias decodificadas como resultados secundarios sujetos a la validación del decoder.

---

## 6. Tabla de Resumen: Cadena Completa

| Hipótesis | Supuesto | Experimento | Resultado que valida | Resultado que refuta | ¿Bloqueante? |
|---|---|---|---|---|---|
| H1 | S1.1 Métrica | EXP-2: Spearman/Hamming | ρ ≥ 0.30 por subtipo | ρ < 0.10 por subtipo | **Sí** |
| H1 | S1.1 Métrica | EXP-1: Normalización | CV norma ≥ 0.05 | CV norma < 0.01 → usar coseno | Condicional |
| H1 | S1.1 Métrica | EXP-3: Comparación métricas | Alguna métrica alcanza ρ ≥ 0.30 | Ninguna métrica supera ρ = 0.15 | **Sí** |
| H1 | S1.2 Suavidad | EXP-6: Interpolación | ≥ 70% interpolaciones monótonas | > 50% con saltos abruptos | Parcial |
| H1 | S1.3 Dimensión | EXP-4 + EXP-5: PCA, TwoNN, MLE | d ∈ [10, 50] | d > 150 | Parcial |
| H2 | S2.1 F_viab | EXP-8: Landscape de viabilidad | Cepas reales > aleatorias en F_viab | Sin correlación F_viab vs. circulación | No |
| H2 | S2.2 F_escape | EXP-8: Campo vectorial | ∇F_escape predice dirección evolutiva | Ortogonal a evolución observada | No |
| H2 | S2.3 Gradientes | EXP-8: Estabilidad de gradientes | Norma acotada, dirección estable | Explosiones de norma, ruido dominante | No |
| H3 | S3.1 Calibración | EXP-9: Validación retrospectiva | Coverage@90 ∈ [0.80, 0.95] | Coverage@90 < 0.50 | No |
| H3 | S3.2 Horizonte largo | EXP-9: Horizontes 1,3,6,12 | AAM(SDE, k=6) < AAM(persist., k=6) | AAM(SDE, k=3) > AAM(persist., k=3) | No |
| H4 | S4.1 Decoder | EXP-7: Decoder OOD | ≥ 80% válidas a ε=0.5 | < 50% válidas a ε=0.1 | Parcial |

### Leyenda de "¿Bloqueante?"

- **Sí**: Si este supuesto falla, la tesis debe pivotar de escenario. No se puede proceder con la SDE euclidiana directa.
- **Parcial**: Si falla, se necesita una adaptación (proyección, constricción, cambio de métrica), pero la tesis sigue siendo viable.
- **No**: Si falla, se pierde una contribución específica, pero la tesis tiene suficiente contenido con las demás.

---

## 7. Cadena de Implicaciones Lógicas

### Cadena positiva (todo funciona)

```
S1.1 ✓ (ρ ≥ 0.30)
  → La distancia euclidiana/coseno es proxy de distancia biológica
    → F_escape tiene interpretación inmunológica (S2.2)
      → ∇F_escape apunta en dirección de escape real (S2.3)
        → El drift combinado genera trayectorias biológicamente plausibles
          → La distribución sobre trayectorias tiene calibración razonable (S3.1)
            → La validación prospectiva 2022-2026 confirma (escenario A)
              → Paper en Bioinformatics / PLOS Comp Bio
```

### Cadena parcial (geometría insuficiente, corregible)

```
S1.1 ✗ (ρ ∈ [0.10, 0.30])
  → La distancia euclidiana cruda no es buen proxy
    → Se entrena capa de proyección g: ℝ^384 → ℝ^d (escenario B)
      → Se recalcula ρ en espacio proyectado
        SI ρ_proyectado ≥ 0.30:
          → Se aplica la SDE en espacio proyectado
            → Doble contribución: diagnóstico geométrico + SDE sobre espacio corregido
              → Paper en JCB / workshop NeurIPS
        SI ρ_proyectado < 0.30:
          → La estructura biológica no se preserva ni con proyección
            → Pivotar a escenario C
```

### Cadena negativa (geometría inadecuada)

```
S1.1 ✗ (ρ < 0.10)
  → El espacio latente de AntigenLM no preserva métrica biológica
    → F_escape no tiene interpretación inmunológica
      → La SDE no tiene fundamento biológico en este espacio
        → Pivotar a escenario C
          → Contribución: primera auditoría geométrica de un LM viral
            → Recomendaciones de diseño para futuros modelos
              → Paper en workshop NeurIPS/ICML (resultado negativo valioso)
```

---

## 8. Mapa de "Qué necesito para cada afirmación"

Cada afirmación que la tesis hace necesita evidencia específica. Si alguna evidencia falta, la afirmación se debilita.

### Afirmación A: "El espacio latente de AntigenLM codifica estructura evolutiva"

| Evidencia | Fuente | Estado |
|---|---|---|
| Clusters por subtipo en UMAP/PCA | EXP-4 | Preliminar (UMAP hecho) |
| Separación temporal dentro de subtipo | EXP-4 | Pendiente (PCA no hecho) |
| ρ(Spearman) > 0.30 por subtipo | EXP-2 | **Pendiente (crítico)** |
| Dimensión intrínseca finita | EXP-5 | Pendiente (error en TwoNN) |

### Afirmación B: "La SDE modela las presiones selectivas sobre Influenza"

| Evidencia | Fuente | Estado |
|---|---|---|
| F_viab discrimina cepas reales de aleatorias | EXP-8 | No iniciado |
| ∇F_escape apunta en dirección de evolución observada | EXP-8 | No iniciado |
| α y β aprendidos son estables entre splits | EXP-9 | No iniciado |
| Ablación α=0 empeora predicción | EXP-9 | No iniciado |
| Ablación β=0 empeora predicción | EXP-9 | No iniciado |

### Afirmación C: "La predicción distribucional supera a la puntual"

| Evidencia | Fuente | Estado |
|---|---|---|
| AAM(SDE) ≤ AAM(AntigenLM) a horizonte 1 mes | EXP-9 | No iniciado |
| AAM(SDE) < AAM(AntigenLM) a horizontes > 3 meses | EXP-9 | No iniciado |
| Coverage@90 ∈ [0.80, 0.95] | EXP-9 | No iniciado |
| CRPS(SDE) < CRPS(persistencia) | EXP-9 | No iniciado |
| Validación prospectiva 2022-2026 confirma | EXP-9 | No iniciado |

### Afirmación D: "Las trayectorias se decodifican a secuencias válidas"

| Evidencia | Fuente | Estado |
|---|---|---|
| ≥ 80% secuencias válidas a ε=0.5 | EXP-7 | No iniciado |
| Hamming(decodificada, referencia) crece con ε | EXP-7 | No iniciado |
| Interpolaciones producen secuencias plausibles | EXP-6 | No iniciado |

---

## 9. Escenarios de Resultado y Redacción Asociada

### Escenario Óptimo: Todo funciona

**Título de tesis:** "Modelado estocástico de la deriva antigénica de Influenza A: distribuciones sobre trayectorias evolutivas en el espacio latente de AntigenLM"

**Contribución principal:** Un marco probabilístico que produce distribuciones calibradas sobre trayectorias antigénicas, superando las predicciones puntuales a horizontes de 3+ meses.

**Narrativa:** "Demostramos que la evolución antigénica de Influenza A puede modelarse como una SDE cuyo drift descompone las presiones selectivas en viabilidad biológica (α) y escape inmunológico (β). El modelo produce distribuciones calibradas sobre trayectorias, con incertidumbre cuantificada, y supera las predicciones puntuales de AntigenLM en horizontes de 3 a 12 meses. La validación prospectiva 2022-2026 confirma estos resultados."

### Escenario Intermedio: Geometría corregible

**Título de tesis:** "Auditoría geométrica y dinámica estocástica sobre el espacio latente corregido de un modelo de lenguaje viral"

**Contribución principal:** (1) Primera auditoría geométrica de un LM viral, que identifica limitaciones de los espacios latentes autorregresivos. (2) Una capa de proyección que corrige la geometría. (3) SDE sobre el espacio corregido.

**Narrativa:** "Mostramos que el espacio latente de AntigenLM no preserva la métrica biológica de forma directa (ρ = X.XX). Proponemos una capa de proyección entrenada con pérdida de preservación de rango que eleva la correlación a ρ = X.XX. Sobre el espacio corregido, formulamos una SDE cuyos resultados [describirlos]."

### Escenario Mínimo: La geometría no soporta la SDE

**Título de tesis:** "¿Pueden los modelos de lenguaje viral soportar dinámicas continuas? Auditoría geométrica del espacio latente de AntigenLM"

**Contribución principal:** La primera evaluación sistemática de las propiedades geométricas de un espacio latente de un modelo de lenguaje viral, con recomendaciones para el diseño de modelos que soporten dinámicas continuas.

**Narrativa:** "Evaluamos si el espacio latente de AntigenLM, un modelo de lenguaje tipo GPT para Influenza A, satisface las condiciones geométricas necesarias para definir una dinámica estocástica continua. Encontramos que [enunciar hallazgos]. Formulamos las condiciones matemáticas que un espacio latente debe cumplir para soportar SDEs, y proponemos estrategias de regularización geométrica para futuros modelos."

**Importante para este escenario:** La tesis debe articularse como una contribución de infraestructura y diagnóstico, no como un "fracaso". El resultado negativo bien documentado tiene valor real: evita que otros investigadores cometan el mismo error, e identifica exactamente qué debe cambiar.

---

## 10. Supuestos No Explicitados (Deuda Argumental)

Hay supuestos que el documento de tesis actual no menciona pero que son necesarios para la cadena argumental. Deben explicitarse en el capítulo de metodología:

**SA — El encoder de AntigenLM produce la misma representación para secuencias biológicamente equivalentes.**
Si dos secuencias difieren en regiones no antigénicas (por ejemplo, diferencias sinónimas), sus embeddings deberían ser cercanos. Si no lo son, la distancia latente incluye ruido no biológico. Verificable con EXP-2 (el Spearman captura indirectamente esto).

**SB — La concatenación HA+NA como input no distorsiona la representación de HA.**
AntigenLM recibe HA+NA concatenados. Si las mutaciones en NA dominan el embedding, la geometría latente no refleja las presiones antigénicas que actúan sobre HA. Verificable separando Spearman para Hamming en HA solo vs. Hamming en HA+NA.

**SC — La cepa "dominante" mensual es representativa de la distribución de cepas circulantes.**
Si la cepa dominante de un mes es un outlier estadístico dentro de la distribución de cepas de ese mes, predecirla correctamente no implica predecir la evolución general. Este supuesto es heredado de AntigenLM.

**SD — La historia inmune H_t es bien aproximada por centroides mensuales de cepas observadas.**
La realidad es que la inmunidad poblacional es heterogénea (diferentes personas tienen diferentes historiales de infección/vacunación). Usar centroides mensuales es una aproximación de campo medio. Este supuesto es estándar en el campo pero debería mencionarse explícitamente.

**SE — El ruido browniano es una buena aproximación de la estocasticidad mutacional.**
Las mutaciones reales son discretas, raras y con efectos epistáticos. El ruido browniano asume perturbaciones continuas e independientes. Esto es una aproximación de campo medio clásica en genética de poblaciones (cf. ecuación de difusión de Kimura), pero merece justificación en el texto.

**SF — Congelar AntigenLM no pierde información necesaria para la dinámica.**
Es posible que un fine-tuning adaptativo del encoder mejoraría las representaciones para la tarea específica de dinámica evolutiva. Al congelarlo, se sacrifica esta posibilidad por simplicidad.

---

## 11. Árbol de Decisión para la Tesis Completa

```
¿EXP-1: Embeddings normalizados?
├── Sí → Usar coseno en todo. CV=0 es trivial, no evidencia de suavidad.
│   ├── EXP-2 con coseno: ¿ρ ≥ 0.30?
│   │   ├── Sí → ESCENARIO A. Proceder con SDE coseno.
│   │   │   ├── EXP-8: ¿Drift aprende algo?
│   │   │   │   ├── Sí → Entrenar SDE completa.
│   │   │   │   │   ├── EXP-9: ¿Supera baselines?
│   │   │   │   │   │   ├── Sí → TESIS ÓPTIMA. Paper journal.
│   │   │   │   │   │   └── No → Marco probabilístico + incertidumbre como contribución.
│   │   │   │   └── No → Drift lineal aprendido, no basado en funcionales.
│   │   │   └── EXP-7: ¿Decoder robusto?
│   │   │       ├── Sí → Reportar secuencias decodificadas.
│   │   │       └── No → Métricas solo en espacio latente.
│   │   ├── 0.20 ≤ ρ < 0.30 → ESCENARIO B. Capa de proyección.
│   │   │   ├── Entrenar proyección → Recalcular ρ
│   │   │   │   ├── ρ_proy ≥ 0.30 → Proceder con SDE sobre proyección.
│   │   │   │   └── ρ_proy < 0.30 → ESCENARIO C.
│   │   └── ρ < 0.20 → ESCENARIO C. Auditoría geométrica como contribución.
└── No → Euclidiana es informativa. Repetir EXP-2 con euclidiana.
    └── (misma ramificación que arriba, con euclidiana)
```

---

*Mapa argumental generado el 26 de abril de 2026. Cada nodo debe verificarse empíricamente en el orden especificado en el plan técnico de la semana 1.*
