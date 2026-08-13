# Plan Técnico de Ejecución — Semana 1

**Proyecto:** Auditoría geométrica del espacio latente de AntigenLM  
**Objetivo:** Determinar en 7 días si el espacio latente soporta una SDE euclidiana  
**Autor:** Carlos Manuel Orrego Franco  
**Fecha:** 26 de abril de 2026  

---

## 0. Arquitectura de archivos

Antes de escribir una línea de código, crea esta estructura:

```
thesis/
├── config.py                  # Constantes globales, paths, seeds
├── data/
│   ├── raw/                   # GISAID, NCBI sin procesar
│   ├── processed/             # Embeddings extraídos, distancias precalculadas
│   └── splits/                # Train ≤2022, prospectivo 2022-2026
├── src/
│   ├── embedding_extractor.py # Extraer z_t de AntigenLM
│   ├── distances.py           # Hamming, euclidiana, coseno
│   ├── geometry/
│   │   ├── normalization_check.py   # EXP-1
│   │   ├── spearman_analysis.py     # EXP-2
│   │   ├── metric_comparison.py     # EXP-3
│   │   ├── pca_analysis.py          # EXP-4
│   │   ├── intrinsic_dim.py         # EXP-5
│   │   ├── interpolation.py         # EXP-6
│   │   └── decoder_validity.py      # EXP-7
│   └── utils.py
├── notebooks/
│   └── 01_geometry_dashboard.ipynb  # Visualización integrada
├── results/
│   ├── tables/
│   └── figures/
├── logs/
│   └── decision_log.md       # Registro fechado de decisiones
└── tests/
    └── test_distances.py
```

`config.py` debe contener:

```python
SEED = 42
EMBEDDING_DIM = 384
SUBTYPES = ["H3N2", "H1N1"]
DATA_CUTOFF_TRAIN = "2022-01-01"
DATA_CUTOFF_PROSPECTIVE_START = "2022-01-01"
DATA_CUTOFF_PROSPECTIVE_END = "2026-04-01"
SPEARMAN_THRESHOLD_GO = 0.30
SPEARMAN_THRESHOLD_MARGINAL = 0.20
SPEARMAN_THRESHOLD_STOP = 0.10
N_INTERPOLATION_PAIRS = 50
N_BOOTSTRAP = 1000
```

**Regla fundamental:** todo script debe fijar la semilla aleatoria al inicio, guardar resultados en `results/`, y loggear decisiones en `decision_log.md` con timestamp.

---

## 1. Checklist de Experimentos

| ID | Experimento | Tiempo estimado | Dependencias | Prioridad |
|---|---|---|---|---|
| EXP-0 | Extracción de embeddings con mean pooling | 3-4 horas | Pesos AntigenLM cargados | Bloqueante |
| EXP-1 | Verificación de normalización | 30 min | EXP-0 | Crítica |
| EXP-2 | Spearman por subtipo con Hamming real en HA | 3-4 horas | EXP-0 | Crítica |
| EXP-3 | Comparación euclidiana vs. coseno vs. Hamming | 1-2 horas | EXP-0, EXP-2 | Alta |
| EXP-4 | PCA y varianza explicada acumulada | 1 hora | EXP-0 | Alta |
| EXP-5 | Dimensión intrínseca (TwoNN + MLE) | 2-3 horas | EXP-0 | Alta |
| EXP-6 | Interpolación con decodificación | 3-4 horas | EXP-0, EXP-1 | Media |
| EXP-7 | Validez del decoder en puntos fuera de distribución | 2-3 horas | EXP-0, EXP-6 | Media |
| DEC-1 | Decisión de escenario (A/B/C) | 1 hora | EXP-1 a EXP-5 | Bloqueante |

---

## 2. Orden de Ejecución

```
Día 1 (lunes):    EXP-0 → EXP-1 → decisión euclidiana vs. coseno
Día 2 (martes):   EXP-2 (completo, ambos subtipos)
Día 3 (miércoles): EXP-3 → EXP-4
Día 4 (jueves):   EXP-5 (TwoNN corregido + MLE)
Día 5 (viernes):  EXP-6 → EXP-7
Día 6 (sábado):   DEC-1 + dashboard integrado + redacción de resultados
Día 7 (domingo):  Congelar decisiones metodológicas para validación prospectiva
```

La lógica del orden: EXP-1 es trivial y determina qué métrica usar en todo lo demás. EXP-2 es el experimento más importante de toda la tesis. EXP-3 y EXP-4 complementan. EXP-5 determina viabilidad computacional de la SDE. EXP-6 y EXP-7 evalúan la cadena completa embeddings → decodificación. DEC-1 sintetiza todo.

---

## 3. Pseudocódigo Detallado

### EXP-0: Extracción de embeddings

Este experimento es el prerequisito de todos los demás. Su objetivo es obtener un embedding z_i ∈ ℝ^384 para cada cepa del dataset, usando mean pooling en lugar del último token.

```
FUNCIÓN extraer_embeddings(model, tokenizer, sequences, batch_size=32):
    """
    IMPORTANTE: Usar mean pooling, NO el último token.
    El último token captura información local (final de secuencia).
    Mean pooling captura información global (toda la secuencia).
    Para geometría del espacio latente, necesitamos información global.
    """
    
    model.eval()
    model.requires_grad_(False)
    
    embeddings = []
    metadata = []   # subtipo, año, mes, cepa_id
    
    PARA cada batch en chunks(sequences, batch_size):
        tokens = tokenizer.encode_batch(batch.ha_sequences)
        
        # Forward pass: obtener hidden states de la última capa
        CON torch.no_grad():
            outputs = model(tokens, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # [B, seq_len, 384]
        
        # Mean pooling sobre tokens (excluyendo padding)
        attention_mask = (tokens != tokenizer.pad_token_id)
        mask_expanded = attention_mask.unsqueeze(-1)  # [B, seq_len, 1]
        sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1)
        mean_pooled = sum_hidden / count  # [B, 384]
        
        embeddings.append(mean_pooled.cpu().numpy())
        metadata.append(batch.metadata)
    
    embeddings = np.concatenate(embeddings)  # [N, 384]
    metadata = pd.concat(metadata)
    
    # Guardar con metadata
    np.save("data/processed/embeddings_mean_pool.npy", embeddings)
    metadata.to_parquet("data/processed/embeddings_metadata.parquet")
    
    # Sanity checks
    ASSERT embeddings.shape[1] == 384
    ASSERT no hay NaN ni Inf
    PRINT f"Extraídos {len(embeddings)} embeddings"
    PRINT f"  H3N2: {sum(metadata.subtype == 'H3N2')}"
    PRINT f"  H1N1: {sum(metadata.subtype == 'H1N1')}"
    
    RETORNAR embeddings, metadata
```

**Nota crítica sobre la secuencia de entrada:** Verificar que se está pasando la secuencia HA completa, no truncada. Si AntigenLM concatena HA+NA, verificar qué parte corresponde a HA y extraer embeddings de esa parte solamente para el cálculo de Hamming (que se hace sobre HA). Para los embeddings usados en la SDE, considerar HA+NA completa como contexto pero enfocarse en los tokens de HA para el pooling si el objetivo es predicción antigénica (que depende principalmente de HA).

**Decisión a documentar:** ¿Se usa mean pooling sobre HA solamente o sobre HA+NA? Justificar en `decision_log.md`.

---

### EXP-1: Verificación de normalización

```
FUNCIÓN verificar_normalizacion(embeddings):
    """
    Determina si los embeddings viven en una esfera.
    Si ||z_i||_2 ≈ constante para todo i, los embeddings están
    normalizados y:
      - CV ≈ 0 en interpolación es trivial (no informativo)
      - Distancia euclidiana es función monótona de distancia coseno
      - Se debe usar coseno como métrica base
    """
    
    normas = np.linalg.norm(embeddings, axis=1)  # [N]
    
    # Estadísticos
    media = np.mean(normas)
    std = np.std(normas)
    cv = std / media
    minimo = np.min(normas)
    maximo = np.max(normas)
    rango_relativo = (maximo - minimo) / media
    
    PRINT "=== EXP-1: Verificación de normalización ==="
    PRINT f"Media de ||z||: {media:.4f}"
    PRINT f"Std de ||z||:  {std:.4f}"
    PRINT f"CV:            {cv:.6f}"
    PRINT f"Rango:         [{minimo:.4f}, {maximo:.4f}]"
    PRINT f"Rango relativo: {rango_relativo:.4f}"
    
    # DECISIÓN
    SI cv < 0.01:
        PRINT "⚠️  EMBEDDINGS NORMALIZADOS (o casi). CV < 1%"
        PRINT "→ Distancia euclidiana es función de coseno."
        PRINT "→ CV = 0 en interpolación es trivial."
        PRINT "→ USAR DISTANCIA COSENO como métrica principal."
        usar_coseno = True
    SINO SI cv < 0.05:
        PRINT "⚠  Embeddings cuasi-normalizados. CV < 5%"
        PRINT "→ Reportar ambas métricas (euclidiana y coseno)."
        usar_coseno = True  # conservador
    SINO:
        PRINT "✓ Embeddings NO normalizados. CV ≥ 5%"
        PRINT "→ Distancia euclidiana es informativa."
        usar_coseno = False
    
    # FIGURA 1: Histograma de normas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(normas, bins=100, edgecolor='black', alpha=0.7)
    axes[0].axvline(media, color='red', linestyle='--', label=f'μ={media:.2f}')
    axes[0].set_xlabel("||z||₂")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribución de normas L2")
    axes[0].legend()
    
    # Norma vs. tiempo (para detectar drift temporal)
    axes[1].scatter(metadata.date, normas, s=1, alpha=0.3)
    axes[1].set_xlabel("Fecha")
    axes[1].set_ylabel("||z||₂")
    axes[1].set_title("Norma L2 vs. tiempo")
    
    plt.tight_layout()
    plt.savefig("results/figures/fig01_norma_embeddings.pdf", dpi=300)
    plt.savefig("results/figures/fig01_norma_embeddings.png", dpi=150)
    
    # TABLA 1
    tabla = {
        "Estadístico": ["N", "Media ||z||", "Std ||z||", "CV", 
                        "Min ||z||", "Max ||z||", "Rango relativo"],
        "Valor": [len(normas), f"{media:.4f}", f"{std:.4f}", 
                  f"{cv:.6f}", f"{minimo:.4f}", f"{maximo:.4f}", 
                  f"{rango_relativo:.4f}"]
    }
    pd.DataFrame(tabla).to_csv("results/tables/tab01_normalizacion.csv", index=False)
    
    RETORNAR usar_coseno, normas
```

---

### EXP-2: Spearman por subtipo con Hamming real en HA

Este es el experimento más importante de la semana. Si sale mal, todo cambia.

```
FUNCIÓN calcular_hamming_ha(seq_i, seq_j):
    """
    Distancia de Hamming normalizada sobre aminoácidos de HA.
    
    CRÍTICO: 
    - Usar secuencia de aminoácidos, no nucleótidos.
    - Si las secuencias tienen diferente longitud, alinear primero.
    - Normalizar por longitud de HA (~566 aa para H3, ~566 para H1).
    """
    ASSERT len(seq_i) == len(seq_j), "Secuencias deben estar alineadas"
    mismatches = sum(1 for a, b in zip(seq_i, seq_j) if a != b)
    RETORNAR mismatches / len(seq_i)


FUNCIÓN spearman_por_subtipo(embeddings, metadata, ha_sequences, usar_coseno):
    """
    Calcula correlación de Spearman entre distancia latente y distancia
    de Hamming en HA, separado por subtipo.
    
    ADVERTENCIA: No mezclar subtipos. La distancia inter-subtipo
    domina y produce correlaciones artificialmente altas.
    """
    
    resultados = {}
    
    PARA cada subtipo EN ["H3N2", "H1N1"]:
        # Filtrar por subtipo
        mask = metadata.subtype == subtipo
        emb_sub = embeddings[mask]
        seq_sub = ha_sequences[mask]
        n = len(emb_sub)
        
        PRINT f"\n=== {subtipo}: {n} cepas ==="
        
        # Submuestreo si n > 5000 (las matrices de distancia serían enormes)
        SI n > 5000:
            idx = np.random.choice(n, 5000, replace=False)
            emb_sub = emb_sub[idx]
            seq_sub = seq_sub[idx]
            n = 5000
            PRINT f"  Submuestreado a {n} cepas"
        
        # Matriz de distancias latentes
        SI usar_coseno:
            # Coseno: 1 - similaridad coseno
            from sklearn.metrics.pairwise import cosine_distances
            D_latente = cosine_distances(emb_sub)
        SINO:
            from scipy.spatial.distance import pdist, squareform
            D_latente = squareform(pdist(emb_sub, metric='euclidean'))
        
        # Matriz de distancias de Hamming en HA
        D_hamming = np.zeros((n, n))
        PARA i en range(n):
            PARA j en range(i+1, n):
                D_hamming[i,j] = calcular_hamming_ha(seq_sub[i], seq_sub[j])
                D_hamming[j,i] = D_hamming[i,j]
        
        # Extraer triángulo superior (evitar diagonal y duplicados)
        triu_idx = np.triu_indices(n, k=1)
        d_lat_vec = D_latente[triu_idx]
        d_ham_vec = D_hamming[triu_idx]
        
        # Spearman global
        rho, pval = scipy.stats.spearmanr(d_lat_vec, d_ham_vec)
        
        # Bootstrap para intervalo de confianza
        rhos_bootstrap = []
        PARA b en range(N_BOOTSTRAP):
            idx_boot = np.random.choice(len(d_lat_vec), len(d_lat_vec), replace=True)
            rho_b, _ = scipy.stats.spearmanr(d_lat_vec[idx_boot], d_ham_vec[idx_boot])
            rhos_bootstrap.append(rho_b)
        
        ci_low = np.percentile(rhos_bootstrap, 2.5)
        ci_high = np.percentile(rhos_bootstrap, 97.5)
        
        PRINT f"  ρ(Spearman) = {rho:.4f}  [{ci_low:.4f}, {ci_high:.4f}]"
        PRINT f"  p-value = {pval:.2e}"
        PRINT f"  N pares = {len(d_lat_vec)}"
        
        # Spearman local por año
        años = sorted(metadata[mask].year.unique())
        rho_por_año = []
        PARA año en años:
            mask_año = metadata[mask].year == año
            SI sum(mask_año) < 30:  # mínimo para Spearman significativo
                CONTINUAR
            emb_año = emb_sub[mask_año[:n]]  # ajustar por submuestreo
            # ... calcular Spearman local ...
            rho_por_año.append({"año": año, "rho": rho_local, "n": n_año})
        
        resultados[subtipo] = {
            "rho_global": rho,
            "pval": pval,
            "ci_95": (ci_low, ci_high),
            "n_cepas": n,
            "n_pares": len(d_lat_vec),
            "rho_por_año": rho_por_año
        }
        
        # FIGURA 2: Scatter plot distancia latente vs. Hamming
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter con densidad (hexbin para muchos puntos)
        axes[0].hexbin(d_ham_vec, d_lat_vec, gridsize=50, cmap='viridis',
                       mincnt=1)
        axes[0].set_xlabel("Distancia de Hamming (HA)")
        metric_name = "coseno" if usar_coseno else "euclidiana"
        axes[0].set_ylabel(f"Distancia {metric_name} (latente)")
        axes[0].set_title(f"{subtipo}: ρ = {rho:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
        plt.colorbar(axes[0].collections[0], ax=axes[0], label="Conteo")
        
        # ρ por año
        if rho_por_año:
            años_plot = [r["año"] for r in rho_por_año]
            rhos_plot = [r["rho"] for r in rho_por_año]
            axes[1].bar(años_plot, rhos_plot, color='steelblue', alpha=0.7)
            axes[1].axhline(0.3, color='green', linestyle='--', 
                           label='Umbral go (0.3)')
            axes[1].axhline(0.1, color='red', linestyle='--', 
                           label='Umbral stop (0.1)')
            axes[1].set_xlabel("Año")
            axes[1].set_ylabel("ρ(Spearman) local")
            axes[1].set_title(f"{subtipo}: correlación por año")
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"results/figures/fig02_spearman_{subtipo}.pdf", dpi=300)
        plt.savefig(f"results/figures/fig02_spearman_{subtipo}.png", dpi=150)
    
    # TABLA 2: Resumen de Spearman
    tabla_rows = []
    PARA subtipo, res en resultados.items():
        tabla_rows.append({
            "Subtipo": subtipo,
            "N cepas": res["n_cepas"],
            "N pares": res["n_pares"],
            "ρ global": f"{res['rho_global']:.4f}",
            "IC 95%": f"[{res['ci_95'][0]:.4f}, {res['ci_95'][1]:.4f}]",
            "p-value": f"{res['pval']:.2e}",
            "Decisión": interpretar_rho(res['rho_global'])
        })
    
    pd.DataFrame(tabla_rows).to_csv("results/tables/tab02_spearman.csv", index=False)
    
    RETORNAR resultados


FUNCIÓN interpretar_rho(rho):
    SI rho >= 0.50: RETORNAR "EXCELENTE → Escenario A directo"
    SI rho >= 0.30: RETORNAR "BUENO → Escenario A viable"
    SI rho >= 0.20: RETORNAR "MARGINAL → Escenario A con cautela, preparar B"
    SI rho >= 0.10: RETORNAR "DÉBIL → Escenario B necesario"
    RETORNAR "INSUFICIENTE → Escenario C, pivotar"
```

---

### EXP-3: Comparación de métricas

```
FUNCIÓN comparar_metricas(embeddings, ha_sequences, metadata):
    """
    Compara tres métricas de distancia latente contra Hamming HA
    para determinar cuál preserva mejor la estructura biológica.
    
    Métricas latentes: euclidiana, coseno, Mahalanobis (con cov empírica)
    Métrica biológica: Hamming normalizada en HA
    """
    
    metricas_latentes = {
        "Euclidiana": lambda X: squareform(pdist(X, 'euclidean')),
        "Coseno": lambda X: cosine_distances(X),
        "Correlación": lambda X: squareform(pdist(X, 'correlation')),
    }
    
    PARA cada subtipo EN ["H3N2", "H1N1"]:
        mask = metadata.subtype == subtipo
        emb_sub = embeddings[mask]
        seq_sub = ha_sequences[mask]
        
        # Submuestreo
        SI len(emb_sub) > 3000:
            idx = np.random.choice(len(emb_sub), 3000, replace=False)
            emb_sub, seq_sub = emb_sub[idx], seq_sub[idx]
        
        # Hamming
        D_ham = calcular_matriz_hamming(seq_sub)
        d_ham = D_ham[np.triu_indices(len(emb_sub), k=1)]
        
        resultados = {}
        PARA nombre, func en metricas_latentes.items():
            D_lat = func(emb_sub)
            d_lat = D_lat[np.triu_indices(len(emb_sub), k=1)]
            rho, p = spearmanr(d_lat, d_ham)
            resultados[nombre] = {"rho": rho, "p": p}
        
        # TABLA 3: Comparación de métricas
        # → Determina qué métrica usar en el resto del proyecto
    
    # La métrica ganadora se usa para EXP-6, EXP-7, y para F_escape
```

---

### EXP-4: PCA y varianza explicada

```
FUNCIÓN analisis_pca(embeddings, metadata):
    """
    PCA para entender la estructura dimensional del espacio latente.
    Dos objetivos:
    1. ¿Cuántos componentes capturan >90% y >95% de varianza?
    2. ¿Los primeros componentes separan subtipos/años?
    """
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Estandarizar (PCA es sensible a escala)
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    
    # PCA completo
    pca = PCA(n_components=min(384, len(embeddings)))
    Z_pca = pca.fit_transform(emb_scaled)
    
    varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
    
    # Dimensiones para 90% y 95%
    d_90 = np.argmax(varianza_acumulada >= 0.90) + 1
    d_95 = np.argmax(varianza_acumulada >= 0.95) + 1
    d_99 = np.argmax(varianza_acumulada >= 0.99) + 1
    
    PRINT f"Dimensiones para 90% varianza: {d_90}"
    PRINT f"Dimensiones para 95% varianza: {d_95}"
    PRINT f"Dimensiones para 99% varianza: {d_99}"
    PRINT f"Top-10 componentes: {varianza_acumulada[9]*100:.1f}% varianza"
    
    # FIGURA 3a: Varianza explicada acumulada (scree plot)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(range(1, len(varianza_acumulada)+1), varianza_acumulada)
    axes[0].axhline(0.90, color='orange', linestyle='--', label='90%')
    axes[0].axhline(0.95, color='red', linestyle='--', label='95%')
    axes[0].axvline(d_90, color='orange', linestyle=':', alpha=0.5)
    axes[0].axvline(d_95, color='red', linestyle=':', alpha=0.5)
    axes[0].set_xlabel("Componente principal")
    axes[0].set_ylabel("Varianza explicada acumulada")
    axes[0].set_title("Scree plot")
    axes[0].set_xlim(0, 100)
    axes[0].legend()
    
    # FIGURA 3b: PC1 vs PC2, coloreado por subtipo
    for st, color in [("H3N2", "tab:blue"), ("H1N1", "tab:red")]:
        mask = metadata.subtype == st
        axes[1].scatter(Z_pca[mask, 0], Z_pca[mask, 1], 
                       s=2, alpha=0.3, c=color, label=st)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].set_title("Subtipos en PCA")
    axes[1].legend()
    
    # FIGURA 3c: PC1 vs PC2 por año (solo H3N2)
    mask_h3 = metadata.subtype == "H3N2"
    scatter = axes[2].scatter(Z_pca[mask_h3, 0], Z_pca[mask_h3, 1],
                              s=2, alpha=0.3, 
                              c=metadata[mask_h3].year, cmap='viridis')
    axes[2].set_xlabel(f"PC1")
    axes[2].set_ylabel(f"PC2")
    axes[2].set_title("H3N2: Evolución temporal en PCA")
    plt.colorbar(scatter, ax=axes[2], label="Año")
    
    plt.tight_layout()
    plt.savefig("results/figures/fig03_pca.pdf", dpi=300)
    plt.savefig("results/figures/fig03_pca.png", dpi=150)
    
    # TABLA 4
    tabla = {
        "Umbral de varianza": ["90%", "95%", "99%"],
        "Componentes necesarios": [d_90, d_95, d_99],
        "Fracción de 384": [f"{d_90/384:.2%}", f"{d_95/384:.2%}", f"{d_99/384:.2%}"]
    }
    pd.DataFrame(tabla).to_csv("results/tables/tab04_pca.csv", index=False)
    
    RETORNAR d_90, d_95, d_99, pca
```

**Interpretación para la SDE:**
- Si d_95 < 50 → la SDE puede operar en un subespacio proyectado (excelente)
- Si d_95 ∈ [50, 150] → factible pero más costoso
- Si d_95 > 200 → los embeddings no tienen estructura de baja dimensión, considerar alternativas

---

### EXP-5: Dimensión intrínseca

```
FUNCIÓN dimension_intrinseca(embeddings, metadata):
    """
    Estima dimensión intrínseca con TwoNN y MLE por subtipo.
    
    TwoNN (Facco et al., 2017): ratio de distancias al 1er y 2do vecino.
    MLE (Levina & Bickel, 2004): máxima verosimilitud con k vecinos.
    
    ERRORES COMUNES A EVITAR:
    - No normalizar datos antes de TwoNN (distorsiona ratios)
    - Usar k demasiado grande en MLE (viola localidad)
    - No separar por subtipo (la mezcla puede inflar dimensión)
    """
    
    from sklearn.neighbors import NearestNeighbors
    
    PARA cada subtipo EN ["H3N2", "H1N1"]:
        mask = metadata.subtype == subtipo
        emb_sub = embeddings[mask]
        n = len(emb_sub)
        
        # --- TwoNN ---
        # Calcular distancias a los 2 vecinos más cercanos
        nn = NearestNeighbors(n_neighbors=3, metric='euclidean')
        nn.fit(emb_sub)
        distances, _ = nn.kneighbors(emb_sub)
        
        # distances[:, 0] = 0 (distancia a sí mismo)
        r1 = distances[:, 1]  # distancia al 1er vecino
        r2 = distances[:, 2]  # distancia al 2do vecino
        
        # Filtrar r1 = 0 (puntos duplicados)
        valid = r1 > 1e-10
        mu = r2[valid] / r1[valid]  # ratio
        
        # Estimar dimensión: mu ~ Pareto con parámetro d
        # log(1 - F(mu)) = -d * log(mu) + constante
        # Ordenar y ajustar
        mu_sorted = np.sort(mu)
        n_valid = len(mu_sorted)
        
        # Estimador de máxima verosimilitud para Pareto
        # d_TwoNN = n / sum(log(mu_i))
        d_twonn = n_valid / np.sum(np.log(mu_sorted))
        
        # Bootstrap para IC
        d_boots = []
        PARA b en range(N_BOOTSTRAP):
            mu_b = np.random.choice(mu, size=n_valid, replace=True)
            d_b = n_valid / np.sum(np.log(mu_b))
            d_boots.append(d_b)
        ci_twonn = (np.percentile(d_boots, 2.5), np.percentile(d_boots, 97.5))
        
        # --- MLE (Levina & Bickel) ---
        # Probar con k = 5, 10, 20, 50
        resultados_mle = []
        PARA k EN [5, 10, 20, 50]:
            nn_k = NearestNeighbors(n_neighbors=k+1)
            nn_k.fit(emb_sub)
            dists_k, _ = nn_k.kneighbors(emb_sub)
            
            # Para cada punto, estimar dimensión local
            d_locals = []
            PARA i en range(n):
                T_k = dists_k[i, k]  # distancia al k-ésimo vecino
                SI T_k < 1e-10:
                    CONTINUAR
                log_sum = sum(np.log(T_k / dists_k[i, j]) 
                              for j in range(1, k))
                d_local = (k - 1) / log_sum si log_sum > 0 sino NaN
                d_locals.append(d_local)
            
            d_mle_k = np.nanmean(d_locals)
            resultados_mle.append({"k": k, "d_MLE": d_mle_k})
        
        PRINT f"\n=== {subtipo} ==="
        PRINT f"  TwoNN: d = {d_twonn:.1f}  [{ci_twonn[0]:.1f}, {ci_twonn[1]:.1f}]"
        PARA r en resultados_mle:
            PRINT f"  MLE(k={r['k']}): d = {r['d_MLE']:.1f}"
    
    # FIGURA 4: Dimensión intrínseca
    # Subplot 1: Distribución empírica de mu (TwoNN) vs. teórica
    # Subplot 2: d_MLE vs. k para detectar estabilidad
    
    # TABLA 5: Resumen de dimensión intrínseca
```

---

### EXP-6: Interpolación con decodificación

```
FUNCIÓN interpolacion_con_decodificacion(model, embeddings, metadata, 
                                          ha_sequences, n_pairs=50):
    """
    Interpola linealmente entre pares de cepas y decodifica los puntos
    intermedios. Evalúa:
    1. Suavidad de la norma (ya hecho, pero verificar si es trivial)
    2. Suavidad de la secuencia decodificada (NUEVO y más informativo)
    3. Validez biológica de las secuencias intermedias
    """
    
    PARA cada subtipo EN ["H3N2", "H1N1"]:
        mask = metadata.subtype == subtipo
        emb_sub = embeddings[mask]
        seq_sub = ha_sequences[mask]
        
        # Seleccionar pares: algunos cercanos, algunos lejanos
        # Para cubrir diferentes regiones del espacio
        pares_cercanos = seleccionar_pares_por_distancia(emb_sub, "cercanos", n=n_pairs//2)
        pares_lejanos = seleccionar_pares_por_distancia(emb_sub, "lejanos", n=n_pairs//2)
        pares = pares_cercanos + pares_lejanos
        
        PARA idx_a, idx_b EN pares:
            z_a = emb_sub[idx_a]
            z_b = emb_sub[idx_b]
            seq_a = seq_sub[idx_a]
            seq_b = seq_sub[idx_b]
            
            lambdas = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
            
            secuencias_interpoladas = []
            hammings_a = []
            hammings_b = []
            
            PARA lam en lambdas:
                z_interp = (1 - lam) * z_a + lam * z_b
                
                # Decodificar (usar el decoder de AntigenLM)
                seq_interp = decodificar(model, z_interp)
                secuencias_interpoladas.append(seq_interp)
                
                # Distancia de Hamming a los extremos
                h_a = calcular_hamming_ha(seq_interp, seq_a)
                h_b = calcular_hamming_ha(seq_interp, seq_b)
                hammings_a.append(h_a)
                hammings_b.append(h_b)
            
            # ¿La Hamming varía monótonamente?
            # Si lambda va de 0 a 1, h_a debería crecer y h_b debería decrecer
            monotonia_a = es_monotona_creciente(hammings_a)
            monotonia_b = es_monotona_decreciente(hammings_b)
            
            # ¿Las secuencias intermedias son "razonables"?
            # Criterio: no deben tener stop codons ni gaps masivos
            validez = [es_secuencia_valida(s) for s in secuencias_interpoladas]
    
    # FIGURA 5: Para 3-4 pares representativos:
    # Lambda vs. Hamming(interpolado, extremo_a) y Hamming(interpolado, extremo_b)
    # Si las curvas son monótonas, la interpolación es semánticamente suave
    
    # TABLA 6: Fracción de interpolaciones monótonas, fracción de secuencias válidas
```

---

### EXP-7: Validez del decoder fuera de distribución

```
FUNCIÓN validez_decoder_ood(model, embeddings, metadata, ha_sequences):
    """
    ¿Qué pasa cuando decodifico puntos que NO son embeddings de cepas reales?
    Esto simula lo que haría la SDE al generar trayectorias.
    
    Se testean tres tipos de puntos OOD:
    1. Perturbaciones pequeñas de cepas reales (ε-vecindad)
    2. Puntos en la interpolación entre cepas (ya cubierto en EXP-6)
    3. Puntos aleatorios en la región del espacio latente
    """
    
    # Seleccionar 100 cepas de referencia
    idx_ref = np.random.choice(len(embeddings), 100, replace=False)
    
    resultados = {"epsilon": [], "frac_valida": [], "hamming_promedio": []}
    
    PARA eps EN [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        validas = 0
        hammings = []
        
        PARA idx EN idx_ref:
            z_ref = embeddings[idx]
            seq_ref = ha_sequences[idx]
            
            # Perturbar: z_pert = z_ref + eps * ruido_unitario
            ruido = np.random.randn(384)
            ruido = ruido / np.linalg.norm(ruido)
            z_pert = z_ref + eps * ruido
            
            # Decodificar
            seq_pert = decodificar(model, z_pert)
            
            SI es_secuencia_valida(seq_pert):
                validas += 1
                hammings.append(calcular_hamming_ha(seq_pert, seq_ref))
        
        frac = validas / len(idx_ref)
        ham_mean = np.mean(hammings) if hammings else float('nan')
        
        resultados["epsilon"].append(eps)
        resultados["frac_valida"].append(frac)
        resultados["hamming_promedio"].append(ham_mean)
        
        PRINT f"  ε = {eps:.2f}: {frac*100:.0f}% válidas, Hamming = {ham_mean:.4f}"
    
    # FIGURA 6: ε vs. fracción válida y ε vs. Hamming promedio
    # Esto define el "radio de confianza" del decoder
    # La SDE debe mantener ||dz|| < ε_max por paso de integración
    
    # TABLA 7: Radio de validez del decoder
```

---

## 4. Tablas y Figuras a Generar

### Tablas

| ID | Contenido | Archivo |
|---|---|---|
| TAB-1 | Estadísticos de normalización de embeddings | `tab01_normalizacion.csv` |
| TAB-2 | Spearman por subtipo (ρ, IC 95%, p-value, decisión) | `tab02_spearman.csv` |
| TAB-3 | Comparación de métricas (euclidiana, coseno, correlación) | `tab03_metricas.csv` |
| TAB-4 | PCA: componentes para 90/95/99% varianza | `tab04_pca.csv` |
| TAB-5 | Dimensión intrínseca (TwoNN, MLE por k) | `tab05_dim_intrinseca.csv` |
| TAB-6 | Interpolación: monotonicidad y validez | `tab06_interpolacion.csv` |
| TAB-7 | Radio de validez del decoder | `tab07_decoder_ood.csv` |
| TAB-8 | **Decisión de escenario** (resumen ejecutivo) | `tab08_decision.csv` |

### Figuras

| ID | Contenido | Archivo |
|---|---|---|
| FIG-1 | Histograma de normas L2 + norma vs. tiempo | `fig01_norma_embeddings.pdf` |
| FIG-2a | Scatter Hamming vs. distancia latente (H3N2) | `fig02_spearman_H3N2.pdf` |
| FIG-2b | Scatter Hamming vs. distancia latente (H1N1) | `fig02_spearman_H1N1.pdf` |
| FIG-3 | PCA: scree plot + subtipos + temporal | `fig03_pca.pdf` |
| FIG-4 | Dimensión intrínseca: distribución mu + d vs. k | `fig04_dim_intrinseca.pdf` |
| FIG-5 | Interpolación: Hamming vs. lambda | `fig05_interpolacion.pdf` |
| FIG-6 | Decoder OOD: fracción válida vs. epsilon | `fig06_decoder_ood.pdf` |

Todas las figuras en PDF (para tesis) y PNG (para presentaciones). Usar matplotlib con estilo consistente.

---

## 5. Umbrales de Decisión

### Tabla de decisión: Escenario A, B o C

```
               ┌─────────────────────────────────────┐
               │       EXP-1: ¿Normalizados?         │
               └──────────┬──────────┬────────────────┘
                    Sí     │         │ No
            Usar coseno    │         │ Usar euclidiana
                          ▼         ▼
               ┌─────────────────────────────────────┐
               │  EXP-2: Spearman por subtipo (ρ)    │
               └──┬──────────┬──────────┬────────────┘
                  │          │          │
            ρ ≥ 0.30     0.20-0.30   ρ < 0.20
                  │          │          │
                  ▼          ▼          ▼
            ESCENARIO A  ESCENARIO B   ¿ρ < 0.10?
            SDE directo  Capa de       │
                         proyección  Sí → ESCENARIO C
                                    No → Intentar B
```

### Umbrales numéricos exactos

| Condición | Umbral | Decisión |
|---|---|---|
| ρ(Spearman) H3N2 con Hamming HA | ≥ 0.30 | **Go** para SDE euclidiana |
| ρ(Spearman) H3N2 con Hamming HA | 0.20 – 0.29 | **Go condicional**: implementar capa de proyección |
| ρ(Spearman) H3N2 con Hamming HA | 0.10 – 0.19 | **Alerta**: capa de proyección obligatoria |
| ρ(Spearman) H3N2 con Hamming HA | < 0.10 | **Stop**: pivotar a escenario C |
| Dimensión intrínseca (TwoNN) | ≤ 50 | SDE factible computacionalmente |
| Dimensión intrínseca (TwoNN) | 50 – 100 | Factible pero considerar proyección PCA |
| Dimensión intrínseca (TwoNN) | > 100 | Proyección PCA obligatoria antes de SDE |
| PCA: componentes para 95% varianza | ≤ 40 | Estructura de baja dimensión (excelente) |
| PCA: componentes para 95% varianza | > 100 | Sin estructura clara de baja dimensión |
| Decoder OOD: fracción válida a ε=0.5 | ≥ 80% | Decoder robusto para SDE |
| Decoder OOD: fracción válida a ε=0.5 | < 50% | Trayectorias SDE necesitan constraint |
| Interpolación: fracción monótona | ≥ 70% | Espacio localmente regular |
| Interpolación: fracción monótona | < 50% | Espacio con discontinuidades semánticas |

### Regla de decisión compuesta

```
FUNCIÓN decidir_escenario(rho_h3n2, rho_h1n1, d_intrinseca, 
                           d_pca_95, frac_decoder, frac_monotona):
    
    # Condiciones para Escenario A (SDE directa)
    escA = (rho_h3n2 >= 0.30 AND
            d_intrinseca <= 50 AND
            frac_decoder >= 0.70 AND
            frac_monotona >= 0.60)
    
    # Condiciones para Escenario B (capa de proyección)
    escB = (rho_h3n2 >= 0.10 AND
            d_intrinseca <= 100 AND
            NOT escA)
    
    # Escenario C (auditoría + marco teórico)
    escC = NOT escA AND NOT escB
    
    SI escA: RETORNAR "A", "SDE euclidiana directa sobre embeddings"
    SI escB: RETORNAR "B", "SDE sobre espacio proyectado con MLP entrenable"
    RETORNAR "C", "Auditoría geométrica como contribución principal + SDE como trabajo futuro"
```

---

## 6. Resultados Mínimos para Seguir con la SDE

Para proceder a la Fase 3 (implementación de SDE) la semana que viene, necesitas **al menos tres de los siguientes cuatro**:

1. ρ(Spearman, H3N2, Hamming HA) ≥ 0.20 con IC 95% que no cruce 0.10.
2. Dimensión intrínseca ≤ 80 (consistente entre TwoNN y MLE).
3. Decoder produce secuencias válidas en ≥ 60% de perturbaciones con ε = 0.5.
4. Interpolación monótona en ≥ 50% de los pares.

Si fallas en dos o más, necesitas la capa de proyección antes de la SDE.
Si fallas en tres o más, pivotas a escenario C.

---

## 7. Cómo Redactar los Resultados

### Si los resultados son favorables (ρ ≥ 0.30)

Estructura del capítulo:

> **4.1 Análisis de la geometría del espacio latente de AntigenLM**
>
> Antes de definir una dinámica estocástica sobre el espacio latente, verificamos que las representaciones internas de AntigenLM satisfacen las propiedades geométricas necesarias. En particular, examinamos si la distancia euclidiana [o coseno] entre embeddings refleja distancia biológica entre las cepas correspondientes.
>
> **4.1.1 Preservación de métrica biológica**
> La correlación de Spearman entre la distancia [euclidiana/coseno] en el espacio latente y la distancia de Hamming en aminoácidos de HA es ρ = X.XX [IC 95%: X.XX, X.XX] para H3N2 (N = ...) y ρ = X.XX para H1N1 (N = ...). Esto indica que el espacio latente preserva la estructura biológica en grado [suficiente/moderado] para soportar la definición de un funcional de escape basado en distancias latentes. [Figura 2]
>
> **4.1.2 Dimensionalidad efectiva**
> [PCA + TwoNN + MLE] → la dimensión intrínseca es d ≈ XX, lo que indica que la dinámica opera sobre una variedad de dimensión manejable.
>
> **4.1.3 Regularidad local**
> La interpolación lineal entre pares de cepas produce secuencias que varían monótonamente en Hamming en el XX% de los casos, lo que sugiere suavidad semántica local.

**Tono:** Factual, sin exceso de entusiasmo. Los números hablan.

### Si los resultados son ambiguos (ρ ∈ [0.15, 0.30])

> **4.1 Análisis de la geometría del espacio latente de AntigenLM**
>
> Encontramos que el espacio latente de AntigenLM preserva parcialmente la estructura biológica. La correlación de Spearman entre distancia [euclidiana/coseno] y distancia de Hamming en HA es moderada para H3N2 (ρ = X.XX) y [comparable/inferior] para H1N1. Esta correlación, si bien estadísticamente significativa, indica que la distancia latente cruda no es un proxy directo de distancia biológica.
>
> Para mejorar la correspondencia entre geometría latente y biología, proponemos una capa de proyección entrenable g: ℝ^384 → ℝ^d cuya pérdida incluye un término de preservación de rango basado en distancias de Hamming. [Describir la capa y su entrenamiento]
>
> Con esta proyección, la correlación mejora a ρ = X.XX, lo que valida el espacio proyectado como soporte para la SDE.

**Tono:** Honesto sobre la limitación, constructivo sobre la solución. La capa de proyección se presenta como contribución metodológica, no como parche.

### Si los resultados son negativos (ρ < 0.15)

> **4.1 Auditoría geométrica del espacio latente de AntigenLM**
>
> Realizamos la primera evaluación sistemática de las propiedades geométricas del espacio latente de un modelo de lenguaje viral entrenado para predicción antigénica. Nuestros resultados revelan que el espacio latente de AntigenLM, tal como fue entrenado con un objetivo de next-token prediction, no preserva la métrica biológica de forma suficiente para soportar directamente una dinámica estocástica euclidiana.
>
> La correlación de Spearman entre distancia latente y distancia de Hamming en HA es ρ = X.XX para H3N2, lo que indica que menos del X% de la varianza en distancia biológica es explicada por la geometría del espacio latente.
>
> **4.1.1 Implicaciones para el modelado dinámico**
> Estos hallazgos sugieren que los modelos de lenguaje de proteínas entrenados con objetivos autorregresivos no producen, sin regularización adicional, espacios latentes adecuados para dinámicas continuas. Esto contrasta con los autoencoders variacionales, donde la regularización KL promueve mayor suavidad geométrica.
>
> **4.1.2 Recomendaciones de diseño**
> Proponemos que futuros modelos de lenguaje viral que pretendan soportar dinámicas en espacio latente incorporen [pérdidas de preservación métrica / regularización geométrica tipo FlatVI].

**Tono:** El resultado negativo se convierte en contribución: es el primer estudio que diagnostica este problema. La tesis pivota a Narrativa 2.

---

## 8. Protocolo de Validación Prospectiva 2022–2026

### Por qué esto cambia la fuerza del proyecto

La validación prospectiva genuina es extremadamente rara en predicción antigénica. La mayoría de los trabajos (incluido AntigenLM) hacen validación retrospectiva: entrenan con datos hasta 2019 y "predicen" 2020, pero los datos de 2020 existían cuando diseñaron el modelo. Tus datos 2022–2026 no existían cuando AntigenLM fue publicado. Esto elimina una de las críticas más comunes en revisión por pares: la posibilidad de overfitting metodológico a datos futuros que el investigador ya conocía.

**Impacto en fuerza internacional:**
- Sin validación prospectiva: proyecto sólido pero convencional → workshop-level.
- Con validación prospectiva bien ejecutada: argumento metodológico fuerte → journal-level.
- Si la SDE acierta en 2022–2026: resultado difícil de cuestionar → paper competitivo.

### Diseño de la validación prospectiva

```
PROTOCOLO DE VALIDACIÓN PROSPECTIVA

Fecha de congelamiento metodológico: [DÍA 7 de esta semana]

1. PARTICIÓN TEMPORAL (no aleatoria)
   - Train:        2000–2021 (inclusive)
   - Calibración:  2019–2021 (subconjunto de train para ajustar α, β, σ)
   - Test retro:   2019–2021 (entrenando con datos ≤2018)
   - Test prosp:   2022–2026 (entrenando con datos ≤2021)

   NOTA: Test retro y calibración NO son el mismo split.
   Calibración usa los datos como train; test retro los usa como evaluación
   con un modelo entrenado sin ellos.

2. MÉTRICAS PRE-REGISTRADAS (congelar antes de mirar 2022–2026)

   Métrica primaria:
   - AAM (Amino Acid Mismatch): distancia de Hamming entre predicción
     puntual (mediana de la distribución) y cepa dominante observada.
   
   Métricas secundarias:
   - CRPS (Continuous Ranked Probability Score): evalúa toda la distribución,
     no solo el punto central.
   - Coverage@90: fracción de meses donde la cepa observada cae dentro
     del intervalo de credibilidad al 90%.
   - Top-5 accuracy: fracción de meses donde la cepa observada está
     entre las 5 predicciones más probables.
   - Log-likelihood de cepa observada bajo distribución predicha.

   CRITICAL: Estas métricas se definen AHORA, no después de ver los resultados.

3. HORIZONTES DE PREDICCIÓN
   - 1 mes (comparable con AntigenLM)
   - 3 meses
   - 6 meses (horizonte relevante para vacunas)
   - 12 meses

4. BASELINES
   - Persistencia (última cepa observada)
   - Consenso (secuencia consenso de los últimos 3 meses)
   - AntigenLM (predicción puntual del paper)
   - LBI
   - ODE (drift sin difusión, σ=0)

5. REGLAS ANTI-CHERRY-PICKING
   - Reportar TODAS las métricas en TODOS los horizontes.
   - No seleccionar el horizonte "que mejor funciona" como resultado principal.
   - Reportar intervalos de confianza (bootstrap) para cada métrica.
   - Si la SDE gana en algunos horizontes y pierde en otros, reportar ambos.
   - Pre-registrar qué horizonte es el "focal" (recomiendo 3 meses:
     es más largo que AntigenLM pero no tan largo como para ser ruidoso).
   - Incluir un apéndice con resultados mensuales desagregados.

6. PREVENCIÓN DE LEAKAGE
   - Verificar que NINGUNA cepa de 2022–2026 aparezca en el tokenizador.
   - Verificar que los pesos de AntigenLM no fueron entrenados con datos post-2022.
   - No usar datos prospectivos para ninguna decisión de hiperparámetros.
   - α, β, σ se ajustan SOLO con datos ≤2021.
   - La arquitectura de la SDE se congela antes de mirar datos prospectivos.
   - Documentar esto explícitamente en la tesis.
```

### Decisiones a congelar esta semana (día 7)

Las siguientes decisiones deben quedar fijas antes de cualquier contacto con datos 2022–2026:

1. **Métrica de distancia latente:** euclidiana o coseno (determinada por EXP-1).
2. **Métricas de evaluación:** AAM primaria, CRPS y Coverage@90 secundarias.
3. **Horizonte focal:** 3 meses (recomendación).
4. **Baselines:** persistencia, consenso, AntigenLM, LBI.
5. **Escenario (A/B/C):** determinado por resultados de geometría.
6. **Si escenario B:** arquitectura de la capa de proyección (dimensión de salida, tipo de pérdida).

**Registrar estas decisiones en `decision_log.md` con fecha y justificación.** Este log es evidencia de que las decisiones fueron pre-data.

### Cómo presentar la validación prospectiva

En la tesis:

> **5.3 Validación prospectiva: 2022–2026**
>
> A diferencia de la mayoría de los trabajos de predicción antigénica, que evalúan retrospectivamente sobre datos que existían al momento del diseño del modelo, realizamos una validación prospectiva genuina. El modelo fue entrenado exclusivamente con datos anteriores a 2022, y se evaluó sobre cepas del periodo 2022–2026, las cuales no existían cuando el modelo base (AntigenLM) fue publicado.
>
> Todas las decisiones metodológicas — incluyendo la elección de métricas, la arquitectura de la SDE, y los valores de los hiperparámetros — fueron congeladas antes de acceder a los datos prospectivos (ver Apéndice X: registro de decisiones).
>
> [Tabla: Métricas por horizonte y por baseline, con intervalos de confianza]

En un eventual paper, esta sección es el argumento más fuerte contra los revisores escépticos. La validación prospectiva es difícil de objetar.

---

## 9. Resumen de la Semana

| Día | Objetivo | Entregable | Go/No-go |
|---|---|---|---|
| Lun | Embeddings + normalización | FIG-1, TAB-1 | ¿Euclidiana o coseno? |
| Mar | Spearman por subtipo | FIG-2, TAB-2 | ¿ρ ≥ 0.20? |
| Mié | Métricas + PCA | FIG-3, TAB-3, TAB-4 | ¿Mejor métrica? ¿d_95? |
| Jue | Dimensión intrínseca | FIG-4, TAB-5 | ¿d ≤ 50? |
| Vie | Interpolación + decoder | FIG-5, FIG-6, TAB-6, TAB-7 | ¿Decoder robusto? |
| Sáb | Decisión + dashboard | TAB-8, notebook integrado | **Escenario A, B o C** |
| Dom | Congelar decisiones | `decision_log.md` completo | Listo para fase 3 o pivote |

**Al final del día 7, debes tener:**
1. Las 7 tablas y 6 figuras generadas.
2. Una decisión clara de escenario (A, B, o C) documentada.
3. Todas las decisiones metodológicas congeladas para validación prospectiva.
4. El primer borrador del capítulo de geometría (al menos la sección de resultados).
5. Un plan de acción para la semana 2 basado en la decisión.

---

## 10. Escenarios de la Semana 2

### Si escenario A (ρ ≥ 0.30, d ≤ 50, decoder robusto):

Semana 2: Implementar ODE como test del drift. Definir F_viab y F_escape operacionalmente. Verificar que el campo vectorial es estable.

### Si escenario B (0.10 ≤ ρ < 0.30, o d > 50):

Semana 2: Implementar capa de proyección g: ℝ^384 → ℝ^d con pérdida de preservación de rango (triplet loss o contrastive loss basada en Hamming). Repetir EXP-2 sobre espacio proyectado. Si ρ mejora a ≥ 0.30, proceder a ODE en semana 3.

### Si escenario C (ρ < 0.10, o decoder masivamente inválido):

Semana 2: Pivotar. Expandir la auditoría geométrica con análisis adicionales (persistent homology, CKA, comparación con ESM-2 embeddings como referencia). Comenzar escritura del capítulo de auditoría como contribución principal. Formular la SDE como contribución teórica (marco + condiciones necesarias) sin implementación empírica completa.

---

*Plan técnico generado el 26 de abril de 2026. Ejecutar en orden. No saltar pasos. Documentar todo.*
