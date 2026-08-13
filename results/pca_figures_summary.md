# Resumen de figuras PCA

Fuente: `results/embeddings_cache_10k_per_subtype_seed42.pkl`.
No se recalcularon embeddings ni se cargo AntigenLM.

| grupo | n | dim | n80 | n90 | n95 | n99 | participation ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| global | 20000 | 384 | 2 | 3 | 4 | 12 | 1.89 |
| H1N1 | 10000 | 384 | 2 | 2 | 3 | 13 | 1.76 |
| H3N2 | 10000 | 384 | 2 | 3 | 6 | 20 | 1.63 |

## Top 10 explained variance ratios

- Global: 0.6975, 0.2024, 0.0458, 0.0163, 0.0089, 0.0053, 0.0039, 0.0029, 0.0027, 0.0019.
- H1N1: 0.7246, 0.2082, 0.0196, 0.0112, 0.0077, 0.0056, 0.0042, 0.0023, 0.0022, 0.0015.
- H3N2: 0.7757, 0.0874, 0.0417, 0.0260, 0.0156, 0.0102, 0.0065, 0.0053, 0.0040, 0.0035.

## Figuras generadas

Estado de verificacion PCA 3D: las figuras `pca_3d_by_subtype` y `pca_3d_by_year`
fueron encontradas ya existentes en PNG y PDF bajo `figures/gisaid/`; no fue necesario
regenerarlas. La verificacion se hizo contra el cache
`results/embeddings_cache_10k_per_subtype_seed42.pkl`, sin recalcular embeddings ni
cargar AntigenLM.

- `figures/gisaid/pca_scree_global.png`
- `figures/gisaid/pca_scree_global.pdf`
- `figures/gisaid/pca_scree_log_global.png`
- `figures/gisaid/pca_scree_log_global.pdf`
- `figures/gisaid/pca_cumulative_global.png`
- `figures/gisaid/pca_cumulative_global.pdf`
- `figures/gisaid/pca_scree_by_subtype.png`
- `figures/gisaid/pca_scree_by_subtype.pdf`
- `figures/gisaid/pca_cumulative_by_subtype.png`
- `figures/gisaid/pca_cumulative_by_subtype.pdf`
- `figures/gisaid/pca_2d_by_subtype.png`
- `figures/gisaid/pca_2d_by_subtype.pdf`
- `figures/gisaid/pca_2d_by_year.png`
- `figures/gisaid/pca_2d_by_year.pdf`
- `figures/gisaid/pca_2d_h1n1_by_year.png`
- `figures/gisaid/pca_2d_h1n1_by_year.pdf`
- `figures/gisaid/pca_2d_h3n2_by_year.png`
- `figures/gisaid/pca_2d_h3n2_by_year.pdf`
- `figures/gisaid/pca_3d_by_subtype.png`
- `figures/gisaid/pca_3d_by_subtype.pdf`
- `figures/gisaid/pca_3d_by_year.png`
- `figures/gisaid/pca_3d_by_year.pdf`

## Interpretacion breve de figuras

- `pca_scree_global`: muestra la concentracion extrema de varianza en las primeras componentes.
- `pca_scree_log_global`: permite inspeccionar la cola espectral, que queda comprimida en escala lineal.
- `pca_cumulative_global`: resume que 2, 3, 4 y 12 componentes alcanzan 80%, 90%, 95% y 99% de varianza explicada, respectivamente.
- `pca_2d_by_subtype`: muestra organizacion clara por subtipo en las dos primeras componentes principales.
- `pca_2d_by_year`, `pca_2d_h1n1_by_year` y `pca_2d_h3n2_by_year`: permiten explorar patron temporal, sin asumir una trayectoria cronologica simple.
- `pca_3d_by_subtype`: proyeccion global PC1-PC2-PC3 coloreada por subtipo; es util como vista exploratoria complementaria, aunque puede ser menos legible que la proyeccion 2D en documento impreso.
- `pca_3d_by_year`: proyeccion global PC1-PC2-PC3 coloreada por anio con colorbar; debe interpretarse como visualizacion descriptiva, no como evidencia de trayectoria temporal lineal.

## Lectura metodologica

- PCA confirma concentracion fuerte de varianza en pocas componentes.
- Esto es compatible con TwoNN y refuerza la hipotesis de baja dimension efectiva.
- La participation ratio baja sugiere anisotropia fuerte.
- Las proyecciones bidimensionales revelan organizacion por subtipo; la coloracion por anio no implica por si misma una dinamica temporal monotona.
- En conjunto con la correlacion fuerte entre distancia latente y Hamming HA/HA+NA, estos resultados sugieren que la geometria latente captura estructura molecular relevante mas que unicamente orden temporal.
- Las figuras son exploratorias/descriptivas y no prueban por si solas que una SDE funcione.
