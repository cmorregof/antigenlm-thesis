# Estructura del proyecto

```text
antigenlm-thesis/
├── thesis/                 # Documento de tesis, borradores y notas
├── papers/                 # Paquetes LaTeX de articulos
├── paper_revision_outputs/ # Scripts, resultados y trazas de revision del articulo
├── figures/                # Figuras generadas por los analisis
├── results/                # Metricas, resumenes, logs y caches locales
├── data/                   # Datos locales privados o derivados; no versionar
├── prediction_sequence/    # Checkpoint local principal de AntigenLM
├── subtype_classifier/     # Checkpoint local auxiliar
├── checkpoints/            # Checkpoints experimentales locales
├── archive/                # Exportaciones antiguas preservadas
├── docs/                   # Guias de uso y organizacion
└── *.py                    # Scripts principales, mantenidos en raiz para no romper rutas existentes
```

Regla practica:

- Tesis escrita: `thesis/`.
- Articulos: `papers/` y `paper_revision_outputs/`.
- Codigo reproducible: scripts `.py` de la raiz y scripts `run_*.py`.
- Resultados que sostienen texto/figuras: `results/` y `figures/`.
- Datos, pesos y entornos: locales, pesados, no versionados.
