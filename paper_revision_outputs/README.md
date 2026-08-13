# Salidas de revision del articulo

La raiz de esta carpeta conserva solo lo que se usa como salida actual o como script reproducible:

- `run_*.py`
- `*_results.json`
- `*_summary.md`

El resto queda separado asi:

```text
paper_revision_outputs/
├── reports/               # Informes numerados y resumenes editoriales
├── manuscript_versions/   # Versiones .tex/.pdf generadas durante revision
└── logs/                  # Logs y auxiliares de compilacion
```

Los enlaces simbolicos `figures` y `references.bib` se mantienen para compilar manuscritos desde esta carpeta si hace falta.
