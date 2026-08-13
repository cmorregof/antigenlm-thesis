#!/usr/bin/env bash
set -euo pipefail

# Full local pipeline for Paper 1 / Checkpoint geométrico.
#
# This script does not train models, modify checkpoints, generate sequences, or
# print complete sequences. It extracts local AntigenLM embeddings, computes
# aggregate geometry diagnostics, and builds an Overleaf-ready draft.

CACHE="results/embeddings_cache_full_all_available.pkl"
PYTHON="venv_antigenlm/bin/python"

mkdir -p results figures/latent_geometry_full papers/paper_1_latent_geometry_full

echo "[1/4] Full embedding cache"
if [[ -f "${CACHE}" && "${FORCE_REBUILD_CACHE:-0}" != "1" ]]; then
  echo "Cache already exists: ${CACHE}"
  echo "Set FORCE_REBUILD_CACHE=1 to rebuild from scratch."
else
  caffeinate -i "${PYTHON}" -u latent_geometry.py \
    --max-per-subtype -1 \
    --sampling-strategy all \
    --seed 42 \
    --save-embeddings-cache \
    --resume-cache \
    --checkpoint-cache-every "${CHECKPOINT_CACHE_EVERY:-5000}" \
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE:-4}" \
    --cache-path "${CACHE}" \
    --skip-umap \
    --skip-interpolation \
    --skip-intrinsic-dim \
    2>&1 | tee results/full_embedding_extraction.log
fi

echo "[2/4] Full-data geometry analysis"
caffeinate -i "${PYTHON}" -u latent_geometry_full_analysis.py \
  --cache-path "${CACHE}" \
  --pair-samples-per-subtype "${PAIR_SAMPLES_PER_SUBTYPE:-100000}" \
  --pair-seeds 42,7,123 \
  --twonn-sample-sizes "${TWONN_SAMPLE_SIZES:-5000,10000,20000,50000}" \
  --twonn-trims 0.01,0.05 \
  --twonn-seeds 42,7,123 \
  --temporal-k-values 5,10,20 \
  --temporal-max-points-per-subtype "${TEMPORAL_MAX_POINTS_PER_SUBTYPE:-0}" \
  --plot-max-points "${PLOT_MAX_POINTS:-50000}" \
  --seed 42 \
  2>&1 | tee results/full_latent_geometry_analysis.log

echo "[3/4] Paper draft"
"${PYTHON}" make_paper_1_latent_geometry_full.py \
  2>&1 | tee results/paper_1_latent_geometry_full_build.log

echo "[4/4] Done"
echo "Summary: results/latent_geometry_full_summary.md"
echo "Data audit: results/full_data_audit_summary.md"
echo "Metrics: results/latent_geometry_full_metrics.json"
echo "Figures: figures/latent_geometry_full/"
echo "Paper: papers/paper_1_latent_geometry_full/main.tex"
echo "Zip: paper_1_latent_geometry_full.zip"
