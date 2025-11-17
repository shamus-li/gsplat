#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SCENE_DIR=${1:-"../gs7/dataset/action-figure/iphone/test"}

if [[ ! -d "$SCENE_DIR/images" || ! -d "$SCENE_DIR/sparse" ]]; then
  cat <<EOF
Usage: scripts/dycheck_smoke_test.sh [COLMAP_SCENE_DIR]

COLMAP_SCENE_DIR must contain 'images/' and 'sparse/' folders (e.g., ../gs7/dataset/scene/iphone/test).
EOF
  exit 1
fi

OUTPUT_DIR=${DYCHECK_SMOKE_OUTPUT:-"$REPO_ROOT/results/dycheck_smoke"}
mkdir -p "$OUTPUT_DIR"
COVI_OUT="$OUTPUT_DIR/covisible"

echo "[smoke] Generating covisible masks under $COVI_OUT"
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
python examples/preprocess_covisible_colmap.py \
  --base_dir "$SCENE_DIR" \
  --support_dir "$SCENE_DIR" \
  --factor ${DYCHECK_SMOKE_FACTOR:-1} \
  --test_every ${DYCHECK_SMOKE_TEST_EVERY:-9999} \
  --support_test_every ${DYCHECK_SMOKE_SUPPORT_EVERY:-9999} \
  --base_split ${DYCHECK_SMOKE_BASE_SPLIT:-val} \
  --support_split ${DYCHECK_SMOKE_SUPPORT_SPLIT:-val} \
  --batch_size ${DYCHECK_SMOKE_BATCH_SIZE:-1} \
  --num_workers ${DYCHECK_SMOKE_NUM_WORKERS:-0} \
  --max_hw ${DYCHECK_SMOKE_MAX_HW:-128} \
  --output_dir "$COVI_OUT" \
  --device ${DYCHECK_SMOKE_DEVICE:-cuda}

echo "[smoke] Running DyCheck metric unit tests"
pytest tests/test_dycheck_metrics.py -q

echo "[smoke] Done"
