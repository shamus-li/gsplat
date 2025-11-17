#!/bin/bash
#SBATCH --job-name=gsplat_ext_eval
#SBATCH --output=slurm_logs/external_eval_%A_%a.out
#SBATCH --error=slurm_logs/external_eval_%A_%a.err
#SBATCH --partition=default_partition
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02,awang-compute-01,kim-compute-03,kim-compute-04,kim-compute-05
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 16gb
#SBATCH -t 04:00:00
#SBATCH --array=0-3
#SBATCH --requeue

# External eval-only refresh of renders using the latest checkpoint from a
# training run. Runs as a 4-task array over {monocular, iphone, stereo, lightfield}.
#
# Usage:
#   sbatch examples/external_eval.slurm SCENE_DIR [EVAL_K]
#
# Arguments:
#   - SCENE_DIR: Path to scene root (expects SCENE_DIR/static/shared exists)
#   - EVAL_K (optional): Number of eval frames to render per camera (default 10; 0 = all)
#
# Environment knobs:
#   PREP_ROOT:          Defaults to "$SCENE_DIR/static/shared"
#   RESULT_ROOT:        Defaults to "results/static"
#   TRAIN_TEST_EVERY:   Defaults to 1
#   EVAL_TEST_EVERY:    Defaults to 1
#   USE_COVISIBLE:      Defaults to 1 (metrics masked if covisible cache present)
#   RENDER_VIDEO:       Defaults to 0 (set 1 to also export a trajectory video)

set -euo pipefail

source /home/wl757/.bashrc
conda activate gaussian_splatting

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
REPO_ROOT="$(pwd)"
ALIGN_SCRIPT="$REPO_ROOT/examples/compute_dataset_alignment.py"

CAMERAS=(monocular iphone stereo lightfield)

SCENE_DIR=${1:-}
EVAL_K=${2:-10}
if [ -z "$SCENE_DIR" ]; then
    echo "ERROR: Missing SCENE_DIR argument."
    echo "Usage: sbatch examples/external_eval.slurm ../gs7/dataset/<scene> [EVAL_K]"
    exit 1
fi
SCENE_DIR=${SCENE_DIR%/}
if [ ! -d "$SCENE_DIR" ]; then
    echo "ERROR: Scene directory not found: $SCENE_DIR"
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set."
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -gt 3 ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID must be in [0,3]"
    exit 1
fi

CAMERA=${CAMERAS[$SLURM_ARRAY_TASK_ID]}
SCENE_NAME=$(basename "$SCENE_DIR")

PREP_ROOT=${PREP_ROOT:-"$SCENE_DIR/static/shared"}
RESULT_ROOT=${RESULT_ROOT:-"results/static"}
TRAIN_TEST_EVERY=${TRAIN_TEST_EVERY:-1}
EVAL_TEST_EVERY=${EVAL_TEST_EVERY:-1}
USE_COVISIBLE=${USE_COVISIBLE:-1}
RENDER_VIDEO=${RENDER_VIDEO:-0}

SENTINEL="$PREP_ROOT/.prep_complete"
if [ ! -f "$SENTINEL" ]; then
    echo "ERROR: Shared dataset sentinel missing at $SENTINEL"
    echo "Run examples/static_preprocessing.slurm first."
    exit 1
fi

TRAIN_DATA_DIR="$PREP_ROOT/subsets/$CAMERA"
EVAL_DIR="$PREP_ROOT/subsets/iphone_eval"
EVAL_SUFFIX=$(basename "$EVAL_DIR")
if [ ! -d "$TRAIN_DATA_DIR" ]; then
    echo "ERROR: Training subset not found for $CAMERA at $TRAIN_DATA_DIR"
    exit 1
fi
if [ ! -d "$EVAL_DIR" ]; then
    echo "ERROR: Eval subset not found at $EVAL_DIR"
    exit 1
fi

TRAIN_RESULT_DIR="$RESULT_ROOT/$SCENE_NAME/$CAMERA"
OUT_DIR="$TRAIN_RESULT_DIR/eval_on_iphone_eval"
mkdir -p "$OUT_DIR" slurm_logs

ALIGN_ROOT="$TRAIN_RESULT_DIR/alignments"
mkdir -p "$ALIGN_ROOT"
TRAIN_ALIGN="$ALIGN_ROOT/train_normalization.npz"
EVAL_ALIGN="$ALIGN_ROOT/${EVAL_SUFFIX}_to_train.npz"
ALIGN_ARG=()
if [ ! -f "$EVAL_ALIGN" ] && [ -d "$TRAIN_DATA_DIR/images" ]; then
    if python "$ALIGN_SCRIPT" \
        --train-dir "$TRAIN_DATA_DIR" \
        --subset-dir "$EVAL_DIR" \
        --train-test-every "$TRAIN_TEST_EVERY" \
        --eval-test-every "$EVAL_TEST_EVERY" \
        --output "$EVAL_ALIGN"; then
        echo "Saved alignment to $EVAL_ALIGN"
    else
        echo "WARNING: Failed to compute alignment for $TRAIN_RESULT_DIR"
        rm -f "$EVAL_ALIGN"
    fi
fi
if [ ! -f "$TRAIN_ALIGN" ] && [ -d "$TRAIN_DATA_DIR/images" ]; then
    if python "$ALIGN_SCRIPT" \
        --train-dir "$TRAIN_DATA_DIR" \
        --train-test-every "$TRAIN_TEST_EVERY" \
        --output "$TRAIN_ALIGN"; then
        echo "Saved train normalization to $TRAIN_ALIGN"
    else
        echo "WARNING: Failed to compute train normalization for $TRAIN_RESULT_DIR"
        rm -f "$TRAIN_ALIGN"
    fi
fi
if [ -f "$EVAL_ALIGN" ]; then
    ALIGN_ARG=(--align_path "$EVAL_ALIGN")
elif [ -f "$TRAIN_ALIGN" ]; then
    ALIGN_ARG=(--align_path "$TRAIN_ALIGN")
fi

# Covisible cache (optional for metrics; renders remain unmasked)
COVI_ROOT="$PREP_ROOT/covisible/$CAMERA/1x"
COVI_ARGS=()
if [ "$USE_COVISIBLE" = "1" ] && [ -d "$COVI_ROOT" ]; then
    COVI_ARGS=(--use_covisible --covisible_dir "$COVI_ROOT")
fi

VIDEO_ARG=()
if [ "$RENDER_VIDEO" = "1" ]; then
    VIDEO_ARG=(--render_video)
fi

echo "=========================================="
echo "GSPLAT EXTERNAL EVAL-ONLY"
echo "=========================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Scene: $SCENE_NAME"
echo "Camera: $CAMERA"
echo "Prep root: $PREP_ROOT"
echo "Train data: $TRAIN_DATA_DIR"
echo "Eval dir: $EVAL_DIR"
echo "Train result dir: $TRAIN_RESULT_DIR"
echo "Out dir: $OUT_DIR"
echo "Eval K: $EVAL_K"
echo "train_test_every: $TRAIN_TEST_EVERY"
echo "eval_test_every: $EVAL_TEST_EVERY"
echo "Use covisible: $USE_COVISIBLE"
echo "Covisible root: $COVI_ROOT"
echo "Render video: $RENDER_VIDEO"
echo "=========================================="
echo ""

CMD=(python scripts/run_external_eval.py
    --train_result_dir "$TRAIN_RESULT_DIR"
    --train_data_dir "$TRAIN_DATA_DIR"
    --eval_dir "$EVAL_DIR"
    --out_dir "$OUT_DIR"
    --k "$EVAL_K"
    --train_test_every "$TRAIN_TEST_EVERY"
    --eval_test_every "$EVAL_TEST_EVERY"
    "${COVI_ARGS[@]}"
    "${VIDEO_ARG[@]}"
    "${ALIGN_ARG[@]}"
)

echo "Executing: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "=========================================="
echo "EXTERNAL EVAL COMPLETE ($CAMERA)"
echo "Results at $OUT_DIR"
echo "=========================================="
