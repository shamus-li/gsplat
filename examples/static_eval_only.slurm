#!/bin/bash
#SBATCH --job-name=gsplat_static_eval
#SBATCH --output=slurm_logs/static_eval_%A_%a.out
#SBATCH --error=slurm_logs/static_eval_%A_%a.err
#SBATCH --partition=default_partition
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02,awang-compute-01,kim-compute-03,kim-compute-04,kim-compute-05
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 24gb
#SBATCH -t 06:00:00
#SBATCH --array=0-7
#SBATCH --requeue

# Static eval-only driver for the eight canonical scenes. Each SLURM array slot
# handles exactly one (scene, camera) combo, so you can blast out many combos in
# parallel or run the script without an array to process everything sequentially.
# Usage:
#   sbatch --array=0-(N-1) examples/static_eval_only.slurm [DATA_ROOT] [RESULT_ROOT]
#   LIST_STATIC_EVAL_COMBOS=1 bash examples/static_eval_only.slurm   # list combos/help size the array

set -euo pipefail

source /home/wl757/.bashrc
conda activate gaussian_splatting

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi
REPO_ROOT="$(pwd)"

mkdir -p slurm_logs
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-python}
ALIGN_SCRIPT="$REPO_ROOT/examples/compute_dataset_alignment.py"

DATA_ROOT_INPUT=${1:-"../gs7/dataset"}
RESULT_ROOT_INPUT=${2:-"results/static"}
DATA_ROOT="$(realpath "$DATA_ROOT_INPUT")"
if [[ "${RESULT_ROOT_INPUT#/}" = "$RESULT_ROOT_INPUT" ]]; then
    RESULT_ROOT="$(pwd)/$RESULT_ROOT_INPUT"
else
    RESULT_ROOT="$RESULT_ROOT_INPUT"
fi

SCENES=(
  action-figure
  ball
  chicken
  dog
  espresso
  optics
  salt-pepper
  shelf
)
DEFAULT_CAMERAS=(monocular iphone stereo lightfield)

if [ -n "${STATIC_EVAL_CAMERAS:-}" ]; then
    IFS=',' read -ra USER_CAMS <<<"$STATIC_EVAL_CAMERAS"
    CAMERAS=()
    for raw in "${USER_CAMS[@]}"; do
        trimmed="${raw//[[:space:]]/}"
        if [ -n "$trimmed" ]; then
            CAMERAS+=("$trimmed")
        fi
    done
    if [ "${#CAMERAS[@]}" -eq 0 ]; then
        CAMERAS=("${DEFAULT_CAMERAS[@]}")
    fi
else
    CAMERAS=("${DEFAULT_CAMERAS[@]}")
fi

PREP_ROOT_OVERRIDE=${PREP_ROOT:-}
unset PREP_ROOT

EVAL_K=${EVAL_K:-0}
EVAL_STRIDE=${EVAL_STRIDE:-8}
TRAIN_TEST_EVERY=${TRAIN_TEST_EVERY:-1}
EVAL_TEST_EVERY=${EVAL_TEST_EVERY:-1}
USE_COVISIBLE=${USE_COVISIBLE:-1}
FORCE_ALIGN_REGEN=${FORCE_ALIGN_REGEN:-0}
VIS_CONCAT_GT=${VIS_CONCAT_GT:-1}

echo "=========================================="
echo "GSPLAT STATIC EVAL-ONLY"
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-manual}"
echo "Array index: ${SLURM_ARRAY_TASK_ID:-all}" 
echo "Data root: $DATA_ROOT"
echo "Result root: $RESULT_ROOT"
echo "Eval frames per camera: $EVAL_K"
echo "Eval stride: $EVAL_STRIDE"
echo "train_test_every: $TRAIN_TEST_EVERY"
echo "eval_test_every: $EVAL_TEST_EVERY"
echo "=========================================="

declare -a COMBO_SCENES=()
declare -a COMBO_CAMERAS=()
declare -a COMBO_TRAIN_DIRS=()
declare -a COMBO_RESULT_DIRS=()
declare -a COMBO_EVAL_DIRS=()
declare -a COMBO_EVAL_SUFFIXES=()

for SCENE in "${SCENES[@]}"; do
    if [[ "$SCENE" = "dynamic" || "$SCENE" = "static" ]]; then
        echo "[skip] Scene '$SCENE' is a meta directory; nothing to evaluate."
        continue
    fi

    SCENE_DIR="$DATA_ROOT/$SCENE"
    if [ ! -d "$SCENE_DIR" ]; then
        echo "[skip] Scene directory not found: $SCENE_DIR"
        continue
    fi

    if [[ -n "$PREP_ROOT_OVERRIDE" ]]; then
        prep_root="$PREP_ROOT_OVERRIDE"
    else
        prep_root="$SCENE_DIR/static/shared"
    fi
    SENTINEL="$prep_root/.prep_complete"
    if [ ! -f "$SENTINEL" ]; then
        echo "[skip] $SCENE: Missing static shared prep at $SENTINEL"
        continue
    fi

    eval_dir="$prep_root/subsets/iphone_eval"
    if [ ! -d "$eval_dir/images" ] || [ ! -d "$eval_dir/sparse" ]; then
        echo "[skip] $SCENE: Eval subset missing at $eval_dir"
        continue
    fi
    eval_suffix="$(basename "$eval_dir")"

    for camera in "${CAMERAS[@]}"; do
        train_dir="$prep_root/subsets/$camera"
        if [ ! -d "$train_dir/images" ] || [ ! -d "$train_dir/sparse" ]; then
            echo "[skip] $SCENE/$camera -> training subset not found at $train_dir"
            continue
        fi

        result_dir="$RESULT_ROOT/$SCENE/$camera"
        ckpt=$(ls -1 "$result_dir"/ckpts/ckpt_*_rank0.pt 2>/dev/null | sort -V | tail -n 1 || true)
        if [ -z "$ckpt" ]; then
            echo "[skip] $SCENE/$camera -> no checkpoints under $result_dir/ckpts"
            continue
        fi

        COMBO_SCENES+=("$SCENE")
        COMBO_CAMERAS+=("$camera")
        COMBO_TRAIN_DIRS+=("$(realpath -m "$train_dir")")
        COMBO_RESULT_DIRS+=("$(realpath -m "$result_dir")")
        COMBO_EVAL_DIRS+=("$(realpath -m "$eval_dir")")
        COMBO_EVAL_SUFFIXES+=("$eval_suffix")
    done
done

COMBO_COUNT=${#COMBO_SCENES[@]}
if [[ "$COMBO_COUNT" -eq 0 ]]; then
    echo "ERROR: No runnable scene/camera combinations found."
    exit 1
fi

echo "Discovered $COMBO_COUNT combinations."
if [[ "${LIST_STATIC_EVAL_COMBOS:-0}" == "1" ]]; then
    for ((idx = 0; idx < COMBO_COUNT; idx++)); do
        printf '[combo %d] %s / %s / %s\n' \
            "$idx" \
            "${COMBO_SCENES[$idx]}" \
            "${COMBO_CAMERAS[$idx]}" \
            "${COMBO_RESULT_DIRS[$idx]}"
    done
    exit 0
fi

declare -a SELECTED_INDICES=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SELECTED_INDICES=("$SLURM_ARRAY_TASK_ID")
else
    for ((idx = 0; idx < COMBO_COUNT; idx++)); do
        SELECTED_INDICES+=("$idx")
    done
fi

for combo_idx in "${SELECTED_INDICES[@]}"; do
    if (( combo_idx < 0 || combo_idx >= COMBO_COUNT )); then
        echo "ERROR: combo index $combo_idx out of range (0..$((COMBO_COUNT-1)))"
        exit 1
    fi

    SCENE="${COMBO_SCENES[$combo_idx]}"
    camera="${COMBO_CAMERAS[$combo_idx]}"
    TRAIN_DIR="${COMBO_TRAIN_DIRS[$combo_idx]}"
    RESULT_DIR="${COMBO_RESULT_DIRS[$combo_idx]}"
    EVAL_DIR="${COMBO_EVAL_DIRS[$combo_idx]}"
    EVAL_SUFFIX="${COMBO_EVAL_SUFFIXES[$combo_idx]}"

    PREP_ROOT_COMBO="$(dirname "$(dirname "$TRAIN_DIR")")"

    CKPT=$(ls -1 "$RESULT_DIR"/ckpts/ckpt_*_rank0.pt 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -z "$CKPT" ]; then
        echo "[skip] combo $combo_idx: no checkpoints under $RESULT_DIR/ckpts"
        continue
    fi

    OUT_DIR="$RESULT_DIR/eval_on_${EVAL_SUFFIX}"
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"
    ALIGN_ROOT="$RESULT_DIR/alignments"
    ALIGN_PATH="$ALIGN_ROOT/${EVAL_SUFFIX}_to_${camera}.npz"
    if [ "$FORCE_ALIGN_REGEN" = "1" ] && [ -f "$ALIGN_PATH" ]; then
        rm -f "$ALIGN_PATH"
    fi
    if [ ! -f "$ALIGN_PATH" ]; then
        mkdir -p "$ALIGN_ROOT"
        "$PYTHON_BIN" "$ALIGN_SCRIPT" \
            --train-dir "$TRAIN_DIR" \
            --subset-dir "$EVAL_DIR" \
            --train-test-every "$TRAIN_TEST_EVERY" \
            --eval-test-every "$EVAL_TEST_EVERY" \
            --output "$ALIGN_PATH" || true
    fi
    ALIGN_ARGS=()
    if [ -f "$ALIGN_PATH" ]; then
        ALIGN_ARGS=(--align_path "$ALIGN_PATH")
    fi

    COVI_DIR="$PREP_ROOT_COMBO/covisible/$camera/1x"
    COVI_ARGS=()
    if [ "$USE_COVISIBLE" = "1" ] && [ -d "$COVI_DIR" ]; then
        COVI_ARGS=(--use_covisible --covisible_dir "$COVI_DIR")
    fi

    VIS_ARGS=()
    if [ "$VIS_CONCAT_GT" = "1" ]; then
        VIS_ARGS=(--vis_concat_gt)
    else
        VIS_ARGS=(--no-vis-concat-gt)
    fi

    echo ""
    echo "[eval] Combo $combo_idx / $((COMBO_COUNT-1)) :: $SCENE / $camera -> $OUT_DIR"
    CMD=(
        "$PYTHON_BIN" scripts/run_external_eval.py
        --train_result_dir "$RESULT_DIR"
        --train_data_dir "$TRAIN_DIR"
        --eval_dir "$EVAL_DIR"
        --out_dir "$OUT_DIR"
        --k "$EVAL_K"
        --eval_stride "$EVAL_STRIDE"
        --train_test_every "$TRAIN_TEST_EVERY"
        --eval_test_every "$EVAL_TEST_EVERY"
        "${ALIGN_ARGS[@]}"
        "${COVI_ARGS[@]}"
        "${VIS_ARGS[@]}"
    )
    echo "Command: ${CMD[*]}"
    "${CMD[@]}"
done

echo ""
echo "=========================================="
echo "STATIC EVAL COMPLETE"
echo "=========================================="
