#!/bin/bash
#SBATCH --job-name=gsplat_dual
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --partition=monakhova
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH --mem 32gb
#SBATCH --array=0-1

# Dual training + optional external-eval SLURM job array
# Array task 0: Full training (all images)
# Array task 1: Filtered training (optional match_string)
#
# Usage: sbatch examples/dual_training.slurm DATA_DIR [MATCH_STRING] [EVAL_DIR]
#   - DATA_DIR:       Path to COLMAP dataset root (expects 'images/' and 'sparse/')
#   - MATCH_STRING:   Optional token to filter training images (regex-safe word match)
#   - EVAL_DIR:       Optional path to an existing COLMAP dataset directory for external eval
#                     (expects 'images/' and 'sparse/'). No preprocessing is performed.
#
# The result base is inferred from the basename of DATA_DIR and written to
# results/<basename>/{combined,filtered}. Set RESULT_ROOT to override the
# results directory prefix (defaults to "results").
#
# Example (filtered run on RIGHT images with external eval dataset dir):
#   sbatch examples/dual_training.slurm ../gs7/input_data/vision-pro-dog RIGHT ../gs7/input_data/dog/eval/

source /home/wl757/.bashrc
conda activate gaussian_splatting

# Get arguments
DATA_DIR=$1
MATCH_STRING=${2:-""}
# Optional separate COLMAP dataset directory to evaluate on
EVAL_DIR=${3:-""}

# Normalize dataset arguments
DATA_DIR=${DATA_DIR%/}
EVAL_DIR=${EVAL_DIR%/}

# Evaluation uses the training match string by default unless overridden.
if [ -n "${EVAL_MATCH_STRING:-}" ]; then
    EVAL_MATCH_STR="$EVAL_MATCH_STRING"
elif [ -n "$EVAL_DIR" ] && [ "$EVAL_DIR" != "$DATA_DIR" ]; then
    EVAL_MATCH_STR=""
else
    EVAL_MATCH_STR="$MATCH_STRING"
fi

# Check required arguments
if [ -z "$DATA_DIR" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: sbatch examples/dual_training.slurm DATA_DIR [MATCH_STRING] [EVAL_DIR]"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory '$DATA_DIR' does not exist."
    exit 1
fi

RESULT_ROOT=${RESULT_ROOT:-results}
DATA_NAME=$(basename "$DATA_DIR")
RESULT_BASE="$RESULT_ROOT/$DATA_NAME"
COMBINED_RESULT_DIR="$RESULT_BASE/combined"
FILTERED_RESULT_DIR="$RESULT_BASE/filtered"

mkdir -p "$COMBINED_RESULT_DIR" "$FILTERED_RESULT_DIR"

# Set result directory and match string based on array task ID
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    RESULT_DIR="$COMBINED_RESULT_DIR"
    JOB_DATA_DIR="$DATA_DIR"
    MATCH_STR=""
    RUN_TYPE="Full (combined)"
else
    RESULT_DIR="$FILTERED_RESULT_DIR"
    JOB_DATA_DIR="$DATA_DIR"
    MATCH_STR="$MATCH_STRING"
    if [ -n "$MATCH_STR" ]; then
        RUN_TYPE="Filtered (match '$MATCH_STR')"
    else
        RUN_TYPE="Filtered (no match string)"
    fi
fi

echo "=========================================="
echo "GSPLAT DUAL TRAINING - ARRAY JOB"
echo "=========================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Run type: $RUN_TYPE"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Data dir: $JOB_DATA_DIR"
echo "Result base: $RESULT_BASE"
echo "Result dir: $RESULT_DIR"
if [ -n "$MATCH_STR" ]; then
    echo "Match string: $MATCH_STR"
else
    echo "Match string: (none)"
fi
if [ -n "$EVAL_MATCH_STR" ]; then
    echo "Eval match string: $EVAL_MATCH_STR"
else
    echo "Eval match string: (none)"
fi
if [ -n "$EVAL_DIR" ]; then
    echo "Eval dataset: $EVAL_DIR"
else
    echo "Eval dataset: (none)"
fi
echo "=========================================="
echo ""

# Optionally compute covisible masks for external evaluation only
TRAIN_TEST_EVERY=${TRAIN_TEST_EVERY:-8}
EVAL_TEST_EVERY=${EVAL_TEST_EVERY:-1}
USE_COVISIBLE_EXTERNAL=${USE_COVISIBLE_EXTERNAL:-1}
COVI_MICRO_CHUNK=${COVI_MICRO_CHUNK:-8}
COVI_DEVICE=${COVI_DEVICE:-cuda}
EVAL_ALIGN_PATH=${EVAL_ALIGN_PATH:-}
ALIGN_ROOT=${ALIGN_ROOT:-$RESULT_DIR/alignments}

# Build the command
CMD="python examples/simple_trainer.py default \
    --disable_viewer \
    --data_factor 1 \
    --data_dir \"$JOB_DATA_DIR\" \
    --result_dir \"$RESULT_DIR\" \
    --save_ply \
    --pose_opt \
    --eval_steps 3000 7000 30000 \
    --ply_steps 3000 7000 30000 \
    --strategy.reset_every 100000 \
    --strategy.pause_refine_after_reset 0 \
    --strategy.prune_scale3d 0.22 \
    --strategy.prune_scale2d 0.12 \
    --strategy.prune_opa 0.006 \
    --strategy.grow_grad2d 0.00035 \
    --strategy.grow_scale3d 0.012 \
    --strategy.refine_stop_iter 26000 \
    --strategy.refine_scale2d_stop_iter 26000 \
    --scale_reg 0.0005 \
    --scales_lr 0.003 \
    --means_lr 0.00012 \
    --antialiased \
    --test_every $TRAIN_TEST_EVERY"

# Add match_string if this is the filtered run and a match string is provided
if [ -n "$MATCH_STR" ]; then
    CMD="$CMD --match_string \"$MATCH_STR\""
fi
if [ -n "$EVAL_MATCH_STR" ]; then
    CMD="$CMD --eval_match_string \"$EVAL_MATCH_STR\""
fi

echo "Executing: $CMD"
echo ""

# Run the training
eval "$CMD"

# Link any precomputed covisible masks from the dataset into this run.
if [ -d "$DATA_DIR/covisible" ]; then
    echo "Linking precomputed covisible caches from $DATA_DIR/covisible"
    mkdir -p "$RESULT_DIR/covisible"
    for subset_path in "$DATA_DIR"/covisible/*; do
        [ -d "$subset_path" ] || continue
        subset_name=$(basename "$subset_path")
        dest="$RESULT_DIR/covisible/$subset_name"
        if [ ! -e "$dest" ]; then
            ln -s "$(realpath "$subset_path")" "$dest"
        fi
    done
fi

# If an eval dataset directory is provided, run eval-only using the latest
# checkpoint produced by this run. Results go to: $RESULT_DIR/eval_on_<eval_dir_basename>
if [ -n "$EVAL_DIR" ]; then
    echo ""
    EVAL_DATA_DIR="$EVAL_DIR"
    EVAL_STEM=$(basename "$EVAL_DATA_DIR")

    if [ -d "$EVAL_DATA_DIR/images" ] && [ -d "$EVAL_DATA_DIR/sparse" ]; then
        echo "Finding latest checkpoint in: $RESULT_DIR/ckpts"
        CKPT=$(ls -1 "$RESULT_DIR"/ckpts/ckpt_*_rank0.pt 2>/dev/null | sort -V | tail -n 1)
        if [ -z "$CKPT" ]; then
            echo "WARNING: No checkpoint found; skipping eval-only."
        else
            EVAL_OUT_DIR="$RESULT_DIR/eval_on_${EVAL_STEM}"
            mkdir -p "$EVAL_OUT_DIR"
            if [ -z "$EVAL_ALIGN_PATH" ]; then
                mkdir -p "$ALIGN_ROOT"
                default_align="$ALIGN_ROOT/${EVAL_STEM}_to_train.npz"
                if [ ! -f "$default_align" ]; then
                    echo "Generating alignment transform at $default_align"
                    python - "$DATA_DIR" "$EVAL_DATA_DIR" "$TRAIN_TEST_EVERY" "$EVAL_TEST_EVERY" "$default_align" <<'PY'
import sys
import numpy as np
from pathlib import Path

sys.path.append('examples')
try:
    from datasets.colmap import Parser  # type: ignore
except Exception as exc:
    print(f"WARNING: could not import datasets.colmap.Parser ({exc}); skipping alignment.")
    sys.exit(0)

data_dir = Path(sys.argv[1]).resolve()
subset_dir = Path(sys.argv[2]).resolve()
train_every = int(sys.argv[3])
eval_every = int(sys.argv[4])
out_path = Path(sys.argv[5]).resolve()

try:
    train_parser = Parser(data_dir=str(data_dir), factor=1, normalize=True, test_every=train_every)
    subset_parser = Parser(data_dir=str(subset_dir), factor=1, normalize=True, test_every=eval_every)
except Exception as exc:
    print(f"WARNING: failed to parse datasets for alignment ({exc}); skipping.")
    sys.exit(0)

align = train_parser.transform @ np.linalg.inv(subset_parser.transform)
out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    out_path,
    align_transform=align.astype(np.float32),
    base_transform=train_parser.transform.astype(np.float32),
    support_transform=subset_parser.transform.astype(np.float32),
)
print(f"Wrote alignment transform to {out_path}")
PY
                fi
                if [ -f "$default_align" ]; then
                    EVAL_ALIGN_PATH="$default_align"
                fi
            fi
            if [ "$USE_COVISIBLE_EXTERNAL" = "1" ]; then
                COVI_OUTPUT_ROOT="$RESULT_DIR/covisible/$EVAL_STEM"
                COVI_MASK_DIR="$COVI_OUTPUT_ROOT/1x/val"
                if [ ! -d "$COVI_MASK_DIR" ]; then
                    echo "Precomputing covisible masks for external eval dataset..."
                    python examples/preprocess_covisible_colmap.py \
                        --base_dir "$EVAL_DATA_DIR" \
                        --support_dir "$JOB_DATA_DIR" \
                        --factor 1 \
                        --test_every 1 \
                        --support_test_every "$TRAIN_TEST_EVERY" \
                        --micro_chunk "$COVI_MICRO_CHUNK" \
                        --device "$COVI_DEVICE" \
                        --output_dir "$COVI_OUTPUT_ROOT" \
                        --base_split val \
                        --support_split train \
                        --chunk 32
                else
                    echo "Covisible masks exist at $COVI_MASK_DIR; skipping generation."
                fi
                if [ -z "$EVAL_ALIGN_PATH" ]; then
                    covi_align="$COVI_OUTPUT_ROOT/alignment.npz"
                    if [ -f "$covi_align" ]; then
                        EVAL_ALIGN_PATH="$covi_align"
                    fi
                fi
            fi
            EVAL_CMD="python examples/simple_trainer.py default \
                --disable_viewer \
                --data_factor 1 \
                --data_dir \"$EVAL_DATA_DIR\" \
                --result_dir \"$EVAL_OUT_DIR\" \
                --ckpt \"$CKPT\" \
                --test_every 1"
            if [ -n "$EVAL_ALIGN_PATH" ] && [ -f "$EVAL_ALIGN_PATH" ]; then
                echo "Using dataset transform: $EVAL_ALIGN_PATH"
                EVAL_CMD="$EVAL_CMD --dataset_transform_path \"$EVAL_ALIGN_PATH\""
            fi
            if [ "$USE_COVISIBLE_EXTERNAL" = "1" ]; then
                EVAL_CMD="$EVAL_CMD --eval_use_covisible --eval_covisible_dir \"$COVI_OUTPUT_ROOT/1x\""
            fi
            echo "Executing external eval: $EVAL_CMD"
            eval "$EVAL_CMD"
        fi
    else
        echo "WARNING: Expected eval dataset at '$EVAL_DATA_DIR' not found (need 'images/' and 'sparse/'); skipping eval-only."
    fi
fi

echo ""
echo "=========================================="
echo "ARRAY TASK $SLURM_ARRAY_TASK_ID COMPLETE"
echo "=========================================="
