#!/bin/bash
#SBATCH --job-name=gsplat_static_train
#SBATCH --output=slurm_logs/static_train_%A_%a.out
#SBATCH --error=slurm_logs/static_train_%A_%a.err
#SBATCH --partition=monakhova
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 32gb
#SBATCH -t 12:00:00
#SBATCH --array=0-3

# Run gsplat training for four camera configurations (monocular, iphone, stereo,
# lightfield) using shared data prepared by static_preprocessing.slurm.

set -euo pipefail

source /home/wl757/.bashrc
conda activate gaussian_splatting

CAMERAS=(monocular iphone stereo lightfield)

SCENE_DIR=${1:-}
if [ -z "$SCENE_DIR" ]; then
    echo "ERROR: Missing SCENE_DIR argument."
    echo "Usage: sbatch examples/static_training.slurm ../gs7/dataset/<scene>"
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
COVI_CHUNK=${COVI_CHUNK:-32}
COVI_MICRO_CHUNK=${COVI_MICRO_CHUNK:-8}
COVI_DEVICE=${COVI_DEVICE:-cuda}

SENTINEL="$PREP_ROOT/.prep_complete"
if [ ! -f "$SENTINEL" ]; then
    echo "ERROR: Shared dataset sentinel missing at $SENTINEL"
    echo "Run examples/static_preprocessing.slurm first."
    exit 1
fi

DATA_DIR="$PREP_ROOT/subsets/$CAMERA"
EVAL_DIR="$PREP_ROOT/subsets/iphone_eval"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Subset directory not found for $CAMERA at $DATA_DIR"
    exit 1
fi
if [ ! -d "$EVAL_DIR" ]; then
    echo "ERROR: Eval subset not found at $EVAL_DIR"
    exit 1
fi

RESULT_DIR="$RESULT_ROOT/$SCENE_NAME/$CAMERA"
mkdir -p "$RESULT_DIR" slurm_logs

COVI_ROOT="$PREP_ROOT/covisible/$CAMERA"
COVI_DIR="$COVI_ROOT/1x"
ALIGN_PATH="$PREP_ROOT/alignments/${CAMERA}_to_eval.npz"

echo "=========================================="
echo "GSPLAT STATIC TRAINING"
echo "=========================================="
echo "Job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Scene: $SCENE_NAME"
echo "Camera: $CAMERA"
echo "Data dir: $DATA_DIR"
echo "Eval dir: $EVAL_DIR"
echo "Result dir: $RESULT_DIR"
echo "Prep root: $PREP_ROOT"
echo "Covisible dir: $COVI_DIR"
echo "Alignment: $ALIGN_PATH"
echo "=========================================="
echo ""

# Ensure covisible masks exist
if [ ! -d "$COVI_DIR/val" ]; then
    echo "Computing covisible masks for $CAMERA..."
    python examples/preprocess_covisible_colmap.py \
        --base_dir "$EVAL_DIR" \
        --support_dir "$DATA_DIR" \
        --base_split val \
        --support_split train \
        --factor 1 \
        --test_every "$EVAL_TEST_EVERY" \
        --support_test_every "$TRAIN_TEST_EVERY" \
        --chunk "$COVI_CHUNK" \
        ${COVI_MICRO_CHUNK:+--micro_chunk "$COVI_MICRO_CHUNK"} \
        --device "$COVI_DEVICE" \
        --output_dir "$COVI_ROOT"
else
    echo "Covisible masks already exist at $COVI_DIR/val"
fi

# Ensure alignment transform exists
if [ ! -f "$ALIGN_PATH" ]; then
    echo "Computing alignment transform for $CAMERA..."
    python - "$DATA_DIR" "$EVAL_DIR" "$ALIGN_PATH" "$TRAIN_TEST_EVERY" "$EVAL_TEST_EVERY" <<'PY'
import os
import sys
import numpy as np
from pathlib import Path
sys.path.append('examples')
from datasets.colmap import Parser

train_dir = Path(sys.argv[1])
eval_dir = Path(sys.argv[2])
out_path = Path(sys.argv[3])
train_every = int(sys.argv[4])
eval_every = int(sys.argv[5])

train_parser = Parser(data_dir=str(train_dir), factor=1, normalize=True, test_every=train_every)
eval_parser = Parser(data_dir=str(eval_dir), factor=1, normalize=True, test_every=eval_every)
align = train_parser.transform @ np.linalg.inv(eval_parser.transform)
out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    out_path,
    align_transform=align.astype(np.float32),
    base_transform=train_parser.transform.astype(np.float32),
    support_transform=eval_parser.transform.astype(np.float32),
)
print(f"Wrote alignment to {out_path}")
PY
else
    echo "Alignment present at $ALIGN_PATH"
fi

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    TRAIN_CMD=(python examples/simple_trainer.py default
        --disable_viewer
        --data_factor 1
        --data_dir "$DATA_DIR"
        --result_dir "$RESULT_DIR"
        --save_ply
        --pose_opt
        --eval_steps 3000 7000 30000
        --ply_steps 3000 7000 30000
        --strategy.reset_every 100000
        --strategy.pause_refine_after_reset 0
        --strategy.prune_scale3d 0.22
        --strategy.prune_scale2d 0.12
        --strategy.prune_opa 0.006
        --strategy.grow_grad2d 0.00035
        --strategy.grow_scale3d 0.012
        --strategy.refine_stop_iter 26000
        --strategy.refine_scale2d_stop_iter 26000
        --scale_reg 0.0005
        --scales_lr 0.003
        --means_lr 0.00012
        --antialiased
        --test_every "$TRAIN_TEST_EVERY"
        --eval_use_covisible
        --eval_covisible_dir "$COVI_DIR"
    )
    echo "Executing training: ${TRAIN_CMD[*]}"
    "${TRAIN_CMD[@]}"
else
    echo "SKIP_TRAIN=1 -> skipping training."
fi

# Symlink covisible cache into results for reference
if [ -d "$COVI_DIR" ]; then
    mkdir -p "$RESULT_DIR/covisible"
    if [ ! -e "$RESULT_DIR/covisible/1x" ]; then
        ln -s "$(realpath "$COVI_DIR")" "$RESULT_DIR/covisible/1x"
    fi
fi

if [ "${SKIP_EVAL:-0}" != "1" ]; then
    CKPT=$(ls -1 "$RESULT_DIR"/ckpts/ckpt_*_rank0.pt 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -z "$CKPT" ]; then
        echo "WARNING: No checkpoint found; skipping external eval."
    else
        EVAL_OUT_DIR="$RESULT_DIR/eval_on_iphone_eval"
        mkdir -p "$EVAL_OUT_DIR"
        EVAL_CMD=(python examples/simple_trainer.py default
            --disable_viewer
            --data_factor 1
            --data_dir "$EVAL_DIR"
            --result_dir "$EVAL_OUT_DIR"
            --ckpt "$CKPT"
            --test_every "$EVAL_TEST_EVERY"
            --eval_use_covisible
            --eval_covisible_dir "$COVI_DIR"
        )
        if [ -f "$ALIGN_PATH" ]; then
            EVAL_CMD+=(--dataset_transform_path "$ALIGN_PATH")
        fi
        echo "Executing external eval: ${EVAL_CMD[*]}"
        "${EVAL_CMD[@]}"
    fi
else
    echo "SKIP_EVAL=1 -> skipping external evaluation."
fi

echo ""
echo "=========================================="
echo "TASK $SLURM_ARRAY_TASK_ID ($CAMERA) COMPLETE"
echo "Results stored in $RESULT_DIR"
echo "=========================================="
