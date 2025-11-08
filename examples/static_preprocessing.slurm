#!/bin/bash
#SBATCH --job-name=gsplat_static_prep
#SBATCH --output=slurm_logs/prep_%j.out
#SBATCH --error=slurm_logs/prep_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem 64gb
#SBATCH -t 06:00:00
#SBATCH --requeue

# Preprocess a static scene on an H100 node to build shared datasets for
# subsequent gsplat training jobs.
#
# Usage:
#   sbatch examples/static_preprocessing.slurm ../gs7/dataset/dog
#
# Environment overrides:
#   PREP_ROOT=/custom/output
#   LF_INNER=2
#   LF_CALIB=/path/to/caldata.tar
#   LF_TO_COLMAP=/path/to/lf_to_colmap.py
#   VGGT_STAGE=both|vggt|ba
#   VGGT_CACHE=/tmp/vggt_cache
#   SKIP_VGGT=1
#   FORCE=1
#   COPY_MODE=symlink|copy|hardlink

set -euo pipefail

source /home/wl757/.bashrc
conda activate gaussian_splatting

SCENE_DIR=${1:-}
if [ -z "$SCENE_DIR" ]; then
    echo "ERROR: Missing SCENE_DIR argument."
    echo "Usage: sbatch examples/static_preprocessing.slurm ../gs7/dataset/<scene>"
    exit 1
fi

SCENE_DIR=${SCENE_DIR%/}
if [ ! -d "$SCENE_DIR" ]; then
    echo "ERROR: Scene directory not found: $SCENE_DIR"
    exit 1
fi

SCENE_NAME=$(basename "$SCENE_DIR")
STATIC_DIR="$SCENE_DIR/static"
if [ ! -d "$STATIC_DIR" ]; then
    echo "ERROR: Missing static/ directory at $STATIC_DIR"
    exit 1
fi

PREP_ROOT=${PREP_ROOT:-"$STATIC_DIR/shared"}
LF_INNER=${LF_INNER:-2}
COPY_MODE=${COPY_MODE:-symlink}

mkdir -p "$(dirname "$PREP_ROOT")" slurm_logs

echo "=========================================="
echo "GSPLAT STATIC PREPROCESSING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Scene: $SCENE_NAME"
echo "Scene dir: $SCENE_DIR"
echo "Prep root: $PREP_ROOT"
echo "LF inner: $LF_INNER"
echo "Copy mode: $COPY_MODE"
echo "Force rebuild: ${FORCE:-0}"
echo "Skip VGGT: ${SKIP_VGGT:-0}"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

CMD=(python "examples/static_training_prepare.py"
    --scene-dir "$SCENE_DIR"
    --output-dir "$PREP_ROOT"
    --lf-inner "$LF_INNER"
    --copy-mode "$COPY_MODE"
)

if [ -n "${LF_CALIB:-}" ]; then
    CMD+=(--lf-calib "$LF_CALIB")
fi
if [ -n "${LF_TO_COLMAP:-}" ]; then
    CMD+=(--lf-to-colmap "$LF_TO_COLMAP")
fi
if [ -n "${LF_DOWNSCALE:-}" ]; then
    CMD+=(--lf-downscale "$LF_DOWNSCALE")
fi
if [ -n "${CONDA_EXE_OVERRIDE:-}" ]; then
    CMD+=(--conda-exe "$CONDA_EXE_OVERRIDE")
fi
if [ -n "${CONDA_ENV_OVERRIDE:-}" ]; then
    CMD+=(--conda-env "$CONDA_ENV_OVERRIDE")
fi
if [ -n "${VGGT_SCRIPT:-}" ]; then
    CMD+=(--vggt-script "$VGGT_SCRIPT")
fi
if [ -n "${VGGT_STAGE:-}" ]; then
    CMD+=(--stage "$VGGT_STAGE")
fi
if [ -n "${VGGT_CACHE:-}" ]; then
    CMD+=(--stage-cache "$VGGT_CACHE")
fi
if [ "${SKIP_VGGT:-0}" = "1" ]; then
    CMD+=(--skip-vggt)
fi
if [ "${FORCE:-0}" = "1" ]; then
    CMD+=(--force)
fi

echo "Executing: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "=========================================="
echo "PREPROCESSING COMPLETE"
echo "Outputs in: $PREP_ROOT"
echo "=========================================="
