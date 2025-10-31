#!/bin/bash
#SBATCH --job-name=gsplat_train
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --partition=default_partition
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 12:00:00
#SBATCH --mem 32gb

# Single training run SLURM script
# Usage: sbatch single_training.slurm DATA_DIR RESULT_DIR [MATCH_STRING]
# Example: sbatch single_training.slurm ../gs7/input_data/IMG_0597/inner_02 results/IMG_0597_tuned

source /home/wl757/.bashrc
conda activate gaussian_splatting

# Get arguments
DATA_DIR=$1
RESULT_DIR=$2
MATCH_STRING=${3:-""}

echo "=========================================="
echo "GSPLAT TRAINING JOB"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Data dir: $DATA_DIR"
echo "Result dir: $RESULT_DIR"
if [ -n "$MATCH_STRING" ]; then
    echo "Match string: $MATCH_STRING"
else
    echo "Match string: (none - using all images)"
fi
echo "=========================================="
echo ""

# Build the command
CMD="python examples/simple_trainer.py default \
    --disable_viewer \
    --data_factor 1 \
    --data_dir $DATA_DIR \
    --result_dir $RESULT_DIR \
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
    --antialiased"

# Add match_string if provided
if [ -n "$MATCH_STRING" ]; then
    CMD="$CMD --match_string $MATCH_STRING"
fi

echo "Executing: $CMD"
echo ""

# Run the training
eval $CMD

echo ""
echo "=========================================="
echo "JOB COMPLETE"
echo "=========================================="
