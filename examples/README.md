# gsplat Training Scripts

SLURM scripts for launching gsplat training runs with custom hyperparameters.

## Scripts

### `single_training.slurm`
Run a single training job with all hyperparameters included.

```bash
sbatch examples/single_training.slurm DATA_DIR RESULT_DIR [MATCH_STRING]
```

**Examples:**
```bash
# All images
sbatch examples/single_training.slurm ../gs7/input_data/IMG_0597/inner_02 results/IMG_0597_tuned

# Filtered images
sbatch examples/single_training.slurm ../gs7/input_data/dog/vision-pro-dog results/dog_right RIGHT
```

### `dual_training.slurm`
Job array that runs TWO training jobs in parallel:
- Task 0: Full dataset (all images)
- Task 1: Filtered dataset (with match_string)

```bash
sbatch examples/dual_training.slurm DATA_DIR [MATCH_STRING] [EVAL_DIR]
```

Where:
- `DATA_DIR` is a COLMAP dataset root (expects `images/` and `sparse/`).
- `MATCH_STRING` (optional) filters the training set by a token in image names.
- `EVAL_DIR` (optional) is an existing COLMAP dataset directory (expects `images/` and `sparse/`) used for external evaluation. No preprocessing is performed.

**Examples:**
```bash
# All images + external eval dataset directory
sbatch examples/dual_training.slurm ../gs7/input_data/vision-pro-dog2 "" ../gs7/input_data/dog/eval/
# Filtered images + external eval dataset directory
sbatch examples/dual_training.slurm ../gs7/input_data/vision-pro-dog2 RIGHT ../gs7/input_data/dog/eval/
```

**Results:**
- `results/<basename>/combined` (task 0: all images)
- `results/<basename>/filtered` (task 1: filtered)
- `results/<basename>/<variant>/eval_on_<eval_dir_basename>` for external evaluation outputs

**Logs:**
- `slurm_logs/JOBID_0.out` (task 0)
- `slurm_logs/JOBID_1.err` (task 1)

## Monitoring Jobs

```bash
# Check status
squeue -u $USER

# Watch logs
tail -f slurm_logs/JOBID_0.out
tail -f slurm_logs/JOBID_1.out

# Cancel jobs
scancel JOBID
```

## Hyperparameters

All scripts include these hyperparameters:
- `--disable_viewer --data_factor 1 --save_ply --pose_opt --antialiased`
- `--eval_steps 3000 7000 30000`
- `--ply_steps 3000 7000 30000`
- `--strategy.reset_every 100000`
- `--strategy.pause_refine_after_reset 0`
- `--strategy.prune_scale3d 0.22`
- `--strategy.prune_scale2d 0.12`
- `--strategy.prune_opa 0.006`
- `--strategy.grow_grad2d 0.00035`
- `--strategy.grow_scale3d 0.012`
- `--strategy.refine_stop_iter 26000`
- `--strategy.refine_scale2d_stop_iter 26000`
- `--scale_reg 0.0005`
- `--scales_lr 0.003`
- `--means_lr 0.00012`

## SLURM Configuration

Each job requests:
- 1 GPU (high-end, excludes problematic nodes)
- 32GB RAM
- 12 hour time limit
- 2 CPUs
- Logs saved to `slurm_logs/`
