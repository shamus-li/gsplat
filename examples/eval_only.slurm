#!/bin/bash
#SBATCH --job-name=gsplat_eval
#SBATCH --output=slurm_logs/gsplat_eval_%A_%a.out
#SBATCH --error=slurm_logs/gsplat_eval_%A_%a.err
#SBATCH --partition=default_partition
#SBATCH --exclude=lil-compute-05,unicorn-compute-01,bhattacharjee-compute-02,awang-compute-01,kim-compute-03,kim-compute-04,kim-compute-05
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu-high"
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 24gb
#SBATCH -t 06:00:00
#SBATCH --requeue

# Unified eval-only SLURM workflow for gsplat runs.
# - Auto-discovers scenes under a dataset root (default ../gs7/dataset)
# - For each scene, iterates cameras (default: iphone, stereo)
# - Detects whether the result layout is single-run (results/<scene>/<camera>)
#   or dual-training (results/<scene>/<camera>/{combined,filtered})
# - Refreshes eval renders for every checkpointed branch using scripts/run_external_eval.py
# - Handles dataset alignment caching, covisible mask generation, and pose-opt forwarding.
#
# Usage examples:
#   sbatch examples/eval_only.slurm                         # iterate all scenes serially
#   sbatch --array=0-7 examples/eval_only.slurm ../gs7/dataset results
#   SCENE_LIST="dog chicken" sbatch --array=0-1 examples/eval_only.slurm
#
# Environment knobs:
#   CAMERAS                Space-separated list of camera folders (default: "iphone stereo")
#   TRAIN_SUBPATH          Sub-folder under each camera containing training data (default: train)
#   EVAL_SUBPATH           Sub-folder for evaluation data (default: test)
#   TRAIN_TEST_EVERY       COLMAP test_every used for train split (default: 8)
#   EVAL_TEST_EVERY        COLMAP test_every used for eval split (default: 1)
#   EVAL_K                 Number of eval frames to render (default: 0 = all)
#   EVAL_STRIDE            Evaluate every Nth frame (default: 8)
#   USE_COVISIBLE          Whether to generate/use covisible masks (default: 1)
#   VIS_CONCAT_GT          Concatenate GT alongside renders (default: 1)
#   COVI_BATCH_SIZE        Batch size for covisible generation (default: 8)
#   COVI_NUM_WORKERS       DataLoader workers for covisible generation (default: 2)
#   COVI_DEVICE            Torch device for covisible script (default: cuda)
#   COVI_MAX_HW            Optional max resolution for covisible script (default: 512)
#   COVI_SUPPORT_SPLIT     Support split passed to covisible script (default: train)
#   SCENE_LIST             Optional whitespace/comma separated list overriding auto-discovery
#   SCENE_FILE             Optional newline-separated file with scenes (overrides auto-discovery)
#   RENDER_VIDEO           Set to 1 to enable eval video renders
#   FORCE_ALIGN_REGEN      Set to 1 to discard cached alignments before recomputing (default: 0)

set -euo pipefail

source /home/wl757/.bashrc
conda activate gaussian_splatting

# Ensure working directory is the repo root so relative paths align.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR" || {
        echo "ERROR: Failed to cd into $SLURM_SUBMIT_DIR" >&2
        exit 1
    }
fi
REPO_ROOT="$(pwd)"

# Arguments
DATA_ROOT_INPUT=${1:-"../gs7/dataset"}
RESULT_ROOT_INPUT=${2:-"results"}

DATA_ROOT="$(realpath -m "$DATA_ROOT_INPUT")"
if [[ "${RESULT_ROOT_INPUT#/}" == "$RESULT_ROOT_INPUT" ]]; then
    RESULT_ROOT="$REPO_ROOT/$RESULT_ROOT_INPUT"
else
    RESULT_ROOT="$RESULT_ROOT_INPUT"
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: DATA_ROOT not found: $DATA_ROOT" >&2
    exit 1
fi
if [[ ! -d "$RESULT_ROOT" ]]; then
    echo "ERROR: RESULT_ROOT not found: $RESULT_ROOT" >&2
    exit 1
fi

IFS=' ' read -r -a CAMERAS <<< "${CAMERAS:-iphone stereo}"
TRAIN_SUBPATH=${TRAIN_SUBPATH:-train}
EVAL_SUBPATH=${EVAL_SUBPATH:-test}
TRAIN_TEST_EVERY=${TRAIN_TEST_EVERY:-8}
EVAL_TEST_EVERY=${EVAL_TEST_EVERY:-1}
EVAL_K=${EVAL_K:-0}
EVAL_STRIDE=${EVAL_STRIDE:-8}
USE_COVISIBLE=${USE_COVISIBLE:-1}
VIS_CONCAT_GT=${VIS_CONCAT_GT:-1}
COVI_BATCH_SIZE=${COVI_BATCH_SIZE:-8}
COVI_NUM_WORKERS=${COVI_NUM_WORKERS:-2}
COVI_DEVICE=${COVI_DEVICE:-cuda}
COVI_MAX_HW=${COVI_MAX_HW:-512}
COVI_SUPPORT_SPLIT=${COVI_SUPPORT_SPLIT:-train}
RENDER_VIDEO=${RENDER_VIDEO:-0}
FORCE_ALIGN_REGEN=${FORCE_ALIGN_REGEN:-0}

SCENE_SOURCE="auto"
if [[ -n "${SCENE_FILE:-}" ]]; then
    if [[ ! -f "$SCENE_FILE" ]]; then
        echo "ERROR: SCENE_FILE '$SCENE_FILE' not found" >&2
        exit 1
    fi
    mapfile -t SCENES <"$SCENE_FILE"
    SCENE_SOURCE="file:$SCENE_FILE"
elif [[ -n "${SCENE_LIST:-}" ]]; then
    IFS=', ' read -r -a SCENES <<< "${SCENE_LIST}" || true
    SCENE_SOURCE="env"
else
    mapfile -t SCENES < <(find -L "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V)
fi

SCENES=($(printf '%s\n' "${SCENES[@]}" | awk 'length' | sort -u))
# Drop umbrella folders (e.g., dynamic/, static/) that merely group scenes.
EXCLUDES=(dynamic static)
FILTERED=()
for scene in "${SCENES[@]}"; do
    skip=0
    for token in "${EXCLUDES[@]}"; do
        if [[ "$scene" == "$token" ]]; then
            skip=1
            break
        fi
    done
    [[ "$skip" -eq 0 ]] && FILTERED+=("$scene")
done
SCENES=("${FILTERED[@]}")
SCENE_COUNT=${#SCENES[@]}
if [[ "$SCENE_COUNT" -eq 0 ]]; then
    echo "ERROR: No scenes discovered under $DATA_ROOT (source=$SCENE_SOURCE)" >&2
    exit 1
fi

echo "=========================================="
echo "GSPLAT EVAL-ONLY"
echo "=========================================="
echo "Data root:  $DATA_ROOT"
echo "Result root: $RESULT_ROOT"
echo "Scenes: ${SCENES[*]}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Array index: ${SLURM_ARRAY_TASK_ID}"
else
    echo "Array index: (none; running all combos serially)"
fi
echo "Cameras: ${CAMERAS[*]}"
echo "Eval stride: $EVAL_STRIDE"
echo "=========================================="

RUN_EXTERNAL_SCRIPT="$REPO_ROOT/scripts/run_external_eval.py"
PREPROCESS_COVI="$REPO_ROOT/examples/preprocess_covisible_colmap.py"
ALIGN_SCRIPT="$REPO_ROOT/examples/compute_dataset_alignment.py"

declare -a COMBO_SCENES=()
declare -a COMBO_CAMS=()
declare -a COMBO_RUN_DIRS=()
declare -a COMBO_TRAIN_DIRS=()
declare -a COMBO_EVAL_DIRS=()
declare -a COMBO_EVAL_SUFFIXES=()

for SCENE in "${SCENES[@]}"; do
    echo "\n---------- Probing scene: $SCENE ----------"
    for CAM in "${CAMERAS[@]}"; do
        train_dir="$DATA_ROOT/$SCENE/$CAM/$TRAIN_SUBPATH"
        eval_dir="$DATA_ROOT/$SCENE/$CAM/$EVAL_SUBPATH"

        if [[ ! -d "$eval_dir/images" || ! -d "$eval_dir/sparse" && ! -d "$eval_dir/sparse/0" ]]; then
            echo "[skip] $SCENE/$CAM: missing eval dataset at $eval_dir"
            continue
        fi
        if [[ ! -d "$train_dir/images" || ! -d "$train_dir/sparse" ]]; then
            alt_train="$DATA_ROOT/$SCENE/$CAM/train"
            if [[ -d "$alt_train/images" && -d "$alt_train/sparse" ]]; then
                train_dir="$alt_train"
            else
                echo "[warn] $SCENE/$CAM: training dataset '$train_dir' missing; alignment may fail"
            fi
        fi

        eval_suffix=$(basename "$eval_dir")
        result_base="$RESULT_ROOT/$SCENE/$CAM"
        declare -a RUN_DIRS=()
        if [[ -d "$result_base/combined" || -d "$result_base/filtered" ]]; then
            for variant in combined filtered; do
                run_dir="$result_base/$variant"
                [[ -d "$run_dir/ckpts" ]] && RUN_DIRS+=("$run_dir")
            done
        else
            [[ -d "$result_base/ckpts" ]] && RUN_DIRS+=("$result_base")
        fi

        if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
            echo "[skip] $SCENE/$CAM: no checkpoints under $result_base"
            continue
        fi

        for RUN_DIR in "${RUN_DIRS[@]}"; do
            ckpt=$(ls -1 "$RUN_DIR"/ckpts/ckpt_*_rank0.pt 2>/dev/null | sort -V | tail -n 1 || true)
            if [[ -z "$ckpt" ]]; then
                echo "[skip] $RUN_DIR: no checkpoints"
                continue
            fi

            COMBO_SCENES+=("$SCENE")
            COMBO_CAMS+=("$CAM")
            COMBO_RUN_DIRS+=("$(realpath -m "$RUN_DIR")")
            COMBO_TRAIN_DIRS+=("$(realpath -m "$train_dir")")
            COMBO_EVAL_DIRS+=("$(realpath -m "$eval_dir")")
            COMBO_EVAL_SUFFIXES+=("$eval_suffix")
        done
    done
done

COMBO_COUNT=${#COMBO_SCENES[@]}
if [[ "$COMBO_COUNT" -eq 0 ]]; then
    echo "ERROR: No runnable (scene, camera, run) combinations discovered."
    exit 1
fi

echo "Discovered $COMBO_COUNT runnable combinations."
echo "Use: sbatch --array=0-$((COMBO_COUNT-1)) examples/eval_only.slurm ..."

if [[ "${LIST_EVAL_COMBOS:-0}" == "1" ]]; then
    for ((combo_idx = 0; combo_idx < COMBO_COUNT; combo_idx++)); do
        printf '[combo %d] %s / %s / %s\n' \
            "$combo_idx" \
            "${COMBO_SCENES[$combo_idx]}" \
            "${COMBO_CAMS[$combo_idx]}" \
            "${COMBO_RUN_DIRS[$combo_idx]#$RESULT_ROOT/}"
    done
    exit 0
fi

declare -a SELECTED_INDICES=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SELECTED_INDICES=("$SLURM_ARRAY_TASK_ID")
else
    for ((combo_idx = 0; combo_idx < COMBO_COUNT; combo_idx++)); do
        SELECTED_INDICES+=("$combo_idx")
    done
fi

for combo_idx in "${SELECTED_INDICES[@]}"; do
    if (( combo_idx < 0 || combo_idx >= COMBO_COUNT )); then
        echo "ERROR: combo index $combo_idx out of range (0..$((COMBO_COUNT-1)))"
        exit 1
    fi

    SCENE="${COMBO_SCENES[$combo_idx]}"
    CAM="${COMBO_CAMS[$combo_idx]}"
    RUN_DIR="${COMBO_RUN_DIRS[$combo_idx]}"
    train_dir="${COMBO_TRAIN_DIRS[$combo_idx]}"
    eval_dir="${COMBO_EVAL_DIRS[$combo_idx]}"
    eval_suffix="${COMBO_EVAL_SUFFIXES[$combo_idx]}"

    if [[ ! -d "$RUN_DIR" ]]; then
        echo "[skip] combo $combo_idx: run dir missing ($RUN_DIR)"
        continue
    fi
    if [[ ! -d "$eval_dir/images" ]]; then
        echo "[skip] combo $combo_idx: eval dataset missing ($eval_dir)"
        continue
    fi

    echo "\n========== Combo $combo_idx / $((COMBO_COUNT-1)) :: $SCENE / $CAM / ${RUN_DIR#$RESULT_ROOT/} =========="

    ckpt=$(ls -1 "$RUN_DIR"/ckpts/ckpt_*_rank0.pt 2>/dev/null | sort -V | tail -n 1 || true)
    if [[ -z "$ckpt" ]]; then
        echo "[skip] $RUN_DIR: no checkpoints"
        continue
    fi

    out_dir="$RUN_DIR/eval_on_${eval_suffix}"
    rm -rf "$out_dir"
    mkdir -p "$out_dir" slurm_logs

    align_root="$RUN_DIR/alignments"
    mkdir -p "$align_root"
    train_norm_path="$align_root/train_normalization.npz"
    eval_align_path="$align_root/${eval_suffix}_to_train.npz"
    if [[ "$FORCE_ALIGN_REGEN" == "1" ]]; then
        if [[ -f "$eval_align_path" ]]; then
            echo "[info] FORCE_ALIGN_REGEN=1, removing $eval_align_path"
            rm -f "$eval_align_path"
        fi
        if [[ -f "$train_norm_path" ]]; then
            echo "[info] FORCE_ALIGN_REGEN=1, removing $train_norm_path"
            rm -f "$train_norm_path"
        fi
    fi
    dataset_transform_arg=()
    if [[ ! -f "$eval_align_path" && -d "$train_dir/images" ]]; then
        if python "$ALIGN_SCRIPT" \
            --train-dir "$train_dir" \
            --subset-dir "$eval_dir" \
            --train-test-every "$TRAIN_TEST_EVERY" \
            --eval-test-every "$EVAL_TEST_EVERY" \
            --output "$eval_align_path"; then
            echo "[info] Wrote alignment to $eval_align_path"
        else
            echo "[warn] Failed to compute alignment for ${RUN_DIR#$RESULT_ROOT/}"
            rm -f "$eval_align_path"
        fi
    fi
    if [[ ! -f "$train_norm_path" && -d "$train_dir/images" ]]; then
        if python "$ALIGN_SCRIPT" \
            --train-dir "$train_dir" \
            --train-test-every "$TRAIN_TEST_EVERY" \
            --output "$train_norm_path"; then
            echo "[info] Wrote train normalization to $train_norm_path"
        else
            echo "[warn] Failed to compute train normalization for ${RUN_DIR#$RESULT_ROOT/}"
            rm -f "$train_norm_path"
        fi
    fi
    if [[ -f "$eval_align_path" ]]; then
        dataset_transform_arg=(--align_path "$eval_align_path")
        echo "[align] Using eval-to-train alignment at $eval_align_path"
    elif [[ -f "$train_norm_path" ]]; then
        dataset_transform_arg=(--align_path "$train_norm_path")
        echo "[align] Falling back to train normalization at $train_norm_path"
    else
        echo "[warn] No alignment metadata available; eval poses may drift."
    fi

    covi_args=()
    if [[ "$USE_COVISIBLE" == "1" ]]; then
        covi_root="$RUN_DIR/covisible/${eval_suffix}"
        covi_stage="$covi_root/1x/val"
        # if [[ -d "$covi_root" ]]; then
        #     echo "[info] Removing stale covisible cache at $covi_root"
        #     rm -rf "$covi_root"
        # fi
        if [[ ! -d "$covi_stage" ]]; then
            support_dir="$train_dir"
            if [[ ! -d "$support_dir/images" ]]; then
                support_dir="$eval_dir"
            fi
            if [[ -d "$support_dir/images" ]]; then
                mkdir -p "$RUN_DIR/covisible"
                echo "[info] Generating covisible masks in $covi_root"
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                python "$PREPROCESS_COVI" \
                    --base_dir "$eval_dir" \
                    --support_dir "$support_dir" \
                    --factor 1 \
                    --test_every "$EVAL_TEST_EVERY" \
                    --support_test_every "$TRAIN_TEST_EVERY" \
                    --base_split val \
                    --support_split "$COVI_SUPPORT_SPLIT" \
                    --batch_size "$COVI_BATCH_SIZE" \
                    --num_workers "$COVI_NUM_WORKERS" \
                    --max_hw "$COVI_MAX_HW" \
                    --device "$COVI_DEVICE" \
                    --output_dir "$covi_root"
            else
                echo "[warn] ${RUN_DIR#$RESULT_ROOT/}: support dataset missing; skipping covisible generation"
            fi
        fi
        if [[ -d "$covi_stage" ]]; then
            covi_args=(--use_covisible --covisible_dir "$covi_root/1x")
        else
            echo "[warn] ${RUN_DIR#$RESULT_ROOT/}: covisible directory missing at $covi_stage"
        fi
    fi

    vis_args=()
    if [[ "$VIS_CONCAT_GT" == "1" ]]; then
        vis_args=(--vis_concat_gt)
    fi

    video_args=()
    if [[ "$RENDER_VIDEO" == "1" ]]; then
        video_args=(--render_video)
    fi

    echo "[eval] ${RUN_DIR#$RESULT_ROOT/} -> $out_dir"
    CMD=(python "$RUN_EXTERNAL_SCRIPT"
        --train_result_dir "$RUN_DIR"
        --train_data_dir "$train_dir"
        --eval_dir "$eval_dir"
        --out_dir "$out_dir"
        --k "$EVAL_K"
        --eval_stride "$EVAL_STRIDE"
        --train_test_every "$TRAIN_TEST_EVERY"
        --eval_test_every "$EVAL_TEST_EVERY")

    if [[ ${#dataset_transform_arg[@]} -gt 0 ]]; then
        CMD+=("${dataset_transform_arg[@]}")
    fi
    if [[ ${#covi_args[@]} -gt 0 ]]; then
        CMD+=("${covi_args[@]}")
    fi
    if [[ ${#vis_args[@]} -gt 0 ]]; then
        CMD+=("${vis_args[@]}")
    fi
    if [[ ${#video_args[@]} -gt 0 ]]; then
        CMD+=("${video_args[@]}")
    fi

    echo "Command: ${CMD[*]}"
    "${CMD[@]}"
done

echo "\n=========================================="
echo "GSPLAT EVAL COMPLETE"
echo "=========================================="
