#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

DATA_ROOT="../gs7/dataset"
IPHONE_MATCH="wide"
STEREO_MATCH="RIGHT"
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
MODALITIES=(iphone stereo)
RESULT_ROOT=${RESULT_ROOT:-results}

mkdir -p slurm_logs

for scene in "${SCENES[@]}"; do
  for modality in "${MODALITIES[@]}"; do
    data_dir="${DATA_ROOT}/${scene}/${modality}/train"
    eval_dir="${DATA_ROOT}/${scene}/${modality}/test"

    if [[ ! -d "${data_dir}" ]]; then
      echo "Skipping ${scene}/${modality}: ${data_dir} is missing" >&2
      continue
    fi
    if [[ ! -d "${eval_dir}/images" || ! -d "${eval_dir}/sparse" ]]; then
      echo "Skipping ${scene}/${modality}: ${eval_dir} missing images/ or sparse/" >&2
      continue
    fi

    if [[ "${modality}" == "iphone" ]]; then
      match_token="${IPHONE_MATCH}"
    else
      match_token="${STEREO_MATCH}"
    fi

    result_base="${RESULT_ROOT}/${scene}/${modality}"
    for split in combined filtered; do
      covisible_dir="${result_base}/${split}/covisible"
      if [[ -d "${covisible_dir}" ]]; then
        echo "Removing stale covisible cache at ${covisible_dir}"
        rm -rf "${covisible_dir}"
      fi
    done

    for variant in combined filtered; do
      echo "Launching ${scene}/${modality} (${variant}) with external eval (match=${match_token})"
      sbatch --requeue examples/dual_training.slurm "${data_dir}" "${match_token}" "${eval_dir}" "${variant}"
    done
  done
done
