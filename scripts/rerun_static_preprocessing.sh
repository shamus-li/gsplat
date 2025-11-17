#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-../gs7/dataset}

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: DATA_ROOT '${DATA_ROOT}' does not exist" >&2
  exit 1
fi

mkdir -p slurm_logs

shopt -s nullglob
scenes=("${DATA_ROOT}"/*)
shopt -u nullglob

if [[ ${#scenes[@]} -eq 0 ]]; then
  echo "No scene directories found under ${DATA_ROOT}" >&2
  exit 1
fi

for scene_dir in "${scenes[@]}"; do
  [[ -d "$scene_dir" ]] || continue
  scene_name=$(basename "$scene_dir")

  static_dir="${scene_dir}/static"
  eval_dir="${scene_dir}/iphone-eval"

  if [[ ! -d "${static_dir}" ]]; then
    echo "Skipping ${scene_name}: missing ${static_dir}" >&2
    continue
  fi
  if [[ ! -d "${eval_dir}" ]]; then
    echo "Skipping ${scene_name}: missing ${eval_dir}" >&2
    continue
  fi

  echo "Submitting static preprocessing for ${scene_name}"
  sbatch --requeue examples/static_preprocessing.slurm "${scene_dir}"
done
