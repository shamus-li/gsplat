#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="../gs7/dataset"
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
RESULT_ROOT=${RESULT_ROOT:-results}

mkdir -p slurm_logs

for scene in "${SCENES[@]}"; do
  scene_dir="${DATA_ROOT}/${scene}"
  prep_root="${scene_dir}/static/shared"
  sentinel="${prep_root}/.prep_complete"

  if [[ ! -d "${scene_dir}" ]]; then
    echo "Skipping ${scene}: scene directory ${scene_dir} not found" >&2
    continue
  fi

  if [[ ! -f "${sentinel}" ]]; then
    echo "Skipping ${scene}: shared static assets missing (${sentinel} not found)" >&2
    continue
  fi

  for cam in iphone stereo lightfield monocular; do
    covi_dir="${RESULT_ROOT}/static/${scene}/${cam}/covisible"
    if [[ -d "${covi_dir}" ]]; then
      echo "Removing stale covisible cache at ${covi_dir}"
      rm -rf "${covi_dir}"
    fi
  done

  echo "Launching static training for ${scene}"
  REBUILD_COVISIBLE=1 sbatch --requeue examples/static_training.slurm "${scene_dir}"
done
