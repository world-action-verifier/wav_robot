#!/usr/bin/env bash
set -euo pipefail

# SpaceTime CLAM (STM) launch script: defaults are overridable via env vars
DATA_DIR="${DATA_DIR:-./IDM/data}"
# ZARR_PATH="${ZARR_PATH:-RoboVerse/data_policy/stack_cubeFrankaL0_obs:joint_pos_act:joint_pos_999.zarr}"
# Example absolute path has been removed for release portability.
ZARR_PATH="${ZARR_PATH:-RoboVerse/data_policy/stack_cube_random_controlFrankaL0_obs:joint_pos_act:joint_pos_1000.zarr}"
DATASET_NAME="${DATASET_NAME:-roboversetfds}"
# DATASET="${DATASET:-stack_cube}"
DATASET="${DATASET:-stack_cube_random_control}"
DATASET_VARIANT="${DATASET_VARIANT:-full_obs}"
SEQ_LEN="${SEQ_LEN:-5}"

PYTHONPATH=IDM python -m udrm.main --config-name train_st_vivit_clam_stm env=roboverse \
  env.dataset_name="${DATASET_NAME}" +env.datasets="[${DATASET}]" \
  data.data_dir="${DATA_DIR}" data.data_type=zarr \
  data.zarr_path="${ZARR_PATH}" \
  data.dataset_variant="${DATASET_VARIANT}" \
  data.seq_len="${SEQ_LEN}" \
  log_rollout_videos=True \
  model.la_dim=32 \
  model.fdm.k_step_pred=1 \
  data.seq_len=5 \
  model.fdm.tcn_loss_weight=0\
  model.fdm.tcn_tau=1.0 \
  "$@"
