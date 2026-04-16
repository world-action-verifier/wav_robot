#!/usr/bin/env bash
set -euo pipefail

# IDM + Action Decoder only training
# Keep defaults aligned with run_st_vivit_clam_stm.sh (overridable via env vars)

CONFIG_NAME="${CONFIG_NAME:-train_st_vivit_clam_stm}"
ENV_NAME="${ENV_NAME:-roboverse}"
DATASET_NAME="${DATASET_NAME:-roboversetfds}"
DATA_DIR="${DATA_DIR:-./IDM/data}"
# DATASET="${DATASET:-stack_cube_random_control}"
DATASET="${DATASET:-stack_cube}"
DATASET_VARIANT="${DATASET_VARIANT:-full_obs}"
DATA_TYPE="${DATA_TYPE:-zarr}"
# ZARR_PATH="${ZARR_PATH:-RoboVerse/data_policy/stack_cube_random_controlFrankaL0_obs:joint_pos_act:joint_pos_1000.zarr}"
ZARR_PATH="${ZARR_PATH:-RoboVerse/data_policy/stack_cubeFrankaL0_obs:joint_pos_act:joint_pos_999.zarr}"
SEQ_LEN="${SEQ_LEN:-5}"
SEQ_LEN="${SEQ_LEN:-5}"

USE_IMAGES="${USE_IMAGES:-True}"
IMAGE_OBS="${IMAGE_OBS:-True}"
LA_DIM="${LA_DIM:-32}"
K_STEP_PRED="${K_STEP_PRED:-1}"
TCN_LOSS_WEIGHT="${TCN_LOSS_WEIGHT:-0}"
TCN_TAU="${TCN_TAU:-1.0}"

POST_STEPS="${POST_STEPS:-20000}"
POST_LOG_EVERY="${POST_LOG_EVERY:-500}"

PYTHONPATH=IDM python IDM/scripts/train_idm_action_decoder.py \
  --config-name "${CONFIG_NAME}" \
  env="${ENV_NAME}" \
  env.dataset_name="${DATASET_NAME}" \
  +env.datasets="[${DATASET}]" \
  data.data_dir="${DATA_DIR}" \
  data.data_type="${DATA_TYPE}" \
  data.zarr_path="${ZARR_PATH}" \
  data.dataset_variant="${DATASET_VARIANT}" \
  data.seq_len="${SEQ_LEN}" \
  data.use_images="${USE_IMAGES}" \
  env.image_obs="${IMAGE_OBS}" \
  data.embedding=False \
  data.use_cache=False \
  data.drop_images_after_obs=True \
  model.la_dim="${LA_DIM}" \
  model.fdm.k_step_pred="${K_STEP_PRED}" \
  model.fdm.tcn_loss_weight="${TCN_LOSS_WEIGHT}" \
  model.fdm.tcn_tau="${TCN_TAU}" \
  post_action_decoder_steps="${POST_STEPS}" \
  post_action_decoder_log_every="${POST_LOG_EVERY}" \
  "$@"
