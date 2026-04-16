#!/usr/bin/env bash
set -euo pipefail

# Simplified launch script: defaults are overridable via env vars
DATA_DIR="${DATA_DIR:-./data}"
DATASET_NAME="${DATASET_NAME:-roboversetfds}"
DATASET="${DATASET:-stack_cube_random_control}"
DATASET_VARIANT="${DATASET_VARIANT:-full_obs}"

# Reuse an existing run config.yaml and train only the post-hoc action decoder
RUN_DIR="${RUN_DIR:-}"
CKPT_STEP="${CKPT_STEP:-200000}"

if [[ "$CKPT_STEP" == "latest" ]]; then
  CKPT_STEP="latest"
elif [[ "$CKPT_STEP" =~ ([0-9]+) ]]; then
  CKPT_STEP="${BASH_REMATCH[1]}"
else
  echo "Could not parse ckpt step from: $CKPT_STEP" >&2
  exit 1
fi

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "Using GPU ${GPU_ID}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: RUN_DIR is empty. Please set RUN_DIR to an existing experiment directory." >&2
  exit 1
fi

PYTHONPATH=IDM python -m udrm.main --config-name train_st_vivit_clam env=roboverse \
  env.dataset_name="${DATASET_NAME}" +env.datasets="[${DATASET}]" \
  data.data_dir="${DATA_DIR}" \
  data.dataset_variant="${DATASET_VARIANT}" \
  data.use_images=True data.embedding=False \
  env.image_obs=True model.use_pretrained_embeddings=False \
  log_rollout_videos=True \
  data.use_cache=False \
  data.drop_images_after_obs=True \
  model.la_dim=16 \
  model.fdm.k_step_pred=4 \
  "$@"

  # load_from_ckpt=True use_ckpt_config=False ckpt_file="${RUN_DIR}" ckpt_step="${CKPT_STEP}" exp_dir="${RUN_DIR}" \
  
  # joint_action_decoder_training=True \
  # num_labelled_trajs=50 \
  # env.action_labelled_dataset=[stack_cube_random_control] \
  # env.action_labelled_dataset_split=[1] \
  # +data.labelled_data_type=trajectory \


