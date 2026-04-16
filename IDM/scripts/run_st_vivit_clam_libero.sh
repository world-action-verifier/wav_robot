#!/usr/bin/env bash
set -euo pipefail

# Simplified launch script: defaults are overridable via env vars
DATA_DIR="${DATA_DIR:-./data}"
DATASET_NAME="${DATASET_NAME:-liberotfds}"
DATASET="${DATASET:-libero_all}"
DATASET_VARIANT="${DATASET_VARIANT:-full_obs}"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "Using GPU ${GPU_ID}"

PYTHONPATH=IDM python -m udrm.main --config-name train_st_vivit_clam env=libero \
  env.dataset_name="${DATASET_NAME}" +env.datasets="[${DATASET}]" \
  data.data_dir="${DATA_DIR}" \
  data.dataset_variant="${DATASET_VARIANT}" \
  data.use_images=True data.embedding=False \
  env.image_obs=True model.use_pretrained_embeddings=False \
  log_rollout_videos=True \
  data.use_cache=False \
  data.drop_images_after_obs=True \
  model.la_dim=128 \
  num_updates=500_000
  "$@"
  
  # joint_action_decoder_training=True \
  # num_labelled_trajs=50 \
  # env.action_labelled_dataset=[stack_cube_random_control] \
  # env.action_labelled_dataset_split=[1] \
  # +data.labelled_data_type=trajectory \


