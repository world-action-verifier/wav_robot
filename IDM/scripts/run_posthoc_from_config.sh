#!/usr/bin/env bash
set -euo pipefail

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

PYTHONPATH=IDM python -m udrm.main --config-name train_clam \
  load_from_ckpt=True use_ckpt_config=True ckpt_file="${RUN_DIR}" ckpt_step="${CKPT_STEP}" exp_dir="${RUN_DIR}" \
  post_action_decoder_only=True post_action_decoder_training=True \
  wandb_on_load=True \
  "$@"
