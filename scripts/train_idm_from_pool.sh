#!/usr/bin/env bash
set -euo pipefail

# General IDM training launcher from SAILOR pool jsonl files.
#
# Supports:
# - explicit TRAIN_POOL_JSONL / EVAL_POOL_JSONL
# - or auto-infer from SUITE/TASK/EXP_NAME/SEED
# - fully configurable core hyperparameters via env vars
# - pass-through extra Hydra overrides via EXTRA_ARGS
#
# Example:
#   SUITE=robomimic TASK=can EXP_NAME=dp_collect_diverse_can_demos50_n30_noise0.10 \
#   NUM_UPDATES=100000 SAVE_EVERY=5000 BATCH_SIZE=64 \
#   ./scripts/train_idm_from_pool.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SUITE="${SUITE:-robomimic}"
TASK="${TASK:-can}"
SEED="${SEED:-0}"
EXP_NAME="${EXP_NAME:-dp_collect_diverse_${TASK}_demos50_n30_noise0.10}"

CONFIG_NAME="${CONFIG_NAME:-train_st_vivit_clam_stm}"
ENV_CONFIG="${ENV_CONFIG:-robomimic_sailor}"
DATA_SOURCE="${DATA_SOURCE:-sailor_pool}"

DATA_DIR="${DATA_DIR:-./IDM/data}"
SEQ_LEN="${SEQ_LEN:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_TRAJS="${NUM_TRAJS:--1}"
NUM_UPDATES="${NUM_UPDATES:-50000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
USE_WANDB="${USE_WANDB:-False}"

IMAGE_OBS="${IMAGE_OBS:-True}"
DATA_USE_IMAGES="${DATA_USE_IMAGES:-${IMAGE_OBS}}"
DROP_IMAGES_AFTER_OBS="${DROP_IMAGES_AFTER_OBS:-True}"

POOLS_ROOT_DEFAULT="scratch_dir/logs/${SUITE}__${TASK}/${EXP_NAME}/seed${SEED}/pools_v1"
POOLS_ROOT="${POOLS_ROOT:-${POOLS_ROOT_DEFAULT}}"
TRAIN_POOL_JSONL="${TRAIN_POOL_JSONL:-${POOLS_ROOT}/train_pool.jsonl}"
EVAL_POOL_JSONL="${EVAL_POOL_JSONL:-${POOLS_ROOT}/eval_pool.jsonl}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -f "${TRAIN_POOL_JSONL}" ]]; then
  echo "ERROR: TRAIN_POOL_JSONL not found: ${TRAIN_POOL_JSONL}"
  exit 1
fi
if [[ ! -f "${EVAL_POOL_JSONL}" ]]; then
  echo "ERROR: EVAL_POOL_JSONL not found: ${EVAL_POOL_JSONL}"
  exit 1
fi

echo "=== Train IDM from pool ==="
echo "SUITE=${SUITE} TASK=${TASK} SEED=${SEED}"
echo "TRAIN_POOL_JSONL=${TRAIN_POOL_JSONL}"
echo "EVAL_POOL_JSONL=${EVAL_POOL_JSONL}"
echo "CONFIG_NAME=${CONFIG_NAME} ENV_CONFIG=${ENV_CONFIG}"
echo "SEQ_LEN=${SEQ_LEN} BATCH_SIZE=${BATCH_SIZE} NUM_UPDATES=${NUM_UPDATES}"
echo

CMD=(
  python IDM/scripts/train_idm_action_decoder.py
  --config-name "${CONFIG_NAME}"
  "env=${ENV_CONFIG}"
  "env.env_id=${TASK}"
  "data.source=${DATA_SOURCE}"
  "data.sailor_pool_train_jsonl=${TRAIN_POOL_JSONL}"
  "data.sailor_pool_eval_jsonl=${EVAL_POOL_JSONL}"
  "data.data_dir=${DATA_DIR}"
  "data.data_type=n_step"
  "data.seq_len=${SEQ_LEN}"
  "data.batch_size=${BATCH_SIZE}"
  "data.num_trajs=${NUM_TRAJS}"
  "env.image_obs=${IMAGE_OBS}"
  "data.use_images=${DATA_USE_IMAGES}"
  "data.drop_images_after_obs=${DROP_IMAGES_AFTER_OBS}"
  "use_wandb=${USE_WANDB}"
  "num_updates=${NUM_UPDATES}"
  "save_every=${SAVE_EVERY}"
)

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARR[@]}")
fi

PYTHONPATH=IDM "${CMD[@]}"
