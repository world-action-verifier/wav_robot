#!/usr/bin/env bash
set -euo pipefail
# WM-only dynamic IDM selection for can + square.
# Example:
# IDM_CKPT_PATH=""

SUITE="${SUITE:-robomimic}"
NUM_EXP_TRAJS="${NUM_EXP_TRAJS:-50}"
WM_ONLY_TRAIN_ITRS="${WM_ONLY_TRAIN_ITRS:-20000}"

SAMPLE_START_ITR="${SAMPLE_START_ITR:-2000}"
SAMPLE_MIX_RATIO="${SAMPLE_MIX_RATIO:-0.3}"
WM_EVAL_EVERY="${WM_EVAL_EVERY:-500}"
OPENLOOP_HORIZON="${OPENLOOP_HORIZON:-8}"
LOG_WM_ROLLOUT_WINDOWS="${LOG_WM_ROLLOUT_WINDOWS:-True}"

SAMPLE_SELECTION_STRATEGY="${SAMPLE_SELECTION_STRATEGY:-idm}" # idm
SAMPLE_SELECT_SIZE="${SAMPLE_SELECT_SIZE:-120}"
SAMPLE_SELECT_SEED="${SAMPLE_SELECT_SEED:-0}"
SAMPLE_REFRESH_EVERY="${SAMPLE_REFRESH_EVERY:-500}"
IDM_SEQ_LEN="${IDM_SEQ_LEN:-8}"

WANDB_ENTITY="${WANDB_ENTITY:-ffeng1017}"
WANDB_PROJECT="${WANDB_PROJECT:-SAILOR}"
USE_WANDB="${USE_WANDB:-True}"

TASKS="${TASKS:-can}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

IDM_CKPT_PATH="${IDM_CKPT_PATH:-}"
if [[ -z "${IDM_CKPT_PATH}" ]]; then
  echo "ERROR: IDM_CKPT_PATH is empty."
  echo "Please provide an IDM checkpoint path, e.g. .../model_ckpts/ckpt_005000.pkl"
  exit 1
fi

IDM_CONFIG_PATH="${IDM_CONFIG_PATH:-}"
IDM_DEVICE="${IDM_DEVICE:-cpu}"
SCORE_KEY="${SCORE_KEY:-idm_wm_latent_mismatch_mse}"

for TASK in ${TASKS}; do
  RUN_NAME="wm_only_${TASK}_pool_v1_${SAMPLE_SELECTION_STRATEGY}_n${SAMPLE_SELECT_SIZE}_mix_from_${SAMPLE_START_ITR}_r${SAMPLE_MIX_RATIO}"
  ROOT="scratch_dir/logs/${SUITE}__${TASK}/overnight_full_${TASK}_demos50/seed0/pools_v1"
  TRAIN_POOL_JSONL="${ROOT}/train_pool.jsonl"
  EVAL_POOL_JSONL="${ROOT}/eval_pool.jsonl"
  SAMPLE_SOURCE_POOL_JSONL="${ROOT}/sample_pool.jsonl"
  DEFAULT_DP_CKPT="scratch_dir/logs/${SUITE}__${TASK}/overnight_full_${TASK}_demos50/seed0/DP_Pretrain_base_policy_latest.pt"
  ALT_DP_CKPT="scratch_dir/logs/${SUITE}__${TASK}/overnight_full_${TASK}_demos50/seed0/latest_base_policy.pt"
  LEGACY_ALT_DP_CKPT="scratch_dir/logs/${SUITE}__${TASK}/overnight_full_demos50/seed0/latest_base_policy.pt"
  DP_CKPT="${DP_CKPT_PATH:-${DEFAULT_DP_CKPT}}"
  if [[ -z "${DP_CKPT_PATH:-}" && ! -f "${DP_CKPT}" && -f "${ALT_DP_CKPT}" ]]; then
    DP_CKPT="${ALT_DP_CKPT}"
  fi
  if [[ -z "${DP_CKPT_PATH:-}" && ! -f "${DP_CKPT}" && -f "${LEGACY_ALT_DP_CKPT}" ]]; then
    DP_CKPT="${LEGACY_ALT_DP_CKPT}"
  fi
  if [[ ! -f "${DP_CKPT}" ]]; then
    echo "ERROR: missing DP checkpoint for task=${TASK}"
    echo "Checked: ${DP_CKPT}"
    echo "Also tried: ${ALT_DP_CKPT}"
    echo "Also tried: ${LEGACY_ALT_DP_CKPT}"
    echo "Hint: run DP collection first, or pass DP_CKPT_PATH=/abs/path/to/ckpt.pt"
    exit 1
  fi
  DP_CKPT_ARG="${DP_CKPT}"
  if [[ "${DP_CKPT_ARG}" != /* ]]; then
    # train_sailor prefixes relative paths with "scratch_dir/", so pass "logs/..."
    DP_CKPT_ARG="${DP_CKPT_ARG#scratch_dir/}"
  fi

  echo "=== Launching WM-only ${SAMPLE_SELECTION_STRATEGY} for task: ${TASK} | run: ${RUN_NAME} ==="
  SUITE="${SUITE}" \
  TASK="${TASK}" \
  RUN_NAME="${RUN_NAME}" \
  TRAIN_POOL_JSONL="${TRAIN_POOL_JSONL}" \
  EVAL_POOL_JSONL="${EVAL_POOL_JSONL}" \
  SAMPLE_SOURCE_POOL_JSONL="${SAMPLE_SOURCE_POOL_JSONL}" \
  SAMPLE_SELECTION_STRATEGY="${SAMPLE_SELECTION_STRATEGY}" \
  SAMPLE_SELECT_SIZE="${SAMPLE_SELECT_SIZE}" \
  SAMPLE_SELECT_SEED="${SAMPLE_SELECT_SEED}" \
  SAMPLE_REFRESH_EVERY="${SAMPLE_REFRESH_EVERY}" \
  SAMPLE_START_ITR="${SAMPLE_START_ITR}" \
  SAMPLE_MIX_RATIO="${SAMPLE_MIX_RATIO}" \
  WM_EVAL_EVERY="${WM_EVAL_EVERY}" \
  OPENLOOP_HORIZON="${OPENLOOP_HORIZON}" \
  LOG_WM_ROLLOUT_WINDOWS="${LOG_WM_ROLLOUT_WINDOWS}" \
  WM_ONLY_TRAIN_ITRS="${WM_ONLY_TRAIN_ITRS}" \
  NUM_EXP_TRAJS="${NUM_EXP_TRAJS}" \
  USE_WANDB="${USE_WANDB}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  DP_CKPT_ARG="${DP_CKPT_ARG}" \
  IDM_CKPT_PATH="${IDM_CKPT_PATH}" \
  IDM_CONFIG_PATH="${IDM_CONFIG_PATH}" \
  IDM_DEVICE="${IDM_DEVICE}" \
  IDM_SEQ_LEN="${IDM_SEQ_LEN}" \
  SCORE_KEY="${SCORE_KEY}" \
  ./scripts/run_wm_only.sh
done
