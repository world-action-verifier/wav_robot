#!/usr/bin/env bash
set -euo pipefail

# Unified WM-only training entry for multiple selection methods:
#   METHOD=random | idm | curiosity | progress
#
# Required assets:
# - built pools under scratch_dir/logs/<suite__task>/<exp_name>/seed<seed>/pools_v1
# - DP checkpoint under the same run (or provide DP_CKPT_PATH explicitly)
#
# Example:
#   METHOD=progress SUITE=robomimic TASKS="can square" \
#   EXP_NAME=dp_collect_diverse_can_demos50_n30_noise0.10 \
#   ./scripts/train_wm_method.sh

METHOD="${METHOD:-random}"
SUITE="${SUITE:-robomimic}"
TASKS="${TASKS:-can}"
SEED="${SEED:-0}"
EXP_NAME="${EXP_NAME:-dp_collect_diverse_can_demos50_n30_noise0.10}"
NUM_EXP_TRAJS="${NUM_EXP_TRAJS:-50}"

WM_ONLY_TRAIN_ITRS="${WM_ONLY_TRAIN_ITRS:-200000}"
SAMPLE_START_ITR="${SAMPLE_START_ITR:-5000}"
SAMPLE_MIX_RATIO="${SAMPLE_MIX_RATIO:-0.3}"
WM_EVAL_EVERY="${WM_EVAL_EVERY:-1000}"
OPENLOOP_HORIZON="${OPENLOOP_HORIZON:-8}"
LOG_WM_ROLLOUT_WINDOWS="${LOG_WM_ROLLOUT_WINDOWS:-True}"
SAMPLE_SELECT_SIZE="${SAMPLE_SELECT_SIZE:-120}"
SAMPLE_SELECT_SEED="${SAMPLE_SELECT_SEED:-0}"
SAMPLE_REFRESH_EVERY="${SAMPLE_REFRESH_EVERY:-5000}"

USE_WANDB="${USE_WANDB:-True}"
WANDB_ENTITY="${WANDB_ENTITY:-ffeng1017}"
WANDB_PROJECT="${WANDB_PROJECT:-SAILOR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

IDM_CKPT_PATH="${IDM_CKPT_PATH:-}"
IDM_CONFIG_PATH="${IDM_CONFIG_PATH:-}"
IDM_DEVICE="${IDM_DEVICE:-cpu}"
IDM_SEQ_LEN="${IDM_SEQ_LEN:-8}"
LOSS_KEY="${LOSS_KEY:-model_loss}"

case "${METHOD}" in
  random)
    SAMPLE_SELECTION_STRATEGY="random"
    SCORE_KEY="${SCORE_KEY:-random_score}"
    ;;
  idm)
    SAMPLE_SELECTION_STRATEGY="idm"
    SCORE_KEY="${SCORE_KEY:-idm_wm_latent_mismatch_mse}"
    if [[ -z "${IDM_CKPT_PATH}" ]]; then
      echo "ERROR: METHOD=idm requires IDM_CKPT_PATH."
      exit 1
    fi
    ;;
  curiosity)
    SAMPLE_SELECTION_STRATEGY="curiosity"
    SCORE_KEY="${SCORE_KEY:-uncertainty_latent_prior_feat_var_mean}"
    ;;
  progress)
    SAMPLE_SELECTION_STRATEGY="progress"
    SCORE_KEY="${SCORE_KEY:-progress_score}"
    ;;
  *)
    echo "ERROR: Unsupported METHOD='${METHOD}'. Use: random | idm | curiosity | progress"
    exit 1
    ;;
esac

for TASK in ${TASKS}; do
  ROOT="scratch_dir/logs/${SUITE}__${TASK}/${EXP_NAME}/seed${SEED}/pools_v1"
  TRAIN_POOL_JSONL="${ROOT}/train_pool.jsonl"
  EVAL_POOL_JSONL="${ROOT}/eval_pool.jsonl"
  SAMPLE_SOURCE_POOL_JSONL="${ROOT}/sample_pool.jsonl"

  if [[ ! -f "${TRAIN_POOL_JSONL}" || ! -f "${SAMPLE_SOURCE_POOL_JSONL}" ]]; then
    echo "ERROR: Missing pool files for task=${TASK} at ${ROOT}"
    echo "Expected train_pool.jsonl and sample_pool.jsonl."
    exit 1
  fi

  DEFAULT_DP_CKPT="scratch_dir/logs/${SUITE}__${TASK}/${EXP_NAME}/seed${SEED}/DP_Pretrain_base_policy_latest.pt"
  ALT_DP_CKPT="scratch_dir/logs/${SUITE}__${TASK}/${EXP_NAME}/seed${SEED}/latest_base_policy.pt"
  DP_CKPT="${DP_CKPT_PATH:-${DEFAULT_DP_CKPT}}"
  if [[ -z "${DP_CKPT_PATH:-}" && ! -f "${DP_CKPT}" && -f "${ALT_DP_CKPT}" ]]; then
    DP_CKPT="${ALT_DP_CKPT}"
  fi
  if [[ ! -f "${DP_CKPT}" ]]; then
    echo "ERROR: missing DP checkpoint for task=${TASK}"
    echo "Checked: ${DP_CKPT}"
    echo "Also tried: ${ALT_DP_CKPT}"
    exit 1
  fi
  DP_CKPT_ARG="${DP_CKPT}"
  if [[ "${DP_CKPT_ARG}" != /* ]]; then
    DP_CKPT_ARG="${DP_CKPT_ARG#scratch_dir/}"
  fi

  RUN_NAME="wm_${METHOD}_${TASK}_sel${SAMPLE_SELECT_SIZE}_mix${SAMPLE_MIX_RATIO}_from${SAMPLE_START_ITR}"
  echo "=== Launching WM-only (${METHOD}) for ${SUITE}__${TASK} | ${RUN_NAME} ==="

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
  LOSS_KEY="${LOSS_KEY}" \
  SCORE_KEY="${SCORE_KEY}" \
  WM_OLD_CKPT_PATH="${WM_OLD_CKPT_PATH:-}" \
  WM_OLD_INIT_CKPT_PATH="${WM_OLD_INIT_CKPT_PATH:-}" \
  EMA_GAMMA="${EMA_GAMMA:-}" \
  ./scripts/run_wm_only.sh
done
