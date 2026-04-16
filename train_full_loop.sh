#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login
fi

# Example:
#   bash train_full_loop.sh
#   CUDA_VISIBLE_DEVICES=7,6 TASKS="lift square" bash train_full_loop.sh
CUDA_VISIBLE_DEVICES=6
SUITE="robomimic"
NUM_EXP_TRAJS=50
WM_ONLY_TRAIN_ITRS=20000

# Sample-mix schedule in wm-only training.
SAMPLE_START_ITR="${SAMPLE_START_ITR:-2000}"
SAMPLE_MIX_RATIO="${SAMPLE_MIX_RATIO:-0.3}"
WM_EVAL_EVERY="${WM_EVAL_EVERY:-500}"
OPENLOOP_HORIZON="${OPENLOOP_HORIZON:-8}"
LOG_WM_ROLLOUT_WINDOWS="${LOG_WM_ROLLOUT_WINDOWS:-True}"
SAMPLE_SELECTION_STRATEGY="${SAMPLE_SELECTION_STRATEGY:-random}"
SAMPLE_SELECT_SIZE="${SAMPLE_SELECT_SIZE:-120}"
SAMPLE_SELECT_SEED="${SAMPLE_SELECT_SEED:-0}"
SAMPLE_REFRESH_EVERY="${SAMPLE_REFRESH_EVERY:-500}"

WANDB_ENTITY="ffeng1017"
WANDB_PROJECT="SAILOR"
USE_WANDB=True

# Space separated task list.
TASKS="${TASKS:-lift}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

for TASK in ${TASKS}; do
  RUN_NAME="wm_only_${TASK}_pool_v1_${SAMPLE_SELECTION_STRATEGY}_n${SAMPLE_SELECT_SIZE}_mix_from_${SAMPLE_START_ITR}_r${SAMPLE_MIX_RATIO}"
  ROOT="scratch_dir/logs/${SUITE}__${TASK}/overnight_full_${TASK}_demos50/seed0/pools_v1"
  TRAIN_POOL_JSONL="${ROOT}/train_pool.jsonl"
  SAMPLE_SOURCE_POOL_JSONL="${ROOT}/sample_pool.jsonl"
  DP_CKPT="logs/${SUITE}__${TASK}/overnight_full_${TASK}_demos50/seed0/DP_Pretrain_base_policy_latest.pt"

  echo "=== Launching task: ${TASK} | run: ${RUN_NAME} ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 train_sailor.py \
    --configs cfg_dp_mppi "${SUITE}" \
    --task "${SUITE}__${TASK}" \
    --num_exp_trajs "${NUM_EXP_TRAJS}" \
    --use_wandb "${USE_WANDB}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_exp_name "${RUN_NAME}" \
    --set wm_only_mode True \
    --set wm_only_pool_jsonl "${TRAIN_POOL_JSONL}" \
    --set wm_only_sample_source_pool_jsonl "${SAMPLE_SOURCE_POOL_JSONL}" \
    --set wm_only_sample_select_strategy "${SAMPLE_SELECTION_STRATEGY}" \
    --set wm_only_sample_select_size "${SAMPLE_SELECT_SIZE}" \
    --set wm_only_sample_select_seed "${SAMPLE_SELECT_SEED}" \
    --set wm_only_sample_refresh_every "${SAMPLE_REFRESH_EVERY}" \
    --set wm_only_sample_start_itr "${SAMPLE_START_ITR}" \
    --set wm_only_sample_mix_ratio "${SAMPLE_MIX_RATIO}" \
    --set wm_only_eval_pool_jsonl "${ROOT}/eval_pool.jsonl" \
    --set wm_eval_every "${WM_EVAL_EVERY}" \
    --set log_openloop_img_pred "${LOG_WM_ROLLOUT_WINDOWS}" \
    --set openloop_img_pred_horizon "${OPENLOOP_HORIZON}" \
    --set wm_only_train_itrs "${WM_ONLY_TRAIN_ITRS}" \
    --set dp.rollout_snapshot_count 0 \
    --set train_dp_mppi_params.use_discrim False \
    --set dp.pretrained_ckpt "${DP_CKPT}"
done

