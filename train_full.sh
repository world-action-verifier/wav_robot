#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
TASK="lift"
SAMPLE_START_ITR=2000
SAMPLE_MIX_RATIO=0.3
WM_EVAL_EVERY=500
OPENLOOP_HORIZON=8
LOG_WM_ROLLOUT_WINDOWS=True
SAMPLE_SELECTION_STRATEGY="${SAMPLE_SELECTION_STRATEGY:-random}"
SAMPLE_SELECT_SIZE="${SAMPLE_SELECT_SIZE:-120}"
SAMPLE_SELECT_SEED="${SAMPLE_SELECT_SEED:-0}"

ROOT="scratch_dir/logs/robomimic__${TASK}/overnight_full_${TASK}_demos50/seed0/pools_v1"
RAW_SAMPLE_POOL_JSONL="${ROOT}/sample_pool.jsonl"
SELECTED_SAMPLE_POOL_JSONL="${ROOT}/sample_pool_selected_${SAMPLE_SELECTION_STRATEGY}_n${SAMPLE_SELECT_SIZE}_seed${SAMPLE_SELECT_SEED}.jsonl"

python3 datasets/data_selection.py \
  --sample-pool-jsonl "${RAW_SAMPLE_POOL_JSONL}" \
  --output-jsonl "${SELECTED_SAMPLE_POOL_JSONL}" \
  --strategy "${SAMPLE_SELECTION_STRATEGY}" \
  --select-size "${SAMPLE_SELECT_SIZE}" \
  --seed "${SAMPLE_SELECT_SEED}"

python3 train_sailor.py \
  --configs cfg_dp_mppi robomimic \
  --task "robomimic__${TASK}" \
  --num_exp_trajs 50 \
  --use_wandb True \
  --wandb_entity ffeng1017 \
  --wandb_project SAILOR \
  --wandb_exp_name "wm_only_${TASK}_pool_v1_${SAMPLE_SELECTION_STRATEGY}_n${SAMPLE_SELECT_SIZE}_mix_from_${SAMPLE_START_ITR}_r${SAMPLE_MIX_RATIO}" \
  --set wm_only_mode True \
  --set wm_only_pool_jsonl "${ROOT}/train_pool.jsonl" \
  --set wm_only_sample_pool_jsonl "${SELECTED_SAMPLE_POOL_JSONL}" \
  --set wm_only_sample_start_itr "${SAMPLE_START_ITR}" \
  --set wm_only_sample_mix_ratio "${SAMPLE_MIX_RATIO}" \
  --set wm_only_eval_pool_jsonl "${ROOT}/eval_pool.jsonl" \
  --set wm_eval_every "${WM_EVAL_EVERY}" \
  --set log_openloop_img_pred "${LOG_WM_ROLLOUT_WINDOWS}" \
  --set openloop_img_pred_horizon "${OPENLOOP_HORIZON}" \
  --set wm_only_train_itrs 20000 \
  --set dp.rollout_snapshot_count 0 \
  --set train_dp_mppi_params.use_discrim False \
  --set dp.pretrained_ckpt "logs/robomimic__${TASK}/overnight_full_${TASK}_demos50/seed0/DP_Pretrain_base_policy_latest.pt"