#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login
fi
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

TASKS="${TASKS:-square}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "${USE_WANDB}" == "True" || "${USE_WANDB}" == "true" || "${USE_WANDB}" == "1" ]]; then
  if ! python3 -c "import wandb" >/dev/null 2>&1; then
    echo "ERROR: USE_WANDB=True but wandb import failed in current environment."
    echo "Fix environment first (wandb/protobuf compatibility), or run with USE_WANDB=False."
    exit 1
  fi
fi

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

  STRATEGY_KWARGS_JSON="{\"suite\":\"${SUITE}\",\"task\":\"${TASK}\",\"device\":\"cuda:0\",\"idm_ckpt_path\":\"${IDM_CKPT_PATH}\",\"idm_device\":\"${IDM_DEVICE}\",\"idm_seq_len\":${IDM_SEQ_LEN},\"score_key\":\"${SCORE_KEY}\""
  if [[ -n "${IDM_CONFIG_PATH}" ]]; then
    STRATEGY_KWARGS_JSON+=",\"idm_config_path\":\"${IDM_CONFIG_PATH}\""
  fi
  STRATEGY_KWARGS_JSON+="}"

  echo "=== Launching WM-only ${SAMPLE_SELECTION_STRATEGY} for task: ${TASK} | run: ${RUN_NAME} ==="
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
    --set wm_only_sample_select_kwargs_json "${STRATEGY_KWARGS_JSON}" \
    --set wm_only_sample_start_itr "${SAMPLE_START_ITR}" \
    --set wm_only_sample_mix_ratio "${SAMPLE_MIX_RATIO}" \
    --set wm_only_eval_pool_jsonl "${EVAL_POOL_JSONL}" \
    --set wm_eval_every "${WM_EVAL_EVERY}" \
    --set log_openloop_img_pred "${LOG_WM_ROLLOUT_WINDOWS}" \
    --set openloop_img_pred_horizon "${OPENLOOP_HORIZON}" \
    --set wm_only_train_itrs "${WM_ONLY_TRAIN_ITRS}" \
    --set dp.rollout_snapshot_count 0 \
    --set train_dp_mppi_params.use_discrim False \
    --set dp.pretrained_ckpt "${DP_CKPT_ARG}"
done
