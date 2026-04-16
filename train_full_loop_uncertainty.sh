#!/usr/bin/env bash
set -euo pipefail

# Example:
#   TASKS="lift square" \
#   bash train_full_loop_uncertainty.sh
#
# Optional fixed ensemble override:
#   TASKS="lift" \
#   WM_CKPT_PATHS="ckpt1.pt,ckpt2.pt,ckpt3.pt,ckpt4.pt,ckpt5.pt" \
#   bash train_full_loop_uncertainty.sh

SUITE="robomimic"
NUM_EXP_TRAJS=50
WM_ONLY_TRAIN_ITRS=20000

SAMPLE_START_ITR="${SAMPLE_START_ITR:-2000}"
SAMPLE_MIX_RATIO="${SAMPLE_MIX_RATIO:-0.3}"
WM_EVAL_EVERY="${WM_EVAL_EVERY:-500}"
OPENLOOP_HORIZON="${OPENLOOP_HORIZON:-8}"
LOG_WM_ROLLOUT_WINDOWS="${LOG_WM_ROLLOUT_WINDOWS:-True}"

SAMPLE_SELECTION_STRATEGY="uncertainty"
SAMPLE_SELECT_SIZE="${SAMPLE_SELECT_SIZE:-120}"
SAMPLE_SELECT_SEED="${SAMPLE_SELECT_SEED:-0}"
SAMPLE_REFRESH_EVERY="${SAMPLE_REFRESH_EVERY:-500}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-5}"

WANDB_ENTITY="${WANDB_ENTITY:-ffeng1017}"
WANDB_PROJECT="${WANDB_PROJECT:-SAILOR}"
USE_WANDB="${USE_WANDB:-True}"

TASKS="${TASKS:-lift}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

# Optional fixed ensemble checkpoints (comma-separated).
# If empty, trainer uses online WM checkpoint history dynamically.
WM_CKPT_PATHS="${WM_CKPT_PATHS:-}"

for TASK in ${TASKS}; do
  RUN_NAME="wm_only_${TASK}_pool_v1_uncertainty_n${SAMPLE_SELECT_SIZE}_mix_from_${SAMPLE_START_ITR}_r${SAMPLE_MIX_RATIO}"
  ROOT="scratch_dir/logs/${SUITE}__${TASK}/overnight_full_${TASK}_demos50/seed0/pools_v1"
  TRAIN_POOL_JSONL="${ROOT}/train_pool.jsonl"
  SAMPLE_SOURCE_POOL_JSONL="${ROOT}/sample_pool.jsonl"
  EVAL_POOL_JSONL="${ROOT}/eval_pool.jsonl"
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
    DP_CKPT_ARG="${DP_CKPT_ARG#scratch_dir/}"
  fi

  STRATEGY_KWARGS_JSON="{\"suite\":\"${SUITE}\",\"task\":\"${TASK}\",\"device\":\"cuda:0\",\"ensemble_size\":${ENSEMBLE_SIZE}}"
  if [[ -n "${WM_CKPT_PATHS}" ]]; then
    IFS=',' read -r -a CKPT_ARRAY <<< "${WM_CKPT_PATHS}"
    if [[ "${#CKPT_ARRAY[@]}" -ne "${ENSEMBLE_SIZE}" ]]; then
      echo "ERROR: WM_CKPT_PATHS must contain exactly ${ENSEMBLE_SIZE} paths, got ${#CKPT_ARRAY[@]}."
      exit 1
    fi
    CKPTS_JSON="["
    for idx in "${!CKPT_ARRAY[@]}"; do
      p="${CKPT_ARRAY[$idx]}"
      p="${p#"${p%%[![:space:]]*}"}"
      p="${p%"${p##*[![:space:]]}"}"
      if [[ -z "${p}" ]]; then
        echo "ERROR: empty checkpoint path at index ${idx}."
        exit 1
      fi
      if [[ "${idx}" -gt 0 ]]; then
        CKPTS_JSON+=", "
      fi
      CKPTS_JSON+="\"${p}\""
    done
    CKPTS_JSON+="]"
    STRATEGY_KWARGS_JSON+=",\"wm_ckpt_paths\":${CKPTS_JSON}"
  fi
  STRATEGY_KWARGS_JSON+="}"

  echo "=== Launching WM-only dynamic uncertainty for task: ${TASK} | run: ${RUN_NAME} ==="
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

