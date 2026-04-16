#!/usr/bin/env bash
set -euo pipefail

# Generic WM-only launcher with dynamic sample selection.
# Expected environment variables:
#   SUITE, TASK, RUN_NAME, TRAIN_POOL_JSONL, EVAL_POOL_JSONL, SAMPLE_SOURCE_POOL_JSONL
#   SAMPLE_SELECTION_STRATEGY, SAMPLE_SELECT_SIZE, SAMPLE_SELECT_SEED, SAMPLE_REFRESH_EVERY
#   SAMPLE_START_ITR, SAMPLE_MIX_RATIO, WM_EVAL_EVERY, OPENLOOP_HORIZON, LOG_WM_ROLLOUT_WINDOWS
#   WM_ONLY_TRAIN_ITRS, NUM_EXP_TRAJS, USE_WANDB, WANDB_ENTITY, WANDB_PROJECT
#   CUDA_VISIBLE_DEVICES, DP_CKPT_ARG
#
# Optional for IDM strategies:
#   IDM_CKPT_PATH, IDM_CONFIG_PATH, IDM_DEVICE, IDM_SEQ_LEN, SCORE_KEY
#
# Optional for progress strategy:
#   LOSS_KEY, SCORE_KEY, WM_OLD_CKPT_PATH, WM_OLD_INIT_CKPT_PATH, EMA_GAMMA
#
# Optional for curiosity/uncertainty strategies:
#   ENSEMBLE_SIZE, WM_CKPT_PATHS_JSON

if [[ -z "${SUITE:-}" || -z "${TASK:-}" || -z "${RUN_NAME:-}" ]]; then
  echo "ERROR: SUITE, TASK, RUN_NAME are required."
  exit 1
fi

if [[ -z "${TRAIN_POOL_JSONL:-}" || -z "${SAMPLE_SOURCE_POOL_JSONL:-}" ]]; then
  echo "ERROR: TRAIN_POOL_JSONL and SAMPLE_SOURCE_POOL_JSONL are required."
  exit 1
fi

if [[ -z "${DP_CKPT_ARG:-}" ]]; then
  echo "ERROR: DP_CKPT_ARG is required."
  exit 1
fi

USE_WANDB="${USE_WANDB:-True}"
if [[ "${USE_WANDB}" == "True" || "${USE_WANDB}" == "true" || "${USE_WANDB}" == "1" ]]; then
  if ! python3 -c "import wandb" >/dev/null 2>&1; then
    echo "ERROR: USE_WANDB=True but wandb import failed in current environment."
    echo "Fix environment first or run with USE_WANDB=False."
    exit 1
  fi
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login
  fi
fi

STRATEGY_KWARGS_JSON="$(python3 - <<'PY'
import json
import os

strategy = os.environ.get("SAMPLE_SELECTION_STRATEGY", "random")
payload = {
    "suite": os.environ["SUITE"],
    "task": os.environ["TASK"],
    "device": "cuda:0",
}
if strategy == "idm":
    payload["idm_ckpt_path"] = os.environ.get("IDM_CKPT_PATH", "")
    payload["idm_device"] = os.environ.get("IDM_DEVICE", "cpu")
    payload["idm_seq_len"] = int(os.environ.get("IDM_SEQ_LEN", "8"))
    payload["score_key"] = os.environ.get("SCORE_KEY", "idm_wm_latent_mismatch_mse")
    if os.environ.get("IDM_CONFIG_PATH"):
        payload["idm_config_path"] = os.environ["IDM_CONFIG_PATH"]
elif strategy == "progress":
    payload["loss_key"] = os.environ.get("LOSS_KEY", "model_loss")
    payload["score_key"] = os.environ.get("SCORE_KEY", "progress_score")
    if os.environ.get("WM_OLD_CKPT_PATH"):
        payload["wm_old_ckpt_path"] = os.environ["WM_OLD_CKPT_PATH"]
    if os.environ.get("WM_OLD_INIT_CKPT_PATH"):
        payload["wm_old_init_ckpt_path"] = os.environ["WM_OLD_INIT_CKPT_PATH"]
    if os.environ.get("EMA_GAMMA"):
        payload["ema_gamma"] = float(os.environ["EMA_GAMMA"])
elif strategy in {"curiosity", "uncertainty"}:
    payload["score_key"] = os.environ.get(
        "SCORE_KEY", "uncertainty_latent_prior_feat_var_mean"
    )
    payload["ensemble_size"] = int(os.environ.get("ENSEMBLE_SIZE", "5"))
    if os.environ.get("WM_CKPT_PATHS_JSON"):
        payload["wm_ckpt_paths"] = json.loads(os.environ["WM_CKPT_PATHS_JSON"])
print(json.dumps(payload))
PY
)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 train_sailor.py \
  --configs cfg_dp_mppi "${SUITE}" \
  --task "${SUITE}__${TASK}" \
  --num_exp_trajs "${NUM_EXP_TRAJS:-50}" \
  --use_wandb "${USE_WANDB}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_project "${WANDB_PROJECT:-SAILOR}" \
  --wandb_exp_name "${RUN_NAME}" \
  --set wm_only_mode True \
  --set wm_only_pool_jsonl "${TRAIN_POOL_JSONL}" \
  --set wm_only_sample_source_pool_jsonl "${SAMPLE_SOURCE_POOL_JSONL}" \
  --set wm_only_sample_select_strategy "${SAMPLE_SELECTION_STRATEGY:-random}" \
  --set wm_only_sample_select_size "${SAMPLE_SELECT_SIZE:-120}" \
  --set wm_only_sample_select_seed "${SAMPLE_SELECT_SEED:-0}" \
  --set wm_only_sample_refresh_every "${SAMPLE_REFRESH_EVERY:-0}" \
  --set wm_only_sample_select_kwargs_json "${STRATEGY_KWARGS_JSON}" \
  --set wm_only_sample_start_itr "${SAMPLE_START_ITR:-0}" \
  --set wm_only_sample_mix_ratio "${SAMPLE_MIX_RATIO:-0.0}" \
  --set wm_only_eval_pool_jsonl "${EVAL_POOL_JSONL:-}" \
  --set wm_eval_every "${WM_EVAL_EVERY:-0}" \
  --set log_openloop_img_pred "${LOG_WM_ROLLOUT_WINDOWS:-True}" \
  --set openloop_img_pred_horizon "${OPENLOOP_HORIZON:-8}" \
  --set wm_only_train_itrs "${WM_ONLY_TRAIN_ITRS:-5000}" \
  --set dp.rollout_snapshot_count 0 \
  --set train_dp_mppi_params.use_discrim False \
  --set dp.pretrained_ckpt "${DP_CKPT_ARG}"
