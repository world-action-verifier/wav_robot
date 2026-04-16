#!/usr/bin/env bash
set -euo pipefail

# Diverse DP rollout collection for robomimic.
# This script trains/loads DP and periodically saves noisy policy rollouts to:
#   scratch_dir/logs/<suite__task>/<exp_name>/seed<seed>/dp_rollouts/step_*/
#
# Defaults are tuned for diversity:
# - rollout_snapshot_noise_std > 0
# - higher rollout_snapshot_count

SUITE="robomimic"
TASKS="${TASKS:-can square}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

NUM_EXP_TRAJS="${NUM_EXP_TRAJS:-50}"
SEED="${SEED:-0}"

USE_WANDB="${USE_WANDB:-True}"
WANDB_ENTITY="${WANDB_ENTITY:-ffeng1017}"
WANDB_PROJECT="${WANDB_PROJECT:-SAILOR}"

DP_TRAIN_STEPS="${DP_TRAIN_STEPS:-24000}"
DP_EVAL_FREQ="${DP_EVAL_FREQ:-12000}"

# Diversity knobs (recommended)
ROLLOUT_SNAPSHOT_COUNT="${ROLLOUT_SNAPSHOT_COUNT:-30}"
ROLLOUT_SNAPSHOT_STEPS="${ROLLOUT_SNAPSHOT_STEPS:-3000}"
ROLLOUT_SNAPSHOT_NOISE_STD="${ROLLOUT_SNAPSHOT_NOISE_STD:-0.10}"  # Suggested: 0.05 ~ 0.20

if [[ "${USE_WANDB}" == "True" || "${USE_WANDB}" == "true" || "${USE_WANDB}" == "1" ]]; then
  if ! python3 -c "import wandb" >/dev/null 2>&1; then
    echo "ERROR: USE_WANDB=True but wandb import failed in current environment."
    exit 1
  fi
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login
  fi
fi

for TASK in ${TASKS}; do
  RUN_NAME="dp_collect_diverse_${TASK}_demos${NUM_EXP_TRAJS}_n${ROLLOUT_SNAPSHOT_COUNT}_noise${ROLLOUT_SNAPSHOT_NOISE_STD}"
  echo "=== Collecting diverse DP rollouts | ${SUITE}__${TASK} | ${RUN_NAME} ==="

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 train_sailor.py \
    --configs cfg_dp_mppi "${SUITE}" \
    --task "${SUITE}__${TASK}" \
    --num_exp_trajs "${NUM_EXP_TRAJS}" \
    --seed "${SEED}" \
    --use_wandb "${USE_WANDB}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_exp_name "${RUN_NAME}" \
    --set train_dp_mppi False \
    --set dp.train_steps "${DP_TRAIN_STEPS}" \
    --set dp.eval_freq "${DP_EVAL_FREQ}" \
    --set dp.rollout_snapshot_count "${ROLLOUT_SNAPSHOT_COUNT}" \
    --set dp.rollout_snapshot_steps "${ROLLOUT_SNAPSHOT_STEPS}" \
    --set dp.rollout_snapshot_noise_std "${ROLLOUT_SNAPSHOT_NOISE_STD}"
done
