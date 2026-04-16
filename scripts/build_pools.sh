#!/usr/bin/env bash
set -euo pipefail

# Flexible wrapper for datasets/build_pools.py
# Supports custom pool sizes, fractions, seed, and paths.
#
# Example:
#   TASK=can SUITE=robomimic EXP_NAME=dp_collect_diverse_can_demos50_n30_noise0.10 \
#   TRAIN_SIZE=400 EVAL_SIZE=120 SAMPLE_SIZE=600 \
#   ./scripts/build_pools.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${RELEASE_DIR}/.." && pwd)"

BUILD_POOLS_PY="${BUILD_POOLS_PY:-${PROJECT_ROOT}/datasets/build_pools.py}"
if [[ ! -f "${BUILD_POOLS_PY}" ]]; then
  echo "ERROR: build_pools.py not found: ${BUILD_POOLS_PY}"
  echo "Set BUILD_POOLS_PY to the correct path."
  exit 1
fi

TASK="${TASK:-can}"
SUITE="${SUITE:-robomimic}"
SEED="${SEED:-0}"

# The experiment directory used during DP collection:
# scratch_dir/logs/<suite__task>/<exp_name>/seed<seed>
EXP_NAME="${EXP_NAME:-dp_collect_diverse_${TASK}_demos50_n30_noise0.10}"

DP_ROLLOUTS_ROOT="${DP_ROLLOUTS_ROOT:-${PROJECT_ROOT}/scratch_dir/logs/${SUITE}__${TASK}/${EXP_NAME}/seed${SEED}/dp_rollouts}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/scratch_dir/logs/${SUITE}__${TASK}/${EXP_NAME}/seed${SEED}/pools_v1}"

# Optional. If empty, build_pools.py will infer robomimic default expert dataset path.
EXPERT_HDF5="${EXPERT_HDF5:-}"

NUM_EXPERT_TRAJS="${NUM_EXPERT_TRAJS:-50}"
TRAIN_SIZE="${TRAIN_SIZE:-500}"
EVAL_SIZE="${EVAL_SIZE:-200}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"

TRAIN_EXPERT_FRAC="${TRAIN_EXPERT_FRAC:-0.6}"
TRAIN_LATE_FRAC="${TRAIN_LATE_FRAC:-0.3}"
TRAIN_REST_FRAC="${TRAIN_REST_FRAC:-0.1}"
SAMPLE_EARLY_FRAC="${SAMPLE_EARLY_FRAC:-0.8}"
SAMPLE_REST_FRAC="${SAMPLE_REST_FRAC:-0.2}"

ALLOW_OVERLAP="${ALLOW_OVERLAP:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -d "${DP_ROLLOUTS_ROOT}" ]]; then
  echo "ERROR: DP rollouts root does not exist: ${DP_ROLLOUTS_ROOT}"
  echo "Please run DP collection first or set DP_ROLLOUTS_ROOT explicitly."
  exit 1
fi

ALLOW_OVERLAP_FLAG=()
if [[ "${ALLOW_OVERLAP}" == "true" || "${ALLOW_OVERLAP}" == "True" || "${ALLOW_OVERLAP}" == "1" ]]; then
  ALLOW_OVERLAP_FLAG=(--allow-overlap)
fi

CMD=(
  python3 "${BUILD_POOLS_PY}"
  --task "${TASK}"
  --dp-rollouts-root "${DP_ROLLOUTS_ROOT}"
  --num-expert-trajs "${NUM_EXPERT_TRAJS}"
  --train-size "${TRAIN_SIZE}"
  --eval-size "${EVAL_SIZE}"
  --sample-size "${SAMPLE_SIZE}"
  --train-expert-frac "${TRAIN_EXPERT_FRAC}"
  --train-late-frac "${TRAIN_LATE_FRAC}"
  --train-rest-frac "${TRAIN_REST_FRAC}"
  --sample-early-frac "${SAMPLE_EARLY_FRAC}"
  --sample-rest-frac "${SAMPLE_REST_FRAC}"
  --seed "${SEED}"
  --output-dir "${OUTPUT_DIR}"
  "${ALLOW_OVERLAP_FLAG[@]}"
)

if [[ -n "${EXPERT_HDF5}" ]]; then
  CMD+=(--expert-hdf5 "${EXPERT_HDF5}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "=== Building pools for ${SUITE}__${TASK} ==="
echo "DP_ROLLOUTS_ROOT=${DP_ROLLOUTS_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TRAIN/EVAL/SAMPLE=${TRAIN_SIZE}/${EVAL_SIZE}/${SAMPLE_SIZE}"
echo "TRAIN FRACTIONS: expert=${TRAIN_EXPERT_FRAC}, late=${TRAIN_LATE_FRAC}, rest=${TRAIN_REST_FRAC}"
echo "SAMPLE FRACTIONS: early=${SAMPLE_EARLY_FRAC}, rest=${SAMPLE_REST_FRAC}"
echo

"${CMD[@]}"

echo
echo "Pool files written to: ${OUTPUT_DIR}"
echo " - train_pool.jsonl"
echo " - sample_pool.jsonl"
echo " - eval_pool.jsonl"
echo " - pool_summary.json"
