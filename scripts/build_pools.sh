#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TASK:?Set TASK}"
: "${DP_ROLLOUTS_ROOT:?Set DP_ROLLOUTS_ROOT to runs/<task>/stages}"
: "${EXPERT_HDF5:?Set EXPERT_HDF5 to the rendered official dataset}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
python3 "${REPO}/datasets/build_pools.py" --task "${TASK}" \
  --dp-rollouts-root "${DP_ROLLOUTS_ROOT}" --expert-hdf5 "${EXPERT_HDF5}" \
  --output-dir "${OUTPUT_DIR}"
