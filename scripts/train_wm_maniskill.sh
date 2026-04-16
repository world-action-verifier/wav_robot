#!/usr/bin/env bash
set -euo pipefail

# Environment preset for maniskill WM-only training.
# All hyperparameters stay fully controllable via env vars.

SUITE="${SUITE:-maniskill}"
TASKS="${TASKS:-pullcube liftpeg pokecube}"
METHOD="${METHOD:-random}"
EXP_NAME="${EXP_NAME:-dp_collect_diverse_pullcube_demos50_n30_noise0.10}"

# Larger defaults (can be overridden by env vars)
WM_ONLY_TRAIN_ITRS="${WM_ONLY_TRAIN_ITRS:-200000}"
SAMPLE_REFRESH_EVERY="${SAMPLE_REFRESH_EVERY:-5000}"
SAMPLE_START_ITR="${SAMPLE_START_ITR:-5000}"

SUITE="${SUITE}" \
TASKS="${TASKS}" \
METHOD="${METHOD}" \
EXP_NAME="${EXP_NAME}" \
WM_ONLY_TRAIN_ITRS="${WM_ONLY_TRAIN_ITRS}" \
SAMPLE_REFRESH_EVERY="${SAMPLE_REFRESH_EVERY}" \
SAMPLE_START_ITR="${SAMPLE_START_ITR}" \
./scripts/train_wm_method.sh
