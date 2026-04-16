# Diverse DP Rollout Collection

This document explains how to collect diverse DP rollouts that are later used to build training/sample pools for world-model training.

Method-specific prerequisites and loading behavior are documented in:

- `README_WM_METHODS.md`

## Goal

Generate richer off-policy trajectory coverage by:

- adding rollout action noise (`dp.rollout_snapshot_noise_std > 0`)
- collecting rollouts at more training checkpoints (`dp.rollout_snapshot_count` higher)

Collected files are saved under:

- `scratch_dir/logs/<suite__task>/<exp_name>/seed<seed>/dp_rollouts/step_*/`

## Scripts

- Robomimic: `scripts/dp_collect_robomimic.sh`
- ManiSkill: `scripts/dp_collect_maniskill.sh`
- Backward-compatible entry: `dp_collection.sh` (calls robomimic script)

## Recommended diversity settings

- `ROLLOUT_SNAPSHOT_NOISE_STD=0.10` (typical range: `0.05 ~ 0.20`)
- `ROLLOUT_SNAPSHOT_COUNT=30`
- `ROLLOUT_SNAPSHOT_STEPS=3000`

If environment dynamics are sensitive, start from `0.05` and increase gradually.

## Usage examples

### 1) Robomimic (can + square)

```bash
cd release
CUDA_VISIBLE_DEVICES=0 \
TASKS="can square" \
ROLLOUT_SNAPSHOT_NOISE_STD=0.10 \
ROLLOUT_SNAPSHOT_COUNT=30 \
ROLLOUT_SNAPSHOT_STEPS=3000 \
./scripts/dp_collect_robomimic.sh
```

### 2) ManiSkill (pullcube + liftpeg + pokecube)

```bash
cd release
CUDA_VISIBLE_DEVICES=0 \
TASKS="pullcube liftpeg pokecube" \
ROLLOUT_SNAPSHOT_NOISE_STD=0.10 \
ROLLOUT_SNAPSHOT_COUNT=30 \
ROLLOUT_SNAPSHOT_STEPS=3000 \
./scripts/dp_collect_maniskill.sh
```

## Optional environment variables

Common knobs (both scripts):

- `NUM_EXP_TRAJS` (default `50`)
- `SEED` (default `0`)
- `USE_WANDB` (default `True`)
- `WANDB_ENTITY`, `WANDB_PROJECT`
- `DP_TRAIN_STEPS` (default `24000`)
- `DP_EVAL_FREQ` (default `12000`)
- `ROLLOUT_SNAPSHOT_COUNT` (default `30`)
- `ROLLOUT_SNAPSHOT_STEPS` (default `3000`)
- `ROLLOUT_SNAPSHOT_NOISE_STD` (default `0.10`)

## Next step after collection

After rollouts are collected, build pools (train/eval/sample) from `dp_rollouts` and launch the WM loop with active data refresh.

## Build pools (train/eval/sample)

Use:

- `scripts/build_pools.sh`

The script wraps `datasets/build_pools.py` and are with flexible knobs:

- pool sizes: `TRAIN_SIZE`, `EVAL_SIZE`, `SAMPLE_SIZE`
- train composition: `TRAIN_EXPERT_FRAC`, `TRAIN_LATE_FRAC`, `TRAIN_REST_FRAC`
- sample composition: `SAMPLE_EARLY_FRAC`, `SAMPLE_REST_FRAC`
- `NUM_EXPERT_TRAJS`, `SEED`, `ALLOW_OVERLAP`
- path overrides: `DP_ROLLOUTS_ROOT`, `OUTPUT_DIR`, `EXPERT_HDF5`

### Example

```bash
cd release
TASK=can \
SUITE=robomimic \
EXP_NAME=dp_collect_diverse_can_demos50_n30_noise0.10 \
TRAIN_SIZE=400 \
EVAL_SIZE=120 \
SAMPLE_SIZE=600 \
TRAIN_EXPERT_FRAC=0.55 \
TRAIN_LATE_FRAC=0.35 \
TRAIN_REST_FRAC=0.10 \
SAMPLE_EARLY_FRAC=0.75 \
SAMPLE_REST_FRAC=0.25 \
SEED=0 \
./scripts/build_pools.sh
```

Generated files:

- `train_pool.jsonl`
- `sample_pool.jsonl`
- `eval_pool.jsonl`
- `pool_summary.json`

## Train WM with different methods

Use:

- `scripts/train_wm_method.sh`
- `scripts/train_wm_robomimic.sh`
- `scripts/train_wm_maniskill.sh`
- `scripts/train_wm_robocasa.sh`

Supported methods:

- `METHOD=random`
- `METHOD=idm`
- `METHOD=curiosity`
- `METHOD=progress` (progress is implemented as curiosity-style loss improvement with a progress score key)

### Example: random

```bash
cd release
METHOD=random \
SUITE=robomimic \
TASKS="can square" \
EXP_NAME=dp_collect_diverse_can_demos50_n30_noise0.10 \
SEED=0 \
./scripts/train_wm_method.sh
```

### Example: IDM

```bash
cd release
METHOD=idm \
SUITE=robomimic \
TASKS="can" \
EXP_NAME=dp_collect_diverse_can_demos50_n30_noise0.10 \
IDM_CKPT_PATH=/abs/path/to/idm_ckpt.pkl \
IDM_CONFIG_PATH=/abs/path/to/idm_config.yaml \
./scripts/train_wm_method.sh
```

### Example: Curiosity / Progress

```bash
cd release
METHOD=progress \
SUITE=robomimic \
TASKS="can" \
EXP_NAME=dp_collect_diverse_can_demos50_n30_noise0.10 \
WM_OLD_CKPT_PATH=/abs/path/to/old_wm_ckpt.pt \
LOSS_KEY=model_loss \
./scripts/train_wm_method.sh
```

Common knobs:

- `SAMPLE_SELECT_SIZE`, `SAMPLE_REFRESH_EVERY`
- `SAMPLE_START_ITR`, `SAMPLE_MIX_RATIO`
- `WM_ONLY_TRAIN_ITRS`, `WM_EVAL_EVERY`
- `NUM_EXP_TRAJS`, `SEED`, `TASKS`, `EXP_NAME`
- `OPENLOOP_HORIZON`, `LOG_WM_ROLLOUT_WINDOWS`
- `USE_WANDB`, `WANDB_ENTITY`, `WANDB_PROJECT`, `CUDA_VISIBLE_DEVICES`
- `DP_CKPT_PATH` (override default DP checkpoint lookup)

Default updates in this release:

- `WM_ONLY_TRAIN_ITRS=200000` (larger WM training budget)
- `SAMPLE_REFRESH_EVERY=5000` (slower / larger selection cycle)
- `SAMPLE_START_ITR=5000` (later, fully controllable start point)
