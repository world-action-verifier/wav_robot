# World Model Selection Methods & Prerequisites

This document explains the WM-only data-selection methods and what must be pretrained / loaded before running each one.

Related launcher:

- `scripts/train_wm_method.sh`

Supported methods:

- `random`
- `progress`
- `curiosity` (mapped to uncertainty proxy in current code)
- `idm`

---

## 1) Method Summary

### `random`

- Selection logic: uniform random sampling from `sample_pool.jsonl`
- Extra model loading: none
- Best for: baseline comparisons

### `progress`

- Selection logic: learning progress score (`old_loss - new_loss`) on candidate episodes
- Extra model loading: world model checkpoints (old/new)
- Inference path: `datasets/data_selection.py` progress strategy
- Best for: prioritize samples where model improvement is highest

### `curiosity`

- Current behavior: uses uncertainty-style scoring (ensemble variance)
- Extra model loading: multiple world model checkpoints (ensemble)
- Inference path: `datasets/data_selection.py` curiosity strategy delegates to uncertainty logic
- Best for: prioritize uncertain samples

### `idm`

- Selection logic: IDM-predicted action vs WM latent rollout mismatch
- Extra model loading:
  - world model checkpoint
  - IDM checkpoint
  - IDM config (explicit path or inferred from checkpoint directory)
- Inference path: `datasets/data_selection.py` IDM strategy
- Best for: action-representation-guided sample filtering

---

## 2) Prerequisites by Stage

All methods share these prerequisites:

1. **DP rollouts collected** (for candidate trajectories)
   - Scripts:
     - `scripts/dp_collect_robomimic.sh`
     - `scripts/dp_collect_maniskill.sh`
2. **Pools built** (`train_pool.jsonl`, `sample_pool.jsonl`, `eval_pool.jsonl`)
   - Script: `scripts/build_pools.sh`
3. **DP checkpoint available** for WM-only training bootstrap
   - Auto-detected in `scripts/train_wm_method.sh` unless overridden

Method-specific prerequisites:

- `random`: no extra artifacts
- `progress`: WM checkpoint history (or explicitly provided old/new checkpoints)
- `curiosity`: WM ensemble checkpoints (>=2, recommended 5)
- `idm`: trained IDM checkpoint + config

---

## 3) What Is Auto-loaded vs User-provided

### Auto-handled by training loop

During refresh, selection service snapshots current WM and injects strategy kwargs:

- `progress`: auto fills `wm_new_ckpt_path`, and `wm_old_ckpt_path` when available
- `curiosity`: auto builds `wm_ckpt_paths` from recent WM snapshots if not provided
- `idm`: auto sets `wm_ckpt_path` to latest WM snapshot

### Must be provided by user

- `idm` requires:
  - `IDM_CKPT_PATH` (required)
  - optionally `IDM_CONFIG_PATH`

If `IDM_CKPT_PATH` is missing, `scripts/train_wm_method.sh` exits with error.

---

## 4) IDM Pretrain Requirement

Before running `METHOD=idm`, train IDM first, then pass checkpoint path.

IDM training scripts (currently located under the historical folder path):

- `IDM/scripts/train_idm_action_decoder.sh`
- `IDM/scripts/train_idm_action_decoder_sailor.sh`
- `scripts/train_idm_from_pool.sh` (general launcher from `train_pool.jsonl` / `eval_pool.jsonl`)

Note: in docs we call this module **IDM**, while code paths may still use `IDM` for compatibility.

Example:

```bash
cd release
METHOD=idm \
SUITE=robomimic \
TASKS="can" \
EXP_NAME=dp_collect_diverse_can_demos50_n30_noise0.10 \
IDM_CKPT_PATH=/abs/path/to/idm_ckpt.pkl \
IDM_CONFIG_PATH=/abs/path/to/config.yaml \
./scripts/train_wm_method.sh
```

### General IDM training from pool (recommended)

If you want a more general and reusable way to train IDM directly from SAILOR pools:

```bash
cd release
SUITE=robomimic \
TASK=can \
EXP_NAME=dp_collect_diverse_can_demos50_n30_noise0.10 \
SEED=0 \
NUM_UPDATES=100000 \
BATCH_SIZE=64 \
SEQ_LEN=8 \
./scripts/train_idm_from_pool.sh
```

You can also pass explicit pool paths:

```bash
cd release
TRAIN_POOL_JSONL=/abs/path/train_pool.jsonl \
EVAL_POOL_JSONL=/abs/path/eval_pool.jsonl \
./scripts/train_idm_from_pool.sh
```

Key controllable knobs:

- `CONFIG_NAME`, `ENV_CONFIG`
- `NUM_UPDATES`, `SAVE_EVERY`
- `BATCH_SIZE`, `SEQ_LEN`, `NUM_TRAJS`
- `IMAGE_OBS`, `DATA_USE_IMAGES`, `DROP_IMAGES_AFTER_OBS`
- `USE_WANDB`
- `EXTRA_ARGS` (free-form Hydra overrides)

---

## 5) Quick Run Examples

### Random

```bash
cd release
METHOD=random SUITE=robomimic TASKS="can square" ./scripts/train_wm_method.sh
```

### Progress

```bash
cd release
METHOD=progress \
SUITE=robomimic \
TASKS="can" \
LOSS_KEY=model_loss \
./scripts/train_wm_method.sh
```

### Curiosity (uncertainty proxy)

```bash
cd release
METHOD=curiosity \
SUITE=robomimic \
TASKS="can" \
ENSEMBLE_SIZE=5 \
./scripts/train_wm_method.sh
```

### IDM

```bash
cd release
METHOD=idm \
SUITE=robomimic \
TASKS="can" \
IDM_CKPT_PATH=/abs/path/to/idm_ckpt.pkl \
./scripts/train_wm_method.sh
```

---

## 6) Common Failure Cases

- `Missing pool files`:
  - run `scripts/build_pools.sh` first
- `missing DP checkpoint`:
  - run DP collection / pretrain first, or set `DP_CKPT_PATH`
- `METHOD=idm requires IDM_CKPT_PATH`:
  - train IDM first, then set `IDM_CKPT_PATH`
- `uncertainty requires >=2 checkpoints`:
  - increase snapshot history / provide explicit checkpoint list
