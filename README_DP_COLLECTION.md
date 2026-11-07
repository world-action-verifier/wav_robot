# Collecting DP rollouts

The collection scripts train a diffusion policy on expert demonstrations and
save rollouts at several points during training. Action noise adds variation to
the collected trajectories.

Run from the repository root in the corresponding simulator environment:

```bash
# Robomimic
TASKS="can square" CUDA_VISIBLE_DEVICES=0 USE_WANDB=False \
ROLLOUT_SNAPSHOT_COUNT=30 ROLLOUT_SNAPSHOT_STEPS=3000 \
ROLLOUT_SNAPSHOT_NOISE_STD=0.10 \
bash scripts/dp_collect_robomimic.sh

# ManiSkill
TASKS="pullcube liftpeg pokecube" CUDA_VISIBLE_DEVICES=0 USE_WANDB=False \
bash scripts/dp_collect_maniskill.sh
```

Prepare the expert demonstrations in the dataset layout expected by the suite
before starting. Both scripts call `train_wm.py` with `train_dp_mppi=False`, which
runs DP training and rollout collection.

## Collection settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `NUM_EXP_TRAJS` | `50` | Expert demonstrations used to train DP |
| `DP_TRAIN_STEPS` | `24000` | DP training updates |
| `DP_EVAL_FREQ` | `12000` | DP evaluation interval |
| `ROLLOUT_SNAPSHOT_COUNT` | `30` | Number of collection points during DP training |
| `ROLLOUT_SNAPSHOT_STEPS` | `3000` | Environment steps collected at each point |
| `ROLLOUT_SNAPSHOT_NOISE_STD` | `0.10` | Action noise during collection |
| `SEED` | `0` | Random seed |

`ROLLOUT_SNAPSHOT_STEPS` counts environment steps, so the number of saved episodes
also depends on episode length and the number of parallel environments.

To log to W&B, set `USE_WANDB=True`, `WANDB_ENTITY`, and `WANDB_PROJECT`.

Rollouts are saved under:

```text
scratch_dir/logs/<suite__task>/<run_name>/seed<seed>/dp_rollouts/step_*/
```

## Build the Robomimic pools

The pool builder reads staged rollouts. Arrange episodes under
`update_<step>/rollouts/*.npz`, retain each episode's JSON metadata, and provide a
`collection.json` containing `horizon` in each `update_<step>` directory. The
expert HDF5 supplies the `mask/train` and `mask/valid` demo lists.

```bash
TASK=can DP_ROLLOUTS_ROOT=/path/to/stages \
EXPERT_HDF5=/path/to/expert.hdf5 OUTPUT_DIR=./scratch_dir/pools/can \
bash scripts/build_pools.sh
```

The builder writes `train_pool.jsonl`, `sample_pool.jsonl`,
`idm_train_pool.jsonl`, `eval_pool.jsonl`, and `pool_summary.json`. All collected
rollouts enter the candidate and IDM pools; the expert masks define the WM
training and evaluation pools.

Continue with [IDM and WM training](README.md#train-the-idm).
