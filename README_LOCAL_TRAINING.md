# Training with staged rollouts

The examples below use staged DP rollouts and an expert HDF5. Set the paths to
match your collection directory.
The candidate pool contains **all** collected rollouts, including successes and
failures. No rollout is held out from this pool. Each task uses a separate pool.

Build each task's manifests:

```bash
TASK=can \
DP_ROLLOUTS_ROOT=/workspace/users/fan/wav_robot_dp/runs/can/stages \
EXPERT_HDF5=/workspace/users/fan/wav_robot_dp/data/can/ph/image_64_shaped_done1_v141.hdf5 \
OUTPUT_DIR=/workspace/users/fan/wav_robot_wm/pools/can \
bash scripts/build_pools.sh
```

Outputs:

- `sample_pool.jsonl`: all collected trajectories for that task, with checkpoint,
  seed, success, stopping reason and file SHA256 metadata.
- `idm_train_pool.jsonl`: the same full rollout pool for IDM pretraining.
- `train_pool.jsonl`: all official expert train demos, for WM bootstrap.
- `eval_pool.jsonl`: official expert valid demos. These do not overlap the
  expert train mask. They do not measure generalization to unseen DP failures.

Counts follow the input data and masks. The builder requires `update_<step>/rollouts/*.npz`, episode
JSON sidecars, and a per-stage `collection.json` with `horizon`. Collection-script
`dp_rollouts/step_*` outputs need staging and this metadata before pool building.

The raw NPZs and DP checkpoints are unchanged. Shared loading fixes legacy final
terminal flags using `stable5_success`: successful collector termination is terminal;
horizon truncation is not. Observations are stored before their corresponding
actions. The final action has no saved successor observation and therefore supplies
no additional next-observation target; RSSM observes/actions remain aligned.

WM training (after activating the Robomimic environment, run from this repository):

```bash
CUDA_VISIBLE_DEVICES=7 USE_WANDB=False \
METHOD=random TASKS=can NUM_EXP_TRAJS=180 NUM_EXP_VAL_TRAJS=20 \
DATADIR=/workspace/users/fan/wav_robot_dp/data \
POOLS_ROOT=/workspace/users/fan/wav_robot_wm/pools/can \
DP_CKPT_PATH=/workspace/users/fan/wav_robot_dp/runs/can/checkpoints/update_024000.pt \
bash scripts/train_wm_method.sh
```

WM-only Robomimic training creates observation/action spaces without launching
rendering simulators. Set budgets explicitly before launching a formal experiment;
the upstream WM launcher still defaults to 200,000 updates. `SAMPLE_SELECT_SIZE`
sets the subset selected per refresh, not the total candidate-pool size.

IDM pretraining (after activating the separate IDM environment):

```bash
CUDA_VISIBLE_DEVICES=7 USE_WANDB=False \
TASK=can POOLS_ROOT=/workspace/users/fan/wav_robot_wm/pools/can \
NUM_UPDATES=50000 \
bash scripts/train_idm_from_pool.sh
```

This launcher now runs full CLAM pretraining followed by the configured post-hoc
action decoder. The persisted decoder is trained with the inverse model frozen.
For `train_st_vivit_clam_stm`, reconstruction gradients into the inverse encoder
are configurable. `model.fdm.detach_idm_latents=true` (default, also the fallback
for older configs) blocks this path; `false` updates the inverse encoder and
latent-action head from reconstruction. This applies to both `SpaceTimeFDM` and
`SpaceTimeFDM_STM` in the STM model module. Supervised action-decoder losses still
update the IDM when enabled; this switch is not a global IDM freeze.

```bash
# Enable reconstruction gradients to IDM (west3 behavior).
DETACH_IDM_LATENTS=false TASK=can POOLS_ROOT=/path/to/pools \
  bash scripts/train_idm_from_pool.sh
# Direct Hydra override: model.fdm.detach_idm_latents=false
```
The legacy supervised inverse entry point updates both the inverse model and decoder.
Checkpoints save input dimensions and the trained decoder for strict downstream
loading. Selection passes relative timesteps and decodes only the final transition
in each context window into one seven-dimensional action.

Curiosity remains a temporal WM-snapshot disagreement proxy. Repeated snapshot
paths are deduplicated; fewer than two distinct snapshots causes an explicitly
recorded random fallback. This is not an independently trained ensemble.

Regression checks (run from the repository root, in a dedicated Python 3.10/3.11
environment with a matching PyTorch / torchvision pair installed):

```bash
pip install -r tests/requirements-smoke.txt
OMP_NUM_THREADS=2 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tests -v
# Optional: use one free GPU for the integration test.
CUDA_VISIBLE_DEVICES=7 WAV_SMOKE_DEVICE=cuda:0 OMP_NUM_THREADS=2 \
  PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tests -p test_offline_pipeline.py -v
```

The tests use temporary synthetic rollouts and an expert HDF5 with explicit masks.
They check both reconstruction-gradient modes, supervised IDM/decoder updates,
uneven gradient accumulation, action/terminal alignment, pool generation, two
small WM updates, strict checkpoint round trips, and scoring/ranking with all four
selectors. Progress and curiosity use two distinct real model snapshots in the
integration test; IDM selection loads a real tiny CLAM and trained decoder. Missing
WM keys must fail loading, and duplicate curiosity snapshots must report fallback.
These are execution checks, not robot-task performance or benchmark measurements.
Pool slicing uses a TensorFlow fallback when optional RLDS/Reverb is unavailable.
