# WAV on MMBench2

Experiments on the original **10 seen + 10 unseen tasks**. The IDM is a frozen
predictor over Base-tokenizer latents, separate from the robot CLAM model.

| Method | Data collection or selection |
| --- | --- |
| WAV offline (`wav-h4-d0125`) | Select 50 trajectories by IDM-guided rollout divergence, horizon 4, d=0.125 |
| WAV online (`wav-cem-alpha075`) | Collect 50 trajectories with action-cycle CEM; train with target-task probability alpha=0.75 |
| u_s (`us-norm-pool50`) | Select by stochastic prediction variance |
| u_r (`ur-norm-pool50`) | Select by motion-normalized reconstruction residual |
| u_f (`uf-norm-pool50`) | Select by flow instability |
| Random (`random-pool50`) | Select uniformly with a fixed seed |

Offline methods share a pool of 250 trajectories per task and select 50 whole
trajectories. The u_s/u_r/u_f rows rank this fixed pool. Online WAV interacts with
the environment and records its actual actions and rewards.

## Setup

Use Linux with an NVIDIA GPU. Run commands from this directory:

```bash
python scripts/setup_upstream.py
conda env create -f upstream/environment.yaml
conda activate mmbench2
```

The setup script fetches the original simulator package at the revision in
[PROVENANCE.json](PROVENANCE.json). Set `MMBENCH2_REPO` to use an existing checkout.
The bundled model and loss code retains the [MMBench2 license](upstream/LICENSE).

## Checkpoints and data

Download the Base feature tokenizer, Coverage-aware initialization and frozen IDM:

```bash
python scripts/download_checkpoints.py --variants base coverage_aware --idm
```

The downloader verifies the hashes in [released-checkpoints.json](released-checkpoints.json).
Use `--variants base combined` to download the authors' reference model pairs.
For authenticated Hub access, use `hf auth login` or `HF_TOKEN`.
Additional archived model pairs are hosted at
[WAV-MMBench-Checkpoint](https://huggingface.co/ffeng1017/WAV-MMBench-Checkpoint).

Download raw training and held-out partitions:

```bash
export MMBENCH2_DATA=/path/to/mmbench2-data
python upstream/src/download_dataset.py --local_dir "$MMBENCH2_DATA" \
  --subset expert mixed-large mixed-small zeros \
  active_random active_curiosity_u_r_norm active_expert active_human active_zero \
  test active_test active_test_human
```

- **Candidate pool:** 50 trajectories from each of the five `active_*` training sources.
- **Background replay:** `expert`, `mixed-large`, `mixed-small`, and `zeros`.
- **Evaluation only:** `test`, `active_test`, and `active_test_human`.

The loaders stream PNG strips directly. Missing replay RGB is recorded in
`raw-index-summary.json`; missing candidate RGB stops preparation.

## Offline WAV and baselines

Index the data, score training trajectories and write the selections:

```bash
python scripts/prepare_offline.py
```

Use `--methods random-pool50` for Random alone; it needs no scoring GPU or IDM.
For parallel scoring, assign distinct `--tasks` to separate processes, then run
`python scripts/prepare_offline.py --select-only` after all scores are ready.

Train and export a setting:

```bash
SETTING=wav-h4-d0125  # or random-pool50, ur-norm-pool50, uf-norm-pool50, us-norm-pool50
torchrun --standalone --nproc_per_node=8 train_stage.py --setting "$SETTING" --stage tokenizer
torchrun --standalone --nproc_per_node=8 train_stage.py --setting "$SETTING" --stage dynamics
python export_models.py --setting "$SETTING"
```

- **Budget:** tokenizer 50k updates, then dynamics 30k updates.
- **Global batches:** 96 / 512; gradient accumulation adapts to the GPU count.
- **Optimizer:** AdamW, learning rate 1e-4; background replay retained, alpha=0.5.

Both stages start from Coverage-aware. Dynamics uses the completed tokenizer.
Training resumes from `latest.pt` and retains optimizer/RMS states and previous
checkpoints. Use `--workers 0` to disable worker prefetch when debugging.

## Online WAV

CEM maximizes disagreement between proposed actions and IDM-recovered actions
from WM predictions. The WM and IDM stay frozen during collection.

```bash
python collect_online.py --task all --episodes 50
python scripts/prepare_offline.py --index-only
python prepare_online.py

torchrun --standalone --nproc_per_node=8 train_online.py --stage tokenizer
torchrun --standalone --nproc_per_node=8 train_online.py --stage dynamics
python export_models.py --online --setting wav-cem-alpha075
```

- **CEM:** 128 candidates, horizon 32, 32 elites, 3 iterations, 2 predictions per candidate.
- **Execution:** replan every 16 steps; assign distinct tasks to GPUs with `--task <name>`.
- **Training:** fixed 50-trajectory snapshot, tokenizer 50k / dynamics 30k, alpha=0.75.

Alpha assigns 75% of task samples to target tasks and 25% to background tasks.
Collection skips validated episodes on reruns; preparation requires all 50
complete episodes per task and checks their seeds and collection protocol.

## Evaluation

Use a trained setting from the table above, or `base`, `coverage_aware`, `combined`:

```bash
python evaluate.py --variant wav-h4-d0125 --kind offline --task all
python evaluate.py --variant wav-h4-d0125 --kind mpc --task all
python evaluate.py --online --variant wav-cem-alpha075 --kind offline --task all
python evaluate.py --online --variant wav-cem-alpha075 --kind mpc --task all
```

- **Offline:** reconstruction PSNR/MSE/LPIPS, rollout PSNR/MSE and action interventions;
  context 8, horizon 16, 8 denoising steps, context noise 0.1.
- **MPC:** three full episodes per task, seeds 731/732/733, horizon 32, replan stride 16.
- **Outputs:** JSON metrics under the run's `evaluation/` directory, with input checkpoint hashes.

MPC uses a fresh process per episode and reuses complete caches only for the same
checkpoint pair. Its default DMControl backend is OSMesa; install `libOSMesa` or
set `MMBENCH2_RENDERER=egl` on a configured EGL host. ManiSkill requires a working
Vulkan renderer. Use the same rendering setup across methods.

## Paths and logging

| Variable | Default, relative to this directory |
| --- | --- |
| `MMBENCH2_REPO` | `_deps/mmbench2` |
| `MMBENCH2_DATA` | `data` |
| `MMBENCH2_MODELS` | `checkpoints` |
| `MMBENCH2_IDM` | `checkpoints/idm/idm_tokenizer.pt` |
| `MMBENCH2_RUN` | `runs/offline` |
| `MMBENCH2_COLLECTION` | `runs/collection` |
| `MMBENCH2_ONLINE_RUN` | `runs/online` |

W&B is disabled by default. Enable it with `WANDB_MODE=online`, `WANDB_ENTITY`
and `WANDB_PROJECT`. Use a new run directory when changing data, checkpoints or rendering settings.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic inputs to check selection budgets, action alignment,
sampling probabilities, model updates, checkpoint loading and cache isolation.
