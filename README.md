<div align="center">
<h2 align="center">
  World Action Verifier <br>
  Self-Improving World Models via Forward-Inverse Asymmetry
</h2>

<p>
  Yuejiang Liu<sup>*</sup>, Fan Feng<sup>*</sup>, Lingjing Kong<sup>*</sup>,
  Weifeng Lu<sup>*</sup>, Jinzhou Tang, Kun Zhang, Kevin Murphy, Chelsea Finn, Yilun Du
</p>
<p><sup>*</sup> Equal contribution</p>
<p><strong>NeurIPS 2026</strong></p>

<a href="https://arxiv.org/abs/2604.01985"><img src="https://img.shields.io/badge/arXiv-2604.01985-b31b1b" alt="arXiv"></a>
<a href="https://world-action-verifier.github.io/"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project website"></a>
<a href="https://huggingface.co/ffeng1017/WAV-MMBench-Checkpoint"><img src="https://img.shields.io/badge/Hugging_Face-Checkpoints-yellow" alt="Hugging Face checkpoints"></a>

</div>

This repository contains the PyTorch implementation of [**WAV**](https://world-action-verifier.github.io/) for Robomimic and ManiSkill.

## Installation

Run commands from the repository root. Use a separate environment for each
simulator suite:

```bash
conda env create -f env_ymls/robomimic_env.yml
# Or:
conda env create -f env_ymls/maniskill_env.yml
```

For IDM training:

```bash
conda env create -f IDM/environment.yaml
```

If the IDM dependencies are already installed, install the package with
`pip install -e IDM`. See [IDM/README.md](IDM/README.md) for model configurations.

## Collect trajectories

Prepare the task's expert demonstrations, activate the simulator environment,
and run the collection script. For Robomimic:

```bash
TASKS="can square" CUDA_VISIBLE_DEVICES=0 USE_WANDB=False \
NUM_EXP_TRAJS=50 DP_TRAIN_STEPS=24000 \
ROLLOUT_SNAPSHOT_COUNT=30 ROLLOUT_SNAPSHOT_STEPS=3000 \
ROLLOUT_SNAPSHOT_NOISE_STD=0.10 \
bash scripts/dp_collect_robomimic.sh
```

For ManiSkill:

```bash
TASKS="pullcube liftpeg pokecube" CUDA_VISIBLE_DEVICES=0 USE_WANDB=False \
bash scripts/dp_collect_maniskill.sh
```

`ROLLOUT_SNAPSHOT_COUNT` sets the number of collection checkpoints during DP
training. `ROLLOUT_SNAPSHOT_STEPS` sets the number of environment steps collected
at each checkpoint; it is not a trajectory count. `ROLLOUT_SNAPSHOT_NOISE_STD`
sets the action noise during collection.

Rollouts are written to:

```text
scratch_dir/logs/<suite__task>/<run_name>/seed<seed>/dp_rollouts/step_*/
```

See [data collection notes](README_DP_COLLECTION.md) for the remaining options.

## Build pools

The staged Robomimic pool builder reads this layout:

```text
stages/
  update_000001/
    collection.json          # contains horizon
    rollouts/
      episode_0.npz
      episode_0.json         # episode metadata, including stable5_success
```

Organize the collection outputs into this layout and retain their episode
metadata. The expert HDF5 needs separate `mask/train` and `mask/valid` demo lists.

```bash
TASK=can DP_ROLLOUTS_ROOT=/path/to/stages \
EXPERT_HDF5=/path/to/expert.hdf5 OUTPUT_DIR=./scratch_dir/pools/can \
bash scripts/build_pools.sh
```

| Manifest | Used for |
| --- | --- |
| `train_pool.jsonl` | Initial WM training on expert train demos |
| `sample_pool.jsonl` | Candidate rollouts for trajectory selection |
| `idm_train_pool.jsonl` | IDM training on the collected rollouts |
| `eval_pool.jsonl` | Evaluation on expert valid demos |

Pool sizes follow the supplied rollouts and expert split. Evaluation on expert
valid demos measures performance on that distribution.

## Train the IDM

Activate the IDM environment, then run:

```bash
TASK=can POOLS_ROOT=./scratch_dir/pools/can USE_WANDB=False \
NUM_UPDATES=50000 DETACH_IDM_LATENTS=false \
bash scripts/train_idm_from_pool.sh
```

This trains CLAM, then fits the action decoder with CLAM frozen. The decoder is
saved with the IDM and used to recover actions during WAV selection.

For `train_st_vivit_clam_stm`, `DETACH_IDM_LATENTS` controls whether the
reconstruction loss updates the inverse encoder:

- `false`: update the IDM from reconstruction loss, as in the command above.
- `true` (launcher default): detach the inverse latents before the FDM.

The equivalent Hydra option is `model.fdm.detach_idm_latents`. This controls the
reconstruction gradient path; supervised IDM/action-decoder training has its own
updates.

## Train the world model

Activate the Robomimic environment. Point `IDM_CKPT_PATH` at the trained IDM run
or its checkpoint, and `DP_CKPT_PATH` at the diffusion-policy checkpoint:

```bash
METHOD=idm SUITE=robomimic TASKS="can" USE_WANDB=False \
POOLS_ROOT=./scratch_dir/pools/can DATADIR=/path/to/expert-data \
DP_CKPT_PATH=/path/to/dp_checkpoint.pt \
IDM_CKPT_PATH=/path/to/idm_run \
SAMPLE_START_ITR=5000 SAMPLE_REFRESH_EVERY=5000 \
SAMPLE_SELECT_SIZE=120 SAMPLE_MIX_RATIO=0.3 \
WM_ONLY_TRAIN_ITRS=200000 \
bash scripts/train_wm_method.sh
```

This starts selection at WM iteration 5,000 and refreshes the selected 120
trajectories every 5,000 iterations. Each training batch uses 50% expert data.
For the other half, `SAMPLE_MIX_RATIO=0.3` gives a 30% chance of sampling from the
selected pool; otherwise it uses the initial training pool. The selected data
therefore accounts for about 15% of training examples on average after selection
starts.

Set `METHOD=random`, `METHOD=progress`, or `METHOD=curiosity` to compare selectors.
Keep the data budget, selection size, mixing probability, and training steps the
same across methods. See [method options](README_WM_METHODS.md) and
[training notes](README_LOCAL_TRAINING.md) for checkpoint loading and evaluation.

## MMBench2 experiments

We study WAV on the 10 seen and 10 unseen tasks used by
[MMBench2](https://github.com/nicklashansen/mmbench2). Seen and unseen refer to
the original WM pretraining split; both groups receive task-specific adaptation
data in these experiments. The IDM here is a frozen tokenizer-latent action
predictor, separate from the CLAM used in the robot experiments.

### Offline WAV

- **Pool:** 250 trajectories per task, 50 each from random, curiosity, expert,
  human, and no-op collection.
- **Selection:** Keep the top 50 by disagreement between observed futures and
  WM rollouts under IDM-inferred actions.
- **Baselines:** Random and u_r/u_f/u_s select 50 from the same pool; Combined
  local uses all 250. Released Base, Coverage-aware, and Combined are references.

### Online WAV

- **Exploration:** CEM searches for action sequences with high mismatch between
  proposed actions and IDM-recovered actions from WM predictions. Both models
  remain frozen during collection.
- **Data:** Collect 50 real-environment trajectories per task for WM training.
- **Task sampling:** alpha=0.75 gives 75% target tasks and 25% background tasks.

### Training and evaluation

Local continuation runs start from Coverage-aware and train the tokenizer for
50k updates, followed by dynamics for 30k updates. Global batch sizes are 96
and 512, respectively. We use AdamW with learning rate 1e-4 and retain background
replay. Offline runs use target-task sampling alpha=0.5; online WAV uses 0.75.
The Base tokenizer supplies the frozen IDM's feature space.

Evaluation covers reconstruction, video rollout prediction, action
interventions, and MPC on all 20 tasks. Offline evaluation uses context 8,
horizon 16, 8 denoising steps, and context noise 0.1. MPC uses three full
episodes per task with planning horizon 32 and replanning every 16 steps.

### Checkpoints

Download the tokenizer and dynamics files for each variant from the original
authors' [Hugging Face model repository](https://huggingface.co/nicklashansen/mmbench2-models).
Use the two files from the same variant together:

| Variant | Tokenizer | Dynamics |
| --- | --- | --- |
| Base | [tokenizer.pt](https://huggingface.co/nicklashansen/mmbench2-models/resolve/b7f7b222526d557f80bf3bcf84ab824cb14f94b0/base/tokenizer.pt?download=true) | [dynamics.pt](https://huggingface.co/nicklashansen/mmbench2-models/resolve/b7f7b222526d557f80bf3bcf84ab824cb14f94b0/base/dynamics.pt?download=true) |
| Combined | [tokenizer.pt](https://huggingface.co/nicklashansen/mmbench2-models/resolve/b7f7b222526d557f80bf3bcf84ab824cb14f94b0/combined/tokenizer.pt?download=true) | [dynamics.pt](https://huggingface.co/nicklashansen/mmbench2-models/resolve/b7f7b222526d557f80bf3bcf84ab824cb14f94b0/combined/dynamics.pt?download=true) |

These links use the release revision from our experiments. Coverage-aware, used
for continuation training, is available in the same model repository under
[coverage_aware](https://huggingface.co/nicklashansen/mmbench2-models/tree/b7f7b222526d557f80bf3bcf84ab824cb14f94b0/coverage_aware).

The model pairs used in our MMBench2 experiments are hosted in
[WAV-MMBench-Checkpoint](https://huggingface.co/ffeng1017/WAV-MMBench-Checkpoint).
The repository contains `default`, `random-pool50`, `ur-norm-pool50`,
`uf-norm-pool50`, and `us-norm-pool50`, each with a tokenizer and dynamics file.
Use the two files from the same directory together.

## Tests

With a matching PyTorch/torchvision pair installed:

```bash
pip install -r tests/requirements-smoke.txt
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 \
TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

The tests cover pool construction, IDM gradient control, WM updates, checkpoint
loading, and all four selectors using small synthetic episodes. CPU tests, GPU
integration, and a short IDM training run have passed. Full robot rollouts and
long training runs are separate experiments.

---

## Acknowledgments

We thank [SAILOR](https://github.com/arnavkj1995/SAILOR),
[MMBench2](https://github.com/nicklashansen/mmbench2), and
[CLAM](https://github.com/clamrobot/clam) for their open-source code and resources.

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{liu2026wav,
  title={World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry},
  author={Liu, Yuejiang and Feng, Fan and Kong, Lingjing and Lu, Weifeng and Tang, Jinzhou and Zhang, Kun and Murphy, Kevin and Finn, Chelsea and Du, Yilun},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  eprint={2604.01985},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2604.01985}
}
```
