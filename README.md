<div align="center">
<h2 align="center">
  World Action Verifier <br> 
  Self-Improving World Models via Forward-Inverse Asymmetry
</h2>

<a href='https://arxiv.org/abs/2604.01985'><img src='https://img.shields.io/badge/ArXiv-2510.10125-red'></a> 
<a href='https://world-action-verifier.github.io/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> 

</div>

This repository contains the official PyTorch implementation of [**WAV**](https://world-action-verifier.github.io/).

This codebase focuses on the controlled setting experiments in Robomimic and Maniskill. 
Note that this is a minimal version; we will update it with a broader range of datasets and models in the coming days.


---

## Installation

This repo has two dependency groups:

1. **Core WM pipeline** (main scripts, built upon [SAILOR](https://github.com/arnavkj1995/SAILOR))
2. **IDM training stack** (`./IDM`)


## Quick Links

- Data collection & pool building: `README_DP_COLLECTION.md`
- Method details & prerequisites: `README_WM_METHODS.md`
- IDM package usage: `IDM/README.md`

### A) Core environments

Create one env per suite:

```bash
cd release

# robomimic
conda env create -f env_ymls/robomimic_env.yml

# maniskill
conda env create -f env_ymls/maniskill_env.yml

```

Core packages (from env files) include:

- `torch`, `torchvision`, `torchaudio` (CUDA wheels)
- `ruamel-yaml`, `termcolor`, `h5py`, `tensorboard`, `wandb`
- `hydra-submitit-launcher`, `omegaconf`, `einops`
- suite-specific:
  - robomimic: `robosuite`, `robomimic`
  - maniskill: `mani-skill`
  - robocasa: `robocasa`, `robomimic`, `robosuite`

### B) IDM stack

Option 1 (recommended): create dedicated IDM env from lock-style file:

```bash
cd release/IDM
conda env create -f environment.yaml
```

Option 2: install package in an existing env after preinstalling deps:

```bash
cd release
pip install -e IDM
```

IDM-specific packages (from `IDM/environment.yaml`) include:

- `tensorflow`, `tensorboard`, `tensorflow-estimator`
- `torch`, `torchvision`
- `hydra`/`omegaconf` stack
- `scikit-learn`, `scipy`
- `imageio`, `imageio-ffmpeg`
- optional model deps such as `dinov2/3`

---

## Pipeline Overview

The workflow is:

1. **Collect diverse DP rollouts**
2. **Build data pools** (`train_pool.jsonl`, `sample_pool.jsonl`, `eval_pool.jsonl`)
3. **Train WM with selection method** (`random`, `progress`, `curiosity`, `idm`)
4. **(Optional) Train IDM from pools**, then use `METHOD=idm`

Core WM methods:

- `random`: random sample selection
- `progress`: learning progress (`old_loss - new_loss`)
- `curiosity`: uncertainty-style proxy (ensemble variance path)
- `idm`: IDM-vs-WM mismatch scoring

---

## Main Scripts

### Data

- `scripts/dp_collect_robomimic.sh`
- `scripts/dp_collect_maniskill.sh`
- `scripts/build_pools.sh`

### WM training

- `scripts/train_wm_method.sh` (general, all methods)
- `scripts/train_wm_robomimic.sh` (env preset)
- `scripts/train_wm_maniskill.sh` (env preset)
- `scripts/train_wm_robocasa.sh` (env preset)

### IDM training

- `scripts/train_idm_from_pool.sh` (general pool-based entry)
- `IDM/scripts/train_idm_action_decoder.sh`
- `IDM/scripts/train_idm_action_decoder_sailor.sh`

---

## Minimal End-to-End Example (robomimic/can)

```bash
cd release

# 1) Collect diverse DP rollouts
TASKS="can" ./scripts/dp_collect_robomimic.sh

# 2) Build pools
TASK=can SUITE=robomimic EXP_NAME=dp_collect_diverse_can_demos50_n30_noise0.10 \
./scripts/build_pools.sh

# 3) Train WM with a method (example: progress)
METHOD=progress SUITE=robomimic TASKS="can" \
EXP_NAME=dp_collect_diverse_can_demos50_n30_noise0.10 \
./scripts/train_wm_method.sh
```

If you plan to run `METHOD=idm`, train IDM first and provide `IDM_CKPT_PATH`.

---

## Configuration

Important hyperparameters are environment-variable configurable in scripts (e.g. WM steps, refresh period, sample size, mix ratio, method-specific checkpoint paths). Defaults are intentionally conservative (for quick eval with minimal computes) and can be overridden per experiment.

See:

- `README_DP_COLLECTION.md` for data/pool knobs
- `README_WM_METHODS.md` for method-specific knobs and required checkpoints

---

## Targets (TODO)

- [ ] More diverse data pools.
- [ ] Dataset and checkpoint release.
- [ ] More datasets (e.g., MimicGen).


---
## Acknowledgement

This codebase is built upon [SAILOR](https://github.com/arnavkj1995/SAILOR) and [CLAM](https://github.com/clamrobot/clam). We thank the authors for their open-source contributions.


## Bibtex 
If you find our work helpful, please leave us a star and cite our paper. Thank you!
```
@article{liu2026wav, 
      title={World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry}, 
      author={Yuejiang Liu and Fan Feng and Lingjing Kong and Weifeng Lu and Jinzhou Tang and Kun Zhang and Kevin Murphy and Chelsea Finn and Yilun Du}, 
      year={2026}, 
      eprint={2604.01985}, 
      archivePrefix={arXiv}, 
      primaryClass={cs.LG}, 
      url={https://arxiv.org/abs/2604.01985}, 
}
```