# IDM Usage Guide (RoboVerse Offline CLAM)

IDM is a streamlined offline version of CLAM. It keeps only the code needed for **RoboVerse training** and **zarr -> TFDS conversion (optionally with R3M embeddings)**, while removing dependencies on metaworld, d4rl, online rollouts, and other unrelated components.

## Core Structure
- `IDM/udrm/`: training code (main IDM package)
- `IDM/udrm/cfg/`: Hydra configs (mainly roboverse + transformer_clam)
- `IDM/udrm/utils/dataloader.py`: data loading and zarr -> TFDS conversion logic
- `IDM/scripts/zarr_to_tfds.py`: optional manual conversion script
- `IDM/r3m/`: R3M model code (embedding extraction)
- `IDM/data/`: default data directory (TFDS is written here)

## Installation
Recommended in your conda environment:
```bash
pip install -e IDM
```
If you do not install it, you can still run with `PYTHONPATH=IDM` (examples below).

## Data Preparation (RoboVerse zarr)
You need a RoboVerse-exported zarr dataset, for example:
```
RoboVerse/data_policy/stack_cubeFrankaL0_obs:joint_pos_act:joint_pos_99.zarr
```

Recommended fields in the zarr:
- `observations` (state)
- `actions`
- `images` (if you want R3M embeddings or image training)

## One-Command Training (Auto zarr -> TFDS + R3M)
The command below automatically converts zarr to TFDS and precomputes R3M embeddings:
```bash
PYTHONPATH=IDM python -m udrm.main --config-name train_transformer_clam env=roboverse \
  env.dataset_name=roboversetfds +env.datasets=[stack_cube] \
  data.data_dir=./IDM/data data.data_type=zarr \
  data.zarr_path=RoboVerse/data_policy/stack_cubeFrankaL0_obs:joint_pos_act:joint_pos_99.zarr \
  data.embedding=True data.r3m_id=resnet50 model.use_pretrained_embeddings=True \
  env.image_obs=True data.use_images=False
```

### Embedding-Related Parameters
- `data.embedding=True`: enable R3M precomputed embeddings and write them to TFDS
- `data.r3m_id=resnet50`: choose R3M backbone (`resnet50` is common)
- `model.use_pretrained_embeddings=True`: use embeddings as training input
- `env.image_obs=True`: must be `True` for embedding replacement path
- `data.use_images=False`: do not keep raw images (keep embeddings only)

If you want to keep images (for visualization or reconstruction), set:
```bash
data.use_images=True
```

## Without Embeddings (State or Image Input)
If you only want **state input**:
```bash
data.embedding=False
env.image_obs=False
data.use_images=False
```

If you want **raw image input** (instead of embeddings), first build TFDS with images:
```bash
data.embedding=False
env.image_obs=True
data.use_images=True
```

## Space-Time Model Example (`st_vivit_clam`)
To run **Space-Time CLAM (ViViT)** with raw image input:
```bash
PYTHONPATH=IDM python -m udrm.main --config-name train_st_vivit_clam env=roboverse \
  env.dataset_name="${DATASET_NAME}" +env.datasets="[${DATASET}]" \
  data.data_dir="${DATA_DIR}" data.data_type=zarr \
  data.zarr_path="${ZARR_PATH}" \
  data.dataset_variant="${DATASET_VARIANT}" \
  data.use_images=True data.embedding=False \
  env.image_obs=True model.use_pretrained_embeddings=False \
  log_rollout_videos=True \
  data.use_cache=False \
  data.drop_images_after_obs=True
```

### Parameter Notes
- `--config-name train_st_vivit_clam`: use Space-Time CLAM config (ViViT version)
- `env=roboverse`: environment config
- `env.dataset_name`: TFDS group name (typically `roboversetfds`)
- `+env.datasets=[stack_cube]`: specific dataset(s) to train on
- `data.data_dir`: TFDS root directory (internally uses `tensorflow_datasets/`)
- `data.data_type=zarr`: auto-convert from zarr if cache does not exist
- `data.zarr_path`: your zarr path
- `data.dataset_variant`: TFDS subdirectory to separate variants (`pre_emb`, `full_obs`, etc.)
- `data.use_images=True`: store and use raw images
- `data.embedding=False`: disable R3M embedding generation
- `env.image_obs=True`: tell the model the input is image-based
- `model.use_pretrained_embeddings=False`: do not use embedding input
- `log_rollout_videos=True`: generate videos during eval (requires `imageio[ffmpeg]`)
- `data.use_cache=False`: avoid high RAM usage from full TFDS cache
- `data.drop_images_after_obs=True`: move image info into observations and drop `images` to save memory

## Script Usage (Recommended)
To avoid very long commands:
```bash
bash IDM/scripts/run_st_vivit_clam.sh
```

Override dataset path/name example:
```bash
DATASET=stack_cube \
DATASET_VARIANT=full_obs \
DATA_DIR=./IDM/data \
ZARR_PATH=RoboVerse/data_policy/stack_cubeFrankaL0_obs:joint_pos_act:joint_pos_99.zarr \
bash IDM/scripts/run_st_vivit_clam.sh
```

Extra args are passed through to `udrm.main`, for example:
```bash
bash IDM/scripts/run_st_vivit_clam.sh num_updates=50000 data.batch_size=8
```

## TFDS Caching
When `data.data_type=zarr`, training checks:
```
<data.data_dir>/tensorflow_datasets/<env.dataset_name>/<dataset>
```
If this directory exists, conversion is skipped and cache is loaded directly.
To force regeneration, remove the directory:
```bash
rm -rf IDM/data/tensorflow_datasets/roboversetfds/stack_cube
```

## Manual Conversion (Optional)
You can also convert zarr to TFDS in advance:
```bash
python IDM/scripts/zarr_to_tfds.py \
  --zarr-path RoboVerse/data_policy/stack_cubeFrankaL0_obs:joint_pos_act:joint_pos_99.zarr \
  --dataset-name roboversetfds \
  --ds-name stack_cube \
  --output-root ./IDM/data \
  --use-images \
  --data.embedding \
  --r3m-id resnet50
```
If embeddings are not needed, remove `--data.embedding`.
After conversion, set `data.data_type=tfds` in training to use cache directly:
```bash
data.data_type=tfds
```

## Main TFDS Fields
Each episode (variable length) includes:
- `observations`: state `(T, obs_dim)`
- `actions`: action `(T, act_dim)`
- `images`: image `(T, H, W, C)` (optional)
- `embeddings`: R3M vector `(T, D)` (optional)
- `discount / is_first / is_last / is_terminal`

Training additionally derives:
- `states`: training input source (when using embeddings, this is embedding-based input context)
- `mask / timestep`: padding and temporal position helper fields

## Common Configuration Knobs
**Data and conversion**
- `data.data_type`: `zarr` or `tfds`
- `data.data_dir`: TFDS root
- `data.zarr_path`: zarr path (required when `data_type=zarr`)
- `data.use_images`: whether to keep raw images in TFDS
- `data.embedding`: whether to compute R3M embeddings
- `data.r3m_id`: R3M backbone
- `data.r3m_batch_size`: R3M inference batch size
- `data.r3m_device`: `cpu` or `cuda` (auto-selected by default)

**Training and model**
- `data.batch_size`: training batch size
- `data.num_trajs`: trajectory limit (`-1` means all)
- `num_updates`: training steps
- `model.use_pretrained_embeddings`: whether to use embeddings as input
- `env.image_obs`: must match embedding/image mode

**Evaluation and logging**
- `eval_every`: evaluation frequency
- `run_eval_rollouts`: whether to run online rollouts (disabled by default in IDM)
- `use_wandb`: whether to enable Weights & Biases
- `wandb_project / wandb_entity / wandb_name / wandb_group`

## Training Outputs and Paths
Default output path generated by Hydra:
```
results/results/<trainer_name>/<timestamp>/
```
Includes:
- `config.yaml`: full training config for this run
- `model_ckpts/`: model checkpoints (`ckpt_*.pkl`, `latest.pkl`, `best.pkl`)
- `model_ckpts/action_decoder_posthoc.pkl`: post-hoc action decoder checkpoint (if enabled)

## WandB Logging (Enabled by Default)
IDM defaults to `use_wandb=True`. Login first:
```bash
wandb login
```
To disable WandB:
```bash
use_wandb=False
```

## Post-hoc Action Decoder (Latent Action Evaluation)
IDM supports freezing the model after training and training only an action decoder that maps latent action `z` to real actions, to evaluate whether `z` captures action structure.

Key config:
```yaml
post_action_decoder_training: True
post_action_decoder_steps: 20000
post_action_decoder_log_every: 500
```
Optional labeled-action dataset:
```yaml
env.action_labelled_dataset: [stack_cube]
env.action_labelled_dataset_split: [1]
```
If not specified, it defaults to `env.datasets`.

## Troubleshooting
1) **Embedding still looks like state (`in_features=9`)**
   - Confirm `env.image_obs=True` and `model.use_pretrained_embeddings=True`
   - Delete old TFDS cache and regenerate (see TFDS Caching)

2) **Dimension mismatch / positional embedding out-of-range**
   - Older TFDS cache may miss `timestep` fixes; remove cache and regenerate

3) **Visualization triggers even when no images exist**
   - With embeddings, prefer `data.use_images=False` and disable image visualization logic

4) **R3M-related errors**
   - Make sure `torch`, `torchvision`, and `r3m` are installed

If you want to further simplify or extend IDM (for example, adding a new environment or dataset format), define your target and iterate from there.
