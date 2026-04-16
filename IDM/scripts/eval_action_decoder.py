#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Register OmegaConf resolvers used in config.yaml
import udrm.resolvers  # noqa: F401

from udrm.models.mlp_policy import MLPPolicy
from udrm.models.utils.clam_utils import get_clam_cls, get_la_dim
from udrm.trainers.clam_trainer import get_labelled_dataloader
from udrm.utils.data_utils import Batch
from udrm.utils.general_utils import to_device
from udrm.utils.logger import log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        default=None,
        help="Training experiment dir containing config.yaml and model_ckpts/",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (if not using --exp-dir)",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to checkpoint .pkl (if not using --exp-dir)",
    )
    parser.add_argument(
        "--ckpt-name",
        default="latest.pkl",
        help="Checkpoint filename under model_ckpts/ when using --exp-dir",
    )
    parser.add_argument("--num-batches", type=int, default=2, help="Num batches to eval")
    parser.add_argument("--batch-size", type=int, default=4, help="Eval batch size")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto if None)")
    parser.add_argument("--print-shapes", action="store_true", help="Print tensor shapes")
    return parser.parse_args()


def load_cfg(config_path: Path):
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


def infer_obs_dim(cfg, observations):
    if cfg.env.image_obs and not cfg.model.use_pretrained_embeddings:
        if cfg.data.data_type == "n_step":
            return observations.shape[2:]
        return observations.shape[1:]
    return observations.shape[-1]


def main():
    args = parse_args()

    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        config_path = Path(args.config) if args.config else exp_dir / "config.yaml"
        ckpt_path = exp_dir / "model_ckpts" / args.ckpt_name
    else:
        if args.config is None or args.ckpt is None:
            raise ValueError("Provide --exp-dir or both --config and --ckpt")
        config_path = Path(args.config)
        ckpt_path = Path(args.ckpt)

    cfg = load_cfg(config_path)
    cfg.use_wandb = False
    cfg.post_action_decoder_training = False
    cfg.joint_action_decoder_training = False
    cfg.data.use_cache = False
    cfg.data.batch_size = args.batch_size

    if not getattr(cfg.env, "action_labelled_dataset", None):
        cfg.env.action_labelled_dataset = cfg.env.datasets
        cfg.env.action_labelled_dataset_split = cfg.env.dataset_split

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"using device: {device}")

    labelled_ds = get_labelled_dataloader(cfg)
    labelled_iter = labelled_ds.repeat().as_numpy_iterator()

    first_batch = next(labelled_iter)
    obs_dim = infer_obs_dim(cfg, first_batch["observations"])

    clam_cls = get_clam_cls(cfg.name)
    la_dim = get_la_dim(cfg)
    model = clam_cls(cfg.model, input_dim=obs_dim, la_dim=la_dim).to(device)
    action_decoder = MLPPolicy(
        cfg=cfg.model.action_decoder,
        input_dim=la_dim,
        output_dim=cfg.env.action_dim,
    ).to(device)

    # Our checkpoint includes OmegaConf objects.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "action_decoder" in ckpt:
        action_decoder.load_state_dict(ckpt["action_decoder"])
    else:
        log("checkpoint has no action_decoder weights; using random init", "yellow")

    use_transformer = "transformer" in cfg.model.idm.net.name
    model.eval()
    action_decoder.eval()

    for i in range(args.num_batches):
        batch_np = first_batch if i == 0 else next(labelled_iter)
        batch_t = to_device(batch_np, device)
        batch = Batch(**batch_t)

        with torch.no_grad():
            if use_transformer:
                clam_output = model(
                    batch.observations, timesteps=batch.timestep, states=batch.states
                )
                la = clam_output.la[:, 1:]
            else:
                clam_output = model(batch.observations)
                la = clam_output.la

            if cfg.model.distributional_la:
                la = model.reparameterize(la)

            action_pred = action_decoder(la)
            gt_actions = batch.actions[:, :-1] if use_transformer else batch.actions

        if args.print_shapes or i == 0:
            log(f"obs shape: {batch.observations.shape}")
            log(f"la shape: {la.shape}")
            log(f"pred shape: {action_pred.shape}")
            log(f"gt shape: {gt_actions.shape}")
            log(f"cfg.env.action_dim: {cfg.env.action_dim}")

        if action_pred.shape != gt_actions.shape:
            log("shape mismatch between action_pred and gt_actions", "red")
            break

        mse = torch.mean((action_pred - gt_actions) ** 2).item()
        mae = torch.mean(torch.abs(action_pred - gt_actions)).item()
        log(f"[batch {i}] action_decoder mse={mse:.6f}, mae={mae:.6f}")


if __name__ == "__main__":
    main()

'''
PYTHONPATH=IDM python IDM/scripts/eval_action_decoder.py \
  --exp-dir results/results/transformer_clam/2025-12-27-16-34-34 \
  --print-shapes --num-batches 2
'''
