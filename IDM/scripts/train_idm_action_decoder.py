import os
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import tensorflow as tf

from udrm.utils.logger import log

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.config.experimental.set_visible_devices([], "GPU")

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from udrm.resolvers import *  # noqa: F401,F403
from udrm.trainers import trainer_to_cls
from udrm.utils.general_utils import omegaconf_to_dict, print_dict


@hydra.main(version_base=None, config_name="config", config_path="../udrm/cfg")
def main(cfg: DictConfig):
    overrides = HydraConfig.get().overrides.task
    log(f"Job overrides: {overrides}")

    # Keep tf quiet in this entrypoint too
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.config.experimental.set_visible_devices([], "GPU")

    if cfg.load_from_ckpt and getattr(cfg, "use_ckpt_config", False):
        ckpt_cfg_file = Path(cfg.ckpt_file) / "config.yaml"
        if ckpt_cfg_file.exists():
            ckpt_cfg = OmegaConf.load(ckpt_cfg_file)
            override_cfg = OmegaConf.create(
                {
                    "load_from_ckpt": cfg.load_from_ckpt,
                    "ckpt_file": cfg.ckpt_file,
                    "ckpt_step": cfg.ckpt_step,
                    "exp_dir": cfg.exp_dir,
                    "wandb_on_load": getattr(cfg, "wandb_on_load", False),
                    "use_wandb": cfg.use_wandb,
                    "mode": cfg.mode,
                }
            )
            cfg = OmegaConf.merge(ckpt_cfg, override_cfg)
            log(f"[ckpt_config] loaded {ckpt_cfg_file}", "green")
        else:
            log(f"[ckpt_config] missing {ckpt_cfg_file}, fallback to current cfg", "yellow")

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    if cfg.name not in trainer_to_cls:
        raise ValueError(f"Invalid trainer name: {cfg.name}")

    log("start")
    trainer = trainer_to_cls[cfg.name](cfg)
    if not hasattr(trainer, "train_idm_action_decoder_only"):
        raise AttributeError("Trainer does not implement train_idm_action_decoder_only()")
    trainer.train_idm_action_decoder_only()
    log("end")
    sys.exit(0)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
