import gc
import os
import pathlib

import numpy as np
import ruamel.yaml as yaml
import torch
from termcolor import cprint

import wav.dreamer.tools as tools
from environments.global_utils import save_demo_videos
from wav.classes.preprocess import Preprocessor
from wav.classes.resnet_encoder import ResNetEncoder
from wav.policies.diffusion_base_policy import DiffusionBasePolicy
from wav.runtime.dataset_loader import load_expert_datasets, load_wm_only_buffers
from wav.runtime.env_factory import build_train_envs
from wav.sailor_trainer import SAILORTrainer


def _setup_logging_and_seed(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser()
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    yaml_writer = yaml.YAML()
    with open(f"{logdir}/config.yaml", "w", encoding="utf-8") as f:
        yaml_writer.dump(vars(config), f)

    config.logdir = logdir
    config.scratch_dir = pathlib.Path(config.scratch_dir).expanduser()
    logger = tools.Logger(config) if config.use_wandb else None
    return logger


def _print_run_banner(config):
    print("---------------------")
    cprint(f"Task: {config.task}", "cyan", attrs=["bold"])
    cprint(f"Logging to: {config.logdir}", "cyan", attrs=["bold"])
    cprint(
        f"Time Limit: {config.time_limit} | Max Env Steps: {config.train_dp_mppi_params['n_env_steps']}",
        "cyan",
        attrs=["bold"],
    )
    if config.visualize_eval:
        cprint(
            f"WARNING: Saving videos of evaluation episodes, please turn off if not needed. High resolution render is {config.high_res_render}",
            "red",
            attrs=["bold"],
        )
    print("---------------------")


def _maybe_visualize_expert_buffer(config, expert_eps):
    if not config.viz_expert_buffer:
        return

    suite, task = config.task.split("__", 1)
    task = task.lower()
    cprint(
        "-----------------Inspecting Expert Dataset, Saving Videos--------------",
        "yellow",
        attrs=["bold"],
    )
    for idx, key in enumerate(expert_eps.keys()):
        frame_successes = np.array(expert_eps[key]["success"]).ravel()
        agent_frames = np.array(expert_eps[key]["agentview_image"])[..., -1]
        robot_frames = np.array(expert_eps[key]["robot0_eye_in_hand_image"])[..., -1]
        save_demo_videos(
            suite=suite,
            task=task,
            id=idx,
            frame_successes=frame_successes,
            agent_frames=agent_frames,
            robot_frames=robot_frames,
        )
    raise SystemExit(0)


def _run_dp_pretrain_if_needed(config, logger, expert_eps, expert_val_eps, envs, state_dim, action_dim):
    log_step = 0
    if config.dp["pretrained_ckpt"] != "":
        return log_step

    cprint(
        "----------------No base policy path provided, begin training diffusion policy--------------",
        "yellow",
        attrs=["bold"],
    )
    preprocessor = Preprocessor(config=config)
    encoder = None if config.state_only else ResNetEncoder()
    base_policy = DiffusionBasePolicy(
        preprocessor=preprocessor,
        encoder=encoder,
        config=config,
        device=config.device,
        state_dim=state_dim,
        action_dim=action_dim,
        logger=logger,
        name="DP_Pretrain",
    )

    expert_dataset_dp = tools.make_dataset(
        expert_eps, batch_length=1, batch_size=config.dp["batch_size"]
    )
    log_step = base_policy.train_base_policy(
        train_dataset=expert_dataset_dp,
        expert_val_eps=expert_val_eps,
        eval_envs=envs,
        log_prefix="dp_pretrain",
    )

    config.dp["pretrained_ckpt"] = os.path.relpath(base_policy.ckpt_file, config.scratch_dir)
    del preprocessor
    del encoder
    del base_policy
    torch.cuda.empty_cache()
    gc.collect()
    return log_step


def run_training_pipeline(config):
    logger = _setup_logging_and_seed(config)
    _print_run_banner(config)

    if "__" not in config.task:
        raise ValueError(f"Task {config.task} must be of form 'env_suite__task'")

    cprint("[Stage] load_datasets", "blue")
    expert_eps, expert_val_eps, _, state_dim, action_dim = load_expert_datasets(config)
    config.state_dim = state_dim
    config.action_dim = action_dim
    cprint(f"Enviroment State Dim: {state_dim}, Action Dim: {action_dim}", "cyan")
    _maybe_visualize_expert_buffer(config, expert_eps)

    cprint("[Stage] build_envs", "blue")
    envs = build_train_envs(config)
    acts = envs.action_space
    print(f"Action Space: {acts}. Low: {acts.low}. High: {acts.high}")
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    cprint("[Stage] pretrain_world_model_backbone (DP if missing)", "blue")
    log_step = _run_dp_pretrain_if_needed(
        config=config,
        logger=logger,
        expert_eps=expert_eps,
        expert_val_eps=expert_val_eps,
        envs=envs,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    if config.train_dp_mppi:
        cprint("[Stage] active_collect_and_retrain_loop", "blue")
        train_eps, sample_eps, wm_eval_eps = load_wm_only_buffers(config)
        trainer = SAILORTrainer(
            config=config,
            expert_eps=expert_eps,
            state_dim=state_dim,
            action_dim=action_dim,
            train_env=envs,
            eval_envs=envs,
            expert_val_eps=expert_val_eps,
            train_eps=train_eps,
            sample_eps=sample_eps,
            wm_eval_eps=wm_eval_eps,
            init_step=log_step,
            logger=logger,
        )
        trainer.train_dp_with_mppi()

    envs.close()
    cprint("--------Finished Everything--------", "yellow", attrs=["bold"])
