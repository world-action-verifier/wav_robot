import copy
import os
import sys
from collections import defaultdict, deque

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

from wav.classes.evaluator import ModelEvaluator
from wav.classes.preprocess import Preprocessor
from wav.classes.rollout_utils import collect_onpolicy_trajs
from wav.diffusion.data4robotics import transforms
from wav.diffusion.data4robotics.models.diffusion_unet import \
    DiffusionUnetAgent
from wav.diffusion.data4robotics.trainers.bc import BehaviorCloning
from wav.diffusion.data4robotics.trainers.utils import optim_builder
from wav.dreamer import tools


class WeigtedActionWrapper:
    IMAGE_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]

    def __init__(self, agent, config, preprocessor: Preprocessor, EXP_WEIGHT=0.0):
        self.config = config
        self.EXP_WEIGHT = EXP_WEIGHT
        self.pred_horizon = config.pred_horizon
        self.weights = np.exp(-self.EXP_WEIGHT * np.arange(self.pred_horizon))
        self.weight_cumsum = np.cumsum(self.weights)
        self.obs_horizon = config.obs_horizon
        self.preprocessor = preprocessor

        if config.dp["num_cams"] == 1:
            self.IMAGE_KEYS = ["agentview_image"]

        self.agent = agent
        self.transform = transforms.get_transform_by_name("preproc")

        assert hasattr(agent, "get_actions")
        self.reset()

    def reset(self):
        # These variables are set the first time get_action is called
        self.image_history = None
        self.act_history = None
        self.num_envs = None
        self.pred_action_history = deque(maxlen=self.pred_horizon)

    def _get_images_and_states(self, obs):
        """
        Input: num_envs x ...
        """
        assert (
            obs["state"].shape[0] == self.num_envs
        ), "Batch size mismatch in observations"

        # Get stacked images based on images seen by the WeigtedActionWrapper
        if not self.config.state_only:
            stacked_images = defaultdict(
                list
            )  # List of length num_envs, each element is stacked obs_horizon images
            for env_id in range(self.num_envs):
                for key in self.IMAGE_KEYS:
                    # If the length of the deque is less than the obs_horizon, fill with current image
                    if len(self.image_history[key][env_id]) == 0:
                        for _ in range(self.obs_horizon):
                            self.image_history[key][env_id].append(obs[key][env_id])

                    self.image_history[key][env_id].append(obs[key][env_id])

                    # Append Shape (obs_horizon, ...)
                    stacked_images[key].append(
                        np.stack(list(self.image_history[key][env_id]), axis=0)
                    )

            # Stack images across env dim to get shape (num_envs, obs_horizon, ...)
            for key in self.IMAGE_KEYS:
                obs[key] = np.stack(stacked_images[key], axis=0)

        # Unsqueeze state for preprocessing
        obs["state"] = obs["state"][:, None, ...]  # (num_envs, 1, ...)

        # Preprocess the data (input shape is [num_envs x obs_horizon x ...])
        obs = self.preprocessor.preprocess_batch(obs, training=False)

        if not self.config.state_only:
            if "robot0_eye_in_hand_image" in obs.keys():
                images = {
                    "cam0": obs["agentview_image"],
                    "cam1": obs["robot0_eye_in_hand_image"],
                }
            else:
                images = {"cam0": obs["agentview_image"]}
        else:
            images = None

        state = obs["state"][:, -1, :].to(torch.float32)  # Take the last state
        return images, state

    def get_weighted_action(self, ac, env_id, get_full_action=False):
        """
        ac: pred_horizon x action_dim
        Return shape:
        - if get_full_action=False: action_dim
        - if get_full_action=True: pred_horizon x action_dim
        """
        # ac: pred_horizon x action_dim
        self.act_history[env_id].append(ac)
        num_actions = len(self.act_history[env_id])

        if not get_full_action:
            # num_actions x action_dim
            curr_act_preds = np.stack(
                [
                    pred_actions[i]
                    for (i, pred_actions) in zip(
                        range(num_actions - 1, -1, -1), self.act_history[env_id]
                    )
                ]
            )

            # more recent predictions get exponentially *less* weight than older predictions
            weights = np.exp(-self.EXP_WEIGHT * np.arange(num_actions))
            weights = weights / weights.sum()

            # compute the weighted average across all predictions for this timestep
            weighted_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        else:
            pred_horizon = ac.shape[0]
            weighted_action = np.zeros_like(ac)
            for i in range(pred_horizon):
                # a(t=i) = (deque[t=0][i]*w[0]== + deque[t=1][1+i]*w[1] + ... + deque[n][n+i]*w[n])/sum(w[:n])
                res_i = np.zeros(ac.shape[1])
                count_i = 0
                for j, pred_actions in zip(
                    range(num_actions - 1, -1, -1), self.act_history[env_id]
                ):
                    if i + j < pred_horizon:
                        res_i += pred_actions[i + j] * self.weights[j]
                        count_i += 1
                weighted_action[i] = res_i / self.weight_cumsum[count_i - 1]
            weighted_action = weighted_action.transpose(
                1, 0
            )  # Shape (action_dim, pred_horizon)

        return weighted_action

    def get_action(self, obs, scale=1.0, weighting=True, get_full_action=False):
        """
        obs orignial shape <data_dim> no stacking
        parallel env obs shape: (num_envs, data_dim)
        ~0.12 seconds for fwd pass on 3090
        """
        # If buffers are None, initialize them and set num_envs
        if self.image_history is None or self.act_history is None:
            self.num_envs = obs["state"].shape[0]
            self.image_history = defaultdict(
                lambda: [deque(maxlen=self.obs_horizon) for _ in range(self.num_envs)]
            )
            self.act_history = [
                deque(maxlen=self.pred_horizon) for _ in range(self.num_envs)
            ]

        images, state = self._get_images_and_states(obs)
        with torch.no_grad():
            # Shape (num_envs, pred_horizon, ...)
            action = self.agent.get_actions(images, state, scale=scale)

        acs = action.cpu().numpy().astype(np.float32)

        if weighting:
            weighted_acts = []
            for env_id in range(self.num_envs):
                ac = self.get_weighted_action(
                    acs[env_id], env_id, get_full_action=get_full_action
                )
                weighted_acts.append(ac)

            action = np.stack(weighted_acts, axis=0)
        else:
            action = acs if get_full_action else acs[:, 0, :]
        return action

    def get_action_batched(self, obs):
        """
        TODO: fix this for mutliple envs
        This function takes an observation as input,
        and saves the chunks of the predicted actions in a deque.
        The goal is to keep taking that action till the prediction horizon is reached.
        """

        if self.image_history is None or self.act_history is None:
            self.num_envs = obs["state"].shape[0]
            self.image_history = defaultdict(
                lambda: [deque(maxlen=self.obs_horizon) for _ in range(self.num_envs)]
            )
            self.act_history = [
                deque(maxlen=self.pred_horizon) for _ in range(self.num_envs)
            ]

        images, state = self._get_images_and_states(obs)

        if len(self.pred_action_history) == 0:
            with torch.no_grad():
                actions = (
                    self.agent.get_actions(images, state)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                for i in range(self.pred_horizon):
                    self.pred_action_history.append(actions[:, i, :])

        ac = self.pred_action_history.popleft()
        return ac


class DiffusionPolicyAgent:
    def __init__(
        self,
        config,
        diffusion_policy,
        noise_std: float,
    ):
        self.config = config
        self.diffusion_policy = diffusion_policy
        self.noise_std = noise_std
        cprint(
            f"Initialized DiffusionPolicyAgent with noise_std: {noise_std}",
            "yellow",
        )

    def get_action(self, obs):
        base_action = self.diffusion_policy.get_action(obs)

        residual_action = np.zeros_like(base_action)

        if self.noise_std > 0:
            residual_action += np.random.normal(
                0, self.noise_std, size=residual_action.shape
            )

        return {
            "base_action": base_action,
            "residual_action": residual_action,
        }

    def reset(self):
        self.diffusion_policy.reset()


class DiffusionBasePolicy:
    def __init__(
        self,
        preprocessor,
        encoder: nn.Module,
        config,
        device: str,
        state_dim,
        action_dim,
        logger: tools.Logger = None,
        name="DP",
    ):
        super(DiffusionBasePolicy, self).__init__()
        self.config = config
        self.preprocessor = preprocessor
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initialize_model(encoder)
        self.initialize_trainer()
        self.policy = WeigtedActionWrapper(
            agent=self.diffusion_unet,
            config=self.config,
            preprocessor=self.preprocessor,
        )
        self.name = name  # For Logging
        self.logger = logger

    def initialize_model(self, encoder):
        # Initialize gpu_transform
        self.gpu_transform = transforms.get_gpu_transform_by_name("gpu_medium")

        # Instantiate the agent
        diffusion_unet = DiffusionUnetAgent(
            features=encoder,
            shared_mlp=self.config.dp["shared_mlp"],
            odim=self.state_dim,
            n_cams=self.config.dp["num_cams"],
            use_obs=True,
            dropout=0.1,
            train_diffusion_steps=100,
            eval_diffusion_steps=16,
            ac_dim=self.action_dim,
            ac_chunk=self.config.dp["ac_chunk"],
            imgs_per_cam=self.config.dp["img_chunk"],
            share_cam_features=False,
            feat_batch_norm=False,
            noise_net_kwargs={
                # # Original Params
                # "diffusion_step_embed_dim": 256,
                # "down_dims": [256, 512, 1024],
                # Our small config
                # "diffusion_step_embed_dim": 128,
                # "down_dims": [32, 64, 128],
                # DPPO
                "diffusion_step_embed_dim": 16,
                "down_dims": [64, 128, 256],
                "kernel_size": 5,
                "n_groups": 8,
            },
            state_only=self.config.state_only,
        )
        self.diffusion_unet = diffusion_unet

        # Print params in diffusion_unet
        num_params = sum(p.numel() for p in self.diffusion_unet.parameters())
        cprint(f"DiffusionUnet + ResNet18 has {num_params} parameters", "green")

    def initialize_trainer(self):
        builder = optim_builder(
            optimizer_type="AdamW",
            optimizer_kwargs={
                "lr": self.config.dp["lr"],
                "betas": [0.95, 0.999],
                "weight_decay": 1.0e-6,
                "eps": 1.0e-8,
            },
        )
        trainer = BehaviorCloning(
            model=self.diffusion_unet,
            device_id=self.device,
            optim_builder=builder,
        )
        self.trainer = trainer

    def eval_policy(self, eval_envs, expert_val_eps, step=0):
        """
        Evaluate Policy:
        1. Compute BC Loss on expert_val_eps
        2. Evaluate the agent on eval_env and get the success rate
        """
        self.trainer.model.set_eval()
        self.trainer.set_eval()

        # Compute BC loss on expert_val_eps
        val_total_mse = 0.0
        if expert_val_eps is not None:
            val_eps_keys = list(expert_val_eps.keys())  # Convert keys to a list
            for i in range(len(val_eps_keys)):
                data_traj_i = expert_val_eps[val_eps_keys[i]].copy()
                for k, v in data_traj_i.items():
                    data_traj_i[k] = np.stack(v)

                # Add dummy BL dimension
                for k, v in data_traj_i.items():
                    data_traj_i[k] = np.expand_dims(v, axis=1)

                (imgs, obs), gt_actions, _ = self.preprocessor.d4r_preprocess_batch(
                    data_traj_i, training=False
                )

                # Get the actions from the model
                with torch.no_grad():
                    model_actions = self.diffusion_unet.get_actions(imgs, obs)

                # Compute the MSE loss (between normalized actions)
                val_total_mse += F.mse_loss(
                    model_actions, gt_actions, reduction="sum"
                ).item() / len(
                    data_traj_i["state"]
                )  # normalize by batch size
            val_total_mse /= len(val_eps_keys)  # normalize by number of episodes
            print(f"Validation BC Loss: {val_total_mse}")

        # Get the Success Rate
        evaluator = ModelEvaluator(
            agent=self,
            envs=eval_envs,
            default_seed=self.config.seed,
            parent_output_dir=self.config.logdir / f"{self.name}_eval_videos/",
            step=step,
            eval_num_runs=self.config.eval_num_runs,
            visualize=self.config.visualize_eval,
        )
        (
            avg_success_rate,
            avg_total_avg_reward,
            episode_length,
            avg_total_orig_reward,
        ) = evaluator.evaluate_agent()

        self.trainer.set_train()
        self.trainer.model.set_train()

        return val_total_mse, avg_success_rate, avg_total_avg_reward, episode_length

    def reset(self):
        """
        Called when we want to evaluate the policy
        Resets self.policy wrapper
        Copies over EMA weights to model
        """
        self.policy.reset()
        self.trainer.model.set_eval()
        self.trainer.set_eval()

    def set_train(self):
        """
        Called after you are done with evaluation
        Restores the model to train mode
        """
        self.trainer.model.set_train()
        self.trainer.set_train()

    def get_action(self, obs, scale=1.0, weighting=True, get_full_action=False):
        """
        obs: B x T x ... (no stacking)
        Policy takes care of all the preprocessing
        """
        # Make a copy of the obs
        obs = copy.deepcopy(obs)

        # Get the actions from the model
        with torch.no_grad():
            model_actions = self.policy.get_action(
                obs, weighting=weighting, get_full_action=get_full_action, scale=scale
            )

        return model_actions

    def get_action_direct(self, obs):
        """
        obs: BS x BL x ... x stack_dim
        Directly call the DDIM sampler to get actions
        Returns of shape BS x BL x pred_horizon x action_dim
        """
        # Make a copy of the obs
        obs = copy.deepcopy(obs)
        assert (
            obs["state"].ndim == 4
        ), "Expected obs['state'] to have 4 dimensions BS x BL x ob_dim x stack_dim"

        # Get the raw actions from the model
        with torch.no_grad():
            BS, BL = obs["state"].shape[0:2]
            (imgs, state), _, _ = self.preprocessor.d4r_preprocess_batch(
                obs, training=False
            )
            direct_action = self.diffusion_unet.get_actions(imgs, state)
            direct_action = einops.rearrange(
                direct_action, "(bs bl) ... -> bs bl ...", bs=BS, bl=BL
            )

        return direct_action

    def get_action_batched(self, obs):
        """
        obs: B x T x ob_chunck x ...
        Policy takes care of all the preprocessing
        """
        # Make a copy of the obs
        obs = copy.deepcopy(obs)

        # Get the actions from the model
        with torch.no_grad():
            model_actions = self.policy.get_action_batched(obs)

        return model_actions

    def train_base_policy(
        self,
        train_dataset,
        expert_val_eps,
        eval_envs,
        init_step=0,
        train_steps=None,
        log_prefix="",
        run_eval=True,
    ):
        """
        Dataset Generator gives data of shape
        obs: B x T x ob_chunck x ...
        acs: B x T x ac_chunk x ...
        expert_val_eps: Validation dataset
        eval_env: Environment for evaluation
        """
        if train_steps is None:
            train_steps = self.config.dp["train_steps"]

        trainer = self.trainer
        trainer.set_train()

        # Move model and EMA to device
        trainer.model.to(trainer.device_id)
        trainer.model.ema.to(trainer.device_id)
        log_step = init_step
        rollout_snapshot_count = int(self.config.dp.get("rollout_snapshot_count", 10))
        rollout_snapshot_steps = int(
            self.config.dp.get(
                "rollout_snapshot_steps", self.config.num_envs * self.config.time_limit
            )
        )
        rollout_snapshot_noise_std = float(
            self.config.dp.get("rollout_snapshot_noise_std", 0.0)
        )
        if rollout_snapshot_count > 0 and train_steps > 0 and rollout_snapshot_steps > 0:
            n_snapshots = min(rollout_snapshot_count, train_steps)
            rollout_snapshot_indices = set(
                np.linspace(0, train_steps - 1, num=n_snapshots, dtype=int).tolist()
            )
        else:
            rollout_snapshot_indices = set()
        for step in range(train_steps):
            log_step = step + init_step
            batch = next(train_dataset)

            d4r_batch = self.preprocessor.d4r_preprocess_batch(batch)

            # Train for a single step
            trainer.optim.zero_grad()
            loss = trainer.training_step(d4r_batch, step)
            loss.backward()
            trainer.optim.step()

            # EMA update
            trainer.model.ema.step(trainer.model.noise_net.parameters())

            if step % self.config.dp["log_freq"] == 0:
                print(f"DP Step: {step}\tLog Step: {log_step}\tLoss: {loss.item()}")
                if self.logger is not None:
                    self.logger.scalar(
                        name=f"{log_prefix}/train/loss", value=loss.item()
                    )
                    self.logger.write(step=log_step, fps=True)

            if step % self.config.dp["schedule_freq"] == 0:
                trainer.step_schedule()

            if run_eval and (
                (step % self.config.dp["eval_freq"] == 0 and step > 0)
                or step == train_steps - 1
            ):
                # Evaluate on environment
                (
                    eval_policy_loss,
                    sr,
                    reward,
                    episode_len,
                ) = self.eval_policy(
                    eval_envs=eval_envs, expert_val_eps=expert_val_eps, step=log_step
                )

                if self.logger is not None:
                    self.logger.scalar(
                        name=f"{log_prefix}/eval/l2_loss", value=eval_policy_loss
                    )
                    self.logger.scalar(
                        name=f"{log_prefix}/eval/dreamer_success_rate", value=sr
                    )
                    self.logger.scalar(
                        name=f"{log_prefix}/eval/dreamer_reward", value=reward
                    )
                    self.logger.scalar(
                        name=f"{log_prefix}/eval/dreamer_episode_len", value=episode_len
                    )
                    self.logger.write(step=log_step, fps=True, flush=True)

                ckpt_file = self.config.logdir / f"{self.name}_base_policy_latest.pt"
                self.ckpt_file = ckpt_file
                trainer.save_checkpoint(ckpt_file, global_step=log_step)

            if step in rollout_snapshot_indices:
                cprint(
                    f"[DP Rollout] Collecting and saving rollout at step {log_step}",
                    "yellow",
                )
                rollout_save_dir = self.config.logdir / "dp_rollouts" / f"step_{log_step}"
                rollout_eps = {}
                collect_onpolicy_trajs(
                    num_steps=rollout_snapshot_steps,
                    max_traj_len=self.config.time_limit if not self.config.debug else 10,
                    base_policy=DiffusionPolicyAgent(
                        config=self.config,
                        diffusion_policy=self,
                        noise_std=rollout_snapshot_noise_std,
                    ),
                    train_env=eval_envs,
                    pred_horizon=self.config.pred_horizon,
                    obs_horizon=self.config.obs_horizon,
                    train_eps=rollout_eps,
                    state_only=self.config.state_only,
                    save_dir=rollout_save_dir,
                    save_episodes_to_file=True,
                )

        return log_step + 1

    def parameters(self):
        # Return parameters of the Diffusion Model
        return self.trainer.model.parameters()
