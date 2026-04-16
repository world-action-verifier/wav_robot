import collections
import os

import torch
from termcolor import cprint

from wav.classes.preprocess import Preprocessor
from wav.classes.resnet_encoder import ResNetEncoder
from wav.classes.rollout_utils import collect_onpolicy_trajs
from wav.dreamer import tools
from wav.dreamer.dreamer_class import Dreamer
from wav.policies.diffusion_base_policy import DiffusionBasePolicy, DiffusionPolicyAgent
from wav.policies.residual_policy import ResidualPolicy
from wav.trainer_utils import count_n_transitions, label_expert_eps
from wav.training.relabel_service import RelabelService
from wav.training.round_loop import RoundLoop
from wav.training.sample_selector_service import SampleSelectorService
from wav.training.trainer_state import TrainerState
from wav.training.wm_trainer import WorldModelTrainer


class SAILORTrainer:
    def __init__(
        self,
        config,
        expert_eps,
        state_dim,
        action_dim,
        train_env,
        eval_envs,
        expert_val_eps,
        train_eps,
        sample_eps,
        wm_eval_eps,
        init_step,
        logger: tools.Logger = None,
    ):
        self.config = config
        self.expert_eps = expert_eps
        self.train_env = train_env
        self.eval_envs = eval_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expert_val_eps = expert_val_eps
        self.train_eps = train_eps
        self.sample_eps = (
            sample_eps if sample_eps is not None else collections.OrderedDict()
        )
        self.wm_eval_eps = (
            wm_eval_eps if wm_eval_eps is not None else collections.OrderedDict()
        )
        self._prev_sample_select_ckpt = None
        self._sample_select_ckpt_history = []
        self._dynamic_sample_selection_enabled = bool(
            getattr(self.config, "wm_only_mode", False)
            and getattr(self.config, "wm_only_sample_source_pool_jsonl", None)
        )
        self.logger = logger
        self.count_n_transitions_fn = count_n_transitions

        self.num_expert_transitions = count_n_transitions(self.expert_eps)
        self.expert_datset = tools.make_dataset(
            self.expert_eps,
            batch_length=self.config.batch_length,
            batch_size=self.config.batch_size,
        )

        self.replay_buffer = (
            train_eps if train_eps is not None else collections.OrderedDict()
        )
        self._step = init_step
        self._env_step = 0
        self.state = TrainerState(step=init_step, env_step=0)
        cprint(f"Initializing SAILORTrainer with init_step: {self._step}")

        self.base_policy = self.init_dp(load_dp_weights=True)
        self.dreamer_class: Dreamer = Dreamer(
            obs_space=self.eval_envs.observation_space,
            base_policy=self.base_policy,
            config=self.config,
            logger=None,
            dataset=None,
            expert_dataset=self.expert_datset,
        ).to(self.config.device)

        self.residual_policy = ResidualPolicy(
            config=self.config,
            dreamer_class=self.dreamer_class,
            expert_eps=self.expert_eps,
            train_eps=self.train_eps,
            train_env=self.train_env,
            eval_envs=self.eval_envs,
            logger=self.logger,
        )

        self.sample_selector = SampleSelectorService()
        self.wm_trainer = WorldModelTrainer()
        self.relabel_service = RelabelService()
        self.round_loop = RoundLoop()

    def init_dp(self, load_dp_weights=False, set_in_dreamer=True):
        encoder = None if self.config.state_only else ResNetEncoder(
            num_cams=self.config.dp["num_cams"]
        )
        preprocessor = Preprocessor(config=self.config)
        base_policy = DiffusionBasePolicy(
            preprocessor,
            encoder=encoder,
            config=self.config,
            device=self.config.device,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            name="DP_Distilled",
            logger=self.logger,
        )

        if load_dp_weights:
            if self.config.dp["pretrained_ckpt"] is None:
                raise ValueError("No pretrained checkpoint provided for DP.")
            full_ckpt_path = os.path.join(
                self.config.scratch_dir, self.config.dp["pretrained_ckpt"]
            )
            base_policy.trainer.load_checkpoint(full_ckpt_path)

        if hasattr(self, "dreamer_class") and set_in_dreamer:
            self.dreamer_class._base_policy = base_policy
            torch.cuda.empty_cache()
        return base_policy

    def train_wm_critic(self, itrs):
        return self.wm_trainer.train_wm_critic(self, itrs)

    def eval_world_model(self, eval_itr):
        return self.wm_trainer.eval_world_model(self, eval_itr)

    def relabel_with_mppi_post(
        self, num_trajs_to_relabel, batch_size=32, select_from_end=True
    ):
        return self.relabel_service.relabel_with_mppi_post(
            self, num_trajs_to_relabel, batch_size=batch_size, select_from_end=select_from_end
        )

    def get_dp_training_buffer(self, num_trajs_to_keep):
        cprint("\nDropping Trajectories from Replay Buffer", "green")
        total_trajectories = len(self.replay_buffer.keys())
        if total_trajectories <= num_trajs_to_keep:
            print(
                f"Total Trajectories: {total_trajectories} <= num_trajs_to_keep: {num_trajs_to_keep}, not subsampling"
            )
            return

        print(
            f"Subsampling {num_trajs_to_keep} trajectories from {total_trajectories} trajectories"
        )
        keys_to_drop = list(self.replay_buffer.keys())[
            : total_trajectories - num_trajs_to_keep
        ]
        for key in keys_to_drop:
            self.replay_buffer.pop(key)

        print(
            f"After Subsampling Trajectories, Total Trajectories in Buffer: {len(self.replay_buffer.keys())}\n"
        )

    def eval_base_policy(self, prefix, round_id, base_policy):
        cprint("\nEvaluating Base Policy", "green")
        eval_policy_loss, sr, reward, episode_len = base_policy.eval_policy(
            eval_envs=self.eval_envs,
            expert_val_eps=self.expert_val_eps,
            step=prefix,
        )
        if self.logger is not None:
            num_buffer_transitions = count_n_transitions(self.replay_buffer)
            self.logger.scalar("eval/dp_l2_loss", eval_policy_loss)
            self.logger.scalar("eval/dp_success_rate", sr)
            self.logger.scalar("eval/dp_reward", reward)
            self.logger.scalar("eval/dp_episode_len", episode_len)
            self.logger.scalar("train/num_buffer_transitions", num_buffer_transitions)
            self.logger.scalar("train/env_step", self._env_step)
            self.logger.scalar("train/n_round", round_id)
            self.logger.write(step=self._step, fps=True)

    def eval_mppi_policy(self, prefix, round_id):
        cprint("\nEvaluating MPPI Policy", "green")
        self.residual_policy.evaluate_agent(step_name=prefix, step=self._step)
        if self.logger is not None:
            num_buffer_transitions = count_n_transitions(self.replay_buffer)
            self.logger.scalar("train/num_buffer_transitions", num_buffer_transitions)
            self.logger.scalar("train/n_round", round_id)
            self.logger.scalar("train/env_step", self._env_step)
            self.logger.write(step=self._step, fps=True)

    def trim_buffer(self, buffer):
        desired_transitions = self.config.num_buffer_transitions
        current_transitions = count_n_transitions(buffer)
        if current_transitions > desired_transitions:
            print(
                f"Trimming buffer from {current_transitions} to {desired_transitions}"
            )
        all_keys = sorted(list(buffer.keys()))
        to_delete = current_transitions - desired_transitions
        deleted_traj_count = 0
        while to_delete > 0 and all_keys:
            key = all_keys.pop(0)
            len_traj = len(buffer[key]["state"])
            del buffer[key]
            to_delete -= len_traj
            deleted_traj_count += 1
        print(
            f"Deleted {deleted_traj_count} trajectories from buffer, final transitions: {count_n_transitions(buffer)}"
        )

    def warm_start_wm(self):
        if not bool(getattr(self.config, "enable_stage_wm_bootstrap", True)):
            cprint(
                "Skipping WM bootstrap stage (enable_stage_wm_bootstrap=False)",
                "yellow",
            )
            self._env_step = count_n_transitions(self.replay_buffer)
            self.state.env_step = self._env_step
            return

        if getattr(self.config, "wm_only_mode", False) and getattr(
            self.config, "wm_only_skip_warmstart_collect", True
        ):
            cprint(
                "Skipping warmstart data collection (wm_only_mode enabled)",
                "yellow",
            )
            self._env_step = count_n_transitions(self.replay_buffer)
            self.state.env_step = self._env_step
            return

        cprint(
            "\n-------------Warmstarting WM + Critic-------------",
            "green",
            attrs=["bold"],
        )
        num_steps_to_collect = int(
            self.config.train_dp_mppi_params["warmstart_percentage_env_steps"]
            * self.config.train_dp_mppi_params["n_env_steps"]
        )
        num_warmstart_itrs = int(
            num_steps_to_collect
            * self.config.train_dp_mppi_params["warmstart_train_ratio"]
        )
        cprint(
            f"Number of steps to collect for warmstart: {num_steps_to_collect}\nNumber of warm start itrs: {num_warmstart_itrs}",
            "green",
        )

        if num_warmstart_itrs <= 0:
            cprint("Skipping warmstart as num_warmstart_itrs <= 0", "yellow")
            return

        cprint("Collecting warm start trajectories...", "yellow")
        collect_onpolicy_trajs(
            num_steps=num_steps_to_collect,
            max_traj_len=self.config.time_limit if not self.config.debug else 10,
            base_policy=DiffusionPolicyAgent(
                config=self.config,
                diffusion_policy=self.base_policy,
                noise_std=self.config.train_dp_mppi_params["data_collect_noise_std"],
            ),
            train_env=self.train_env,
            pred_horizon=self.config.pred_horizon,
            obs_horizon=self.config.obs_horizon,
            train_eps=self.replay_buffer,
            save_dir=None,
            state_only=self.config.state_only,
        )

        label_expert_eps(
            expert_eps=self.expert_eps,
            dreamer_class=self.dreamer_class,
        )
        self.train_wm_critic(itrs=num_warmstart_itrs)
        self._env_step += count_n_transitions(self.replay_buffer)
        self.state.env_step = self._env_step

    def collect_trajs(self):
        if not bool(getattr(self.config, "enable_stage_active_collection", True)):
            cprint(
                "Skipping active collection stage (enable_stage_active_collection=False)",
                "yellow",
            )
            return int(self.config.train_dp_mppi_params["min_env_steps_per_round"])

        init_transitions = count_n_transitions(self.replay_buffer)
        n_steps_collected = self.residual_policy.collect_residual_onpolicy_trajs(
            num_steps=self.config.train_dp_mppi_params["min_env_steps_per_round"],
            buffer=self.replay_buffer,
        )
        final_transitions = count_n_transitions(self.replay_buffer)
        self._env_step += final_transitions - init_transitions
        self.state.env_step = self._env_step
        return n_steps_collected

    def train_dp_with_mppi(self):
        return self.round_loop.run(self)
