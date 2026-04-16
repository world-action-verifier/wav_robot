import numpy as np
from termcolor import cprint

import wav.dreamer.tools as tools
from wav.classes.evaluator import ModelEvaluator
from wav.classes.rollout_utils import collect_onpolicy_trajs
from wav.dreamer.dreamer_class import Dreamer


class ResidualAgent:
    """
    Wraps a dreamer class as a residual agent, that can given the next action and keeps track of recurrent state
    """

    def __init__(
        self,
        agent: Dreamer,
        add_acts=False,
    ):
        self.agent = agent
        self.add_acts = add_acts

        # These stay for the entire instance for a class (therefore per evaluation)
        self.base_bin_edges = np.linspace(-1, 1, 21)  # 20 bins
        self.residual_bin_edges = np.linspace(-0.5, 0.5, 21)  # 20 bins
        self.residual_counts = np.zeros(20)
        self.base_counts = np.zeros(20)

        cprint(
            "Initialized ResidualAgent with Base Policy + MPPI correction",
            "yellow",
        )

        self.reset()

    def get_action(self, obs, step=None):
        """
        obs: n_envs x ...
        """
        policy_output_dict, self.state = self.agent.get_action(
            obs_orig=obs,
            state=self.state,
        )

        # Detach and convert to numpy
        detached_action_dict = {
            k: v.detach().cpu().numpy() for k, v in policy_output_dict.items()
        }

        if "reward_output" in detached_action_dict.keys():
            self.reward_output = detached_action_dict["reward_output"]

        # Update freq counters
        self.update_counts(detached_action_dict)

        if self.add_acts:
            detached_action_dict = np.clip(
                detached_action_dict["base_action"]
                + detached_action_dict["residual_action"],
                -1,
                1,
            )

        return detached_action_dict

    def reset(self):
        # For dreamer residual actor
        self.state = None
        self.agent.reset()

    def update_counts(self, action_dict):
        # Update residual counts
        residual_hist = np.histogram(
            action_dict["residual_action"], bins=self.residual_bin_edges
        )[0]
        self.residual_counts += residual_hist

        # Update base counts
        base_hist = np.histogram(action_dict["base_action"], bins=self.base_bin_edges)[
            0
        ]
        self.base_counts += base_hist

    def get_action_range(self):
        # return dict of np.histogram
        return {
            "residual_counts": (self.residual_counts, self.residual_bin_edges),
            "base_counts": (self.base_counts, self.base_bin_edges),
        }


class ResidualPolicy:
    def __init__(
        self,
        config,
        dreamer_class: Dreamer,
        expert_eps,
        train_eps,
        train_env,
        eval_envs,
        logger: tools.Logger = None,
    ):
        self.config = config
        self.dreamer_class = dreamer_class
        self.expert_eps = expert_eps
        self.train_eps = train_eps
        self.train_env = train_env
        self.eval_envs = eval_envs
        self.logger = logger
        self._step = 0

    def evaluate_agent(self, step_name, step=None):
        residual_agent = ResidualAgent(
            agent=self.dreamer_class,
            add_acts=True,
        )
        # Get the Success Rate
        evaluator = ModelEvaluator(
            agent=residual_agent,
            envs=self.eval_envs,
            default_seed=self.config.seed,
            parent_output_dir=self.config.logdir / "SAILOR_eval_videos/",
            step=step_name,
            eval_num_runs=self.config.eval_num_runs,
            visualize=self.config.visualize_eval,
        )
        (
            avg_success_rate,
            avg_total_avg_reward,
            episode_length,
            avg_total_orig_reward,
        ) = evaluator.evaluate_agent()
        eval_video_dir = self.config.logdir / "SAILOR_eval_videos" / f"step_{step_name}"

        hist_data = residual_agent.get_action_range()
        self._print_action_histograms(hist_data)

        if self.logger is not None:
            self.logger.scalar(f"residual_eval/mppi_success_rate", avg_success_rate)
            self.logger.scalar(f"residual_eval/mppi_avg_reward", avg_total_avg_reward)
            self.logger.scalar(f"residual_eval/mppi_eps_length", episode_length)
            self._log_eval_videos(eval_video_dir)
            self._log_action_histograms(hist_data)

            self.logger.write(step=step, fps=True)

        return avg_success_rate, episode_length

    def collect_residual_onpolicy_trajs(self, num_steps, buffer):
        """
        Collect on-policy trajectories with base policy and store in buffer.
        """
        n_step_collected = collect_onpolicy_trajs(
            num_steps=num_steps,
            max_traj_len=self.config.time_limit if not self.config.debug else 10,
            base_policy=ResidualAgent(
                self.dreamer_class,
            ),
            train_env=self.train_env,
            pred_horizon=self.config.pred_horizon,
            obs_horizon=self.config.obs_horizon,
            train_eps=buffer,
            save_dir=self.config.logdir / "train_eps",
            state_only=self.config.state_only,
        )
        return n_step_collected

    def _print_action_histograms(self, hist_data):
        for key, np_hist in hist_data.items():
            print(f"[Histogram], Key: {key}, Counts: {np_hist[0]}")

    def _log_eval_videos(self, eval_video_dir):
        if not (self.config.visualize_eval and eval_video_dir.exists()):
            return
        video_paths = sorted(eval_video_dir.glob("*.mp4"))
        max_videos = int(getattr(self.config, "wandb_eval_video_limit", 2))
        for idx, video_path in enumerate(video_paths[:max_videos]):
            self.logger.video_file(
                name=f"residual_eval/video_{idx}",
                path=video_path,
                fps=30,
            )

    def _log_action_histograms(self, hist_data):
        for key, np_hist in hist_data.items():
            self.logger.histogram(key, np_hist=np_hist)
