import numpy as np
import torch
import torch.nn as nn
from termcolor import cprint

from wav.classes.rollout_utils import select_latest_obs
from wav.dreamer import tools
from wav.dreamer.imag_behavior import ImagBehavior
from wav.dreamer.wm import WorldModel

to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(
        self,
        obs_space,
        base_policy,
        config,
        logger,
        dataset,
        expert_dataset=None,
    ):
        super(Dreamer, self).__init__()

        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)

        self._metrics = {}
        self._step = logger.step // config.action_repeat if logger is not None else 0
        self._dataset = dataset
        self._expert_dataset = expert_dataset
        self._base_policy = base_policy

        # Create Models
        self._wm = WorldModel(
            obs_space=obs_space,
            step=self._step,
            config=config,
        )
        self._task_behavior = ImagBehavior(config, self._wm, base_policy=base_policy)

        # Compile if necessary
        if config.compile:
            cprint("Compiling models", "green")
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)

    def get_action(self, obs_orig, state):
        """
        Called during evaluation
        """
        if state is None:
            latent = action = None
        else:
            latent, action = state

        # Create obs_dreamer = BS x BL x ...
        obs_dreamer = {k: np.expand_dims(v, axis=1) for k, v in obs_orig.items()}

        obs_dreamer = self._wm.preprocess(obs_dreamer)
        embed = self._wm.encoder(obs_dreamer)  # BS x BL x (1024 + encoding_dim)
        embed = embed.squeeze(1)  # Remove BL dim

        # Add action
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs_dreamer["is_first"]
        )
        feat = self._wm.dynamics.get_feat(latent)

        action_dict = self._task_behavior.get_action(
            obs=obs_orig, feat=feat, latent=latent
        )

        latent = {k: v.detach() for k, v in latent.items()}
        action_dict = {k: v.detach() for k, v in action_dict.items()}
        action_sum = self._task_behavior.get_action_sum(
            action_dict["base_action"], action_dict["residual_action"]
        )
        state = (latent, action_sum)
        return action_dict, state

    def reset(self):
        self._task_behavior.reset()

    def _train(self, data, training_step):
        # Obs shape BS x BL x ... x stack_dim
        metrics = {}
        data_wm = select_latest_obs(data)  # Select only last obs and remove stacking
        post, context, mets = self._wm._train(data_wm)
        metrics.update(mets)
        if getattr(self._config, "wm_only_mode", False):
            return metrics
        start = {"obs_orig": data, "post": post}

        if self._config.train_dp_mppi_params["use_discrim"]:
            if self._config.train_dp_mppi_params["discrim_state_only"]:
                reward = lambda f, s, a: self._wm.get_reward(
                    self._wm.dynamics.get_feat(s)
                ).mode()
            else:
                reward = lambda f, s, a: self._wm.get_reward(
                    torch.cat([f, a], dim=-1)
                ).mode()
        else:
            # Disable learned-reward path entirely; keep value training numerically stable.
            reward = lambda f, s, a: torch.zeros(
                (*f.shape[:-1], 1), device=f.device, dtype=f.dtype
            )

        metrics.update(
            self._task_behavior._train(
                start,
                reward,
                training_step,
            )[-1]
        )
        return metrics

    # Public stable API used by trainer/services.
    def train_step(self, data, training_step):
        return self._train(data, training_step)

    def evaluate_wm_batch_metrics(self, batch):
        return self._wm.evaluate_batch_metrics(batch)

    def preprocess_for_wm(self, obs_batch):
        return self._wm.preprocess(obs_batch)

    def encode_wm_obs(self, obs_batch):
        return self._wm.encoder(obs_batch)

    def obs_step_wm(self, latent, action, embed, is_first):
        return self._wm.dynamics.obs_step(latent, action, embed, is_first)

    def get_mppi_actions(self, latent, base_action):
        return self._task_behavior.mppi_actions(latent=latent, base_action=base_action)

    def get_last_openloop_video(self):
        return getattr(self._wm, "_last_openloop_video", None)

    def get_last_rollout_videos(self):
        videos = getattr(self._wm, "_last_rollout_videos", None)
        return videos if videos is not None else {}

    def compute_base_action_direct(self, data_traj_slice):
        return self._task_behavior.base_policy.get_action_direct(data_traj_slice)

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        cprint("Saved dreamer checkpoint to {}".format(path), "green")

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        cprint("Loaded dreamer checkpoint from {}".format(path), "green")
