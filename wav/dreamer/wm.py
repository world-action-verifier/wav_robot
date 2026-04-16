import numpy as np
import torch
import torch.optim
from einops import rearrange
from termcolor import cprint
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from tqdm import trange

import wav.dreamer.networks as networks
import wav.dreamer.tools as tools


def to_np(x):
    return x.detach().cpu().numpy()


def gradient_penalty(
    learner_sa: torch.Tensor,
    expert_sa: torch.Tensor,
    f: nn.Module,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Calculates the gradient penalty for the given learner and expert state-action tensors.

    Args:
        learner_sa (torch.Tensor): The state-action tensor from the learner.
        expert_sa (torch.Tensor): The state-action tensor from the expert.
        f (nn.Module): The discriminator network.
        device (str, optional): The device to use. Defaults to "cuda".

    Returns:
        torch.Tensor: The gradient penalty.
    """
    batch_size = expert_sa.size()[0]

    alpha = torch.rand(batch_size, 1).to(device)
    alpha = alpha.expand_as(expert_sa)

    interpolated = alpha * expert_sa.data + (1 - alpha) * learner_sa.data

    interpolated = Variable(interpolated, requires_grad=True).to(device)

    f_interpolated = f(interpolated.float()).mode().to(device)

    gradients = torch_grad(
        outputs=f_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(f_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
    )[0].to(device)

    gradients = gradients.view(batch_size, -1)

    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    # 2 * |f'(x_0)|
    return ((gradients_norm - 0.4) ** 2).mean()


class WorldModel(nn.Module):
    def __init__(self, obs_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        self.always_frozen_layers = []

        # Initialize the RSSM + Encoder + Decoder
        self.encoder = networks.MultiEncoder(
            shapes, **config.encoder, state_only=config.state_only
        )
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
            pred_horizon=config.pred_horizon,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder, state_only=config.state_only
        )

        # Initialize the continue predictor
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )

        # Initialize the Discriminator and its optimizer
        net_size = (
            feat_size
            if config.train_dp_mppi_params["discrim_state_only"]
            else feat_size + config.num_actions
        )
        self.reward_discrim = networks.MLP(
            net_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward_Discrim",
        )
        self.discrim_opt = tools.Optimizer(
            "discrim",
            self.reward_discrim.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer discrim_opt has {sum(param.numel() for param in self.reward_discrim.parameters())} variables."
        )

        # Scales for losses, others are scaled by 1.0.
        self._scales = dict(
            cont=config.cont_head["loss_scale"],
        )
        self._last_openloop_video = None
        self._last_rollout_videos = {}

    def parameters(self):
        # Return parameters of self.encoder, self.dynamics, self.heads
        return (
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.heads.parameters())
        )

    def _get_post(
        self,
        data,
    ):
        # Get the post from the world model, dont use during training
        data = self.preprocess(data)

        with torch.no_grad():
            embed = self.encoder(data)
            post, prior = self.dynamics.observe(embed, data["action"], data["is_first"])
            post = {k: v.detach() for k, v in post.items()}

        return post

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self, always_frozen_layers=self.always_frozen_layers):
            aux_metrics = {}
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss

                if bool(getattr(self._config, "add_openloop_img_pred_loss", False)):
                    openloop_loss = self._compute_openloop_image_loss_tensor(
                        data=data,
                        post=post,
                        horizon=int(
                            getattr(self._config, "openloop_img_pred_horizon", 8)
                        ),
                    )
                    if openloop_loss is not None:
                        model_loss = model_loss + (
                            float(
                                getattr(self._config, "openloop_img_pred_loss_weight", 0.0)
                            )
                            * openloop_loss
                        )
                        aux_metrics["openloop_img_pred_train_loss"] = float(
                            openloop_loss.detach().cpu().item()
                        )

                if bool(getattr(self._config, "add_latent_rollout_loss", False)):
                    latent_rollout_loss = self._compute_latent_rollout_loss(
                        data=data,
                        post=post,
                        horizon=int(getattr(self._config, "latent_rollout_horizon", 8)),
                    )
                    if latent_rollout_loss is not None:
                        model_loss = model_loss + (
                            float(
                                getattr(self._config, "latent_rollout_loss_weight", 0.0)
                            )
                            * latent_rollout_loss
                        )
                        aux_metrics["latent_rollout_train_loss"] = float(
                            latent_rollout_loss.detach().cpu().item()
                        )
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())
            metrics.update(aux_metrics)

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        openloop_loss = self._compute_openloop_image_loss(data, post)
        if openloop_loss is not None:
            metrics["openloop_img_pred_loss"] = openloop_loss
        with torch.amp.autocast("cuda", enabled=self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}

        # Discriminator Update
        self._step += 1
        if (
            self._config.train_dp_mppi_params["use_discrim"]
            and self._step % self._config.train_dp_mppi_params["upate_discrim_every"]
            == 0
        ):
            # Get expert and learner data ids
            expert_data_ids = torch.where(torch.all(data["reward"] == 1, dim=1))[0]
            learner_data_ids = torch.where(torch.all(data["reward"] == -1, dim=1))[0]

            # Assert union of expert and learner data ids is equal to the total number of data points
            assert (
                torch.cat([expert_data_ids, learner_data_ids]).shape[0]
                == data["reward"].shape[0]
            )
            feat = self.dynamics.get_feat(post)

            # Get learner and expert state-action pairs
            learner_s = feat[learner_data_ids]
            expert_s = feat[expert_data_ids]

            if self._config.train_dp_mppi_params["discrim_state_only"]:
                learner_sa = learner_s
                expert_sa = expert_s
            else:
                learner_a = data["action"][learner_data_ids]
                learner_sa = torch.cat(
                    [learner_s, learner_a], dim=-1
                )  # BS/2 x BL x feat_dim + act_dim

                expert_a = data["action"][expert_data_ids]
                expert_sa = torch.cat(
                    [expert_s, expert_a], dim=-1
                )  # BS/2 x BL x feat_dim + act_dim

            # Merge BS and BL dimensions
            learner_sa = learner_sa.view(-1, learner_sa.shape[-1])
            expert_sa = expert_sa.view(-1, expert_sa.shape[-1])

            # Calculate discriminator loss
            f_learner = self.reward_discrim(learner_sa.float())
            f_expert = self.reward_discrim(expert_sa.float())
            gp = gradient_penalty(
                learner_sa, expert_sa, self.reward_discrim, device=self._config.device
            )  # Scalar
            pure_loss = torch.mean(f_learner.mode() - f_expert.mode())  # Scalar
            loss = pure_loss + 10 * gp  # Scalar
            metrics["discrim_gp"] = gp.item()
            metrics["discrim_pure_loss"] = pure_loss.item()
            metrics.update(self.discrim_opt(loss, self.reward_discrim.parameters()))

        return post, context, metrics

    def _compute_openloop_image_loss_tensor(self, data, post, horizon):
        image_keys = [k for k in data.keys() if "image" in k]
        if len(image_keys) == 0:
            return None

        time_dim = int(data["action"].shape[1])
        horizon = min(int(horizon), time_dim - 1)
        if horizon <= 0:
            return None

        state = {k: v[:, 0] for k, v in post.items()}
        total_loss = 0.0
        n_terms = 0
        for t in range(horizon):
            action_t = data["action"][:, t]
            state = self.dynamics.img_step(state, action_t)
            feat = self.dynamics.get_feat(state)
            dec = self.heads["decoder"](feat[:, None])
            for key in image_keys:
                pred_t = dec[key].mode().squeeze(1)
                gt_t = data[key][:, t + 1]
                total_loss = total_loss + torch.mean((pred_t - gt_t) ** 2)
                n_terms += 1

        return total_loss / max(n_terms, 1)

    def _compute_latent_rollout_loss(self, data, post, horizon):
        time_dim = int(data["action"].shape[1])
        horizon = min(int(horizon), time_dim - 1)
        if horizon <= 0:
            return None

        state = {k: v[:, 0] for k, v in post.items()}
        gt_feat_seq = self.dynamics.get_feat(post).detach()
        loss = 0.0
        for t in range(horizon):
            action_t = data["action"][:, t]
            state = self.dynamics.img_step(state, action_t)
            pred_feat_t = self.dynamics.get_feat(state)
            gt_feat_t = gt_feat_seq[:, t + 1]
            loss = loss + torch.mean((pred_feat_t - gt_feat_t) ** 2)
        return loss / horizon

    def _compute_openloop_image_loss(self, data, post):
        self._last_openloop_video = None
        self._last_rollout_videos = {}
        if not getattr(self._config, "log_openloop_img_pred", True):
            return None

        image_keys = [k for k in data.keys() if "image" in k]
        if len(image_keys) == 0:
            return None

        max_horizon = int(getattr(self._config, "openloop_img_pred_horizon", 8))
        time_dim = int(data["action"].shape[1])
        horizon = min(max_horizon, time_dim - 1)
        if horizon <= 0:
            return None

        with torch.no_grad():
            state = {k: v[:, 0].detach() for k, v in post.items()}
            pred_imgs = {key: [] for key in image_keys}
            gt_imgs = {key: [] for key in image_keys}

            for t in range(horizon):
                action_t = data["action"][:, t]
                state = self.dynamics.img_step(state, action_t)
                feat = self.dynamics.get_feat(state)
                dec = self.heads["decoder"](feat[:, None])
                for key in image_keys:
                    pred_t = dec[key].mode().squeeze(1)
                    gt_t = data[key][:, t + 1]
                    pred_imgs[key].append(pred_t)
                    gt_imgs[key].append(gt_t)

            total_loss = 0.0
            for key in image_keys:
                pred = torch.stack(pred_imgs[key], dim=1)
                gt = torch.stack(gt_imgs[key], dim=1)
                total_loss += torch.mean((pred - gt) ** 2)
            total_loss = total_loss / len(image_keys)

            # Visualization: first sample, side-by-side GT and prediction.
            primary_key = "agentview_image" if "agentview_image" in image_keys else image_keys[0]
            pred_vis = torch.stack(pred_imgs[primary_key], dim=1)[0]  # H x W x C over time
            gt_vis = torch.stack(gt_imgs[primary_key], dim=1)[0]
            vis = torch.cat([gt_vis, pred_vis], dim=2)  # concat on width
            self._last_openloop_video = vis.unsqueeze(0).detach().cpu().numpy()

            # Additional rollout visualizations.
            # 1) Random 8-step open-loop window.
            # 2) Open-loop rollout from random anchor to end of chunk.
            # 3) Closed-loop decode (posterior features) from same anchor to end.
            max_start = max(time_dim - 2, 0)
            start_t = int(torch.randint(low=0, high=max_start + 1, size=(1,)).item())
            max_steps_from_start = max((time_dim - 1) - start_t, 0)
            random_horizon = min(max_horizon, max_steps_from_start)
            to_end_horizon = max_steps_from_start

            def _openloop_window_video(start, horizon_steps):
                if horizon_steps <= 0:
                    return None
                state_t = {k: v[:, start].detach() for k, v in post.items()}
                pred_seq = []
                gt_seq = []
                for dt in range(horizon_steps):
                    action_t = data["action"][:, start + dt]
                    state_t = self.dynamics.img_step(state_t, action_t)
                    feat_t = self.dynamics.get_feat(state_t)
                    dec_t = self.heads["decoder"](feat_t[:, None])
                    pred_t = dec_t[primary_key].mode().squeeze(1)
                    gt_t = data[primary_key][:, start + dt + 1]
                    pred_seq.append(pred_t)
                    gt_seq.append(gt_t)
                pred_vis_local = torch.stack(pred_seq, dim=1)[0]
                gt_vis_local = torch.stack(gt_seq, dim=1)[0]
                vis_local = torch.cat([gt_vis_local, pred_vis_local], dim=2)
                return vis_local.unsqueeze(0).detach().cpu().numpy()

            def _closedloop_window_video(start, horizon_steps):
                if horizon_steps <= 0:
                    return None
                feat_seq = self.dynamics.get_feat(post).detach()
                pred_seq = []
                gt_seq = []
                for dt in range(horizon_steps):
                    t = start + dt + 1
                    dec_t = self.heads["decoder"](feat_seq[:, t : t + 1])
                    pred_t = dec_t[primary_key].mode().squeeze(1)
                    gt_t = data[primary_key][:, t]
                    pred_seq.append(pred_t)
                    gt_seq.append(gt_t)
                pred_vis_local = torch.stack(pred_seq, dim=1)[0]
                gt_vis_local = torch.stack(gt_seq, dim=1)[0]
                vis_local = torch.cat([gt_vis_local, pred_vis_local], dim=2)
                return vis_local.unsqueeze(0).detach().cpu().numpy()

            video_random8 = _openloop_window_video(start_t, random_horizon)
            if video_random8 is not None:
                self._last_rollout_videos["openloop_random8_vs_gt"] = video_random8

            video_to_end = _openloop_window_video(start_t, to_end_horizon)
            if video_to_end is not None:
                self._last_rollout_videos["openloop_to_end_vs_gt"] = video_to_end

            video_closed_loop = _closedloop_window_video(start_t, to_end_horizon)
            if video_closed_loop is not None:
                self._last_rollout_videos["closedloop_to_end_vs_gt"] = video_closed_loop

        return float(total_loss.detach().cpu().item())

    def get_reward(self, data):
        if self._config.train_dp_mppi_params["use_discrim"]:
            return self.reward_discrim(data)
        # Keep interface stable when discriminator reward is disabled.
        zeros = torch.zeros((data.shape[0], 1), device=data.device, dtype=data.dtype)
        return tools.MSEDist(zeros)

    def evaluate_batch_metrics(self, data):
        data = self.preprocess(data)
        metrics = {}
        with torch.no_grad():
            embed = self.encoder(data)
            post, prior = self.dynamics.observe(embed, data["action"], data["is_first"])
            kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                post,
                prior,
                self._config.kl_free,
                self._config.dyn_scale,
                self._config.rep_scale,
            )

            preds = {}
            feat = self.dynamics.get_feat(post)
            for name, head in self.heads.items():
                pred = head(feat)
                if type(pred) is dict:
                    preds.update(pred)
                else:
                    preds[name] = pred

            losses = {}
            for name, pred in preds.items():
                losses[name] = -pred.log_prob(data[name])

            scaled = {
                key: value * self._scales.get(key, 1.0) for key, value in losses.items()
            }
            model_loss = sum(scaled.values()) + kl_loss

            if bool(getattr(self._config, "add_openloop_img_pred_loss", False)):
                openloop_loss_tensor = self._compute_openloop_image_loss_tensor(
                    data=data,
                    post=post,
                    horizon=int(getattr(self._config, "openloop_img_pred_horizon", 8)),
                )
                if openloop_loss_tensor is not None:
                    model_loss = model_loss + (
                        float(getattr(self._config, "openloop_img_pred_loss_weight", 0.0))
                        * openloop_loss_tensor
                    )
                    metrics["openloop_img_pred_eval_train_loss"] = float(
                        openloop_loss_tensor.detach().cpu().item()
                    )

            if bool(getattr(self._config, "add_latent_rollout_loss", False)):
                latent_rollout_loss = self._compute_latent_rollout_loss(
                    data=data,
                    post=post,
                    horizon=int(getattr(self._config, "latent_rollout_horizon", 8)),
                )
                if latent_rollout_loss is not None:
                    model_loss = model_loss + (
                        float(getattr(self._config, "latent_rollout_loss_weight", 0.0))
                        * latent_rollout_loss
                    )
                    metrics["latent_rollout_eval_loss"] = float(
                        latent_rollout_loss.detach().cpu().item()
                    )

            metrics["model_loss"] = float(torch.mean(model_loss).detach().cpu().item())
            metrics["kl"] = float(torch.mean(kl_value).detach().cpu().item())
            metrics["dyn_loss"] = float(torch.mean(dyn_loss).detach().cpu().item())
            metrics["rep_loss"] = float(torch.mean(rep_loss).detach().cpu().item())
            for name, loss in losses.items():
                metrics[f"{name}_loss"] = float(torch.mean(loss).detach().cpu().item())

            metrics["prior_ent"] = float(
                torch.mean(self.dynamics.get_dist(prior).entropy()).detach().cpu().item()
            )
            metrics["post_ent"] = float(
                torch.mean(self.dynamics.get_dist(post).entropy()).detach().cpu().item()
            )

            openloop_loss = self._compute_openloop_image_loss(data, post)
            if openloop_loss is not None:
                metrics["openloop_img_pred_loss"] = float(openloop_loss)

        return metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()

        # Remove stacking dimension (select last state, and first action)
        if len(obs["state"].shape) == 4:
            if "agentview_image" in obs.keys():
                obs["agentview_image"] = obs["agentview_image"][..., -1]
            if "robot0_eye_in_hand_image" in obs.keys():
                obs["robot0_eye_in_hand_image"] = obs["robot0_eye_in_hand_image"][
                    ..., -1
                ]
            obs["state"] = obs["state"][..., -1]
        if "action" in obs.keys() and len(obs["action"].shape) == 4:
            obs["action"] = obs["action"][..., 0]

        for key in obs.keys():
            # If key contains 'image', normalize the image
            if "image" in key:
                obs[key] = torch.Tensor(obs[key]) / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs
