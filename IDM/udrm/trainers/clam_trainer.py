import collections
import contextlib
import copy
import os
from pathlib import Path

import einops
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
import imageio
import matplotlib.pyplot as plt
from matplotlib import colormaps as mpl_cmaps
from matplotlib import colors as mcolors
import colorsys
from rich.pretty import pretty_repr

from udrm.models.mlp_policy import MLPPolicy
from udrm.models.utils.clam_utils import get_clam_cls, get_la_dim
from udrm.trainers.offline_trainer import OfflineTrainer
from udrm.utils.data_utils import Batch
from udrm.utils.dataloader import get_dataloader
from udrm.utils.general_utils import to_device, to_numpy
from udrm.utils.logger import log
from typing import Sequence, Optional

def get_labelled_dataloader(cfg):
    if cfg.data.labelled_data_type == "trajectory":
        log("loading expert data for training action decoder", "blue")
        log(f"num trajs to load: {cfg.num_labelled_trajs}")
        cfg.data.num_trajs = cfg.num_labelled_trajs
        cfg.data.num_examples = -1
        labelled_dataloader, *_ = get_dataloader(
            cfg=cfg,
            dataset_names=cfg.env.action_labelled_dataset,
            dataset_split=cfg.env.action_labelled_dataset_split,
            shuffle=True,
        )
    elif cfg.data.labelled_data_type == "transition":
        log("loading random transition data for training action decoder", "blue")
        cfg.data.num_trajs = -1
        cfg.data.num_examples = cfg.num_labelled_trajs * cfg.env.max_episode_steps
        log(f"num examples to load: {cfg.data.num_examples}")
        labelled_dataloader, *_ = get_dataloader(
            cfg=cfg,
            dataset_names=cfg.env.action_labelled_dataset,
            dataset_split=cfg.env.action_labelled_dataset_split,
            shuffle=True,
        )
    else:
        raise ValueError(f"Unknown labelled data type {cfg.data.labelled_data_type}")

    labelled_dataloader = tf.data.Dataset.sample_from_datasets(
        list(labelled_dataloader.values())
    )
    return labelled_dataloader


class CLAMTrainer(OfflineTrainer):
    def __init__(self, cfg):
        self.use_transformer = "transformer" in cfg.model.idm.net.name
        self.use_soda = "soda" in cfg.model.idm.net.name

        log(f"Using transformer: {self.use_transformer}")
        log(f"Using SODA: {self.use_soda}")

        super().__init__(cfg)
        self.loss_fn = nn.MSELoss(reduction="none")

        if cfg.joint_action_decoder_training:
            log("loading labelled data for training action decoder")
            cfg_cpy = copy.deepcopy(cfg)
            self.labelled_dataloader = get_labelled_dataloader(cfg_cpy)

            self.labelled_dataloader_train = (
                self.labelled_dataloader.repeat().as_numpy_iterator()
            )

        # ---------------------------------------------------------------------
        # [Fix] Video Evaluation Dataset Setup
        # ---------------------------------------------------------------------
        # 1) Eval group path
        eval_ds_path = getattr(self.cfg.env, "eval_dataset_name", None)
        if eval_ds_path in (None, "null"):
            eval_ds_path = self.cfg.env.dataset_name

        # 2) Eval dataset list (actual names)
        eval_ds_list = getattr(self.cfg.env, "eval_datasets", None)
        if eval_ds_list is None or len(eval_ds_list) == 0:
            eval_ds_list = getattr(self.cfg.env, "datasets", [])

        # 3) Select the first actually available eval dataset for video
        self.video_ds_name = eval_ds_list[0]

        log(f"[VideoEval] Path: '{eval_ds_path}', Target Dataset: '{self.video_ds_name}'", "blue")

        cfg2 = copy.deepcopy(self.cfg)
        cfg2.data.shuffle = False
        cfg2.data.batch_size = 1
        cfg2.data.num_trajs = 10 # disable trajectory-count initialization constraint
        cfg2.data.num_examples = -1
        cfg2.data.pad_dataset = True

        cfg2.env.dataset_name = eval_ds_path
        cfg2.env.datasets = [self.video_ds_name]
        print("this is cfg2 for visualization.")
        ds_dict, *_ = get_dataloader(
            cfg=cfg2,
            dataset_names=[self.video_ds_name],
            dataset_split=[1],
            shuffle=False,
        )
            
        if not ds_dict:
            log(f"[VideoEval] Failed to load chunk '{self.video_ds_name}'. Dictionary is empty.", "red")
            self.video_seq_ds = None
        else:
            self.video_seq_ds = tf.data.Dataset.sample_from_datasets(
                list(ds_dict.values())
            )
            log(f"[VideoEval] Successfully loaded video dataset.", "green")

    def setup_action_decoder(self):
        log("---------------------- Initializing Action Decoder ----------------------")
        la_dim = get_la_dim(self.cfg)
        action_decoder = MLPPolicy(
            cfg=self.cfg.model.action_decoder,
            input_dim=la_dim,
            output_dim=self.cfg.env.action_dim,
        )
        action_decoder = action_decoder.to(self.device)
        log(f"action decoder: {action_decoder}")
        return action_decoder

    def train_idm_action_decoder_only(self):
        log("IDM + action decoder training (freeze FDM)", "blue")
        self.mse_loss_fn = nn.MSELoss(reduction="none")

        self.model.train()
        if hasattr(self.model, "fdm"):
            self.model.fdm.eval()
        # self.action_decoder.train()

        # self.action_decoder.action_activation=null


        # Optimizer for IDM
        opt_cls = getattr(torch.optim, self.cfg.optimizer.name)
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfg.lr_scheduler.name)
        self.idm_optimizer = opt_cls(
            self.model.idm.parameters(), **self.cfg.optimizer.params
        )
        self.idm_scheduler = scheduler_cls(
            self.idm_optimizer, **self.cfg.lr_scheduler.params
        )

        train_iter = self.train_dataloader.repeat().as_numpy_iterator()
        eval_iter = self.eval_dataloader.repeat().as_numpy_iterator()
        num_eval_batches = int(getattr(self, "num_eval_batches", 0))

        num_steps = int(getattr(self.cfg, "num_updates", 50000))
        log_every = int(getattr(self.cfg, "post_action_decoder_log_every", 100))
        save_every = int(getattr(self.cfg, "save_every", 0))
        eval_batches = int(getattr(self.cfg, "post_action_decoder_eval_batches", -1))
        if eval_batches <= 0:
            eval_batches = num_eval_batches

        for step in range(num_steps):
            self.train_step = step
            batch_np = next(train_iter)
            batch_np = to_device(batch_np, self.device)
            batch = Batch(**batch_np)

            self.idm_optimizer.zero_grad()
            # self.action_decoder_optimizer.zero_grad()

            autocast_ctx = (
                torch.amp.autocast(device_type=self.device.type)
                if self.device.type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                loss, metrics = self.compute_action_decoder_loss(
                    batch, train=True, freeze_model=False, step = step
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.idm.parameters(), max_norm=self.cfg.clip_grad_norm
            )

            self.idm_optimizer.step()

            if hasattr(self, "idm_scheduler"):
                self.idm_scheduler.step()
            # if hasattr(self, "action_decoder_scheduler"):
            #     self.action_decoder_scheduler.step()


            if step % log_every == 0:
                log_metrics = dict(metrics)
                log_metrics["step"] = step

                if step % 3000 == 0:
                    eval_losses = []
                    for _ in range(eval_batches):
                        eval_np = next(eval_iter)
                        eval_np = to_device(eval_np, self.device)
                        eval_batch = Batch(**eval_np)
                        with torch.no_grad():
                            eval_loss, _ = self.compute_action_decoder_loss(
                                eval_batch, train=False, freeze_model=False
                            )
                        eval_losses.append(eval_loss.item())

                    if eval_losses:
                        log_metrics["eval_action_decoder_loss"] = float(
                            sum(eval_losses) / len(eval_losses)
                        )

                self.log_to_wandb(
                    log_metrics, prefix="idm_action_decoder/", step=step + 1
                )
                log(f"idm_action_decoder step: {step}, {metrics}")

            if save_every > 0 and ((step + 1) % save_every == 0):
                ckpt_metrics = {"action_decoder_loss": float(loss.detach().item())}
                self.save_model(
                    ckpt_dict=self.save_dict,
                    metrics=ckpt_metrics,
                    iter=step + 1,
                )

        # Final safeguard save so downstream WM selection always has latest.pkl.
        self.save_model(
            ckpt_dict=self.save_dict,
            metrics={"action_decoder_loss": float(loss.detach().item())},
            iter=num_steps,
        )

    # Initialize model; when load_from_ckpt is enabled, load from ckpt_file and ckpt_step
    def setup_model(self):
        clam_cls = get_clam_cls(self.cfg.name)
        la_dim = get_la_dim(self.cfg)
        model = clam_cls(self.cfg.model, input_dim=self.obs_shape, la_dim=la_dim)

        if self.cfg.load_from_ckpt:
            cfg, ckpt = model.load_from_ckpt(
                self.cfg.ckpt_file, ckpt_step=self.cfg.ckpt_step, key="model"
            )

        if self.cfg.joint_action_decoder_training or getattr(
            self.cfg, "post_action_decoder_training", False
        ):
            self.action_decoder = self.setup_action_decoder()
            self.action_decoder_loss_fn = nn.MSELoss(reduction="none")
            if self.cfg.load_from_ckpt:
                cfg, ckpt = self.action_decoder.load_from_ckpt(
                    self.cfg.ckpt_file, ckpt_step=self.cfg.ckpt_step, key="action_decoder",
                )
        return model

    def setup_optimizer_and_scheduler(self):
        opt_cls = getattr(torch.optim, self.cfg.optimizer.name)
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfg.lr_scheduler.name)
        clam_optimizer, clam_scheduler = super().setup_optimizer_and_scheduler()

        if self.cfg.joint_action_decoder_training:
            self.action_decoder_optimizer = opt_cls(
                self.action_decoder.parameters(), **self.cfg.decoder_optimizer.params
            )
            self.action_decoder_scheduler = scheduler_cls(
                self.action_decoder_optimizer, **self.cfg.lr_scheduler.params
            )
        return clam_optimizer, clam_scheduler

    def compute_ad_loss(self, model, decoder, batch, loss_fn):
        """Simple latent-action → action regression."""
        obs = batch.observations
        la = model.idm(obs, timesteps=batch.timestep, states=None)  # (B, T, D_la)
        la = la.la
        pred = decoder(la[:, 1:])  # align with action targets (drop t0)
        target = batch.actions[:, :-1]
        loss = loss_fn(pred, target).mean()
        return loss, {"action_decoder_loss": loss.item()}

    def compute_probe_loss(self, model, decoder, batch, loss_fn, order: int = 1, kind=None):
        """
        order = 1: s_t,a_t -> s_{t+1}
        order = 2: s_t,a_t -> s_{t+2}
        order = 3: s_t,a_t -> s_{t+3}
        """
        if order < 1:
            raise ValueError(f"order must be >=1, got {order}")

        obs = batch.observations
        la = model.idm(obs, timesteps=batch.timestep, states=None)  # (B, N, D_la)
        la = la.la
        B, N, _ = la.shape

        # Alignment: la starts at t=1, so at most N-order samples are usable
        max_t = N - order
        if max_t <= 0:
            raise ValueError(f"order {order} too large for sequence length {N}")

        la_win = la[:, 1:1 + max_t]              # (B, max_t, D_la)
        pre_state = batch.states[:, 0:max_t]      # s_t
        next_state = batch.states[:, order:order + max_t]  # s_{t+order}

        # Flatten to (B*max_t, ·)
        la_flat = la_win.reshape(B * max_t, -1)
        pre_flat = pre_state.reshape(B * max_t, -1)
        next_flat = next_state.reshape(B * max_t, -1)

        if "cheat" in kind:
            probe_input = pre_flat
        else:    
            probe_input = torch.cat([la_flat, pre_flat], dim=1)

        pred = decoder(probe_input)
        loss = loss_fn(pred, next_flat).mean()
        return loss, {"probe_loss": loss.item(), "order": order}


    def train_posthoc_evaluator(
        self,
        kind,
        model,
        train_loader,
        eval_loader,
        device,
        decoder_factory,
        optimizer_cls=torch.optim.AdamW,
        optimizer_params=None,
        scheduler_cls=None,
        scheduler_params=None,
        num_steps=10000,
        log_every=500,
        log_fn=print,
        wandb_log_fn=None,
        probe_order: int | None = None,
        start_step: int = 0,
    ):
        optimizer_params = optimizer_params or {"lr": 1e-3, "eps": 1e-5, "weight_decay": 0.01}
        loss_fn = nn.MSELoss(reduction="none")

        # ---- bootstrap a batch for shapes ----
        train_iter = train_loader.repeat().as_numpy_iterator()
        first_np = next(train_iter)
        first_batch = Batch(**to_device(first_np, device))
        with torch.no_grad():
            la_sample = model.idm(first_batch.observations, timesteps=first_batch.timestep, states=None)
            la_sample = la_sample.la
        la_dim = la_sample.shape[-1]
        state_dim = first_batch.states.shape[-1] if hasattr(first_batch, "states") else 0
        action_dim = first_batch.actions.shape[-1]

        # new decoder per run (factory can use inferred dims)
        decoder = decoder_factory(kind, la_dim, state_dim, action_dim).to(device)
        decoder.train()

        # freeze CLAM
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        opt = optimizer_cls(decoder.parameters(), **optimizer_params)
        sched = scheduler_cls(opt, **scheduler_params) if scheduler_cls else None

        eval_iter = eval_loader.repeat().as_numpy_iterator()

        def step_loss(batch):
            if "probe" in kind or "cheat" in kind:
                order = probe_order
                if order is None:
                    order_token = str(kind).split("order")[-1]
                    order = int(order_token) if order_token.isdigit() else 1
                return self.compute_probe_loss(model, decoder, batch, loss_fn, order=order, kind=kind)
            return self.compute_ad_loss(model, decoder, batch, loss_fn)

        history = []
        for step in range(num_steps):
            if step == 0:
                batch = first_batch
            else:
                batch_np = next(train_iter)
                batch = Batch(**to_device(batch_np, device))

            opt.zero_grad()
            loss, train_metrics = step_loss(batch)
            loss.backward()
            opt.step()
            if sched:
                sched.step()

            if step % log_every == 0:
                # eval
                eval_batch = Batch(**to_device(next(eval_iter), device))
                eval_loss, _ = step_loss(eval_batch)
                log_step = start_step + step
                metrics = {
                    "step": log_step,
                    "train_loss": loss.item(),
                    "eval_loss": eval_loss.item(),
                    "kind": kind,
                }
                metrics.update(train_metrics)
                history.append(metrics)
                log_fn(f"[posthoc-{kind}] step={log_step} train={loss.item():.4f} eval={eval_loss.item():.4f}")
                if wandb_log_fn:
                    wandb_log_fn(metrics, prefix=f"posthoc/{kind}/", step=None)
        return decoder, history

    def compute_loss(self, batch, train: bool = True):
        k_step = self.cfg.model.fdm.k_step_pred
        # print(f"[Compute_Loss] Computing loss with k_step = {k_step}")
        # print("[Compute_Loss] Context Length:", self.cfg.model.context_len)
        if k_step > 1:
            def splice(x, start, end):
                return x[:, start:end] if x is not None else x
            total_loss = 0.0
            obs_recons = []
            metrics = collections.defaultdict(float) # dictionary with default value 0.0
            for step in range(k_step):
                end = self.cfg.model.context_len + 1 + step
                batch_splice = dict(map(lambda kv: (kv[0], splice(kv[1], start=step, end=end)), batch.__dict__.items()))
                batch_splice = Batch(**batch_splice)
                # print(f"[Compute_Loss] Step {step+1}/{k_step}, Batch Splice Observations Shape: {batch_splice.observations.shape}")
                if len(obs_recons) > 0:
                    new_observations = batch_splice.observations.clone()
                    # torch.Size([32, 3, 3, 128, 128])
                    # print("new_observations: ", new_observations.shape)
                    # print("obs_recons: ", np.shape(obs_recons))
                    
                    new_observations[:, :-1] = obs_recons[-1] # in current window, use the frame that predicts the last frame as input
                    batch_splice.observations = new_observations
                step_metrics, obs_recon, step_loss = self.compute_step_loss(batch_splice, train=train)
                # print("obs_recon: ", obs_recon.shape)

                for k, v in step_metrics.items():
                    metrics[k] += v
                obs_recons.append(obs_recon)
                total_loss += step_loss
            obs_recons = torch.stack(obs_recons, dim=-1)
            metrics = {k: v / k_step for k, v in metrics.items()}
        else:
            # Default k_step_pred=1; STM typically enters this branch
            metrics, obs_recon, total_loss = self.compute_step_loss(batch, train=train)
        metrics["total_loss"] = float(total_loss.detach().item())
        return metrics, total_loss

    def compute_action_decoder_loss(
        self,
        batch,
        train: bool = True,
        freeze_model: bool = False,
        step=None,
    ):
        # Keep backward compatibility with call sites that pass
        # freeze_model/step during post action-decoder training.
        del train, freeze_model, step
        # (기존 코드와 동일하여 생략 가능하지만 전체 코드를 위해 포함)
        obs = batch.observations
        k_step = self.cfg.model.fdm.k_step_pred
        if k_step > 1:
            action_preds = []
            for step in range(k_step):
                end = self.cfg.model.context_len + 1 + step
                obs_splice = obs[:, step:end]
                if self.use_transformer:
                    clam_output = self.model(obs_splice, timesteps=batch.timestep)
                else:
                    clam_output = self.model(obs_splice)
                la = clam_output.la
                if self.cfg.model.distributional_la:
                    la = self.model.reparameterize(la)
                action_pred = self.action_decoder(la)
                action_preds.append(action_pred)
            action_pred = torch.stack(action_preds, dim=1)
            if self.use_transformer:
                action_pred = action_pred[:, 0, :-1]
                gt_actions = batch.actions[:, :-1]
            else:
                gt_actions = batch.actions[:, -2 : (-2 + k_step)]
        else:
            if self.use_transformer:
                clam_output = self.model(
                    obs, timesteps=batch.timestep, states=batch.states
                )
                la = clam_output.la[:, 1:]
                action_pred = self.action_decoder(la)
                gt_actions = batch.actions[:, :-1]
            else:
                clam_output = self.model(obs)
                la = clam_output.la
                action_pred = self.action_decoder(la)
                gt_actions = batch.actions[:, self.cfg.model.context_len :].squeeze()

        assert action_pred.shape == gt_actions.shape
        gt_actions = to_device(gt_actions, self.device)
        action_decoder_loss = self.action_decoder_loss_fn(action_pred, gt_actions).mean()
        return action_decoder_loss, {"action_decoder_loss": action_decoder_loss.item()}


    def compute_step_loss(self, batch, train: bool = True):
        if self.use_transformer:
            clam_output = self.model(
                batch.observations, timesteps=batch.timestep, states=batch.states
            )
            # print("[Compute_Step_Loss] Transformer mode: obs shape:", batch.observations.shape)
            # [128, 5, 3, 128, 128]
            obs_gt = batch.observations[:, 1:]
            cheat_pred = batch.observations[:, :-1]
            cheat_loss = self.loss_fn(cheat_pred, obs_gt).mean()
        else:
            clam_output = self.model(batch.observations)
            if not self.cfg.model.fdm.predict_target_embedding:
                obs_gt = batch.observations[:, self.cfg.model.context_len :].squeeze()
            cheat_pred = batch.observations[:, self.cfg.model.context_len - 1 : -1].squeeze()
            cheat_loss = self.loss_fn(cheat_pred, obs_gt).mean()

        # Used for simulator debugging when directly predicting residuals
        if self.cfg.use_res_mseloss:
            obs_recon_res = clam_output.reconstructed_obs
            B, T, _, _ , _ = obs_recon_res.size()
            if self.cfg.model.FDM_TYPE == "SpaceTimeFDM":
                obs_recon = obs_recon_res + cheat_pred
            else:
                obs_recon = obs_recon_res + batch.observations[:, :1].expand(B, T, -1, -1, -1)
        else:
            obs_recon = clam_output.reconstructed_obs
        # [128, 4, 3, 128, 128]
        # print("[Compute_Step_Loss] obs_recon shape:", obs_recon.shape)
        la = clam_output.la
        assert obs_recon.shape == obs_gt.shape, f"{obs_recon.shape}, {obs_gt.shape}"

        recon_loss = self.loss_fn(obs_recon, obs_gt).mean()
        total_loss = self.cfg.model.recon_loss_weight * recon_loss

        if self.cfg.model.distributional_la and self.cfg.model.kl_loss_weight:
            la_mean = la[:, : self.cfg.model.la_dim]
            la_logvar = la[:, self.cfg.model.la_dim :]
            posterior = torch.distributions.Normal(la_mean, torch.exp(0.5 * la_logvar))
            prior = torch.distributions.Normal(0, 1)
            kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).mean()
            total_loss += self.cfg.model.kl_loss_weight * kl_loss
        else:
            kl_loss = torch.tensor(0.0)

        # Optional soft-L1 sparsity regularization on latent action.
        # Uses Charbonnier penalty: sqrt(x^2 + eps^2), smoother than plain |x|.
        la_l1_weight = float(getattr(self.cfg.model, "la_l1_weight", 0.0))
        la_l1_eps = float(getattr(self.cfg.model, "la_l1_eps", 1e-6))
        la_l1_loss = torch.tensor(0.0, device=total_loss.device)
        if la_l1_weight > 0:
            la_target = getattr(clam_output.idm_output, "la", la)
            if la_target.ndim >= 2 and la_target.shape[1] > 1:
                la_target = la_target[:, 1:]
            la_l1_loss = torch.sqrt(la_target.pow(2) + la_l1_eps**2).mean()
            total_loss += la_l1_weight * la_l1_loss

        metrics = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "cheat_loss": cheat_loss.item(),
            "la_l1_loss": la_l1_loss.item(),
        }

        # If AdaLN is used, compute adjacent-frame similarity loss
        fdm_beta = getattr(clam_output.idm_output, "fdm_beta", None)
        if fdm_beta is not None:
            with torch.no_grad():
                beta = fdm_beta
                if beta.shape[1] >= 3:
                    beta = F.normalize(beta, dim=-1)
                    tau = float(getattr(self.cfg.model.fdm, "tcn_tau", 1.0))
                    sim_pos = (beta[:, :-2] * beta[:, 1:-1]).sum(-1) / tau
                    sim_neg = (beta[:, :-2] * beta[:, 2:]).sum(-1) / tau
                    beta_sim_pos = sim_pos.mean()
                    beta_sim_neg = sim_neg.mean()
                    beta_tcn_loss = -torch.log(
                        torch.exp(sim_pos)
                        / (torch.exp(sim_pos) + torch.exp(sim_neg) + 1e-8)
                    ).mean()
                    metrics["beta_sim_pos"] = beta_sim_pos.item()
                    metrics["beta_sim_neg"] = beta_sim_neg.item()
                    metrics["beta_tcn_loss"] = beta_tcn_loss.item()
                else:
                    metrics["beta_sim_pos"] = 0.0
                    metrics["beta_sim_neg"] = 0.0
                    metrics["beta_tcn_loss"] = 0.0

        tcn_weight = float(getattr(self.cfg.model.fdm, "tcn_loss_weight", 0.0))
        # if tcn_weight > 0:
        # print("clam_output.la[:, 1:]: ", clam_output.la.shape)
        state_seq = getattr(clam_output.idm_output, "state_seq", None)
        if state_seq is None:
            state_seq = clam_output.la[:, 1:]
        feat = state_seq
        if feat.shape[1] >= 3:
            feat = F.normalize(feat, dim=-1)
            tau = float(getattr(self.cfg.model.fdm, "tcn_tau", 1.0))
            # Positive similarity: t and t+1 (adjacent frames).
            sim_pos = (feat[:, :-2] * feat[:, 1:-1]).sum(-1) / tau
            # Negative similarity: t and t+2 (one frame apart).
            sim_neg = (feat[:, :-2] * feat[:, 2:]).sum(-1) / tau
            tcn_loss = -torch.log(
                torch.exp(sim_pos)
                / (torch.exp(sim_pos) + torch.exp(sim_neg) + 1e-8)
            ).mean()
            total_loss += tcn_weight * tcn_loss
            metrics["tcn_loss"] = tcn_loss.item()
        else:
            metrics["tcn_loss"] = 0.0
        # else:
        #     metrics["tcn_loss"] = 0.0

        # vq loss and metrics
        # ---------------------------------------------------------------------
        if self.cfg.model.idm.quantize_la:
            if clam_output.idm_output.vq_loss is not None:
                total_loss += 1.0 * clam_output.idm_output.vq_loss
            metrics.update(clam_output.idm_output.vq_metrics)

        if train and self.train_step > 0:
            revival_freq = 50 if self.train_step < 2000 else 500
            if self.train_step % revival_freq == 0:
                if hasattr(self.model.idm, "vq") and self.model.idm.vq is not None:
                    if hasattr(self.model.idm.vq, "replace_unused_codebooks"):
                        self.model.idm.vq.replace_unused_codebooks(num_batches=revival_freq)
        # ---------------------------------------------------------------------

        if (self.cfg.joint_action_decoder_training and self.train_step % self.cfg.train_action_decoder_every == 0 and train):
            labelled_batch = next(self.labelled_dataloader_train)
            labelled_batch = to_device(labelled_batch, self.device)
            labelled_batch = Batch(**labelled_batch)
            action_decoder_loss, action_decoder_metrics = self.compute_action_decoder_loss(labelled_batch)
            total_loss += self.cfg.action_decoder_loss_weight * action_decoder_loss
            metrics.update(action_decoder_metrics)

        return metrics, obs_recon, total_loss

    @property
    def save_dict(self):
        state_dict = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        if self.cfg.joint_action_decoder_training or getattr(self.cfg, "post_action_decoder_training", False):
            if hasattr(self, "action_decoder"):
                state_dict["action_decoder"] = self.action_decoder.state_dict()
            if hasattr(self, "action_decoder_optimizer"):
                state_dict["action_decoder_opt"] = self.action_decoder_optimizer.state_dict()
        return state_dict

    def _ensure_cmap_lut(self, name: str = "magma"):
        if not hasattr(self, "_cmap_lut") or self._cmap_lut is None:
            lut_np = mpl_cmaps.get_cmap(name)(np.linspace(0, 1, 256))[..., :3]
            self._cmap_lut = torch.tensor(lut_np, device=self.device, dtype=torch.float32)

    def _require_torchvision(self):
        if torchvision is None:
            raise ImportError(
                "torchvision import failed. Disable rollout/video visualization "
                "(e.g., log_rollout_videos=False) or install a torch/torchvision compatible pair."
            )

    # -------------------------------------------------------------------------
    # [Fix] Corrected Video Generation Methods
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def make_episode_video(
        self,
        ds_name: str,
        save_path: str,
        fps: int = 8,
        max_steps: int | None = None,
        step: int | None = None,
    ):
        if self.video_seq_ds is None: return None
        self._require_torchvision()
        if max_steps is None: max_steps = 50

        seq_ds = self.video_seq_ds
        it = seq_ds.as_numpy_iterator()

        frames = []
        steps = 0
        for batch_np in it:
            if steps >= max_steps:
                break
            batch = to_device(batch_np, self.device)
            batch = Batch(**batch)
            pred0 = None
            if self.use_transformer:
                # print("batch.observations.shape:", batch.observations.shape)
                # # [1, 5, 3, 128, 128]
                out  = self.model(batch.observations, timesteps=batch.timestep, states=batch.states)
                # [BUG FIX] Removed ':-1' slicing. Output is already T-1.
                pred = out.reconstructed_obs[0]        # (T-1,C,H,W)
                pred0 = pred
                # [4, 3, 128, 128]
                # print("pred.shape:", pred.shape)
                gt   = batch.observations[0, 1:]       # (T-1,C,H,W)
                # [4, 3, 128, 128]
                # print("gt.shape:", gt.shape)
            else:
                out  = self.model(batch.observations)
                pred = out.reconstructed_obs[0][None]
                gt   = batch.observations[0, -1:] # ?
            
            if self.cfg.use_res_mseloss:
                if self.cfg.model.FDM_TYPE == "SpaceTimeFDM":
                    # print("self.cfg.model.FDM_TYPE:", self.cfg.model.FDM_TYPE)
                    pred = pred + batch.observations[0, :-1]
                else:
                    T = pred.size()[0]
                    pred = batch.observations[0, :1].expand(T, -1, -1, -1) + pred

            # Handle frame-stacking compatibility
            if self.cfg.env.n_frame_stack > 1:
                # print(self.cfg.env.n_frame_stack)
                C = pred.shape[-3]
                pred = pred[:, C-3:C]; gt = gt[:, C-3:C]

            # Diff visualization
            diff_all = (pred - gt).abs().mean(dim=1, keepdim=True)
            vmax = diff_all.max().clamp(min=1e-8)
            self._ensure_cmap_lut("magma")
            self._cmap_lut = self._cmap_lut.to(device=pred.device, dtype=pred.dtype)

            Tprime = pred.shape[0] # seq length - 1
            for t in range(Tprime):
                diff_t = diff_all[t] / vmax
                diff_t = torch.nan_to_num(diff_t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
                idx = (diff_t.squeeze(0) * 255).floor().clamp(0, 255).long()
                diff_rgb = self._cmap_lut[idx].permute(2,0,1).contiguous()
                
                pred_t = (pred[t].clamp(-1, 1) + 1) / 2
                gt_t   = (gt[t].clamp(-1, 1) + 1) / 2
                pred0_t = (pred0[t].clamp(-1, 1) + 1) / 2

                tile = torch.stack([diff_rgb, pred0_t, pred_t, gt_t], dim=0)
                grid = torchvision.utils.make_grid(tile, nrow=3)
                grid = einops.rearrange(grid, "c h w -> h w c")
                img  = (torch.clamp(grid, 0, 1).cpu().numpy() * 255).astype(np.uint8)
                frames.append(img)
                steps += 1
                if t != 0:
                    _ = next(it)
                if steps >= max_steps:
                    break

        if len(frames) == 0:
            log(f"[make_episode_video] no frames produced for {ds_name}", "yellow")
            return None

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with imageio.get_writer(save_path, format="mp4", fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)

        if self.wandb_run is not None and wandb is not None:
            self.log_to_wandb(
                {f"videos/episode_{ds_name}": wandb.Video(save_path, fps=fps, format="mp4")},
                step=step,
            )
        log(f"[make_episode_video] saved: {save_path}", "green")
        return save_path

    @torch.no_grad()
    def make_dreamer_rollout_video(
        self,
        ds_name: str,
        save_path: str,
        context_len: int = 5,
        fps: int = 8,
        max_steps: int = 50,
        step: int | None = None,
    ):
        """
        [Fix #3]
        - Assume model/gt are in [-1,1]
        - Convert to [0,1] with (x+1)/2 only for visualization
        """
        if not hasattr(self, "video_seq_ds") or self.video_seq_ds is None:
            return None
        self._require_torchvision()

        seq_ds = self.video_seq_ds
        it = seq_ds.as_numpy_iterator()

        full_gt_seq = []
        first_batch = next(it, None)
        if first_batch is None:
            return None
        obs_chunk = first_batch["observations"][0]
        for i in range(obs_chunk.shape[0]):
            full_gt_seq.append(torch.tensor(obs_chunk[i]))

        for batch_np in it:
            if len(full_gt_seq) >= max_steps + 5:
                break
            last_obs = torch.tensor(batch_np["observations"][0, -1])
            full_gt_seq.append(last_obs)

        if len(full_gt_seq) < 2:
            return None

        gt_seq = torch.stack(full_gt_seq).to(self.device)[:max_steps]

        if not hasattr(self.model, "visualize_dreamer_style_rollout"):
            return None
        # try:
        recons = self.model.visualize_dreamer_style_rollout(gt_seq, idm_len=3)
        # except Exception as e:
        #     log(f"[DreamerVis] Error: {e}", "red")
        #     return None

        if recons is None or len(recons) == 0:
            return None

        # print("[make_dreamer_rollout_video] recons.shape:", recons.shape)
        # print("[make_dreamer_rollout_video] gt_seq.shape:", gt_seq.shape)

        T_pred = recons.shape[0]
        gt_match = gt_seq[1 : 1 + T_pred]  # GT frame 1.. align

        frames = []
        self._ensure_cmap_lut("magma")
        if self._cmap_lut.device != recons.device:
            self._cmap_lut = self._cmap_lut.to(device=recons.device, dtype=recons.dtype)

        diff_all = (recons - gt_match).abs().mean(dim=1, keepdim=True)
        vmax = diff_all.max().clamp(min=1e-8)

        def to_disp(x):
            # [-1,1] -> [0,1]
            return (x.clamp(-1, 1) + 1) / 2

        for t in range(T_pred):
            recon_raw = recons[t]
            gt_raw = gt_match[t]

            diff_t = (diff_all[t] / vmax).clamp(0, 1)
            idx = (diff_t.squeeze(0) * 255).long().clamp(0, 255)
            diff_rgb = self._cmap_lut[idx].permute(2, 0, 1)

            recon_t = to_disp(recon_raw)
            gt_t = to_disp(gt_raw)
            
            # Mark which frames are context frames
            current_step = t + 1
            is_context = current_step < context_len
            color = torch.tensor([0.0, 1.0, 0.0], device=recons.device) if is_context else torch.tensor([1.0, 0.0, 0.0], device=recons.device)
            border = color.view(3, 1, 1).repeat(1, 3, recon_t.shape[2])
            recon_t[:, :3, :] = border

            tile = torch.stack([diff_rgb, recon_t, gt_t], dim=0)
            grid = torchvision.utils.make_grid(tile, nrow=3, padding=2)
            grid = einops.rearrange(grid, "c h w -> h w c")
            img = (torch.clamp(grid, 0, 1).cpu().numpy() * 255).astype(np.uint8)
            frames.append(img)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with imageio.get_writer(save_path, format="mp4", fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)

        if self.wandb_run is not None and wandb is not None:
            self.log_to_wandb(
                {f"videos/dreamer_rollout_{ds_name}": wandb.Video(save_path, fps=fps, format="mp4")},
                step=step,
            )
        return save_path

    @torch.no_grad()
    def target_vis(self, sample_dataloader):
        """
        [Fix #3]
        - Compute diff by treating pred/gt as [-1,1]
        - Convert to [0,1] with (x+1)/2 only for visualization
        """
        batch = self._take_one_batch(sample_dataloader)
        self._require_torchvision()
        b = 0
        if self.use_transformer:
            out = self.model(batch.observations, timesteps=batch.timestep, states=batch.states)
            pred = out.reconstructed_obs[b]   # (T-1,C,H,W)
            gt   = batch.observations[b, 1:]  # (T-1,C,H,W)
        else:
            out  = self.model(batch.observations)
            pred = out.reconstructed_obs[b][None]
            gt   = batch.observations[b, -1:]

        if self.cfg.env.n_frame_stack > 1:
            C = pred.shape[-3]
            pred = pred[:, C-3:C]; gt = gt[:, C-3:C]

        diff = (pred - gt).abs().mean(dim=1, keepdim=True)

        self._ensure_cmap_lut("magma")
        self._cmap_lut = self._cmap_lut.to(device=pred.device, dtype=pred.dtype)

        vmax = diff.max().clamp(min=1e-8)
        tiles = []

        def to_disp(x):
            return (x.clamp(-1, 1) + 1) / 2

        for t in range(pred.shape[0]):
            diff_t = (diff[t] / vmax).clamp(0, 1)
            idx = (diff_t.squeeze(0) * 255).round().long().clamp(0, 255)
            diff_rgb = self._cmap_lut[idx].permute(2, 0, 1).contiguous()

            pred_t = to_disp(pred[t])
            gt_t   = to_disp(gt[t])

            tiles += [diff_rgb, pred_t, gt_t]

        grid = torchvision.utils.make_grid(torch.stack(tiles, dim=0), nrow=3)
        grid = einops.rearrange(grid, "c h w -> h w c")
        return (torch.clamp(grid, 0, 1) * 255).byte().cpu().numpy()


    # (make_correlation_vq etc. remain unchanged)
    @torch.no_grad()
    def make_correlation_vq(
        self,
        ds_names,
        save_path="results/vis/figure13_vq.png",
        max_batches=20000,
        max_points=500000,
        wandb_prefix="figures/",
        step: int | None = None,
    ):
        if isinstance(ds_names, str): ds_names = [ds_names]
        self.model.eval()
        all_actions_xy, all_codes, num_points = [], [], 0

        for name in ds_names:
            if hasattr(self, "eval_ds") and name in self.eval_ds: ds = self.eval_ds[name]
            elif hasattr(self, "train_ds") and name in self.train_ds: ds = self.train_ds[name]
            else: continue
            
            for batch_np in ds.as_numpy_iterator():
                if num_points >= max_points: break
                batch = Batch(**to_device(batch_np, self.device))
                out = self.model(batch.observations, timesteps=batch.timestep, states=batch.states)
                idm_out = out.idm_output
                if idm_out is None or idm_out.vq_outputs is None or "indices" not in idm_out.vq_outputs: continue
                
                codes = idm_out.vq_outputs["indices"][:, 1:]
                actions = batch.actions[:, :-1, :2]
                all_actions_xy.append(actions.detach().cpu().numpy().reshape(-1, 2))
                all_codes.append(codes.detach().cpu().numpy().reshape(-1))
                num_points += actions.shape[0] * actions.shape[1]
            if num_points >= max_points: break

        if not all_actions_xy: return
        actions_xy = np.concatenate(all_actions_xy, axis=0)
        codes = np.concatenate(all_codes, axis=0)

        if actions_xy.shape[0] > max_points:
            idx = np.random.choice(actions_xy.shape[0], max_points, replace=False)
            actions_xy = actions_xy[idx]
            codes = codes[idx]

        vocab_size = int(codes.max()) + 1
        cmap = mcolors.ListedColormap([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(0, 1, vocab_size, endpoint=False)])
        norm = mcolors.BoundaryNorm(np.arange(vocab_size + 1) - 0.5, cmap.N)

        plt.figure(figsize=(8, 8))
        plt.scatter(actions_xy[:, 0], actions_xy[:, 1], c=codes, s=4, alpha=1.0, cmap=cmap, norm=norm)
        plt.colorbar(label="Latent Code Index")
        plt.title(f"Latent Action Correlation (Datasets: {', '.join(ds_names)})")
        plt.xlabel("action x"); plt.ylabel("action y"); plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200); plt.close()
        if (
            hasattr(self, "log_to_wandb")
            and os.path.exists(save_path)
            and self.wandb_run is not None
            and wandb is not None
        ):
             self.log_to_wandb(
                 {f"{wandb_prefix}figure13_{'_'.join(ds_names)}": wandb.Image(save_path)},
                 step=step,
             )

    def eval(self, step: int):
        super().eval(step=step)
        if self.cfg.joint_action_decoder_training:
            # (Action Decoder Evaluation skipped for brevity)
            pass

        if (
            self.cfg.env.image_obs
            and self.cfg.data.use_images
            and not self.cfg.model.use_pretrained_embeddings
            and not self.cfg.model.fdm.predict_target_embedding
            and self.cfg.log_rollout_videos
        ):
            log("visualizing image reconstructions", "blue")
            video_target = self.video_ds_name
            os.makedirs("results/vis", exist_ok=True)
            ep_path = f"results/vis/oxe_eval_ep0_step{step:06d}.mp4"
            dream_path = f"results/vis/dreamer_open_loop_step{step:06d}.mp4"
            
            self.make_episode_video(
                ds_name=video_target,
                save_path=ep_path,
                fps=8,
                max_steps=100,
                step=step,
            )
            
            self.make_dreamer_rollout_video(
                ds_name=video_target,
                save_path=dream_path,
                context_len=self.cfg.model.context_len,
                fps=8,
                max_steps=100,
                step=step,
            )

            chunk_list = [f"chunk-00{i}" for i in range(5)] if "chunk" in video_target else [video_target]
            self.make_correlation_vq(
                ds_names=chunk_list,
                save_path="results/vis/figure13_vq_chunks000_005.png",
                max_points=50000,
                step=step,
            )
