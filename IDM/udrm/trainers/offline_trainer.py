import collections
import contextlib
import os
import time
from collections import defaultdict
from typing import Sequence

import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
from omegaconf import DictConfig
from rich.pretty import pretty_repr

import udrm.utils.general_utils as gutl
from udrm.trainers.base_trainer import BaseTrainer
from udrm.utils.data_utils import Batch
from udrm.utils.logger import log

class MLP(nn.Module):
    """
    A simple MLP (feed-forward network).

    Args:
        in_dim: input feature dimension
        out_dim: output feature dimension
        hidden_dims: hidden layer dimensions, e.g. (128, 64)
        activation: activation module class, e.g. nn.ReLU
        dropout: dropout probability (0.0 means no dropout)
        use_bias: whether Linear layers use bias
        activate_last: whether to apply activation after the last Linear
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        use_bias: bool = True,
        activate_last: bool = False,
    ):
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive integers.")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError("All hidden_dims must be positive integers.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")

        dims = [in_dim, *list(hidden_dims), out_dim]
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            is_last = (i == len(dims) - 2)
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=use_bias))

            # Add activation/dropout after each layer except last (unless activate_last=True)
            if (not is_last) or activate_last:
                layers.append(activation())
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))

        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (..., in_dim)
        returns: shape (..., out_dim)
        """
        if x.size(-1) != self.in_dim:
            raise ValueError(f"Expected last dim {self.in_dim}, got {x.size(-1)}")
        return self.net(x)
    

class OfflineTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.train_step = 0

        # Must initialize eval iterator (use repeat to avoid StopIteration)
        self._eval_iter = self.eval_dataloader.repeat().as_numpy_iterator()
        self._rollout_vis_done = False

    def run_posthoc_suite(self):
        """Run post-hoc probes/action decoder suite."""
        if not hasattr(self, "train_posthoc_evaluator"):
            return

        def decoder_factory(kind, la_dim, state_dim, action_dim):
            if "probe" in kind:
                return MLP(in_dim=la_dim + state_dim, out_dim=state_dim, hidden_dims=(64, 32))
            if "cheat" in kind:
                return MLP(in_dim=state_dim, out_dim=state_dim, hidden_dims=(64, 32))
            return MLP(in_dim=la_dim, out_dim=action_dim, hidden_dims=(64, 32))

        suite = [
            ("probe_order1", 1),
            ("cheat_order1", 1),
            ("probe_order2", 2),
            ("cheat_order2", 2),
            # ("probe_order3", 3),
            ("action_decoder", None),
        ]
        base_step = 0
        steps = int(getattr(self.cfg, "post_action_decoder_steps", 5000))
        log_every = int(getattr(self.cfg, "post_action_decoder_log_every", 200))

        for kind, order in suite:
            log(f"[posthoc] running {kind}", "blue")
            self.train_posthoc_evaluator(
                kind=kind,
                model=self.model,
                train_loader=self.train_dataloader,
                eval_loader=self.eval_dataloader,
                device=self.device,
                decoder_factory=decoder_factory,
                num_steps=steps,
                log_every=log_every,
                log_fn=log,
                wandb_log_fn=self.log_to_wandb if hasattr(self, "log_to_wandb") else None,
                probe_order=order,
                start_step=base_step,
            )
            base_step += steps

    def train(self):
        # Evaluate action decoder directly
        if getattr(self.cfg, "post_action_decoder_only", False):
            log("post_action_decoder_only=True: skip main training loop", "yellow")
            if self.cfg.ckpt_step is not None:
                if str(self.cfg.ckpt_step).isdigit():
                    self.train_step = int(self.cfg.ckpt_step)-1
            self.run_posthoc_suite()

            if self.wandb_run is not None:
                self.wandb_run.finish()
            return

        # With accelerate, run eval only on main process (avoid duplicates/conflicts)
        if (not self.cfg.skip_first_eval) and (
            (not self.cfg.accelerate.use) or self.accelerator.is_main_process
        ):
            self.eval(step=0)

        self.model.train()
        if hasattr(self, "action_decoder"):
            self.action_decoder.train()

        train_iter = self.train_dataloader.repeat().as_numpy_iterator()

        for self.train_step in tqdm.tqdm(
            range(self.cfg.num_updates),
            desc=f"{self.cfg.name} train batches",
            disable=False,
            total=self.cfg.num_updates,
        ):
            batch_load_time = time.time()
            batch_np = next(train_iter)

            # put the batch on the device
            batch_np = gutl.to_device(batch_np, self.device)
            batch_load_time = time.time() - batch_load_time
            batch = Batch(**batch_np)

            update_time = time.time()

            self.optimizer.zero_grad()
            if hasattr(self, "action_decoder_optimizer"):
                self.action_decoder_optimizer.zero_grad()

            if self.cfg.accelerate.use:
                with self.accelerator.autocast():
                    metrics, total_loss = self.compute_loss(batch, train=True)

                self.accelerator.backward(total_loss)

                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad_norm
                )
                self.optimizer.step()

                if hasattr(self, "action_decoder_optimizer") and (
                    self.train_step % self.cfg.train_action_decoder_every == 0
                ):
                    self.accelerator.clip_grad_norm_(
                        self.action_decoder.parameters(), self.cfg.clip_grad_norm
                    )
                    self.action_decoder_optimizer.step()
            else:
                autocast_ctx = (
                    torch.amp.autocast(device_type=self.device.type)
                    if self.device.type == "cuda"
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    metrics, total_loss = self.compute_loss(batch, train=True)

                self.scaler.scale(total_loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.cfg.clip_grad_norm
                )

                self.scaler.step(self.optimizer)

                if hasattr(self, "action_decoder_optimizer") and (
                    self.train_step % self.cfg.train_action_decoder_every == 0
                ):
                    self.scaler.unscale_(self.action_decoder_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.action_decoder.parameters(),
                        max_norm=self.cfg.clip_grad_norm,
                    )
                    self.scaler.step(self.action_decoder_optimizer)

                self.scaler.update()

            self.scheduler.step()

            metrics["time/batch_load"] = batch_load_time
            metrics["time/update"] = time.time() - update_time
            metrics["lr"] = self.scheduler.get_last_lr()[0]

            if hasattr(self, "action_decoder_scheduler"):
                metrics["action_decoder_lr"] = self.action_decoder_scheduler.get_last_lr()[0]

            self.log_to_wandb(metrics, prefix="train/", step=self.train_step)

            # Eval only on the main process
            if (not self.cfg.accelerate.use) or self.accelerator.is_main_process:
                if ((self.train_step + 1) % self.eval_every) == 0:
                    self.eval(step=self.train_step + 1)
                    self.model.train()
                    if hasattr(self, "action_decoder"):
                        self.action_decoder.train()

                if ((self.train_step + 1) % self.cfg.log_terminal_every) == 0:
                    log(f"step: {self.train_step}, train:")
                    log(f"{pretty_repr(metrics)}")

        # Final evaluation (main process only)
        if (not self.cfg.accelerate.use) or self.accelerator.is_main_process:
            self.eval(step=self.cfg.num_updates)

        # Run action decoder training after main training (main process only)
        if getattr(self.cfg, "post_action_decoder_training", False):
            if (not self.cfg.accelerate.use) or self.accelerator.is_main_process:
                # Run an extra posthoc suite (probe / action decoder)
                self.run_posthoc_suite()

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def eval(self, step: int):
        # With accelerate, evaluate on main process only
        if self.cfg.accelerate.use and (not self.accelerator.is_main_process):
            return {}

        log("running evaluation", "blue")

        self.model.eval()
        if hasattr(self, "action_decoder"):
            self.action_decoder.eval()

        eval_time = time.time()
        eval_metrics = collections.defaultdict(list)

        # Restore iterator if it is missing or None
        if not hasattr(self, "_eval_iter") or self._eval_iter is None:
            self._eval_iter = self.eval_dataloader.repeat().as_numpy_iterator()

        for _ in range(self.num_eval_batches):
            batch_np = next(self._eval_iter)
            batch_np = gutl.to_device(batch_np, self.device)
            batch = Batch(**batch_np)

            with torch.no_grad():
                metrics, _total_eval_loss = self.compute_loss(batch, train=False)
            

            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().item()
                eval_metrics[k].append(v)

            del batch
            del batch_np

        for k, v in eval_metrics.items():
            eval_metrics[k] = float(np.mean(np.array(v)))

        eval_metrics["time"] = time.time() - eval_time
        self.log_to_wandb(eval_metrics, prefix="eval/", step=step)

        with open(self.log_dir / "eval.txt", "a+") as f:
            f.write(f"{step}, {eval_metrics}\n")

        log(f"eval: {pretty_repr(eval_metrics)}")

        self.save_model(ckpt_dict=self.save_dict, metrics=eval_metrics, iter=step)
        return eval_metrics
