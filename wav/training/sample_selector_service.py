import collections
import copy
import json
import pathlib

import h5py
import numpy as np
from termcolor import cprint


class SampleSelectorService:
    def should_refresh_sample_pool(self, trainer, n_wm_itr):
        start_itr = int(getattr(trainer.config, "wm_only_sample_start_itr", 0))
        refresh_every = int(getattr(trainer.config, "wm_only_sample_refresh_every", 0))
        if n_wm_itr < start_itr:
            return False
        if n_wm_itr == start_itr:
            return True
        if refresh_every <= 0:
            return False
        return ((n_wm_itr - start_itr) % refresh_every) == 0

    def refresh_sample_pool(self, trainer, n_wm_itr):
        from datasets.data_selection import SelectionRequest, run_selection

        source_jsonl = getattr(trainer.config, "wm_only_sample_source_pool_jsonl", None)
        if not source_jsonl:
            return False

        strategy = str(getattr(trainer.config, "wm_only_sample_select_strategy", "random"))
        select_size = int(getattr(trainer.config, "wm_only_sample_select_size", 0))
        if select_size <= 0:
            return False

        seed_base = int(getattr(trainer.config, "wm_only_sample_select_seed", 0))
        kwargs_raw = getattr(trainer.config, "wm_only_sample_select_kwargs_json", "{}")
        if isinstance(kwargs_raw, str):
            try:
                strategy_kwargs = json.loads(kwargs_raw)
            except Exception:
                strategy_kwargs = {}
        elif isinstance(kwargs_raw, dict):
            strategy_kwargs = copy.deepcopy(kwargs_raw)
        else:
            strategy_kwargs = {}

        suite, task = str(trainer.config.task).split("__", 1)
        strategy_kwargs.setdefault("suite", suite)
        strategy_kwargs.setdefault("task", task.lower())
        strategy_kwargs.setdefault("device", str(trainer.config.device))
        strategy_kwargs.setdefault(
            "image_size", int(getattr(trainer.config, "image_size", 64))
        )
        strategy_kwargs.setdefault(
            "state_only", bool(getattr(trainer.config, "state_only", False))
        )

        ckpt_dir = trainer.config.logdir / "sample_selection_ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        current_ckpt = ckpt_dir / f"wm_itr_{n_wm_itr}.pt"
        trainer.dreamer_class.save_checkpoint(path=current_ckpt)
        trainer._sample_select_ckpt_history.append(current_ckpt)

        if strategy == "progress":
            strategy_kwargs["wm_new_ckpt_path"] = str(current_ckpt)
            if (
                "wm_old_ckpt_path" not in strategy_kwargs
                and trainer._prev_sample_select_ckpt is not None
            ):
                strategy_kwargs["wm_old_ckpt_path"] = str(
                    trainer._prev_sample_select_ckpt
                )
        elif strategy in ("uncertainty", "curiosity"):
            if "wm_ckpt_paths" not in strategy_kwargs:
                ensemble_size = int(strategy_kwargs.get("ensemble_size", 5))
                hist = [
                    str(p) for p in trainer._sample_select_ckpt_history[-ensemble_size:]
                ]
                if len(hist) == 0:
                    hist = [str(current_ckpt)]
                strategy_kwargs["wm_ckpt_paths"] = hist[:ensemble_size]
                strategy_kwargs["ensemble_size"] = ensemble_size
        elif strategy == "idm":
            strategy_kwargs.setdefault("wm_ckpt_path", str(current_ckpt))

        selection_dir = trainer.config.logdir / "sample_selection"
        selection_dir.mkdir(parents=True, exist_ok=True)
        output_jsonl = selection_dir / f"selected_itr_{n_wm_itr}_{strategy}.jsonl"

        cprint(
            f"[Sample Selection] itr={n_wm_itr} strategy={strategy} size={select_size}",
            "cyan",
        )
        request = SelectionRequest(
            sample_pool_jsonl=pathlib.Path(source_jsonl),
            output_jsonl=output_jsonl,
            strategy=strategy,
            select_size=select_size,
            seed=seed_base + n_wm_itr,
            strategy_kwargs=strategy_kwargs,
        )
        result = run_selection(request=request, context=None)
        new_sample_eps = self.load_pool_episodes_for_wm_local(trainer, output_jsonl)
        if len(new_sample_eps.keys()) == 0:
            cprint(
                f"[Sample Selection] empty selection at itr {n_wm_itr}, keeping previous sample pool.",
                "red",
            )
            return False
        trainer.sample_eps = new_sample_eps
        trainer._prev_sample_select_ckpt = current_ckpt
        cprint(
            f"[Sample Selection] selected episodes: {len(trainer.sample_eps.keys())}",
            "cyan",
        )
        if trainer.logger is not None:
            trainer.logger.scalar(
                "wm_critic_train/sample_selection_num_selected",
                float(result.metadata.get("num_selected", len(trainer.sample_eps.keys()))),
            )
            trainer.logger.scalar(
                "wm_critic_train/sample_selection_num_scored",
                float(result.metadata.get("num_scored", 0)),
            )
            trainer.logger.scalar(
                "wm_critic_train/sample_selection_itr",
                float(n_wm_itr),
            )
            trainer.logger.write(step=trainer._step, fps=False)
        return True

    def load_pool_episodes_for_wm_local(self, trainer, pool_jsonl_path):
        from wav.runtime.dataset_loader import load_pool_episodes_for_wm
        return load_pool_episodes_for_wm(trainer.config, pool_jsonl_path)
