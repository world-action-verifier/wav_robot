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
                while len(hist) < max(2, ensemble_size):
                    hist.append(hist[-1])
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
        pool_jsonl_path = pathlib.Path(pool_jsonl_path)
        if not pool_jsonl_path.exists():
            return collections.OrderedDict()

        with open(pool_jsonl_path, "r", encoding="utf-8") as f:
            refs = [json.loads(line) for line in f if line.strip()]

        cache = collections.OrderedDict()
        expert_refs = []
        npz_refs = []
        for ref in refs:
            if ref.get("source_type") == "dp_rollout_npz":
                npz_refs.append(ref)
            elif ref.get("source_type") == "expert_hdf5":
                expert_refs.append(ref)

        for ref in npz_refs:
            npz_path = pathlib.Path(ref["file_path"])
            if not npz_path.exists():
                continue
            with np.load(npz_path, allow_pickle=False) as ep:
                episode = {k: ep[k] for k in ep.files}
            episode = self.normalize_episode_fields_local(episode)
            cache[ref["episode_id"]] = episode

        if expert_refs:
            suite, _task = trainer.config.task.split("__", 1)
            if suite != "robomimic":
                return cache
            from environments.robomimic.utils import add_traj_to_cache, create_shape_meta

            shape_meta = create_shape_meta(
                img_size=trainer.config.image_size, include_state=True
            )
            obs_keys = shape_meta["obs"].keys()
            pixel_keys = sorted([key for key in obs_keys if "image" in key])
            state_keys = sorted([key for key in obs_keys if "image" not in key])

            by_file = collections.defaultdict(list)
            for ref in expert_refs:
                by_file[ref["file_path"]].append(ref)

            for file_path, file_refs in by_file.items():
                file_path = pathlib.Path(file_path)
                if not file_path.exists():
                    continue
                with h5py.File(file_path, "r") as f:
                    for idx, ref in enumerate(file_refs):
                        demo_key = ref.get("demo_key")
                        if demo_key is None or demo_key not in f["data"]:
                            continue
                        add_traj_to_cache(
                            traj_id=f"{demo_key}_{idx}",
                            demo=demo_key,
                            cache=cache,
                            f=f,
                            config=trainer.config,
                            pixel_keys=pixel_keys,
                            state_keys=state_keys,
                            norm_dict=None,
                        )
                        key = f"exp_traj_{demo_key}_{idx}"
                        if key in cache:
                            cache[key] = self.normalize_episode_fields_local(cache[key])
        return cache

    def normalize_episode_fields_local(self, ep):
        if "action" in ep:
            if "base_action" not in ep:
                ep["base_action"] = np.zeros_like(ep["action"])
            if "residual_action" not in ep:
                ep["residual_action"] = np.zeros_like(ep["action"])
        if "success" not in ep and "reward" in ep:
            ep["success"] = np.zeros_like(ep["reward"], dtype=bool)
        return ep
