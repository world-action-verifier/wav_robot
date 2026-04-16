import collections
import json
import pathlib

import h5py
import numpy as np
from termcolor import cprint


def load_expert_datasets(config):
    suite, _task = config.task.split("__", 1)
    if suite == "robomimic":
        from environments.robomimic.utils import get_train_val_datasets

        return get_train_val_datasets(config)
    if suite == "robocasa":
        from environments.robocasa.utils import get_train_val_datasets

        return get_train_val_datasets(config)
    if suite == "maniskill":
        from environments.maniskill.utils import get_train_val_datasets_maniskill

        return get_train_val_datasets_maniskill(config)
    raise ValueError(f"Unknown env suite {suite}")


def _zeros_like_action(action_val):
    if isinstance(action_val, list):
        return [np.zeros_like(x) for x in action_val]
    return np.zeros_like(action_val)


def _normalize_episode_fields(ep):
    if "action" in ep:
        if "base_action" not in ep:
            ep["base_action"] = _zeros_like_action(ep["action"])
        if "residual_action" not in ep:
            ep["residual_action"] = _zeros_like_action(ep["action"])
    if "success" not in ep and "reward" in ep:
        if isinstance(ep["reward"], list):
            ep["success"] = [False for _ in ep["reward"]]
        else:
            ep["success"] = np.zeros_like(ep["reward"], dtype=bool)
    return ep


def load_pool_episodes_for_wm(config, pool_jsonl_path):
    pool_jsonl_path = pathlib.Path(pool_jsonl_path)
    if not pool_jsonl_path.exists():
        raise FileNotFoundError(f"Pool jsonl not found: {pool_jsonl_path}")

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
        cache[ref["episode_id"]] = _normalize_episode_fields(episode)

    if expert_refs:
        suite, _task = config.task.split("__", 1)
        if suite != "robomimic":
            cprint(
                "Skipping expert_hdf5 refs: only robomimic expert loading is implemented in wm-only mode",
                "red",
            )
            return cache

        from environments.robomimic.utils import add_traj_to_cache, create_shape_meta

        shape_meta = create_shape_meta(img_size=config.image_size, include_state=True)
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
                        config=config,
                        pixel_keys=pixel_keys,
                        state_keys=state_keys,
                        norm_dict=None,
                    )
                    key = f"exp_traj_{demo_key}_{idx}"
                    if key in cache:
                        cache[key] = _normalize_episode_fields(cache[key])
    return cache


def load_wm_only_buffers(config):
    train_eps = collections.OrderedDict()
    sample_eps = collections.OrderedDict()
    wm_eval_eps = collections.OrderedDict()

    if getattr(config, "wm_only_mode", False) and config.wm_only_pool_jsonl:
        train_eps = load_pool_episodes_for_wm(config, config.wm_only_pool_jsonl)
        cprint(
            f"Loaded {len(train_eps.keys())} episodes into wm-only replay buffer",
            "yellow",
        )

        if getattr(config, "wm_only_sample_pool_jsonl", None):
            sample_eps = load_pool_episodes_for_wm(config, config.wm_only_sample_pool_jsonl)
            cprint(
                f"Loaded {len(sample_eps.keys())} episodes into wm-only sample buffer",
                "yellow",
            )

        if getattr(config, "wm_only_sample_source_pool_jsonl", None):
            cprint(
                f"Configured dynamic sample source pool: {config.wm_only_sample_source_pool_jsonl}",
                "yellow",
            )

        if getattr(config, "wm_only_eval_pool_jsonl", None):
            wm_eval_eps = load_pool_episodes_for_wm(config, config.wm_only_eval_pool_jsonl)
            cprint(
                f"Loaded {len(wm_eval_eps.keys())} episodes into wm-only eval buffer",
                "yellow",
            )

    return train_eps, sample_eps, wm_eval_eps
