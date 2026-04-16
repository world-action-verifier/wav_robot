import json
import os
from functools import partial
from pathlib import Path
from typing import List

import einops
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from udrm.utils.logger import log

#Hayden
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

# import zarr
import rlds

# NOTE: TFDS data dir is determined by data.data_dir in configs; no hard-coded path here.


def get_dataset_dir(
    base_dir: Path, group_name: str, ds_name: str, variant: Optional[str]
) -> Path:
    """Build TFDS dataset path with optional variant subfolder."""
    if variant:
        return base_dir / group_name / ds_name / variant
    return base_dir / group_name / ds_name


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_sailor_episode_from_ref(ref):
    source_type = ref.get("source_type", "")
    if source_type != "dp_rollout_npz":
        return None
    file_path = Path(ref.get("file_path", ""))
    if not file_path.exists():
        return None
    with np.load(file_path, allow_pickle=False) as ep:
        episode = {k: ep[k] for k in ep.files}

    if "state" not in episode or "action" not in episode:
        return None

    obs = np.asarray(episode["state"])
    act = np.asarray(episode["action"])
    if obs.ndim == 3:
        obs = obs[..., -1]
    if act.ndim == 3:
        act = act[..., 0]

    out = {
        "observations": obs.astype(np.float32),
        "actions": act.astype(np.float32),
    }
    if "reward" in episode:
        out["rewards"] = np.asarray(episode["reward"]).astype(np.float32)
    if "agentview_image" in episode:
        img = np.asarray(episode["agentview_image"])
        if img.ndim == 5:
            img = img[..., -1]
        out["images"] = img
    return out


def _build_tf_episode_dataset_from_sailor_rows(rows, max_episodes=-1):
    episodes = []
    for ref in rows:
        ep = _load_sailor_episode_from_ref(ref)
        if ep is not None and len(ep["actions"]) > 2:
            episodes.append(ep)
        if max_episodes != -1 and len(episodes) >= max_episodes:
            break
    if len(episodes) == 0:
        raise ValueError("No valid dp_rollout_npz episodes found in sailor pool jsonl.")

    first = episodes[0]
    obs_shape = first["observations"].shape
    act_shape = first["actions"].shape
    output_signature = {
        "observations": tf.TensorSpec(
            shape=(None,) + tuple(obs_shape[1:]), dtype=tf.float32
        ),
        "actions": tf.TensorSpec(shape=(None,) + tuple(act_shape[1:]), dtype=tf.float32),
    }
    if "rewards" in first:
        output_signature["rewards"] = tf.TensorSpec(
            shape=(None,) + tuple(first["rewards"].shape[1:]), dtype=tf.float32
        )
    if "images" in first:
        output_signature["images"] = tf.TensorSpec(
            shape=(None,) + tuple(first["images"].shape[1:]), dtype=tf.uint8
        )

    def _gen():
        for ep in episodes:
            item = {
                "observations": ep["observations"],
                "actions": ep["actions"],
            }
            if "rewards" in ep:
                item["rewards"] = ep["rewards"]
            if "images" in ep:
                item["images"] = ep["images"]
            yield item

    ds = tf.data.Dataset.from_generator(_gen, output_signature=output_signature)
    return ds, len(episodes)


def _get_dataloader_sailor_pool(
    cfg: DictConfig,
    dataset_names: List[str],
    shuffle: bool = True,
):
    data_cfg = cfg.data.copy()
    train_jsonl = Path(data_cfg.sailor_pool_train_jsonl)
    eval_jsonl = (
        Path(data_cfg.sailor_pool_eval_jsonl)
        if getattr(data_cfg, "sailor_pool_eval_jsonl", None)
        else None
    )
    if not train_jsonl.exists():
        raise FileNotFoundError(f"sailor_pool_train_jsonl not found: {train_jsonl}")

    train_rows = _read_jsonl(train_jsonl)
    train_base_ds, n_train_eps = _build_tf_episode_dataset_from_sailor_rows(
        train_rows, max_episodes=data_cfg.num_trajs
    )
    log(f"[sailor_pool] loaded train episodes: {n_train_eps} from {train_jsonl}", "yellow")

    if eval_jsonl is not None and eval_jsonl.exists():
        eval_rows = _read_jsonl(eval_jsonl)
        eval_base_ds, n_eval_eps = _build_tf_episode_dataset_from_sailor_rows(
            eval_rows, max_episodes=-1
        )
        log(f"[sailor_pool] loaded eval episodes: {n_eval_eps} from {eval_jsonl}", "yellow")
    else:
        n_take = int(n_train_eps * float(data_cfg.train_frac))
        eval_base_ds = train_base_ds.skip(n_take)
        train_base_ds = train_base_ds.take(n_take)
        log(
            f"[sailor_pool] split train/eval from train pool: train={n_take}, eval={max(0, n_train_eps - n_take)}",
            "yellow",
        )

    cfg_train = data_cfg.copy()
    cfg_eval = data_cfg.copy()
    cfg_eval.num_trajs = -1
    cfg_eval.num_examples = -1

    train_proc = process_dataset(
        cfg_train,
        train_base_ds,
        env_name=cfg.env.env_name,
        shuffle=shuffle,
        use_pretrained_embeddings=cfg.model.use_pretrained_embeddings,
    )
    eval_proc = process_dataset(
        cfg_eval,
        eval_base_ds,
        env_name=cfg.env.env_name,
        shuffle=False,
        use_pretrained_embeddings=cfg.model.use_pretrained_embeddings,
    )

    key = dataset_names[0] if len(dataset_names) > 0 else "sailor_pool"
    return {key: train_proc}, {key: eval_proc}


def episode_to_step_custom(episode, size, shift):
    episode = tf.data.Dataset.from_tensor_slices(episode)
    if rlds is not None:
        return rlds.transformations.batch(
            episode, size=size, shift=shift, drop_remainder=True
        )

    # Fallback when rlds is unavailable: create sliding windows via tf.data.window.
    windows = episode.window(size=size, shift=shift, drop_remainder=True)

    def _batch_window(window_dict):
        return tf.data.Dataset.zip(
            {
                key: value.batch(size, drop_remainder=True)
                for key, value in window_dict.items()
            }
        )

    return windows.flat_map(_batch_window)


# add additional fields to the dataset
def add_new_fields(x):
    x["mask"] = tf.ones_like(x["actions"])
    x["timestep"] = tf.range(tf.shape(x["actions"])[0])
    return x


def use_image_observations(
    x,
    channel_first: bool = False,
    use_pretrained_embeddings: bool = False,
    image_shape: List[int] = [224, 224],
    drop_images_after_obs: bool = False,
):
    if "embeddings" in x and use_pretrained_embeddings:
        # Preserve original state observations for logging/eval, replace inputs with embeddings
        state = x.get("observations", None)
        x["observations"] = x["embeddings"]
        if state is not None:
            x["states"] = state

        has_framestack = len(x["observations"].shape) == 3

        if has_framestack:
            # flatten the dimensions
            x["observations"] = einops.rearrange(x["observations"], "B F D -> B (F D)")

    elif "images" in x:
        state = x["observations"]
        x["observations"] = x["images"]
        x["states"] = state  # add the state information here in case we want to use it

        # If images are channel-first (CHW), transpose to HWC for TF image ops
        if x["observations"].shape.rank == 4:
            if (
                x["observations"].shape[1] in (1, 3, 4)
                and x["observations"].shape[-1] not in (1, 3, 4)
            ):
                x["observations"] = tf.transpose(x["observations"], perm=[0, 2, 3, 1])
        elif x["observations"].shape.rank == 5:
            if (
                x["observations"].shape[2] in (1, 3, 4)
                and x["observations"].shape[-1] not in (1, 3, 4)
            ):
                x["observations"] = tf.transpose(x["observations"], perm=[0, 1, 3, 4, 2])

        if channel_first:
            # has framestack
            has_framestack = len(x["observations"].shape) == 5

            if has_framestack:
                # take the last frame first
                x["observations"] = einops.rearrange(
                    x["observations"], "B F H W C -> B (F C) H W"
                )
            else:
                # reshape images here
                # TODO: remove this later
                x["observations"] = tf.image.resize(
                    x["observations"], OmegaConf.to_container(image_shape)
                )
                x["observations"] = tf.transpose(x["observations"], perm=[0, 3, 1, 2])

        if tf.reduce_max(x["observations"]) > 1:
            x["observations"] = tf.cast(x["observations"], tf.float32) / 255.0
        else:
            x["observations"] = tf.cast(x["observations"], tf.float32)

        # optional: drop raw images to save memory
        if drop_images_after_obs:
            x.pop("images", None)
    return x


def process_state(x, cfg, env_name):
    states = x["observations"]
    has_framestack = len(states.shape) == 3

    if has_framestack:
        # take the last state!
        states = states[:, -1]

    x["observations"] = states
    x["states"] = states

    # for calvin, add scene_obs
    if env_name == "calvin":
        x["observations"] = tf.concat([x["observations"], x["scene_obs"]], axis=-1)

    return x


def pad_dataset(x, pad):
    # create a dataset from tensor with padded shapes
    for key in x:
        x[key] = tf.concat([x[key], pad[key]], axis=0)
    return x


# remove trajectories where the number of steps is less than 2
def filter_fn(traj):
    return tf.math.greater(tf.shape(traj["observations"])[0], 2)


def process_dataset(
    cfg: DictConfig,
    ds: tf.data.Dataset,
    shuffle: bool = True,
    env_name: str = None,
    drop_remainder: bool = False,
    use_pretrained_embeddings: bool = False,
):
    """
    Applies transformations to base tfds such as batching, shuffling, etc.
    """
    ds = ds.filter(filter_fn)
    # Debug shapes of element_spec (disabled by default)
    # spec = ds.element_spec
    # log("dataset element_spec shapes:", "yellow")
    # for k, v in spec.items():
    #     try:
    #         log(f"\t{k}: {v.shape}", "yellow")
    #     except Exception:
    #         log(f"\t{k}: {v}", "yellow")

    #Hayden - data parallelization
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    ds = ds.with_options(options)

    # if cfg.use_cache:
    #     ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(100, reshuffle_each_iteration=False)

    ds = ds.take(cfg.num_trajs)
    log(f"\ttaking {cfg.num_trajs} trajectories")

    if cfg.use_cache:
        print("using cache")
        ds = ds.cache()

    # compute return of the trajectories
    if "rewards" in ds.element_spec:
        returns = list(
            ds.map(
                lambda episode: tf.reduce_sum(episode["rewards"])
            ).as_numpy_iterator()
        )
        traj_lens = list(
            ds.map(lambda episode: tf.shape(episode["rewards"])[0]).as_numpy_iterator()
        )
        if len(returns) > 0:
            log(
                f"\tN: {len(returns)} | Average return: {sum(returns) / len(returns)} | Max return: {max(returns)} | Min return: {min(returns)} | Average traj len: {sum(traj_lens) / len(traj_lens)}",
                "yellow",
            )

    #ds = ds.map(add_new_fields)
    ds = ds.map(add_new_fields, num_parallel_calls=tf.data.AUTOTUNE)


    # replace observations with images
    if cfg.image_obs:
        log("replace observations with images", "yellow")
        ds = ds.map(
            partial(
                use_image_observations,
                channel_first=True,
                use_pretrained_embeddings=use_pretrained_embeddings,
                drop_images_after_obs=cfg.drop_images_after_obs,
                image_shape=cfg.image_shape,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(
            partial(process_state, cfg=cfg, env_name=env_name),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    # maybe pad the dataset here
    if cfg.pad_dataset:
        if rlds is None:
            # Keep training/eval runnable in environments without rlds.
            log(
                "pad_dataset=True but rlds is not installed; skip padding for this run.",
                "yellow",
            )
        else:
            # TODO: what happens if we use transitions here
            # add extra timesteps to each trajectory
            log("padding dataset", "yellow")
            pad = rlds.transformations.zeros_from_spec(ds.element_spec)

            def repeat_padding(pad, batch_size):
                return tf.nest.map_structure(
                    lambda x: tf.repeat(x, repeats=batch_size, axis=0), pad
                )

            # padding needed when we use shift
            # pad = repeat_padding(pad, cfg.seq_len - 1)
            pad = repeat_padding(pad, 5)

            #ds = ds.map(partial(pad_dataset, pad=pad))
            ds = ds.map(partial(pad_dataset, pad=pad), num_parallel_calls=tf.data.AUTOTUNE)


    # quick assert
    if cfg.data_type == "n_step":
        assert not cfg.load_latent_actions, (
            "Cannot use n_step with loading latent actions"
        )

        ds = ds.flat_map(
            partial(episode_to_step_custom, size=cfg.seq_len, shift=cfg.shift)
        )
    elif cfg.data_type == "transitions":
        ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
    else:
        raise ValueError(f"unknown data type: {cfg.data_type}")

    # Reset timesteps to be local within each window to avoid out-of-range pos embeddings
    def reset_timestep(x):
        x["timestep"] = tf.range(tf.shape(x["actions"])[0])
        return x

    ds = ds.map(reset_timestep, num_parallel_calls=tf.data.AUTOTUNE)

    # Debug shapes after slicing episodes into fixed-length windows/transitions (disabled by default)
    # post_spec = ds.element_spec
    # log("post-slice element_spec shapes:", "yellow")
    # for k, v in post_spec.items():
    #     try:
    #         log(f"\t{k}: {v.shape}", "yellow")
    #     except Exception:
    #         log(f"\t{k}: {v}", "yellow")

    # shuffle the full dataset one more time
    if shuffle:  # shuffle here is for transitions
        log("\tshuffling dataset")
        ds = ds.shuffle(1000, reshuffle_each_iteration=False)

    if cfg.num_examples != -1:
        log(f"\ttaking {cfg.num_examples} examples")
        ds = ds.take(cfg.num_examples)

        # recommended to do dataset.take(k).cache().repeat()
        #ds = ds.cache()

    ds = ds.batch(cfg.batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds



def get_dataloader(
    cfg: DictConfig,
    dataset_names: List[str],
    dataset_split: List[int],
    shuffle: bool = True,
    eval_dataset_names: Optional[List[str]] = None,
):
    """
    Returns a dictionary containing the training and validation datasets.
    train_ds: {ds_name: tf.data.Dataset}
    eval_ds : {ds_name: tf.data.Dataset}
    """
    if str(getattr(cfg.data, "source", "tfds")) == "sailor_pool":
        return _get_dataloader_sailor_pool(
            cfg=cfg,
            dataset_names=dataset_names,
            shuffle=shuffle,
        )

    # local copy so we can tweak data_type when converting zarr
    data_cfg = cfg.data.copy()
    data_dir = Path(data_cfg.data_dir) / "tensorflow_datasets"
    dataset_variant = getattr(data_cfg, "dataset_variant", None)
    if isinstance(dataset_variant, str):
        dataset_variant = dataset_variant.strip()
        if dataset_variant == "" or dataset_variant.lower() == "null":
            dataset_variant = None
    log(f"loading tfds dataset from: {data_dir}")

    env_id = cfg.env.env_id

    train_group_name = cfg.env.dataset_name
    # Use eval_dataset_name if provided; otherwise use the same directory as train
    eval_group_name = getattr(cfg.env, "eval_dataset_name", train_group_name)

    # load train dataset
    datasets = {}
    dataset_split = dataset_split[: len(dataset_names)] # since now we use seperate eval dataset, it should be 1.0 
    dataset_ratio = [x / sum(dataset_split) for x in dataset_split]

    log(
        f"loading TRAIN dataset for {env_id}, "
        f"group: {train_group_name}, "
        f"num datasets: {len(dataset_names)}, "
        f"ratios: {dataset_ratio}"
    )

    total_trajs = 0
    ds_to_len = {}
    for ds_name in dataset_names:
        save_file = get_dataset_dir(
            data_dir, train_group_name, ds_name, dataset_variant
        )
        ds = tf.data.experimental.load(str(save_file))
        log(f"\t[TRAIN RAW] {ds_name}, num trajs: {len(ds)}")

        if data_cfg.load_latent_actions:
            mapping_file = data_dir / train_group_name / ds_name / "la_map.json"

            if mapping_file.exists():
                log(f"Loading latent actions mapping from {mapping_file}", "yellow")
                with open(mapping_file, "r") as f:
                    la_map = json.load(f)
            else:
                raise ValueError(
                    f"Latent actions mapping file not found: {mapping_file}"
                )

            if not hasattr(cfg, "lam_ckpt"):
                raise ValueError("lam_ckpt not found in config")

            lam_ckpt = cfg.lam_ckpt
            id_ = la_map[lam_ckpt]
            la_file = save_file / f"latent_actions_{id_}"

            log(f"Loading latent actions relabelled from {la_file}", "yellow")

            if la_file.exists():
                latent_actions_ds = tf.data.experimental.load(str(la_file))
            else:
                raise ValueError(f"Latent actions file not found: {la_file}")

            combined_ds = tf.data.Dataset.zip((ds, latent_actions_ds))
            combined_ds = combined_ds.map(
                lambda x, y: {**x, "latent_actions": y["latent_actions"]}
            )
            ds = combined_ds

        datasets[ds_name] = ds
        total_trajs += len(ds)
        ds_to_len[ds_name] = len(ds)

    log(f"total TRAIN trajectories: {total_trajs}")
    for ds_name in dataset_names:
        log(f"\t{ds_name}: {ds_to_len[ds_name]} trajs")

    train_ds = {}
    eval_ds = {}

    log("split TRAIN dataset into train and eval (base split): ")
    for i, ds_name in enumerate(dataset_names):
        num_take = int(ds_to_len[ds_name] * cfg.data.train_frac)
        num_eval = ds_to_len[ds_name] - num_take
        log(f"\t{ds_name}: num train trajs: {num_take}, num eval trajs: {num_eval}")
        train_ds[ds_name] = datasets[ds_name].take(num_take)
        eval_ds[ds_name] = datasets[ds_name].skip(num_take)

    # pre-processing
    log("creating TRAIN datasets (processed)")
    for i, ds_name in enumerate(dataset_names):
        cfg_train = data_cfg.copy()
        if cfg.data.num_trajs != -1:
            cfg_train.num_trajs = int(cfg.data.num_trajs * dataset_ratio[i])
        if cfg.data.num_examples != -1:
            cfg_train.num_examples = int(cfg.data.num_examples * dataset_ratio[i])

        log(
            f"\t[TRAIN PROC] {ds_name}: num_trajs={cfg_train.num_trajs}, num_examples={cfg_train.num_examples}"
        )
        train_ds[ds_name] = process_dataset(
            cfg_train,
            train_ds[ds_name],
            env_name=cfg.env.env_name,
            shuffle=shuffle,
            use_pretrained_embeddings=cfg.model.use_pretrained_embeddings,
        )

    # Eval dataset pre-processing
    #    - eval_dataset_names is None  → using eval_ds splited from train dataset
    #    - eval_dataset_names exist     → load eval datasets
    print("Eval dataset pre-processing", eval_dataset_names)
    if eval_dataset_names is None:
        log("creating EVAL datasets (from TRAIN split)")
        cfg_eval = data_cfg.copy()
        cfg_eval.num_trajs = -1
        cfg_eval.num_examples = -1
        for ds_name in dataset_names:
            log(f"\t[EVAL PROC] {ds_name}: use all split eval trajs/examples")
            eval_ds[ds_name] = process_dataset(
                cfg_eval,
                eval_ds[ds_name],
                env_name=cfg.env.env_name,
                shuffle=False,
                use_pretrained_embeddings=cfg.model.use_pretrained_embeddings,
            )
    else:
        log(
            f"creating EVAL datasets from separate eval group: {eval_group_name}",
            "yellow",
        )
        eval_ds = {}
        cfg_eval = data_cfg.copy()
        cfg_eval.num_trajs = -1
        cfg_eval.num_examples = -1

        for ds_name in eval_dataset_names:
            save_file = get_dataset_dir(
                data_dir, eval_group_name, ds_name, dataset_variant
            )
            ds_e = tf.data.experimental.load(str(save_file))
            log(f"\t[EVAL RAW] {ds_name}, num trajs: {len(ds_e)}")

            if data_cfg.load_latent_actions:
                mapping_file = data_dir / eval_group_name / ds_name / "la_map.json"

                if mapping_file.exists():
                    log(f"Loading latent actions mapping from {mapping_file}", "yellow")
                    with open(mapping_file, "r") as f:
                        la_map = json.load(f)
                else:
                    raise ValueError(
                        f"Latent actions mapping file not found: {mapping_file}"
                    )

                if not hasattr(cfg, "lam_ckpt"):
                    raise ValueError("lam_ckpt not found in config")

                lam_ckpt = cfg.lam_ckpt
                id_ = la_map[lam_ckpt]
                la_file = save_file / f"latent_actions_{id_}"

                log(f"Loading latent actions relabelled from {la_file}", "yellow")

                if la_file.exists():
                    latent_actions_ds = tf.data.experimental.load(str(la_file))
                else:
                    raise ValueError(f"Latent actions file not found: {la_file}")

                combined_ds = tf.data.Dataset.zip((ds_e, latent_actions_ds))
                combined_ds = combined_ds.map(
                    lambda x, y: {**x, "latent_actions": y["latent_actions"]}
                )
                ds_e = combined_ds

            log(f"\t[EVAL PROC] {ds_name}: process all trajs/examples")
            eval_ds[ds_name] = process_dataset(
                cfg_eval,
                ds_e,
                env_name=cfg.env.env_name,
                shuffle=False,
                use_pretrained_embeddings=cfg.model.use_pretrained_embeddings,
            )

    return train_ds, eval_ds

if __name__ == "__main__":
    pass
