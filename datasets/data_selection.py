"""Active sample selection over a fixed candidate pool.
Four strategies are supported, matching ``scripts/train_wm_method.sh``:

``random``       uniform sample, no model loading (the baseline)
``progress``     learning progress, ``old_loss - new_loss`` on the candidate
``curiosity``    ensemble disagreement over WM checkpoint snapshots
``idm``          IDM-vs-WM action mismatch (the forward-inverse asymmetry score)

"""

from __future__ import annotations
import torch
import collections
import dataclasses
import json
import pathlib
import types
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from datasets.episode_io import action_dim, load_rollout

__all__ = [
    "SelectionRequest",
    "SelectionResult",
    "run_selection",
    "register_strategy",
    "STRATEGIES",
]


# --------------------------------------------------------------------------
# Public interface (exactly what sample_selector_service.py constructs)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class SelectionRequest:
    """One refresh of the active sample pool."""

    sample_pool_jsonl: pathlib.Path
    output_jsonl: pathlib.Path
    strategy: str
    select_size: int
    seed: int = 0
    strategy_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SelectionResult:
    """Outcome of a refresh.

    ``metadata`` must contain ``num_selected`` and ``num_scored``: the caller
    logs both to wandb (see ``sample_selector_service.refresh_sample_pool``).
    """

    selected_refs: List[Dict[str, Any]]
    scores: Dict[str, float]
    metadata: Dict[str, Any]


# --------------------------------------------------------------------------
# Pool I/O
# --------------------------------------------------------------------------


def read_pool_refs(pool_jsonl: pathlib.Path) -> List[Dict[str, Any]]:
    """Read a pool jsonl into a list of episode references.

    Each line is one episode reference as written by ``datasets/build_pools.py``:
    ``{"episode_id": str, "source_type": "dp_rollout_npz"|"expert_hdf5",
       "file_path": str, "demo_key": str|None, ...}``
    Extra keys are preserved verbatim so the output file round-trips.
    """
    pool_jsonl = pathlib.Path(pool_jsonl)
    if not pool_jsonl.exists():
        raise FileNotFoundError(f"sample pool not found: {pool_jsonl}")

    refs: List[Dict[str, Any]] = []
    for lineno, line in enumerate(pool_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            ref = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{pool_jsonl}:{lineno} is not valid JSON") from exc
        if "episode_id" not in ref:
            raise ValueError(f"{pool_jsonl}:{lineno} has no 'episode_id'")
        refs.append(ref)
    return refs


def write_selected(output_jsonl: pathlib.Path, refs: Sequence[Dict[str, Any]]) -> None:
    """Write the selected refs, one JSON object per line.

    The output must stay a strict subset of the input schema: the trainer
    re-reads it with ``load_pool_episodes_for_wm_local``, which dispatches on
    ``source_type`` and resolves ``file_path`` / ``demo_key``.
    """
    output_jsonl = pathlib.Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for ref in refs:
            f.write(json.dumps(ref) + "\n")


# --------------------------------------------------------------------------
# Episode loading
# --------------------------------------------------------------------------


def load_episode(ref: Dict[str, Any], **loader_kwargs) -> Optional[Dict[str, np.ndarray]]:
    """Materialise one episode referenced by ``ref``.

    Returns ``None`` when the episode cannot be loaded (missing file, or an
    ``expert_hdf5`` reference without the optional robomimic dependencies).
    Callers must treat ``None`` as "unscorable", not as "score 0".
    """
    source_type = ref.get("source_type")
    file_path = pathlib.Path(ref.get("file_path", ""))
    if not file_path.exists():
        return None

    if source_type == "dp_rollout_npz":
        return _normalize_episode_fields(load_rollout(ref))

    if source_type == "expert_hdf5":
        return _load_expert_episode(ref, file_path, **loader_kwargs)

    return None


def _load_expert_episode(ref, file_path, **loader_kwargs) -> Optional[Dict[str, np.ndarray]]:
    """Load one demo out of a robomimic hdf5 file.

    Expert references normally live in ``train_pool.jsonl`` rather than the
    candidate pool, so this path is optional: it needs ``h5py`` plus the
    robomimic helpers, and it needs a config object to know the image size.
    Without them we report the episode as unscorable rather than guessing at
    the preprocessing (a mismatched image pipeline would produce silently wrong
    world-model scores).
    """
    config = loader_kwargs.get("config")
    if config is None:
        return None
    try:
        import h5py

        from environments.robomimic.utils import add_traj_to_cache, create_shape_meta
    except ImportError:
        return None

    demo_key = ref.get("demo_key")
    if demo_key is None:
        return None

    shape_meta = create_shape_meta(img_size=config.image_size, include_state=True)
    obs_keys = shape_meta["obs"].keys()
    pixel_keys = sorted([k for k in obs_keys if "image" in k])
    state_keys = sorted([k for k in obs_keys if "image" not in k])

    cache: Dict[str, Any] = collections.OrderedDict()
    with h5py.File(file_path, "r") as f:
        if demo_key not in f["data"]:
            return None
        add_traj_to_cache(
            traj_id=f"{demo_key}_0",
            demo=demo_key,
            cache=cache,
            f=f,
            config=config,
            pixel_keys=pixel_keys,
            state_keys=state_keys,
            norm_dict=None,
        )
    key = f"exp_traj_{demo_key}_0"
    if key not in cache:
        return None
    return _normalize_episode_fields(cache[key])


def _normalize_episode_fields(ep: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Mirror of the trainer-side normalisation, so scoring sees the same keys.

    Kept byte-compatible with ``dataset_loader._normalize_episode_fields`` and
    ``sample_selector_service.normalize_episode_fields_local``.
    """
    if "action" in ep:
        if "base_action" not in ep:
            ep["base_action"] = np.zeros_like(ep["action"])
        if "residual_action" not in ep:
            ep["residual_action"] = np.zeros_like(ep["action"])
    if "success" not in ep and "reward" in ep:
        ep["success"] = np.zeros_like(ep["reward"], dtype=bool)
    return ep


def iter_chunks(episode: Dict[str, np.ndarray], batch_length: int) -> Iterable[Dict[str, np.ndarray]]:
    """Split an episode into batch-1 chunks with the fields the WM requires.

    This mirrors ``WorldModelTrainer.iter_eval_chunks`` so that scoring uses the
    exact temporal segmentation the world model is evaluated with elsewhere.
    """
    if "action" not in episode:
        return
    total = int(len(episode["action"]))
    if total < 2:
        return

    for start in range(0, total - 1, batch_length):
        end = min(start + batch_length, total)
        if end - start < 2:
            continue
        chunk: Dict[str, np.ndarray] = {}
        for key, value in episode.items():
            if "log_" in key:
                continue
            arr = np.asarray(value)
            if arr.shape[0] < end:
                continue
            chunk[key] = arr[start:end].copy()

        if "action" not in chunk:
            continue
        t = int(len(chunk["action"]))
        if t < 2:
            continue

        if "is_first" not in chunk:
            chunk["is_first"] = np.zeros((t,), dtype=bool)
        chunk["is_first"][0] = True
        if "is_terminal" not in chunk:
            chunk["is_terminal"] = np.zeros((t,), dtype=bool)
        if "discount" not in chunk:
            chunk["discount"] = np.ones((t,), dtype=np.float32)
        if "reward" not in chunk:
            chunk["reward"] = np.zeros((t,), dtype=np.float32)

        yield {k: np.expand_dims(v, axis=0) for k, v in chunk.items()}


# --------------------------------------------------------------------------
# World-model reconstruction
# --------------------------------------------------------------------------


def _obs_space_from_episode(episode: Dict[str, np.ndarray], state_only: bool):
    """Build the minimal ``obs_space`` duck-type that ``WorldModel`` needs.

    ``WorldModel.__init__`` only ever does
    ``{k: tuple(v.shape) for k, v in obs_space.spaces.items()}``, so a namespace
    holding shape-carrying stubs is enough -- we do not need a live gym env just
    to score episodes offline.

    Shapes are taken *after* the stacking dimension is dropped, matching
    ``WorldModel.preprocess`` (which selects ``[..., -1]`` for observations).
    """
    spaces: Dict[str, Any] = {}
    for key, value in episode.items():
        arr = np.asarray(value)
        if key == "state":
            shape = arr.shape[1:]
            if len(shape) == 2:  # (dim, stack) -> (dim,)
                shape = shape[:-1]
            spaces["state"] = types.SimpleNamespace(shape=tuple(shape))
        elif "image" in key and not state_only:
            shape = arr.shape[1:]
            if len(shape) == 4:  # (h, w, c, stack) -> (h, w, c)
                shape = shape[:-1]
            spaces[key] = types.SimpleNamespace(shape=tuple(shape))
    if not spaces:
        raise ValueError("episode has neither 'state' nor image keys; cannot infer obs_space")
    return types.SimpleNamespace(spaces=spaces)


def _load_run_config(ckpt_path: pathlib.Path, explicit_path: Optional[str] = None):
    """Recover the training config that produced a WM checkpoint.

    ``train_loop._setup_logging_and_seed`` dumps ``vars(config)`` to
    ``<logdir>/config.yaml``, and the selector writes its snapshots to
    ``<logdir>/sample_selection_ckpts/``, so the config sits two levels up.
    Note the dump happens *before* ``state_dim`` / ``action_dim`` /
    ``num_actions`` are assigned, so those are filled in by the caller from the
    episode data.
    """
    import ruamel.yaml as yaml

    if explicit_path:
        cfg_file = pathlib.Path(explicit_path)
    else:
        cfg_file = pathlib.Path(ckpt_path).parent.parent / "config.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"could not find the run config for {ckpt_path}; "
            f"looked at {cfg_file}. Pass wm_config_path in strategy_kwargs."
        )

    loader = yaml.YAML(typ="safe", pure=True)
    cfg_dict = loader.load(cfg_file.read_text(encoding="utf-8"))
    return types.SimpleNamespace(**cfg_dict)


def load_world_model(
    ckpt_path: str,
    reference_episode: Dict[str, np.ndarray],
    device: str,
    state_only: bool,
    wm_config_path: Optional[str] = None,
):

    from wav.dreamer.wm import WorldModel

    ckpt_path = pathlib.Path(ckpt_path)
    config = _load_run_config(ckpt_path, wm_config_path)
    config.device = device
    config.state_only = state_only

    config.num_actions = action_dim(reference_episode['action'])
    config.action_dim = config.num_actions
    if "state" in reference_episode:
        state_shape = np.asarray(reference_episode["state"]).shape[1:]
        config.state_dim = int(state_shape[0])

    obs_space = _obs_space_from_episode(reference_episode, state_only=state_only)
    wm = WorldModel(obs_space=obs_space, step=0, config=config).to(device)

    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = {}
    for key, value in raw.items():
        clean = key.replace("_orig_mod.", "")
        if clean.startswith("_wm."):
            state_dict[clean[len("_wm.") :]] = value

    if len(state_dict) == 0:
        raise RuntimeError(
            f"{ckpt_path} contained no '_wm.' tensors; is it a Dreamer checkpoint?"
        )
    wm.load_state_dict(state_dict, strict=True)

    wm.eval()
    return wm, config


# --------------------------------------------------------------------------
# Strategy: random
# --------------------------------------------------------------------------


def score_random(refs, request, **_) -> Tuple[Dict[str, float], Dict[str, Any]]:
    rng = np.random.default_rng(request.seed)
    scores = {ref["episode_id"]: float(rng.random()) for ref in refs}
    return scores, {"score_key": "random_score"}


# --------------------------------------------------------------------------
# Strategy: progress
# --------------------------------------------------------------------------


def score_progress(refs, request, **_) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Learning progress: ``old_loss - new_loss`` per candidate episode.

    A high score means the world model improved a lot on this episode between
    the previous and the current snapshot, i.e. it is still actively learnable.
    The service injects ``wm_new_ckpt_path`` every refresh and
    ``wm_old_ckpt_path`` from the second refresh onwards; on the very first
    refresh there is no "old" model, so we fall back to random rather than
    ranking on a degenerate signal.
    """
    kwargs = request.strategy_kwargs
    new_ckpt = kwargs.get("wm_new_ckpt_path")
    old_ckpt = kwargs.get("wm_old_ckpt_path") or kwargs.get("wm_old_init_ckpt_path")
    loss_key = kwargs.get("loss_key", "model_loss")

    if not new_ckpt:
        raise ValueError("progress strategy requires wm_new_ckpt_path")
    if not old_ckpt:
        scores, meta = score_random(refs, request)
        meta.update({"fallback": "random", "reason": "no wm_old_ckpt_path on first refresh"})
        return scores, meta

    new_losses, n_scored = _episode_losses(refs, request, new_ckpt, loss_key)
    old_losses, _ = _episode_losses(refs, request, old_ckpt, loss_key)

    scores = {}
    for episode_id, new_loss in new_losses.items():
        if episode_id in old_losses:
            scores[episode_id] = float(old_losses[episode_id] - new_loss)

    scores = _apply_ema(scores, request)
    return scores, {
        "score_key": kwargs.get("score_key", "progress_score"),
        "loss_key": loss_key,
        "num_episodes_evaluated": n_scored,
    }


def _episode_losses(refs, request, ckpt_path, loss_key) -> Tuple[Dict[str, float], int]:
    """Mean per-episode world-model loss under one checkpoint."""
    import torch

    kwargs = request.strategy_kwargs
    device = kwargs.get("device", "cuda:0")
    state_only = bool(kwargs.get("state_only", False))
    batch_length = int(kwargs.get("batch_length", 8))

    losses: Dict[str, float] = {}
    wm = None
    for ref in refs:
        episode = load_episode(ref, config=kwargs.get("config"))
        if episode is None:
            continue
        if wm is None:
            wm, _ = load_world_model(
                ckpt_path,
                reference_episode=episode,
                device=device,
                state_only=state_only,
                wm_config_path=kwargs.get("wm_config_path"),
            )

        values = []
        with torch.no_grad():
            for chunk in iter_chunks(episode, batch_length):
                metrics = wm.evaluate_batch_metrics(chunk)
                if loss_key in metrics:
                    values.append(float(np.mean(metrics[loss_key])))
        if values:
            losses[ref["episode_id"]] = float(np.mean(values))

    del wm
    _free_cuda()
    return losses, len(losses)


def _apply_ema(scores: Dict[str, float], request) -> Dict[str, float]:
    #Optionally smooth scores across refreshes.

    gamma = request.strategy_kwargs.get("ema_gamma")
    if gamma is None:
        return scores

    gamma = float(gamma)
    state_file = pathlib.Path(request.output_jsonl).parent / "progress_ema_state.json"
    previous: Dict[str, float] = {}
    if state_file.exists():
        try:
            previous = json.loads(state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}

    smoothed = {
        episode_id: gamma * previous[episode_id] + (1.0 - gamma) * raw
        if episode_id in previous
        else raw
        for episode_id, raw in scores.items()
    }
    state_file.write_text(json.dumps(smoothed), encoding="utf-8")
    return smoothed


# --------------------------------------------------------------------------
# Strategy: curiosity / uncertainty
# --------------------------------------------------------------------------


def score_uncertainty(refs, request, **_) -> Tuple[Dict[str, float], Dict[str, Any]]:
    # Ensemble disagreement over the prior latent features.

    kwargs = request.strategy_kwargs
    ckpt_paths = list(dict.fromkeys(str(p) for p in kwargs.get("wm_ckpt_paths", [])))
    if len(ckpt_paths) < 2:
        if not ckpt_paths:
            raise ValueError('uncertainty requires WM checkpoints')
        scores, meta = score_random(refs, request)
        meta.update(fallback='random', reason='fewer than two distinct WM snapshots')
        return scores, meta

    if len(set(str(p) for p in ckpt_paths)) < 2:
        scores, meta = score_random(refs, request)
        meta.update({
            "fallback": "random",
            "reason": "ensemble members are all the same checkpoint",
        })
        return scores, meta

    import torch

    device = kwargs.get("device", "cuda:0")
    state_only = bool(kwargs.get("state_only", False))
    batch_length = int(kwargs.get("batch_length", 8))

    # feats[episode_id][member] -> stacked prior features for that episode
    feats: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    for ckpt_path in ckpt_paths:
        wm = None
        for ref in refs:
            episode = load_episode(ref, config=kwargs.get("config"))
            if episode is None:
                continue
            if wm is None:
                wm, _ = load_world_model(
                    ckpt_path,
                    reference_episode=episode,
                    device=device,
                    state_only=state_only,
                    wm_config_path=kwargs.get("wm_config_path"),
                )
            per_chunk = []
            with torch.no_grad():
                for chunk in iter_chunks(episode, batch_length):
                    data = wm.preprocess(chunk)
                    embed = wm.encoder(data)
                    _post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
                    per_chunk.append(wm.dynamics.get_feat(prior).cpu().numpy())
            if per_chunk:
                feats[ref["episode_id"]].append(np.concatenate(per_chunk, axis=1))
        del wm
        _free_cuda()

    scores = {}
    for episode_id, members in feats.items():
        if len(members) < 2:
            continue
        length = min(m.shape[1] for m in members)
        stacked = np.stack([m[:, :length] for m in members], axis=0)
        scores[episode_id] = float(np.mean(np.var(stacked, axis=0)))

    return scores, {
        "score_key": kwargs.get("score_key", "uncertainty_latent_prior_feat_var_mean"),
        "ensemble_size": len(ckpt_paths),
    }


# --------------------------------------------------------------------------
# Strategy: idm (forward-inverse asymmetry)
# --------------------------------------------------------------------------


def score_idm(refs, request, **_) -> Tuple[Dict[str, float], Dict[str, Any]]:
    # IDM-vs-WM mismatch.
    
    kwargs = request.strategy_kwargs
    wm_ckpt = kwargs.get("wm_ckpt_path")
    idm_ckpt = kwargs.get("idm_ckpt_path")
    if not wm_ckpt:
        raise ValueError("idm strategy requires wm_ckpt_path")
    if not idm_ckpt:
        raise ValueError("idm strategy requires idm_ckpt_path (train the IDM first)")

    import torch

    device = kwargs.get("device", "cuda:0")
    idm_device = kwargs.get("idm_device", "cpu")
    state_only = bool(kwargs.get("state_only", False))
    batch_length = int(kwargs.get("batch_length", int(kwargs.get("idm_seq_len", 8))))
    subtract_baseline = bool(kwargs.get("subtract_action_baseline", False))

    idm_model, action_decoder, idm_cfg = load_idm(
        idm_ckpt, kwargs.get("idm_config_path"), idm_device
    )
    context_len = int(getattr(idm_cfg.model, "context_len", 1))

    scores: Dict[str, float] = {}
    baseline_scores: Dict[str, float] = {}
    wm = None
    for ref in refs:
        episode = load_episode(ref, config=kwargs.get("config"))
        if episode is None:
            continue
        if wm is None:
            wm, _ = load_world_model(
                wm_ckpt,
                reference_episode=episode,
                device=device,
                state_only=state_only,
                wm_config_path=kwargs.get("wm_config_path"),
            )

        mismatches, baselines = [], []
        with torch.no_grad():
            for chunk in iter_chunks(episode, batch_length):
                data = wm.preprocess(chunk)
                embed = wm.encoder(data)
                post, _prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
                gt_feat = wm.dynamics.get_feat(post)
                horizon = int(data["action"].shape[1]) - 1

                for t in range(horizon):
                    obs_window = _idm_obs_window(data, t, context_len, idm_cfg, state_only)
                    if obs_window is None:
                        continue
                    a_hat = predict_idm_action(idm_model, action_decoder, idm_cfg,
                                               obs_window.to(idm_device)).to(device)
                    if a_hat.shape != data['action'][:, t].shape:
                        raise ValueError(f'IDM action shape {a_hat.shape} does not match WM action')

                    state_t = {k: v[:, t] for k, v in post.items()}
                    pred = wm.dynamics.get_feat(wm.dynamics.img_step(state_t, a_hat))
                    target = gt_feat[:, t + 1]
                    mismatches.append(float(torch.mean((pred - target) ** 2)))

                    if subtract_baseline:
                        rolled = wm.dynamics.img_step(state_t, data["action"][:, t])
                        base_feat = wm.dynamics.get_feat(rolled)
                        baselines.append(float(torch.mean((base_feat - target) ** 2)))

        if mismatches:
            raw = float(np.mean(mismatches))
            scores[ref["episode_id"]] = raw
            if baselines:
                baseline_scores[ref["episode_id"]] = float(np.mean(baselines))

    if subtract_baseline and baseline_scores:
        scores = {
            episode_id: value - baseline_scores.get(episode_id, 0.0)
            for episode_id, value in scores.items()
        }

    del wm, idm_model, action_decoder
    _free_cuda()
    return scores, {
        "score_key": kwargs.get("score_key", "idm_wm_latent_mismatch_mse"),
        "subtract_action_baseline": subtract_baseline,
        "idm_context_len": context_len,
    }


def load_idm(idm_ckpt_path: str, idm_config_path: Optional[str], device: str):
    """Load a trained CLAM inverse model plus its post-hoc action decoder.

    ``IDM_CKPT_PATH`` may point either at a ``*.pkl`` checkpoint or at a Hydra
    run directory (which holds ``config.yaml`` and ``model_ckpts/``); both are
    accepted.  The action decoder is stored in the same pickle under the
    ``action_decoder`` key (see ``CLAMTrainer.save_dict``) and is required --
    without it the latent action cannot be mapped into the world model's action
    space.
    """
    import torch
    import sys
    from omegaconf import OmegaConf

    idm_package = str(pathlib.Path(__file__).resolve().parents[1] / 'IDM')
    if idm_package not in sys.path:
        sys.path.insert(0, idm_package)

    from udrm.models.mlp_policy import MLPPolicy
    from udrm.models.utils.clam_utils import get_clam_cls, get_la_dim

    ckpt_path = pathlib.Path(idm_ckpt_path)
    if ckpt_path.is_dir():
        run_dir = ckpt_path
        ckpt_file = run_dir / "model_ckpts" / "latest.pkl"
    else:
        ckpt_file = ckpt_path
        run_dir = ckpt_path.parent.parent

    cfg_file = pathlib.Path(idm_config_path) if idm_config_path else run_dir / "config.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"IDM config not found at {cfg_file}; pass idm_config_path explicitly."
        )
    cfg = OmegaConf.load(cfg_file)

    if getattr(cfg.model, "use_pretrained_embeddings", False):
        raise NotImplementedError(
            "This IDM was trained on precomputed R3M embeddings; scoring would "
            "need the R3M stack, which is not part of this release. Retrain with "
            "model.use_pretrained_embeddings=False or supply an embedding hook."
        )

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)

    clam_cls = get_clam_cls(cfg.name)
    input_dim = ckpt.get('input_dim')
    if input_dim is None:
        input_dim = (3, *list(cfg.env.image_shape)) if cfg.env.image_obs else ckpt.get('state_dim')
    if input_dim is None:
        raise ValueError('IDM checkpoint lacks input_dim; re-save with the repaired trainer')
    model = clam_cls(cfg.model, input_dim=input_dim, la_dim=get_la_dim(cfg))
    model.load_state_dict(_strip_compile_prefix(ckpt["model"]))
    model = model.to(device).eval()

    if "action_decoder" not in ckpt:
        raise KeyError(
            f"{ckpt_file} has no 'action_decoder'. Train it with "
            "post_action_decoder_training=True before using METHOD=idm."
        )
    decoder = MLPPolicy(cfg=cfg.model.action_decoder, input_dim=get_la_dim(cfg),
                        output_dim=cfg.env.action_dim)
    decoder.load_state_dict(_strip_compile_prefix(ckpt["action_decoder"]))
    decoder = decoder.to(device).eval()

    return model, decoder, cfg


def _strip_compile_prefix(params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the ``_orig_mod.`` prefix ``torch.compile`` leaves on state dicts."""
    return {k.replace("_orig_mod.", ""): v for k, v in params.items()}


def predict_idm_action(model, decoder, cfg, observations):
    """Decode the final observed transition, preserving the batch dimension."""
    transformer = cfg.name in {'transformer_clam', 'st_vivit_clam', 'st_vivit_clam_stm',
                               'nsvq_clam', 'tssm_clam'}
    if getattr(cfg.model, 'concatenate_gripper_state', False):
        raise NotImplementedError('IDM scoring with gripper-state concatenation needs explicit states')
    if transformer:
        timesteps = torch.arange(observations.shape[1], device=observations.device)[None]
        timesteps = timesteps.expand(observations.shape[0], -1)
        output = model.idm(observations, timesteps=timesteps, states=None)
    else:
        output = model.idm(observations)
    la = output.la
    if getattr(cfg.model, 'distributional_la', False):
        # Deterministic posterior mean for repeatable selection scores.
        la = la.chunk(2, dim=-1)[0]
    if transformer:
        la = la[:, -1]
    return decoder(la)


def _idm_obs_window(data, t: int, context_len: int, idm_cfg, state_only: bool):
    """Assemble the ``(o_{t-c..t}, o_{t+1})`` window the IDM expects.

    The IDM consumes its own observation view, so this is the one place where
    the two pipelines have to be reconciled.  Returns ``None`` near the start of
    a chunk where there is not enough context.
    """
    start = t - context_len + 1
    if start < 0:
        return None

    if getattr(idm_cfg.env, "image_obs", False) and not state_only:
        image_keys = sorted([k for k in data.keys() if "image" in k])
        if not image_keys:
            return None
        # (B, T, H, W, C) -> (B, T, C, H, W) for the conv encoder
        frames = data[image_keys[0]][:, start : t + 2]
        frames = frames.permute(0, 1, 4, 2, 3)
        target = tuple(idm_cfg.env.image_shape)
        if tuple(frames.shape[-2:]) != target:
            flat = frames.flatten(0, 1)
            flat = torch.nn.functional.interpolate(flat, size=target, mode='bilinear', align_corners=False)
            frames = flat.reshape(frames.shape[0], frames.shape[1], *flat.shape[1:])
        return frames

    return data["state"][:, start : t + 2]


# --------------------------------------------------------------------------
# Dispatch and ranking
# --------------------------------------------------------------------------

ScoreFn = Callable[..., Tuple[Dict[str, float], Dict[str, Any]]]

STRATEGIES: Dict[str, ScoreFn] = {
    "random": score_random,
    "progress": score_progress,
    "uncertainty": score_uncertainty,
    "curiosity": score_uncertainty,  # curiosity delegates to the uncertainty proxy
    "idm": score_idm,
}


def register_strategy(name: str, fn: ScoreFn) -> None:
    """Register a custom scorer (used by tests and for new methods)."""
    STRATEGIES[name] = fn


def run_selection(request: SelectionRequest, context: Any = None) -> SelectionResult:
    """Score the candidate pool, keep the top ``select_size``, write them out.

    Higher score always means "more worth training on", for every strategy.
    """
    refs = read_pool_refs(request.sample_pool_jsonl)
    if request.select_size <= 0:
        raise ValueError(f"select_size must be positive, got {request.select_size}")

    strategy = request.strategy
    if strategy not in STRATEGIES:
        raise KeyError(
            f"unknown strategy '{strategy}'; available: {sorted(STRATEGIES)}"
        )

    scores, score_meta = STRATEGIES[strategy](refs, request, context=context)

    scored = [ref for ref in refs if ref["episode_id"] in scores]
    unscored = [ref for ref in refs if ref["episode_id"] not in scores]

    scored.sort(key=lambda ref: (-scores[ref["episode_id"]], str(ref["episode_id"])))
    selected = scored[: request.select_size]

    num_backfilled = 0
    if len(selected) < request.select_size and unscored:
        rng = np.random.default_rng(request.seed)
        order = rng.permutation(len(unscored))
        need = request.select_size - len(selected)
        fill = [unscored[i] for i in order[:need]]
        num_backfilled = len(fill)
        selected.extend(fill)
        print(
            f"[data_selection] only {len(scored)}/{len(refs)} episodes could be scored; "
            f"back-filled {num_backfilled} unscored episodes to reach select_size."
        )

    write_selected(request.output_jsonl, selected)

    metadata = {
        "strategy": strategy,
        "num_candidates": len(refs),
        "num_scored": len(scored),
        "num_selected": len(selected),
        "num_unscored": len(unscored),
        "num_backfilled": num_backfilled,
        "seed": request.seed,
        "output_jsonl": str(request.output_jsonl),
    }
    metadata.update(score_meta)
    if selected:
        chosen = [scores[r["episode_id"]] for r in selected if r["episode_id"] in scores]
        if chosen:
            metadata["score_max"] = float(np.max(chosen))
            metadata["score_min"] = float(np.min(chosen))
            metadata["score_mean"] = float(np.mean(chosen))

    return SelectionResult(selected_refs=selected, scores=scores, metadata=metadata)


def _free_cuda() -> None:
    """Release model memory between checkpoints; a no-op without torch/CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample_pool_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--strategy", default="random", choices=sorted(STRATEGIES))
    parser.add_argument("--select_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy_kwargs_json", default="{}")
    args = parser.parse_args(argv)

    request = SelectionRequest(
        sample_pool_jsonl=pathlib.Path(args.sample_pool_jsonl),
        output_jsonl=pathlib.Path(args.output_jsonl),
        strategy=args.strategy,
        select_size=args.select_size,
        seed=args.seed,
        strategy_kwargs=json.loads(args.strategy_kwargs_json),
    )
    result = run_selection(request=request, context=None)
    print(json.dumps(result.metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
