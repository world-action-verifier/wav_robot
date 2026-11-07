"""Frozen-model CEM collection of real-environment WAV trajectories."""
import os,time,json,hashlib
from pathlib import Path
import numpy as np
import torch
from config import MODELS,IDM_CKPT,atomic_json
from common import metadata
from model import temporal_patchify,pack_bottleneck_to_spatial
from train_dynamics import load_frozen_tokenizer_from_pt_ckpt,make_tau_schedule
from interactive import load_dynamics_from_ckpt
from uncertainty import URNormScorer,sample_predictions_for_actions
from idm_hallucination import load_idm_tokenizer_head
from wav_action_cycle import WAVActionCycleScorer,masked_action_error
import curiosity_wav
PROFILES={'original':dict(n_candidates=128,horizon=32,n_elite=32,n_cem_iters=3,n_samples=2,candidate_batch_size=32)}
save=atomic_json

def config(args,models):
    return dict(policy=args.policy,task=args.task,seed=args.seed,profile=args.profile,
                stage=args.stage,max_env_steps=args.steps,parameters=dict(PROFILES[args.profile]),
                replan_stride=16,context_window=24,eval_d=.25,tau_ctx=.1,
                initial_world_model='coverage_aware',idm_feature_tokenizer='base',
                idm_meta=models['idm_meta'],idm_frozen=True,world_model_frozen=True,
                candidate_aggregation='0.5 mean + 0.5 max',action_mask=True)

def seed(value):
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

def sync():
    torch.cuda.synchronize()

@torch.no_grad()
def encode(encoder, frame):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = encoder(temporal_patchify(frame, 14))[0]
        return pack_bottleneck_to_spatial(z, n_spatial=32, k=2).float()

def load_models():
    device = torch.device("cuda")
    models = MODELS
    enc, dec, cfg = load_frozen_tokenizer_from_pt_ckpt(
        str(models / "coverage_aware/tokenizer.pt"), device=device)
    base_enc, base_dec, base_cfg = load_frozen_tokenizer_from_pt_ckpt(
        str(models / "base/tokenizer.pt"), device=device)
    del base_dec
    dyn, rew, pol, meta = load_dynamics_from_ckpt(
        str(models / "coverage_aware/dynamics.pt"), device=device,
        d_bottleneck=cfg["d_bottleneck"], n_latents=cfg["n_latents"], packing_factor=2)
    del rew, pol
    head, idm_meta = load_idm_tokenizer_head(
        str(IDM_CKPT),
        device=device, n_latents=base_cfg["n_latents"], d_bottleneck=base_cfg["d_bottleneck"])
    for module in [enc, dec, base_enc, dyn, head]:
        module.eval().requires_grad_(False)
    sched = make_tau_schedule(k_max=meta["k_max"], schedule="shortcut", d=.25)
    wav = WAVActionCycleScorer(dec, base_enc, head)
    ur = URNormScorer(enc, dec, patch=14, packing_factor=2, n_spatial=32, H=224, W=224)
    return dict(enc=enc, dyn=dyn, wav=wav, ur=ur, sched=sched,
                k_max=meta["k_max"], idm_meta=idm_meta)

def task_args(task):
    m = metadata(task)
    if not m or "text_embedding" not in m:
        raise ValueError("Missing task metadata: " + task)
    mask = torch.zeros(16, device="cuda")
    mask[:int(m["action_dim"])] = 1
    lang = torch.tensor(m["text_embedding"], device="cuda").view(1, -1)
    return mask, lang

def sample_next(models, z, actions, mask, lang):
    return sample_predictions_for_actions(
        models["dyn"], past_packed=z, candidate_actions=actions,
        k_max=models["k_max"], sched=models["sched"], act_mask=mask,
        tau_ctx=.1, lang_emb=lang, n_samples=2, use_kv_cache=True)

def plan(models, policy, z, actions, mask, lang, profile, horizon=None):
    p = dict(PROFILES[profile])
    if horizon is not None:
        p["horizon"] = min(p["horizon"], horizon)
    return curiosity_wav.curiosity_mpc_action(
        models["dyn"], past_packed=z, past_actions=actions,
        scorer=models["wav"] if policy == "wav_actioncycle" else models["ur"],
        k_max=models["k_max"], sched=models["sched"], act_mask=mask,
        lang_emb=lang, tau_ctx=.1, action_dim=16, device=torch.device("cuda"),
        use_kv_cache=True, **p)

@torch.no_grad()
def collect(args, models, cfg, folder):
    from env_wrapper import EnvCollector, save_raw_format
    seed(args.seed)
    mask, lang = task_args(args.task)
    env = EnvCollector(args.task, seed=args.seed)
    obs, info = env.env.reset()
    frames, actions, rewards, zs, diagnostics, plans, terminals = [], [], [float("nan")], [], [], [], [False]
    obs = env._extract_rgb(obs)
    frames.append(obs)
    zs.append(encode(models["enc"], torch.from_numpy(obs.copy()).to("cuda").float()[None, None] / 255)[:, 0])
    planned, offset = [], 0
    stride = cfg["replan_stride"]
    started = time.time()
    try:
        for step in range(args.steps):
            start = max(0, len(zs) - 24)
            z_ctx = torch.stack(zs[start:], 1)
            incoming = [np.zeros(16, np.float32)] + actions
            a_ctx = torch.from_numpy(np.stack(incoming[start:])).to("cuda")[None]
            image = torch.from_numpy(obs.copy()).to("cuda").float()[None, None] / 255
            models["wav"].set_root_frame(image)
            if args.policy == "random":
                action = np.random.uniform(-1, 1, 16).astype(np.float32) * mask.cpu().numpy()
            else:
                if offset >= len(planned):
                    seed(args.seed * 10000 + step)
                    sync()
                    begin = time.perf_counter()
                    result = plan(models, args.policy, z_ctx, a_ctx, mask, lang,
                                  args.profile, horizon=args.steps-step)
                    sync()
                    seq = result.get("action_sequence", result["action"][None])
                    planned = seq[:stride].cpu().numpy()
                    offset = 0
                    plans.append(dict(step=step, seconds=time.perf_counter()-begin,
                                      plan_score=float(result["uncertainty"]), executed_steps=len(planned)))
                    print("PLAN", args.policy, args.task, json.dumps(plans[-1]), flush=True)
                action = planned[offset]
                offset += 1
            requested = torch.from_numpy(action.copy()).to("cuda")[None]
            # Separate deterministic diagnostic noise from CEM/Random RNG state.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(7000000 + args.seed * 1000 + step)
                predictions = sample_next(models, z_ctx, torch.cat([a_ctx, requested[:, None]], 1), mask, lang)
            ur = models["ur"](predictions, z_ctx[:, -1])
            wav = models["wav"].score_independent(predictions, z_ctx[:, -1], requested, mask)
            next_obs, reward, terminated, truncated, info = env.env.step(action[:env._real_action_dim])
            next_rgb = env._extract_rgb(next_obs)
            next_image = torch.from_numpy(next_rgb.copy()).to("cuda").float()[None, None] / 255
            z_next = encode(models["enc"], next_image)[:, 0]
            error = (predictions.mean(1).float() - z_next).square().mean().sqrt()
            real_next_base = models["wav"].encode_image(next_image)
            real_inferred = models["wav"].infer(models["wav"].root_features, real_next_base)
            real_idm_error = masked_action_error(real_inferred, requested, mask)
            row = dict(step=step, wav=float(wav), u_r_norm=float(ur),
                       actual_wm_latent_rms=float(error), real_transition_idm_mse=float(real_idm_error),
                       action_magnitude=float(requested[:, :env._real_action_dim].abs().mean()),
                       reward=float(reward))
            diagnostics.append(row)
            zs.append(z_next)
            frames.append(next_rgb)
            actions.append(action.copy())
            rewards.append(float(reward))
            terminals.append(bool(terminated or truncated))
            obs = next_rgb
            save(folder / "progress.json", dict(step=step+1, planned=len(plans), time=time.time(), latest=row))
            print("STEP", args.policy, args.task, step+1, "wm_error", float(error), flush=True)
            if terminated or truncated:
                break
        # Keep true environment actions/rewards; no IDM pseudo-label training.
        episode = dict(frames=np.stack(frames), actions=np.stack(actions + [np.full(16, np.nan, np.float32)]),
                       rewards=np.array(rewards, np.float32), episode_id=0)
        raw = save_raw_format([episode], str(folder / "raw"), args.task)
        # Original save_raw_format supplies placeholder terminal flags. Preserve
        # actual terminal/truncation flags for the collected dataset.
        demo = torch.load(raw["demo_path"], weights_only=False)
        demo["terminated"] = torch.tensor(terminals)
        torch.save(demo, raw["demo_path"])
        return dict(config=cfg, steps=len(actions), episode_done=bool(terminals[-1]),
                    collection_seconds=time.time()-started, raw=raw, plans=plans,
                    diagnostics=diagnostics, collection_return=float(np.nansum(rewards)),
                    interpretation="Collection diagnostics; task performance is measured separately by held-out MPC evaluation")
    finally:
        env.close()
