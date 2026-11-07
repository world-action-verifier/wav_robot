"""Deterministic, sample-weighted diagnostics for visual transformer IDMs."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch


def pool_action_mean(pool_jsonl):
    """Fit the constant-action baseline on train actions only (no validation labels)."""
    total, count = np.zeros(7, dtype=np.float64), 0
    for line in Path(pool_jsonl).read_text().splitlines():
        ref = json.loads(line)
        if ref['source_type'] != 'dp_rollout_npz':
            raise ValueError('Train-mean diagnostics expect a rollout pool')
        with np.load(ref['file_path'], allow_pickle=False) as episode:
            action = episode['action']
            if action.ndim == 3:
                action = action[..., 0]
            total += action.sum(axis=0, dtype=np.float64)
            count += len(action)
    if not count:
        raise ValueError('Empty training pool')
    return total / count, count


def _batch(raw, device):
    return {key: torch.tensor(raw[key], device=device) for key in
            ['observations', 'actions', 'timestep', 'states'] if key in raw}


def _latents(model, cfg, output):
    latent = output.la
    if cfg.model.distributional_la:
        latent = latent.chunk(2, dim=-1)[0]
    return latent


def _reconstruction(model, cfg, batch, output):
    prediction = model.fdm(batch['observations'], idm_output=output,
                           timesteps=batch['timestep'], states=batch.get('states'))
    if cfg.use_res_mseloss:
        if cfg.model.FDM_TYPE == 'SpaceTimeFDM':
            prediction = prediction + batch['observations'][:, :-1]
        else:
            prediction = prediction + batch['observations'][:, :1]
    return prediction


@torch.no_grad()
def evaluate_idm(model, decoder, cfg, dataset, action_mean, device, max_batches=None):
    """Evaluate all windows, then shuffle latents across the entire evaluated set.

    Reconstruction uses both transitions per 3-frame window. last_action_mse
    additionally evaluates just the last transition used by the WAV selector.
    Dataset order must be repeatable, as with shuffle=False validation pools.
    """
    model.eval()
    decoder.eval()
    totals = {key: np.zeros(7, dtype=np.float64) for key in
              ['action', 'constant', 'zero_action', 'shuffle_action']}
    pixels = {key: 0.0 for key in ['recon', 'copy', 'zero_recon', 'shuffle_recon']}
    pixel_count, transitions, windows, correct = 0, 0, 0, 0
    last_error, last_count = 0.0, 0
    latents = []
    mean = torch.tensor(action_mean, device=device, dtype=torch.float32)

    def iterator():
        source = dataset.take(max_batches) if max_batches else dataset
        return source.as_numpy_iterator()

    for raw in iterator():
        batch = _batch(raw, device)
        obs = batch['observations']
        output = model.idm(obs, timesteps=batch['timestep'], states=batch.get('states'))
        latent = _latents(model, cfg, output)
        if latent.shape[-1] != output.la.shape[-1]:
            output = replace(output, la=latent)
        prediction = decoder(latent[:, 1:])
        target = batch['actions'][:, :-1]
        assert prediction.shape == target.shape
        error = (prediction - target).square()
        totals['action'] += error.sum(dim=(0, 1)).cpu().double().numpy()
        totals['constant'] += (mean - target).square().sum(dim=(0, 1)).cpu().double().numpy()
        correct += int(((prediction[..., 6] >= 0) == (target[..., 6] >= 0)).sum())
        last_error += float(error[:, -1].sum())
        last_count += error.shape[0] * error.shape[-1]
        truth = obs[:, 1:]
        pixels['recon'] += float((_reconstruction(model, cfg, batch, output) - truth).square().sum())
        pixels['copy'] += float((obs[:, :-1] - truth).square().sum())
        pixel_count += truth.numel()
        transitions += int(target.shape[0] * target.shape[1])
        windows += int(target.shape[0])
        latents.append(latent.cpu())
        zero = replace(output, la=torch.zeros_like(latent))
        totals['zero_action'] += (decoder(zero.la[:, 1:]) - target).square().sum(dim=(0, 1)).cpu().double().numpy()
        pixels['zero_recon'] += float((_reconstruction(model, cfg, batch, zero) - truth).square().sum())

    if not windows:
        raise ValueError('No evaluation windows')
    latent_pool = torch.cat(latents)
    order = np.random.RandomState(17).permutation(windows)
    offset = 0
    for raw in iterator():
        batch = _batch(raw, device)
        obs = batch['observations']
        output = model.idm(obs, timesteps=batch['timestep'], states=batch.get('states'))
        count = obs.shape[0]
        shuffled = latent_pool[order[offset:offset + count]].to(device)
        output = replace(output, la=shuffled)
        target = batch['actions'][:, :-1]
        totals['shuffle_action'] += (decoder(shuffled[:, 1:]) - target).square().sum(dim=(0, 1)).cpu().double().numpy()
        pixels['shuffle_recon'] += float((_reconstruction(model, cfg, batch, output) - obs[:, 1:]).square().sum())
        offset += count
    assert offset == windows, 'Evaluation dataset order/length changed between passes'
    per_dim = totals['action'] / transitions
    result = dict(windows=windows, transitions=transitions,
                  action_mse=float(per_dim.mean()), per_dim_action_mse=per_dim.tolist(),
                  translation_mse=float(per_dim[:3].mean()), rotation_mse=float(per_dim[3:6].mean()),
                  gripper_mse=float(per_dim[6]), gripper_sign_accuracy=correct/transitions,
                  last_action_mse=last_error/last_count,
                  constant_action_mse=float(totals['constant'].mean()/transitions),
                  constant_per_dim_mse=(totals['constant']/transitions).tolist(),
                  zero_action_mse=float(totals['zero_action'].mean()/transitions),
                  shuffle_action_mse=float(totals['shuffle_action'].mean()/transitions),
                  latent_abs_mean=float(latent_pool[:, 1:].abs().mean()),
                  latent_std=latent_pool[:, 1:].reshape(-1, latent_pool.shape[-1]).std(dim=0).tolist())
    result.update({key + '_mse': value/pixel_count for key, value in pixels.items()})
    if not all(np.isfinite(value) for value in [result['action_mse'], *result['per_dim_action_mse']]):
        raise FloatingPointError('Non-finite diagnostic metrics')
    return result
