"""Shared rollout semantics for WM training, selection and IDM loading."""
import json
from pathlib import Path

import numpy as np


def action_dim(action):
    shape = np.asarray(action).shape
    if len(shape) not in (2, 3):
        raise ValueError(f"Expected (T,A) or (T,A,H) actions, got {shape}")
    return int(shape[1])


def load_rollout(ref):
    path = Path(ref['file_path'])
    with np.load(path, allow_pickle=False) as handle:
        episode = {key: handle[key] for key in handle.files}
    count = len(episode['action'])
    if count < 2 or any(len(value) != count for value in episode.values()):
        raise ValueError(f"Invalid episode lengths: {path}")
    metadata = dict(ref)
    sidecar = path.with_suffix('.json')
    if sidecar.exists():
        metadata.update(json.loads(sidecar.read_text()))
    # Legacy staged data marked ALL final rows terminal, including timeouts.
    # Only rewrite when the collector's explicit stopping reason is available.
    if 'stable5_success' in metadata:
        terminal = bool(metadata['stable5_success'])
        episode['is_terminal'] = np.zeros(count, dtype=bool)
        episode['is_terminal'][-1] = terminal
        episode['is_truncated'] = np.zeros(count, dtype=bool)
        episode['is_truncated'][-1] = not terminal
    return episode


def official_demo_split(handle, train_count=None, valid_count=None):
    def read(name):
        values = handle[f'mask/{name}'][()]
        return [v.decode() if isinstance(v, bytes) else str(v) for v in values]
    train, valid = read('train'), read('valid')
    if len(set(train)) != len(train) or len(set(valid)) != len(valid):
        raise ValueError('Duplicate demos in official masks')
    if set(train) & set(valid):
        raise ValueError('Official train/valid masks overlap')
    if not set(train + valid) <= set(handle['data']):
        raise ValueError('Official masks reference missing demos')
    def take(values, count):
        if count is None or count == -1:
            return values
        if count < 0 or count > len(values):
            raise ValueError(f'Requested {count} demos from mask containing {len(values)}')
        return values[:count]
    return take(train, train_count), take(valid, valid_count)
