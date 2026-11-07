"""Select whole training episodes from a common pool; never read test partitions."""
import json
import numpy as np
import torch
from config import RUN, TARGETS, SOURCES, SETTINGS, config, atomic_json

def candidate_pool(index, task):
    rows = []
    for part in SOURCES:
        spans = index[task][part]['spans']
        ids = [int(eid) for eid, start, end in spans]
        if len(ids) != 50 or len(set(ids)) != 50:
            raise ValueError(f'{task}/{part}: expected 50 distinct episodes')
        rows.extend(dict(task=task, partition=part, episode=eid) for eid in sorted(ids))
    return rows

def identity(row):
    return row['task'], row['partition'], int(row['episode'])

def choose_episodes(pool, method, seed, count=50):
    if count > len(pool) or len({identity(r) for r in pool}) != len(pool):
        raise ValueError('Invalid candidate pool or selection budget')
    if method == 'random':
        indices = np.random.default_rng(seed).permutation(len(pool))[:count]
        return [pool[int(i)] for i in indices]
    if not all(np.isfinite(r[method]) for r in pool):
        raise ValueError('Nonfinite trajectory score')
    return sorted(pool, key=lambda r: r[method], reverse=True)[:count]

def select(settings=None):
    settings = list(SETTINGS) if settings is None else settings
    index = torch.load(RUN / 'raw-index.pt', mmap=True, weights_only=False)
    for setting in settings:
        selected = []
        method = SETTINGS[setting]['method']
        for ti, task in enumerate(TARGETS):
            pool = candidate_pool(index, task)
            if method != 'random':
                scored = json.loads((RUN / 'scores' / (task + '.json')).read_text())
                if scored['task'] != task or scored['denoising_d'] != SETTINGS[setting]['score_d']:
                    raise ValueError('Scoring task or denoising setting mismatch')
                if len(scored['episodes']) != len(pool) or {identity(r) for r in scored['episodes']} != {identity(r) for r in pool}:
                    raise ValueError('Scored episodes differ from the training pool')
                pool = scored['episodes']
            selected.extend(choose_episodes(pool, method, seed=731 + ti))
        atomic_json(RUN / 'selections' / (setting + '.json'), dict(config=config(setting), episodes=selected))
        print('SELECTED', setting, len(selected), 'whole episodes', flush=True)
