"""Resume complete MPC episodes while giving each new episode its own process."""
import json
from pathlib import Path
from mpc_validation import PROTOCOL, valid_episode, valid_result


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def has_protocol(data):
    return isinstance(data, dict) and all(data.get('protocol', {}).get(k) == v for k, v in PROTOCOL.items())


def cached_episode(data, task, variant, index, fingerprints=None):
    return (has_protocol(data) and (fingerprints is None or data.get('checkpoint_fingerprints') == fingerprints)
            and len(data.get('episodes', [])) == 1
            and valid_episode(data['episodes'][0], task, variant, index))


def run_isolated(task, variant, out, launch, checkpoint_fingerprints=None):
    out = Path(out)
    cache = out.parent.parent / 'mpc-episode-cache'
    existing = read(out)
    if has_protocol(existing) and (checkpoint_fingerprints is None or existing.get('checkpoint_fingerprints') == checkpoint_fingerprints):
        for row in existing.get('episodes', []):
            index = row.get('episode')
            if index in range(3) and valid_episode(row, task, variant, index):
                path = cache / f'{variant}.{task}.ep{index}.json'
                if not cached_episode(read(path), task, variant, index, checkpoint_fingerprints):
                    save(path, dict(episodes=[row], protocol=existing['protocol'],
                                    mean_normalized_score=row['normalized_score'],checkpoint_fingerprints=checkpoint_fingerprints))
    episodes, protocol = [], None
    for index in range(3):
        path = cache / f'{variant}.{task}.ep{index}.json'
        data = read(path)
        if not cached_episode(data, task, variant, index, checkpoint_fingerprints):
            launch(index)
            data = read(path)
        if not cached_episode(data, task, variant, index, checkpoint_fingerprints):
            raise ValueError(f'Episode {index} did not complete with the original protocol and seed')
        if protocol is not None and data['protocol'] != protocol:
            raise ValueError('Episode protocols differ; refusing to combine their scores')
        protocol = data['protocol']
        episodes.append(data['episodes'][0])
    combined = dict(episodes=episodes, protocol=protocol,checkpoint_fingerprints=checkpoint_fingerprints,
                    mean_normalized_score=sum(row['normalized_score'] for row in episodes) / 3,
                    execution=dict(episode_process_isolation=True, reused_complete_episodes=True,
                                   reason='DMControl renderer aborted during a second episode in one process'))
    if not valid_result(combined, task, variant):
        raise ValueError('Merged MPC output failed full-result validation')
    save(out, combined)
