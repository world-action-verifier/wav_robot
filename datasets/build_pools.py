"""Index ALL collected rollouts; expert masks provide separate bootstrap/eval pools."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
from datasets.episode_io import official_demo_split


def write_rows(path, rows):
    temporary = path.with_suffix('.preparing')
    temporary.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    temporary.replace(path)


def build(task, rollouts_root, expert_hdf5, output_dir):
    rollouts_root, expert_hdf5 = Path(rollouts_root).resolve(), Path(expert_hdf5).resolve()
    output_dir = Path(output_dir).resolve()
    files = sorted(rollouts_root.glob('update_*/rollouts/*.npz'))
    if not files:
        raise ValueError(f'No staged rollouts found under {rollouts_root}')
    refs = []
    for path in files:
        metadata = json.loads(path.with_suffix('.json').read_text())
        stage = path.parent.parent
        collection = json.loads((stage / 'collection.json').read_text())
        refs.append(dict(episode_id=f'{task}/{stage.name}/{path.stem}', task=task,
                         source_type='dp_rollout_npz', file_path=str(path),
                         checkpoint_update=int(stage.name.split('_')[1]),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         horizon=collection['horizon'], **metadata))
    with h5py.File(expert_hdf5, 'r') as handle:
        train, valid = official_demo_split(handle)
    def experts(names, split):
        return [dict(episode_id=f'{task}/expert/{name}', task=task, source_type='expert_hdf5',
                     file_path=str(expert_hdf5), demo_key=name, split=split) for name in names]
    train_refs, eval_refs = experts(train, 'train'), experts(valid, 'valid')
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / 'sample_pool.jsonl', refs)
    write_rows(output_dir / 'train_pool.jsonl', train_refs)
    write_rows(output_dir / 'eval_pool.jsonl', eval_refs)
    # IDM learns from all rollouts; eval uses only the official expert valid mask.
    write_rows(output_dir / 'idm_train_pool.jsonl', refs)
    summary = dict(task=task, rollouts=len(refs), expert_train=len(train), expert_valid=len(valid),
                   candidate_pool='all collected rollouts, successes and failures',
                   eval_distribution='official expert valid demos; no held-out DP rollouts',
                   source_files_unchanged=True)
    (output_dir / 'pool_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--dp-rollouts-root', required=True)
    parser.add_argument('--expert-hdf5', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.task, args.dp_rollouts_root, args.expert_hdf5, args.output_dir)))
