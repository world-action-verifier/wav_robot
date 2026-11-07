"""Paths, task splits and the five equal-budget offline settings."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('MMBENCH2_REPO', HERE / '_deps/mmbench2')).expanduser().resolve()
DATA = Path(os.environ.get('MMBENCH2_DATA', HERE / 'data')).expanduser().resolve()
MODELS = Path(os.environ.get('MMBENCH2_MODELS', HERE / 'checkpoints')).expanduser().resolve()
RUN = Path(os.environ.get('MMBENCH2_RUN', HERE / 'runs/offline')).expanduser().resolve()
IDM_CKPT = Path(os.environ.get('MMBENCH2_IDM', MODELS / 'idm/idm_tokenizer.pt')).expanduser().resolve()
SPACE = HERE
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(HERE / 'upstream/src'))
from task_set import TASK_SET, SEEN_TASK_SET, UNSEEN_TASK_SET

TARGETS = list(SEEN_TASK_SET) + list(UNSEEN_TASK_SET)
ALL_TASKS = list(TASK_SET) + list(UNSEEN_TASK_SET)
SOURCES = ['active_random', 'active_curiosity_u_r_norm', 'active_expert', 'active_human', 'active_zero']
REPLAY = ['expert', 'mixed-large', 'mixed-small', 'zeros']
ENTITY = os.environ.get('WANDB_ENTITY') or None
PROJECT = os.environ.get('WANDB_PROJECT', 'mmbench2-wav')
GROUP = os.environ.get('WANDB_RUN_GROUP', 'mmbench2-offline')
os.environ.setdefault('WANDB_MODE', 'disabled')

SETTINGS = {
    'wav-h4-d0125': dict(method='wav_h4', score_horizon=4, score_d=.125),
    'random-pool50': dict(method='random', score_horizon=1, score_d=.125),
    'ur-norm-pool50': dict(method='u_r', score_horizon=1, score_d=.125),
    'uf-norm-pool50': dict(method='u_f', score_horizon=1, score_d=.125),
    'us-norm-pool50': dict(method='u_s', score_horizon=1, score_d=.125),
}

def display_name(label):
    return f'WAV MMBench2 | {label}'

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)

def config(setting):
    return dict(
        setting=setting, **SETTINGS[setting], seed=0,
        experiment_type='offline_episode_selection',
        pool_sources=SOURCES, pool_episodes_per_task=250, selected_episodes_per_task=50,
        n_seen=10, n_unseen=10, tokenizer_updates=50000, dynamics_updates=30000,
        tokenizer_effective_batch=96, dynamics_effective_batch=512, sequence_length=24,
        world_model_lr=1e-4, weight_decay=.01, initial_world_model='coverage_aware',
        idm_frozen=True, idm_feature_tokenizer='base', replay_partitions=REPLAY,
        task_weighting='targeted', targeted_alpha=.5,
        eval_context=8, eval_horizon=16, eval_d=.125, tau_ctx=.1,
        mpc_horizon=32, mpc_stride=16, mpc_population=32, mpc_iters=3,
        mpc_rollouts_per_candidate=2, mpc_episodes=3,
    )

if __name__ == '__main__':
    atomic_json(RUN / 'settings.json', {k: config(k) for k in SETTINGS})
