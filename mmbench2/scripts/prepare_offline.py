"""Index raw data, score training candidates and select 50 episodes per task."""
import argparse
from pathlib import Path
import subprocess
import sys
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

def main():
    from config import RUN, TARGETS, SETTINGS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', nargs='+', default=['all'])
    p.add_argument('--methods', nargs='+', choices=list(SETTINGS), default=list(SETTINGS))
    p.add_argument('--index-only', action='store_true')
    p.add_argument('--select-only', action='store_true')
    a = p.parse_args()
    if a.index_only and a.select_only:
        p.error('--index-only and --select-only are mutually exclusive')
    tasks = TARGETS if a.tasks == ['all'] else a.tasks
    if len(set(tasks)) != len(tasks) or any(t not in TARGETS for t in tasks):
        p.error('Use distinct target task names')
    if not (RUN / 'raw-index.pt').exists():
        subprocess.run([sys.executable, str(HERE / 'stream_data.py')], check=True)
    if a.index_only:
        return
    if not a.select_only and any(m != 'random-pool50' for m in a.methods):
        for task in tasks:
            subprocess.run([sys.executable, str(HERE / 'score_pool.py'), '--task', task], check=True)
    if a.select_only or set(tasks) == set(TARGETS):
        from selection import select
        select(a.methods)
    else:
        print('Task subset scored. Run --select-only after all task scores are ready.')

if __name__ == '__main__':
    main()
