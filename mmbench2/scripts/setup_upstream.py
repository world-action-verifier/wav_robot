"""Fetch the pinned MMBench2 simulator package without altering existing checkouts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parents[1]
REVISION = '3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9'
URL = 'https://github.com/nicklashansen/mmbench2.git'

def validate(target):
    if not (target / 'src/envs/__init__.py').is_file():
        raise RuntimeError('Expected an upstream checkout containing src/envs')
    metadata = json.loads((target / 'tasks.json').read_text())
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError('Invalid upstream task metadata')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, default=Path(os.environ.get('MMBENCH2_REPO', HERE / '_deps/mmbench2')))
    a = p.parse_args()
    target = a.target.expanduser().resolve()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + f'.download-{os.getpid()}')
        if temporary.exists():
            raise FileExistsError(temporary)
        for attempt in range(3):
            try:
                subprocess.run(['git', '-c', 'http.lowSpeedLimit=512', '-c', 'http.lowSpeedTime=30',
                                'clone', '--filter=blob:none', '--no-checkout', URL, str(temporary)], check=True, timeout=180)
                subprocess.run(['git', '-C', str(temporary), 'checkout', '--detach', REVISION], check=True, timeout=180)
                validate(temporary)
                temporary.replace(target)
                break
            except (subprocess.SubprocessError, OSError):
                shutil.rmtree(temporary, ignore_errors=True)
                if attempt == 2:
                    raise
                print(f'Download failed; retrying ({attempt + 2}/3)', flush=True)
                time.sleep(attempt + 1)
    validate(target)
    print('MMBench2 simulator checkout:', target)
    if target != HERE / '_deps/mmbench2':
        print('Set MMBENCH2_REPO=' + str(target) + ' when running experiments.')

if __name__ == '__main__':
    main()
