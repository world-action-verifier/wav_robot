"""Bind evaluation outputs and episode caches to the exact input weight files."""
import hashlib
import json
from pathlib import Path
from config import atomic_json

def fingerprints(model_dir):
    result = {}
    for stage in ['tokenizer', 'dynamics']:
        path = Path(model_dir) / (stage + '.pt')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        result[stage] = dict(sha256=h.hexdigest(), bytes=path.stat().st_size)
    return result

def bind(output_dir, variant, identity):
    output_dir = Path(output_dir)
    record = output_dir / 'checkpoint-identities' / (variant + '.json')
    if record.exists():
        if json.loads(record.read_text()) != identity:
            raise ValueError('Checkpoint files changed; use a new MMBENCH2_RUN or MMBENCH2_ONLINE_RUN for evaluation')
    else:
        atomic_json(record, identity)
