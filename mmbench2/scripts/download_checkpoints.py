"""Download paired Base/Combined/Coverage-aware weights from the public release."""
import argparse,hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',nargs='+',choices=['base','combined','coverage_aware'],default=['base','combined'])
    p.add_argument('--local-dir',type=Path,default=Path(os.environ.get('MMBENCH2_MODELS',str(HERE/'checkpoints'))))
    p.add_argument('--idm',action='store_true',help='Also fetch the frozen tokenizer IDM head')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();release=json.loads((HERE/'released-checkpoints.json').read_text())
    jobs=[]
    for variant in a.variants:
        for stage in ['tokenizer','dynamics']:
            jobs.append(dict(repo_id=release['repo_id'],repo_type='model',revision=release['revision'],
                             filename=f'{variant}/{stage}.pt',destination=f'{variant}/{stage}.pt',
                             sha256=release['variants'][variant][stage]['sha256']))
    if a.idm:jobs.append(dict(repo_id='ffeng1017/World-Action-Verifier',repo_type='space',revision='main',
                             filename='assets/idm_tokenizer.pt',destination='idm/idm_tokenizer.pt',sha256=release['idm']['sha256']))
    if a.dry_run:
        print(json.dumps(jobs,indent=2));return
    from huggingface_hub import hf_hub_download
    import shutil
    for job in jobs:
        destination=a.local_dir/job['destination'];destination.parent.mkdir(parents=True,exist_ok=True)
        cache=hf_hub_download(repo_id=job['repo_id'],repo_type=job['repo_type'],revision=job['revision'],
                              filename=job['filename'],token=os.environ.get('HF_TOKEN'))
        actual=sha256(Path(cache))
        if job.get('sha256') and actual!=job['sha256']:raise ValueError('Checkpoint hash mismatch: '+job['filename'])
        if destination.exists():
            if sha256(destination)!=actual:raise FileExistsError(f'Refusing to replace different weights: {destination}')
        else:
            tmp=destination.with_suffix('.pt.part');shutil.copyfile(cache,tmp);tmp.replace(destination)
        print(destination)
        manifest=destination.with_suffix('.download.json')
        manifest.write_text(json.dumps({**job,'downloaded_sha256':actual},indent=2))

if __name__=='__main__':main()
