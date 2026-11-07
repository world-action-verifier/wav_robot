"""Collect full online WAV episodes with the original-search-budget CEM planner."""
import argparse,json,os,subprocess,sys,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',default='all')
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--candidate-batch',type=int,default=32)
    p.add_argument('--episode-index',type=int,help=argparse.SUPPRESS)
    a=p.parse_args()
    import config as c
    from round_config import COLLECTION
    tasks=c.TARGETS if a.task=='all' else [a.task]
    if any(t not in c.TARGETS for t in tasks):raise ValueError('Use one of the 20 target tasks')
    if not 1<=a.episodes<=50 or a.candidate_batch<1:raise ValueError('Invalid collection budget')
    protocol=dict(policy='wav_actioncycle',n_candidates=128,horizon=32,n_elite=32,n_cem_iters=3,
                  n_samples=2,candidate_batch_size=a.candidate_batch,replan_stride=16,
                  context_window=24,denoising_d=.25,tau_ctx=.1,seed_rule='620000 + task_index*1000 + episode',
                  source_hashes={n:hashlib.sha256((c.HERE/n).read_bytes()).hexdigest() for n in ['collection_runtime.py','wav_action_cycle.py','curiosity_wav.py']})
    COLLECTION.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (COLLECTION/'protocol.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cache_path=COLLECTION/'checkpoint-fingerprints.json'
        cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
        release=json.loads((c.HERE/'released-checkpoints.json').read_text())
        model_paths={name:c.MODELS/name for name in ['base/tokenizer.pt','coverage_aware/tokenizer.pt','coverage_aware/dynamics.pt']}
        model_paths['idm_tokenizer.pt']=c.IDM_CKPT
        fingerprints={}
        for name,path in model_paths.items():
            stat=path.stat();key=str(path.resolve());signature=[stat.st_size,stat.st_mtime_ns]
            record=cache.get(key)
            if not record or record['signature']!=signature:
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
                record=dict(signature=signature,sha256=h.hexdigest());cache[key]=record
            expected=release['idm']['sha256'] if name=='idm_tokenizer.pt' else release['variants'][name.split('/')[0]][Path(name).stem]['sha256']
            if record['sha256']!=expected:raise ValueError('Frozen collection checkpoint differs from the recorded experiment: '+name)
            fingerprints[name]=dict(sha256=record['sha256'],bytes=stat.st_size)
        c.atomic_json(cache_path,cache)
        protocol['checkpoint_fingerprints']=fingerprints
        digest=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        plan=COLLECTION/'plan.json'
        if plan.exists():
            if json.loads(plan.read_text())['protocol_sha256']!=digest:raise ValueError('Collection settings differ; choose a new MMBENCH2_COLLECTION directory')
        else:c.atomic_json(plan,dict(protocol=protocol,protocol_sha256=digest,tasks=c.TARGETS))
    if a.episode_index is None:
        for task in tasks:
            for ep in range(a.episodes):
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--task',task,'--episodes',str(a.episodes),
                                '--candidate-batch',str(a.candidate_batch),'--episode-index',str(ep)],check=True)
        return
    if len(tasks)!=1 or not 0<=a.episode_index<a.episodes:raise ValueError('Invalid episode assignment')
    task=tasks[0];ep=a.episode_index;seed=620000+c.TARGETS.index(task)*1000+ep
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    os.environ.setdefault('SDL_VIDEODRIVER','dummy')
    from env_wrapper import EnvCollector
    env=EnvCollector(task,seed=seed)
    max_steps,action_dim=env._max_episode_steps,env._real_action_dim
    env.close()
    from dataset_io import validate_episode
    folder=COLLECTION/'episodes'/task/f'episode_{ep:03d}'
    if validate_episode(folder,task,ep,max_steps,action_dim):
        print('ALREADY_VALID',task,ep);return
    from types import SimpleNamespace
    import collection_runtime as collector
    import torch
    torch.set_num_threads(4)
    collector.PROFILES['original']['candidate_batch_size']=a.candidate_batch
    args=SimpleNamespace(stage='collect',policy='wav_actioncycle',task=task,profile='original',seed=seed,steps=max_steps)
    models=collector.load_models();cfg=collector.config(args,models)
    cfg.update(experiment_type='online_active_collection_full_task',pilot=False,
               protocol_sha256=digest,episode_index=ep,original_max_episode_steps=max_steps)
    c.atomic_json(folder/'config.json',cfg)
    result=collector.collect(args,models,cfg,folder)
    labels_path=folder/'raw'/(task+'.pt')
    labels=torch.load(labels_path,map_location='cpu',weights_only=False);labels['episode'].fill_(ep)
    tmp=labels_path.with_suffix('.pt.tmp');torch.save(labels,tmp);tmp.replace(labels_path)
    result.update(task=task,episode=ep,seed=seed,protocol_sha256=digest)
    c.atomic_json(folder/'result.json',result)
    valid=validate_episode(folder,task,ep,max_steps,action_dim)
    if not valid:raise RuntimeError('Collected episode did not pass validation')
    c.atomic_json(folder/'validated.json',valid)
    print('VALIDATED',task,ep,result['steps'])

if __name__=='__main__':main()
