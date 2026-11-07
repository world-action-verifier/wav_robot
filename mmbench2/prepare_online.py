"""Freeze 50 validated online episodes per target task for alpha=0.75 training."""
import argparse,json,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.parse_args()
    import torch
    from config import TARGETS,atomic_json
    from round_config import ROOT,COLLECTION,PART
    from dataset_io import validate_episode,png_frames
    out=ROOT/'formal50'
    if (out/'snapshot.json').exists():raise FileExistsError('Snapshot already exists; use a new output directory to change data')
    plan=json.loads((COLLECTION/'plan.json').read_text())
    foreground={};records=[];episodes=[]
    for task in TARGETS:
        parts=[];files=[];bounds=[0];spans=[];offset=0
        for ep in range(50):
            folder=COLLECTION/'episodes'/task/f'episode_{ep:03d}'
            cfg=json.loads((folder/'config.json').read_text())
            if cfg['protocol_sha256']!=plan['protocol_sha256']:raise ValueError('Collection protocol mismatch')
            result=json.loads((folder/'result.json').read_text())
            raw=folder/'raw'/(task+'.pt')
            data=torch.load(raw,map_location='cpu',weights_only=False)
            action_dim=int((data['action'][1:].abs().sum(0)>0).sum())
            # Use task metadata for masks; an action dimension may remain zero by chance.
            from common import metadata
            action_dim=int(metadata(task)['action_dim'])
            valid=validate_episode(folder,task,ep,cfg['original_max_episode_steps'],action_dim)
            if not valid:raise ValueError(f'Incomplete episode: {task}/{ep}')
            expected_seed=620000+TARGETS.index(task)*1000+ep
            if valid['seed']!=expected_seed:raise ValueError('Seed mismatch')
            spans.append((ep,offset,offset+len(data['episode'])));offset+=len(data['episode']);parts.append(data)
            for path in valid['files']:files.append(path);bounds.append(bounds[-1]+png_frames(path))
            records.append(dict(task=task,episode=ep,seed=expected_seed,label_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),frames=len(data['episode'])))
            episodes.append(dict(task=task,partition=PART,episode=ep))
        if bounds[-1]!=offset:raise ValueError('Image/label lengths differ')
        foreground[task]={PART:dict(actions=torch.cat([d['action']for d in parts]).float(),
            rewards=torch.cat([d['reward']for d in parts]).float().reshape(-1),spans=spans,files=files,bounds=bounds,n_frames=offset)}
    out.mkdir(parents=True,exist_ok=True)
    tmp=out/'online-index.tmp';torch.save(foreground,tmp);tmp.replace(out/'online-index.pt')
    body=dict(episodes_per_task=50,total_episodes=len(records),records=records,collection_protocol_sha256=plan['protocol_sha256'])
    body['snapshot_sha256']=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    atomic_json(out/'selections/wav-cem-alpha075.json',dict(episodes=episodes,snapshot_sha256=body['snapshot_sha256']))
    atomic_json(out/'snapshot.json',body)
    print('SNAPSHOT_READY',len(records),body['snapshot_sha256'])

if __name__=='__main__':main()
