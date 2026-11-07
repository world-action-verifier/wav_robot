import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

from config import REPO,SPACE,DATA,MODELS,RUN,HERE
OUT=RUN/'evaluation'
# config installs the bundled model modules and the upstream simulator package.
from task_set import SEEN_TASK_SET,UNSEEN_TASK_SET
from interactive import TEST_TASK_SET
TASKS=list(SEEN_TASK_SET)+list(UNSEEN_TASK_SET)
META=json.loads(((REPO/'tasks.json') if (REPO/'tasks.json').exists() else HERE/'upstream/tasks.json').read_text())
torch.set_num_threads(4)

def log(*args):print(time.strftime('%H:%M:%S'),*args,flush=True)
def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)
def metadata(task):return META.get(task,META.get(TEST_TASK_SET.get(task,'')))

def windows(part,task,length=24,n_per_ep=3,episode_subset=None):
    """Load whole real windows; no crossing episodes or mixing goal images."""
    from torchvision.io import read_image
    p=DATA/part
    raw=torch.load(p/(task+'.pt'),map_location='cpu',weights_only=False)
    ep=raw['episode'].reshape(-1).numpy()
    entries=[]
    for eid in np.unique(ep):
        if episode_subset is not None and int(eid) not in episode_subset:continue
        ids=np.flatnonzero(ep==eid)
        if len(ids)<length:raise ValueError(f'{task}/{part}/{eid}: episode shorter than {length}')
        choices=np.unique(np.linspace(int(ids[0]),int(ids[-1])-length+1,n_per_ep,dtype=int))
        for start in choices:
            assert np.all(ep[start:start+length]==eid)
            assert torch.isfinite(raw['action'][start+1:start+length]).all()
            entries.append(dict(partition=part,episode=int(eid),start=int(start),task=task))
    required=sorted({j for e in entries for j in range(e['start'],e['start']+length)})
    selected={};offset=0
    files=[f for f in p.glob(task+'-*.png') if f.stem[len(task)+1:].isdigit()]
    for f in sorted(files,key=lambda f:int(f.stem.rsplit('-',1)[1])):
        image=read_image(str(f));c,h,w=image.shape
        assert c==3 and h==224 and w%224==0
        n=w//224
        for j in required:
            if offset<=j<offset+n:selected[j]=image[:,:,224*(j-offset):224*(j-offset+1)].contiguous()
        offset+=n;del image
        if len(selected)==len(required):break
    assert len(selected)==len(required),(task,part,len(selected),len(required))
    m=metadata(task)
    assert m is not None,task
    ad=int(m['action_dim']);mask=torch.zeros(16);mask[:ad]=1
    frames=[];actions=[];rewards=[]
    for e in entries:
        s=e['start']
        frames.append(torch.stack([selected[j] for j in range(s,s+length)]))
        a=torch.nan_to_num(raw['action'][s:s+length].float()).clone()*mask
        a[0]=0;assert a.abs().max()<=1.001
        actions.append(a)
        rewards.append(torch.nan_to_num(raw['reward'][s:s+length].float()).reshape(length))
    return dict(frames=torch.stack(frames),actions=torch.stack(actions),rewards=torch.stack(rewards),
        mask=mask,lang=torch.tensor(m['text_embedding'],dtype=torch.float32),manifest=entries)

def prep_eval(task):
    path=OUT/'eval-data'/(task+'.pt')
    if path.exists():return
    parts=['test' if task in SEEN_TASK_SET else 'active_test','active_test_human']
    ds=[windows(p,task) for p in parts]
    assert len(ds[0]['frames'])==len(ds[1]['frames']),(task,'Unequal expert/human windows')
    result={k:torch.cat([d[k] for d in ds]) for k in ['frames','actions','rewards']}
    result.update(mask=ds[0]['mask'],lang=ds[0]['lang'],manifest=sum([d['manifest'] for d in ds],[]))
    path.parent.mkdir(exist_ok=True,parents=True);torch.save(result,path)
    save('manifests/'+task+'.json',result['manifest'])
    log('prepared eval',task,len(result['frames']),'windows')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args();prep_eval(a.task)
