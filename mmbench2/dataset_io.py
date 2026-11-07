"""Validate and publish collected incoming-action labels + PNG strips."""
from pathlib import Path
import json,struct,hashlib,os,time,torch

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def png_frames(path):
 with Path(path).open('rb')as stream:h=stream.read(24)
 if h[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Invalid PNG signature')
 w,height=struct.unpack('>II',h[16:24])
 if height!=224 or w%224:raise ValueError('Invalid frame strip geometry')
 return w//224

def validate_episode(folder,task,episode,max_steps,action_dim):
 folder=Path(folder);p=folder/'result.json'
 if not p.exists():return None
 r=json.loads(p.read_text());d=torch.load(folder/'raw'/(task+'.pt'),map_location='cpu',weights_only=False)
 n=len(d['episode']);steps=n-1
 assert 1<=steps<=max_steps and steps==r['steps'],(task,episode,steps)
 assert r['episode_done']or steps==max_steps,'A short budget-truncated pilot is not a full trajectory'
 assert d['action'].shape==(n,16)and len(d['reward'])==n
 assert torch.isnan(d['action'][0]).all()and torch.isnan(d['reward'][0]),'Invalid initial incoming labels'
 assert torch.isfinite(d['action'][1:]).all()and torch.isfinite(d['reward'][1:]).all()
 assert d['action'][1:].abs().max()<=1.001
 assert (d['action'][1:,action_dim:]==0).all(),'Nonzero inactive action'
 assert (d['episode']==episode).all(),'Episode ID mismatch'
 pngs=sorted((folder/'raw').glob(task+'-*.png'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
 assert sum(png_frames(p)for p in pngs)==n,'RGB/label frame mismatch'
 return dict(task=task,episode=episode,steps=steps,frames=n,seed=r['seed'],files=[str(p)for p in pngs],result=str(p),episode_done=r['episode_done'])

def assemble_task(root,task,max_steps,action_dim,episodes=50):
 root=Path(root);partition='active_wav_actioncycle_cem_v1'
 out=root/'dataset'/partition;out.mkdir(parents=True,exist_ok=True)
 rows=[];labels=[];files=[];bounds=[0];index=0
 for ep in range(episodes):
  folder=root/'episodes'/task/f'episode_{ep:03d}'
  row=validate_episode(folder,task,ep,max_steps,action_dim)
  if row is None:return None
  rows.append(row);labels.append(torch.load(folder/'raw'/(task+'.pt'),map_location='cpu',weights_only=False))
  for source in row['files']:
   target=out/f'{task}-{index}.png';source=Path(source)
   if target.exists():
    assert source.stat().st_size==target.stat().st_size,'Conflicting published file'
    if source.stat().st_ino!=target.stat().st_ino:assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(target.read_bytes()).digest()
   else:os.link(source,target)
   files.append(str(target));bounds.append(bounds[-1]+png_frames(target));index+=1
 merged={key:torch.cat([d[key]for d in labels])for key in ['episode','action','reward','terminated']}
 tmp=out/(task+'.pt.tmp');torch.save(merged,tmp);tmp.replace(out/(task+'.pt'))
 ep_tensor=merged['episode'];changes=torch.where(ep_tensor[1:]!=ep_tensor[:-1])[0]+1
 starts=torch.cat([torch.zeros(1,dtype=torch.long),changes]);ends=torch.cat([changes,torch.tensor([len(ep_tensor)])])
 spans=[(int(ep_tensor[s]),int(s),int(e))for s,e in zip(starts,ends)]
 assert len(spans)==episodes and bounds[-1]==len(ep_tensor)
 assert {eid for eid,s,e in spans}==set(range(episodes))
 valid_dynamics_windows=sum(max(0,e-s-24)for eid,s,e in spans)
 target_index=dict(actions=merged['action'].float().contiguous(),rewards=merged['reward'].float().contiguous(),spans=spans,files=files,bounds=bounds,n_frames=len(ep_tensor))
 path=root/'training-index'/f'{task}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save({task:{partition:target_index}},path)
 manifest=dict(task=task,partition=partition,episodes=episodes,frames=len(ep_tensor),steps=sum(r['steps']for r in rows),
  max_episode_steps=max_steps,action_dim=action_dim,valid_dynamics_windows_T24=valid_dynamics_windows,
  dataset_path=str(out),training_index=str(path),complete=True,time=time.time(),episode_records=rows)
 atomic(root/'task-complete'/f'{task}.json',manifest)
 return manifest
