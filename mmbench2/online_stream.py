"""Bounded-memory PNG streaming adapter for the released raw dataset.

It preserves incoming-action labels, episode boundaries for dynamics, and
targeted task probabilities. Tokenizer frames are virtually concatenated within
each selected source, as they would be when writing a selected-data shard.
"""
from bisect import bisect_right
from collections import OrderedDict
import json
import os
import random
import struct
import numpy as np
import torch
from torch.utils.data import Dataset,get_worker_info
from config import *
from interactive import TEST_TASK_SET
from round_config import BACKGROUND,PART as ONLINE_PART
def load_online_index():
 original=torch.load(BACKGROUND/'raw-index.pt',mmap=True,weights_only=False)
 index={t:{part:d for part,d in parts.items()if part in REPLAY}for t,parts in original.items()}
 index={t:parts for t,parts in index.items()if parts}
 foreground=torch.load(RUN/'online-index.pt',mmap=True,weights_only=False)
 assert set(foreground)==set(TARGETS)
 for task,parts in foreground.items():index.setdefault(task,{}).update(parts)
 return index


def build_index():
 result={};issues=[]
 for ti,task in enumerate(ALL_TASKS):
  parts={}
  for part in REPLAY+(SOURCES if task in TARGETS else []):
   p=DATA/part/(task+'.pt')
   if not p.exists():continue
   d=torch.load(p,map_location='cpu',weights_only=False)
   pngs=[p for p in (DATA/part).glob(task+'-*.png') if p.stem[len(task)+1:].isdigit()]
   bounds=[0];files=[]
   for f in sorted(pngs,key=lambda x:int(x.stem.rsplit('-',1)[1])):
    with f.open('rb') as stream:header=stream.read(24)
    assert header[:8]==b'\x89PNG\r\n\x1a\n'
    w,h=struct.unpack('>II',header[16:24]);assert h==224 and w%224==0
    bounds.append(bounds[-1]+w//224);files.append(str(f))
   ep=d['episode'].reshape(-1).long().contiguous()
   if bounds[-1]!=len(ep):
    issue=dict(task=task,partition=part,label_frames=len(ep),rgb_frames=bounds[-1],reason='released RGB files missing or incomplete')
    issues.append(issue)
    atomic_json(RUN/'raw-data-issues.json',issues)
    if part in SOURCES:raise ValueError(issue)
    # Never substitute state observations or misalign labels to available RGB.
    # The original preprocessing also cannot load a source without its PNGs.
    print('UNAVAILABLE_REPLAY_SOURCE',issue,flush=True)
    continue
   changes=torch.where(ep[1:]!=ep[:-1])[0]+1
   starts=torch.cat([torch.zeros(1,dtype=torch.long),changes])
   ends=torch.cat([changes,torch.tensor([len(ep)])])
   spans=[(int(ep[s]),int(s),int(e)) for s,e in zip(starts,ends)]
   parts[part]=dict(actions=d['action'].float().contiguous(),rewards=d['reward'].float().reshape(-1).contiguous(),
     spans=spans,files=files,bounds=bounds,n_frames=len(ep))
  if parts:result[task]=parts
  if ti%20==0:print('INDEX',ti,'/',len(ALL_TASKS),flush=True)
 path=RUN/'raw-index.pt';path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');torch.save(result,tmp);tmp.replace(path)
 atomic_json(RUN/'raw-index-summary.json',dict(tasks=len(result),unavailable_replay_sources=issues,
   replay_frames=sum(v['n_frames'] for task in result for k,v in result[task].items() if k in REPLAY),
   candidate_frames=sum(v['n_frames'] for task in result for k,v in result[task].items() if k in SOURCES)))

class RawStreamDataset(Dataset):
 def __init__(self,setting,stage,seq_len=24,cache_size=2,samples_per_tile=16):
  assert stage in ['tokenizer','dynamics']
  self.setting=setting;self.stage=stage;self.T=seq_len
  self.index=load_online_index()
  self.tasks=[t for t in ALL_TASKS if t in self.index]
  missing=set(TARGETS)-set(self.tasks)
  if missing:raise ValueError(f'Missing target tasks in raw index: {sorted(missing)}')
  if not set(self.tasks)-set(TARGETS):raise ValueError('Background replay is required for targeted task sampling')
  selected=json.loads((RUN/'selections'/(setting+'.json')).read_text())['episodes']
  choose={}
  for e in selected:choose.setdefault((e['task'],e['partition']),set()).add(e['episode'])
  self.segments=[];self.weights=[]
  for ti,task in enumerate(self.tasks):
   segments=[]
   for part,d in self.index[task].items():
    wanted=None if part in REPLAY else choose.get((task,part),set())
    if wanted is not None and not wanted:continue
    spans=[(eid,s,e) for eid,s,e in d['spans'] if wanted is None or eid in wanted]
    if stage=='dynamics':
     arrays=[]
     for eid,s,e in spans:
      if e-s<self.T+1:continue
      # Initial action/reward is undefined; a window uses the next T labels.
      valid=torch.isfinite(d['actions'][s+1:e]).all(-1)&torch.isfinite(d['rewards'][s+1:e])
      for_start=(valid.unfold(0,self.T,1).all(1)).nonzero().reshape(-1)+s
      arrays.append(for_start)
     valid_starts=torch.cat(arrays) if arrays else torch.empty(0,dtype=torch.long)
     frame_map=None
    else:
     frame_map=None if wanted is None else torch.cat([torch.arange(s,e) for _,s,e in spans])
     count=d['n_frames'] if frame_map is None else len(frame_map)
     valid_starts=torch.arange(max(0,count-self.T+1))
    if not len(valid_starts):continue
    actual_start=valid_starts if frame_map is None else frame_map[valid_starts]
    tile_ids=torch.bucketize(actual_start,torch.tensor(d['bounds'][1:]),right=True)
    tile_ranges=[]
    for tile in tile_ids.unique().tolist():
     ids=torch.where(tile_ids==tile)[0]
     tile_ranges.append((int(ids[0]),int(ids[-1])+1))
    segments.append(dict(task_idx=ti,task=task,part=part,data=d,starts=valid_starts,
      frame_map=frame_map,tile_ranges=tile_ranges,weight=len(valid_starts)))
   self.segments.append(segments)
   alpha=config(setting)['targeted_alpha']
   self.weights.append(alpha/len(TARGETS) if task in TARGETS else (1-alpha)/(len(self.tasks)-len(TARGETS)))
  assert all(self.segments)
  online_fraction=sum(w*sum(s['weight']for s in segs if s['part']==ONLINE_PART)/sum(s['weight']for s in segs)for w,segs in zip(self.weights,self.segments))
  self.audit=dict(tasks=len(self.tasks),target_task_probability=sum(w for w,t in zip(self.weights,self.tasks)if t in TARGETS),expected_online_window_fraction=online_fraction,background_replay_retained=True,online_episodes_per_task=config(setting)['selected_episodes_per_task'])
  self.cache=OrderedDict();self.cache_size=cache_size;self.n_draws=samples_per_tile
  self.current=None;self.remaining=0;self.rng=None
  self.metadata=json.loads(((REPO/'tasks.json') if (REPO/'tasks.json').exists() else HERE/'upstream/tasks.json').read_text())

 def __len__(self):return 1_000_000

 def _rng(self):
  if self.rng is None:self.rng=random.Random(torch.initial_seed()+int(os.environ.get('RANK','0'))*100000)
  return self.rng

 def _frame(self,d,j):
  from torchvision.io import read_image
  tile=bisect_right(d['bounds'],j)-1
  path=d['files'][tile]
  if path not in self.cache:
   image=read_image(path)
   frames=image.view(3,224,-1,224).permute(2,0,1,3)
   self.cache[path]=frames
   while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
  self.cache.move_to_end(path)
  return self.cache[path][j-d['bounds'][tile]]

 def __getitem__(self,unused):
  rng=self._rng()
  if self.remaining<=0:
   ti=rng.choices(range(len(self.tasks)),weights=self.weights,k=1)[0]
   seg=rng.choices(self.segments[ti],weights=[s['weight'] for s in self.segments[ti]],k=1)[0]
   lo,hi=rng.choices(seg['tile_ranges'],weights=[h-l for l,h in seg['tile_ranges']],k=1)[0]
   self.current=(seg,lo,hi);self.remaining=self.n_draws
  self.remaining-=1
  seg,lo,hi=self.current
  start=int(seg['starts'][rng.randrange(lo,hi)])
  n=self.T+(self.stage=='dynamics')
  idx=list(range(start,start+n)) if seg['frame_map'] is None else seg['frame_map'][start:start+n].tolist()
  frames=torch.stack([self._frame(seg['data'],j) for j in idx])
  if self.stage=='tokenizer':return dict(frames=frames.float()/255,task_idx=torch.tensor(seg['task_idx']))
  task=seg['task'];meta=self.metadata.get(task,self.metadata.get(TEST_TASK_SET.get(task,'')))
  ad=int(meta['action_dim']);mask=torch.zeros(self.T,16);mask[:,:ad]=1
  a=seg['data']['actions'][start+1:start+1+self.T].clone()*mask
  r=seg['data']['rewards'][start+1:start+1+self.T].clone()
  return dict(obs=frames,act=a,act_mask=mask,rew=r,
    lang_emb=torch.tensor(meta['text_embedding'],dtype=torch.float32),emb_id=torch.tensor(seg['task_idx']))

if __name__=='__main__':build_index()
