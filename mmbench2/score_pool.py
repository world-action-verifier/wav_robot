"""Score the full, common 250-episode training pool for each target task."""
import argparse
import json
import math
import sys
import numpy as np
import torch
from config import *
# The already-validated raw-window reader is shared by the evaluation adapters.
sys.path.insert(0,str(Path(__file__).parent))
from common import windows,metadata,log
from train_dynamics import load_frozen_tokenizer_from_pt_ckpt,make_tau_schedule,sample_one_timestep_packed
from interactive import load_dynamics_from_ckpt
from model import temporal_patchify,pack_bottleneck_to_spatial,unpack_spatial_to_bottleneck
from uncertainty import URNormScorer
sys.path.append(str(SPACE))
from idm_hallucination import load_idm_tokenizer_head

@torch.no_grad()
def run(task,node,d=.125):
 torch.set_num_threads(4);torch.manual_seed(731);np.random.seed(731)
 dst=RUN/('scores' if d==.125 else 'scores-d025')/(task+'.json')
 if dst.exists():return
 device=torch.device('cuda')
 eb,db,cb=load_frozen_tokenizer_from_pt_ckpt(str(MODELS/'base/tokenizer.pt'),device=device)
 del db
 ew,dw,cw=load_frozen_tokenizer_from_pt_ckpt(str(MODELS/'coverage_aware/tokenizer.pt'),device=device)
 f,r,p,meta=load_dynamics_from_ckpt(str(MODELS/'coverage_aware/dynamics.pt'),device=device,
   d_bottleneck=cw['d_bottleneck'],n_latents=cw['n_latents'],packing_factor=2)
 del r,p
 h,_=load_idm_tokenizer_head(str(IDM_CKPT),device=device,n_latents=64,d_bottleneck=64)
 ur=URNormScorer(ew,dw,patch=14,packing_factor=2,n_spatial=32,H=224,W=224)
 schedule=make_tau_schedule(k_max=meta['k_max'],schedule='shortcut',d=d)
 rows=[]
 for si,part in enumerate(SOURCES):
  data=windows(part,task,length=16,n_per_ep=3)
  assert len({x['episode'] for x in data['manifest']})==50
  for s in range(0,len(data['frames']),8):
   x=data['frames'][s:s+8].to(device).float()/255
   zb=eb(temporal_patchify(x,14))[0]
   zw=ew(temporal_patchify(x,14))[0]
   z=pack_bottleneck_to_spatial(zw,n_spatial=32,k=2)
   mask=data['mask'].to(device).view(1,16).expand(len(x),-1)
   lang=data['lang'].to(device).view(1,-1).expand(len(x),-1)
   actions=data['actions'][s:s+8].to(device)
   inferred=h(zb[:,:-1],zb[:,1:]).float()*mask[:,None]
   ai=actions.clone();ai[:,8:]=inferred[:,7:]
   past=z[:,:8];dist=[]
   for k in range(4):
    torch.manual_seed(100000+si*1000+s*10+k)
    zn=sample_one_timestep_packed(f,past_packed=past,k_max=meta['k_max'],sched=schedule,
      actions=ai[:,:9+k],act_mask=mask[:,None].expand(-1,9+k,-1),lang_emb=lang,tau_ctx=.1,use_kv_cache=True)
    dist.append((zn.float()-z[:,8+k]).square().mean((1,2)).sqrt())
    past=torch.cat([past,zn[:,None]],1)
   dist=torch.stack(dist,1)
   # Runtime predictors use true candidate actions, as appropriate for the
   # explicitly labelled offline candidate pool. No test data is read.
   traces=[]
   handle=f.register_forward_hook(lambda mod,args,out:traces.append(out[0][:,-1].detach().float()))
   torch.manual_seed(200000+si*1000+s)
   q=sample_one_timestep_packed(f,past_packed=z[:,:8],k_max=meta['k_max'],sched=schedule,
      actions=actions[:,:9],act_mask=mask[:,None].expand(-1,9,-1),lang_emb=lang,tau_ctx=.1,use_kv_cache=False)
   handle.remove()
   delta=torch.stack([(traces[i]-traces[i-1]).square().mean((1,2)).sqrt() for i in range(1,len(traces))])
   uf=delta[len(delta)//2:].mean(0)
   torch.manual_seed(300000+si*1000+s)
   q2=sample_one_timestep_packed(f,past_packed=z[:,:8],k_max=meta['k_max'],sched=schedule,
      actions=actions[:,:9],act_mask=mask[:,None].expand(-1,9,-1),lang_emb=lang,tau_ctx=.1,use_kv_cache=False)
   u=ur.score_components(torch.stack([q,q2],1),z[:,7])['u_r']
   us=torch.stack([q,q2],1).var(1).mean((1,2))
   motion=(z[:,8]-z[:,7]).square().mean((1,2)).sqrt()
   for i in range(len(x)):
    e=data['manifest'][s+i]
    rows.append(dict(**e,wav_h4=float(dist[i].mean()),u_r=float(u[i]),u_f=float(uf[i]),u_s=float(us[i]),motion=float(motion[i])))
  log('scored',task,part)
 # Paper task-average dynamism normalization is constant within a task; it
 # does not alter the ranking when every task has the same trajectory budget.
 mo=max(float(np.mean([x['motion'] for x in rows])),1e-3)
 ep=[]
 for part in SOURCES:
  ids=sorted({r['episode'] for r in rows if r['partition']==part})
  for eid in ids:
   rs=[r for r in rows if r['partition']==part and r['episode']==eid]
   episode_row=dict(task=task,partition=part,episode=eid)
   for key in ['wav_h4','u_r','u_f','u_s']:
    episode_row[key]=float(np.mean([r[key] for r in rs]))/(mo if key.startswith('u_') else 1.)
   ep.append(episode_row)
 assert len(ep)==250
 atomic_json(dst,dict(task=task,node=node,idm_tokenizer='base',world_model='coverage_aware',
   context=8,denoising_d=d,windows=rows,episodes=ep,task_mean_motion=mo))
 log('complete pool',task)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--task',required=True,choices=TARGETS)
 a=p.parse_args();run(a.task,0,.125)
