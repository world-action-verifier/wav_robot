"""Two-stage continuation using original model and loss implementations.

An update is counted only after optimizer.step(). All logged batch sizes are
effective global sequence counts. No credentials enter the saved config.
"""
import argparse
from contextlib import ExitStack
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb
from config import *
from stream_data import RawStreamDataset
import train_tokenizer as tt
import train_dynamics as td
from model import Encoder,Decoder,Tokenizer,EmaRms,temporal_patchify,pack_bottleneck_to_spatial
from interactive import load_dynamics_from_ckpt

def make_tokenizer(device):
 ck=torch.load(MODELS/'coverage_aware/tokenizer.pt',map_location='cpu',weights_only=False,mmap=True)
 a=ck['args'];p=int(a['patch']);dm=int(a['d_model']);nl=int(a['n_latents']);db=int(a['d_bottleneck'])
 common=dict(d_model=dm,n_heads=int(a['n_heads']),depth=int(a['depth']),n_latents=nl,n_patches=(224//p)**2,
   dropout=float(a.get('dropout',.05)),mlp_ratio=float(a['mlp_ratio']),time_every=int(a['time_every']))
 enc=Encoder(patch_dim=p*p*3,d_bottleneck=db,mae_p_min=0.,mae_p_max=.9,**common)
 dec=Decoder(d_patch=p*p*3,d_bottleneck=db,**common)
 m=Tokenizer(enc,dec);m.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in ck['model'].items()},strict=True)
 return m.to(device),dict(a),int(ck.get('step',330000)),ck.get('rms_state',{})

def plain(m):return m.module if hasattr(m,'module') else m
def cpu_state(m):return {k:v.detach().cpu() for k,v in plain(m).state_dict().items()}

def run(args):
 world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
 torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(4)
 if world>1:dist.init_process_group('nccl',timeout=timedelta(hours=2))
 torch.manual_seed(731+rank);torch.cuda.manual_seed_all(731+rank)
 stage=args.stage;cfg=config(args.setting)
 # Use bounded prefetch when a container has a small /dev/shm allocation.
 import shutil
 requested_workers=args.workers
 shm_bytes=shutil.disk_usage('/dev/shm').total
 memfd_transport=bool(requested_workers and shm_bytes<1024**3)
 cfg.update(data_loader_requested_workers=requested_workers,data_loader_workers=args.workers,
   data_loader_shm_capacity_bytes=shm_bytes,
   data_loader_transport='memfd-prefetch'if memfd_transport else('in-process'if args.workers==0 else'multiprocessing'),
   data_loader_prefetch_factor=1 if memfd_transport else(2 if args.workers else None))
 target_batch=96 if stage=='tokenizer' else 512
 if args.micro_batch:
  micro=args.micro_batch
 else:
  micro=4 if stage=='tokenizer' else 16
 assert target_batch%(world*micro)==0,(target_batch,world,micro)
 accum=target_batch//(world*micro)
 cfg.update(stage=stage,node=args.node,world_size=world,micro_batch=micro,gradient_accumulation=accum,
   effective_batch=target_batch,optimizer_updates=args.updates,smoke=args.smoke,
   seed_index=cfg['seed'],random_seed_base=731,rank_seed_rule='731 + distributed rank')
 suffix='smoke' if args.smoke else 'training'
 folder=RUN/suffix/args.setting/f'seed{cfg["seed"]}'/stage
 if args.smoke:folder=folder/f'w{world}-micro{micro}'
 folder.mkdir(parents=True,exist_ok=True)
 rms={k:EmaRms().to(device) for k in (['mse','lpips'] if stage=='tokenizer' else ['flow','rew','bc'])}
 if stage=='tokenizer':
  tok,model_args,initial_step,old_rms=make_tokenizer(device)
  models=[tok]
  import lpips
  lp=lpips.LPIPS(net='alex').to(device).eval()
  for p in lp.parameters():p.requires_grad_(False)
 else:
  tok_ckpt=RUN/'training'/args.setting/'seed0/tokenizer'/f'update_{cfg["tokenizer_updates"]:06d}.pt'
  if args.smoke and not tok_ckpt.exists():tok_ckpt=MODELS/'coverage_aware/tokenizer.pt'
  if not tok_ckpt.exists():raise FileNotFoundError(f'Complete tokenizer training before dynamics: {tok_ckpt}')
  enc,dec,tok_args=td.load_frozen_tokenizer_from_pt_ckpt(str(tok_ckpt),device=device)
  del dec
  dyn,rew,policy,meta=load_dynamics_from_ckpt(str(MODELS/'coverage_aware/dynamics.pt'),device=device,
     d_bottleneck=64,n_latents=64,packing_factor=2)
  assert rew is not None and policy is not None
  models=[dyn,rew,policy]
  initial=torch.load(MODELS/'coverage_aware/dynamics.pt',map_location='cpu',weights_only=False,mmap=True)
  model_args=dict(initial['args']);initial_step=int(initial.get('step',180000));old_rms=initial.get('rms_state',{})
 for k,m in rms.items():
  source_key='lp' if k=='lpips' and 'lp' in old_rms else k
  if source_key in old_rms:m.load_state_dict(old_rms[source_key])
 for m in models:
  m.train()
  for p in m.parameters():p.requires_grad_(True)
 params=[p for m in models for p in m.parameters()]
 opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01,betas=(.9,.999))
 start=0
 if (folder/'latest.pt').exists():
  resume=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
  if stage=='tokenizer':models[0].load_state_dict(resume['model'])
  else:
   for m,k in zip(models,['dynamics','rew_head','policy_head']):m.load_state_dict(resume[k])
  opt.load_state_dict(resume['opt']);start=int(resume['optimizer_updates'])
  for k,m in rms.items():m.load_state_dict(resume['rms_state'][k])
 if start>args.updates:raise ValueError('Saved checkpoint is beyond the requested update budget')
 if world>1:models=[DDP(m,device_ids=[local],broadcast_buffers=False) for m in models]
 dataset=RawStreamDataset(args.setting,stage,seq_len=24,cache_size=2,samples_per_tile=16)
 if memfd_transport:
  from memfd_prefetch import MemfdDataLoader
  import memfd_prefetch
  cfg['data_loader_transport_source_sha256']=hashlib.sha256(Path(memfd_prefetch.__file__).read_bytes()).hexdigest()
  loader_class=MemfdDataLoader
 else:loader_class=DataLoader
 loader=loader_class(dataset,batch_size=micro,shuffle=False,num_workers=args.workers,pin_memory=True,
    drop_last=True,persistent_workers=args.workers>0,prefetch_factor=2 if args.workers else None)
 iterator=iter(loader)
 if rank==0:
  (RUN/'wandb').mkdir(parents=True,exist_ok=True)
  cfg['resume_optimizer_updates']=start
  cfg['dataset_audit']=dataset.audit
  atomic_json(folder/'config.json',cfg)
  smoke_key=f':w{world}:micro{micro}' if args.smoke else ''
  ident=hashlib.sha256(f'{GROUP}:{args.setting}:{stage}:{suffix}:seed0{smoke_key}'.encode()).hexdigest()[:12]
  run=wandb.init(entity=ENTITY,project=PROJECT,group=GROUP,id=ident,resume='allow',
   name=display_name(f'{args.setting}-{stage}-s0-node{args.node}'+(f'-SMOKE-w{world}-b{micro}' if args.smoke else '')),
   job_type='smoke' if args.smoke else stage,
   tags=[args.setting,stage,f'node{args.node}','seed0','seen10-unseen10',cfg['experiment_type'],'paper-budget'],
   config=cfg,dir=str(RUN/'wandb'),settings=wandb.Settings(init_timeout=90))
  run.config.update(cfg,allow_val_change=True)
  atomic_json(folder/'wandb.json',dict(url=run.url,id=run.id,entity=ENTITY,project=PROJECT))
 else:run=None
 if world>1:dist.barrier()
 t0=time.time()
 for update in range(start,args.updates):
  opt.zero_grad(set_to_none=True);loss_sum=0.;aux_sum={};input_wait_seconds=0.
  for mi in range(accum):
   input_started=time.perf_counter()
   try:b=next(iterator)
   except StopIteration:iterator=iter(loader);b=next(iterator)
   input_wait_seconds+=time.perf_counter()-input_started
   with ExitStack() as stack:
    if world>1 and mi<accum-1:
     for m in models:stack.enter_context(m.no_sync())
    if stage=='tokenizer':
     x=b['frames'].to(device,non_blocking=True);patches=temporal_patchify(x,14)
     with torch.autocast('cuda',dtype=torch.bfloat16):pred,mask,keep=models[0](patches)
     mse=tt.recon_loss_from_mae(pred,patches,mask)
     rms['mse'].update(mse)
     lper=tt.lpips_on_mae_recon(lp,pred,patches,mask,H=224,W=224,C=3,patch=14,subsample_frac=.5)
     rms['lpips'].update(lper)
     loss=rms['mse'].normalize(mse)+.2*rms['lpips'].normalize(lper)
     aux={'mse':mse.detach(),'lpips':lper.detach()}
    else:
     frames,actions,act_mask,rewards,lang=td.parse_batch(b,use_rewards=True,device=device)
     with torch.no_grad():
      zs=[]
      for si in range(0,len(frames),8):
       z,_=enc(temporal_patchify(frames[si:si+8],14));zs.append(z)
      z1=pack_bottleneck_to_spatial(torch.cat(zs),n_spatial=32,k=2)
     with torch.autocast('cuda',dtype=torch.bfloat16):
      loss,aux=td.dynamics_pretrain_loss(models[0],z1=z1,actions=actions,act_mask=act_mask,
        k_max=64,B_self=int(round(.25*micro)),step=initial_step+update,bootstrap_start=0,
        lang_emb=lang,rew_head=models[1],rewards=rewards,reward_weight=1.,
        policy_head=models[2],bc_weight=1.,rms_flow=rms['flow'],rms_rew=rms['rew'],rms_bc=rms['bc'])
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
    (loss/accum).backward()
   loss_sum+=float(loss.detach())/accum
   for k,v in aux.items():
    if torch.is_tensor(v) and v.numel()==1:aux_sum[k]=aux_sum.get(k,0.)+float(v)/accum
  norm=[]
  if stage=='tokenizer':
   norm=[float(torch.nn.utils.clip_grad_norm_(plain(models[0]).encoder.parameters(),1.)),
         float(torch.nn.utils.clip_grad_norm_(plain(models[0]).decoder.parameters(),1.))]
  else:
   norm=[float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.)) for m in models]
  if not all(math.isfinite(x) for x in norm):raise RuntimeError('Nonfinite gradient norm')
  # Fixed paper learning rate; preserve the dynamics trainer's 1k-update warmup.
  rate=1e-4*(min(1.,(update+1)/1000) if stage=='dynamics' else 1.)
  for pg in opt.param_groups:pg['lr']=rate
  opt.step()
  for m in rms.values():m.sync(world)
  completed=update+1
  if completed%20==0 or completed==start+1 or completed==args.updates:
   stats=torch.tensor([loss_sum],device=device)
   if world>1:dist.all_reduce(stats,op=dist.ReduceOp.AVG)
   if rank==0:
    row=dict(optimizer_updates=completed,global_step=initial_step+completed,loss=float(stats[0]),
      lr=rate,effective_batch=target_batch,gradient_accumulation=accum,
      seconds=time.time()-t0,session_start_update=start,session_optimizer_updates=completed-start,
      input_wait_seconds_rank0=input_wait_seconds,
      **{f'loss/{k}':v for k,v in aux_sum.items()})
    run.log(row,step=completed)
    atomic_json(folder/'progress.json',row)
    print('TRAIN',args.setting,stage,completed,'/',args.updates,'loss',row['loss'],flush=True)
  if completed%1000==0 or completed==args.updates:
   if rank==0:
    payload=dict(step=initial_step+completed,epoch=0,optimizer_updates=completed,
      opt=opt.state_dict(),args=model_args,experiment_config=cfg,
      rms_state={k:m.state_dict() for k,m in rms.items()})
    if stage=='tokenizer':payload['model']=cpu_state(models[0])
    else:
     for m,k in zip(models,['dynamics','rew_head','policy_head']):payload[k]=cpu_state(m)
    from checkpoint_safe import save_checkpoint
    save_checkpoint(payload,folder,completed)
   if world>1:dist.barrier()
 if rank==0:
  atomic_json(folder/'complete.json',dict(optimizer_updates=args.updates,completed=True))
  run.finish()
 if world>1:dist.destroy_process_group()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--setting',required=True,choices=SETTINGS);p.add_argument('--stage',required=True,choices=['tokenizer','dynamics'])
 p.add_argument('--experiment-node',dest='node',type=int,default=0);p.add_argument('--updates',type=int);p.add_argument('--micro_batch',type=int,default=0);p.add_argument('--workers',type=int,default=2);p.add_argument('--smoke',action='store_true')
 a=p.parse_args()
 if a.updates is None:a.updates=50000 if a.stage=='tokenizer' else 30000
 (RUN/'wandb').mkdir(parents=True,exist_ok=True);run(a)
