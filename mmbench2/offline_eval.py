"""Reuse the original reconstruction, rollout and attribution evaluators.

Paper inference d=.125 overrides CLI default d=.25. Expert and human test
trajectories have equal weight. Test sets never select IDM hyperparameters.
"""
import argparse
from types import SimpleNamespace
import math
import numpy as np
import torch
from common import *
import train_dynamics as td
import train_tokenizer as tt
from model import Tokenizer
from interactive import load_dynamics_from_ckpt

def evaluate(task,variant):
    torch.manual_seed(731);np.random.seed(731)
    device=torch.device('cuda')
    enc,dec,cfg=td.load_frozen_tokenizer_from_pt_ckpt(str(MODELS/variant/'tokenizer.pt'),device=device)
    dyn,rew,pol,meta=load_dynamics_from_ckpt(str(MODELS/variant/'dynamics.pt'),device=device,
        d_bottleneck=cfg['d_bottleneck'],n_latents=cfg['n_latents'],packing_factor=2)
    del rew,pol
    data=torch.load(OUT/'eval-data'/(task+'.pt'),weights_only=False)
    n=len(data['frames']);batches=[]
    for s in range(0,n,4):
        f=data['frames'][s:s+4].to(device).float()/255
        a=data['actions'][s:s+4].to(device)
        m=data['mask'].to(device).view(1,1,16).expand(len(f),24,-1)
        l=data['lang'].to(device).view(1,-1).expand(len(f),-1)
        batches.append((f,a,m,l))
    records=[]
    # Capture numeric W&B outputs locally; preserve the original numerical
    # evaluation functions, while disabling only their image-upload side effect.
    original_log=td.wandb.log
    td.wandb.log=lambda payload,**kw: records.append({k:float(v) for k,v in payload.items() if isinstance(v,(int,float))})
    td.log_dynamics_eval_wandb=lambda **kw:None
    tok=Tokenizer(enc,dec).eval()
    lpips_fn=None
    try:
        import lpips
        lpips_fn=lpips.LPIPS(net='alex').to(device).eval()
    except Exception as e:
        save(f'offline/{variant}.{task}.lpips-error.json',dict(error=str(e)))
    args=SimpleNamespace(patch=cfg['patch'],H=224,W=224,C=3,lpips_weight=.2,lpips_frac=1.)
    with torch.no_grad():
        recon,_,_=tt.validate(model=tok,val_loader=[b[0] for b in batches],device=device,use_amp=True,
            lpips_fn=lpips_fn,args=args,max_batches=len(batches),key_prefix='reconstruction')
        for i,(f,a,m,l) in enumerate(batches):
            torch.manual_seed(10000+i)
            td.run_dynamics_eval(encoder=enc,decoder=dec,dyn=dyn,frames=f,actions=a,act_mask=m,
                H=224,W=224,C=3,patch=cfg['patch'],packing_factor=2,k_max=meta['k_max'],
                ctx_length=8,horizon=16,sched=td.make_tau_schedule(k_max=meta['k_max'],schedule='shortcut',d=.125),
                max_items=0,step=i,tau_ctx=.1,lang_emb=l,tag='rollout')
            if i%8==0:log('rollout',variant,task,i,'/',len(batches))
        # Chunking keeps original pool-level shuffling and a bounded pool size.
        # Each task is a separate pool, so shuffled actions remain in the task's
        # actual embodiment and mask. Report this explicitly in the protocol.
        td.run_action_attribution_eval(encoder=enc,dyn=dyn,batches=batches,
            patch=cfg['patch'],packing_factor=2,k_max=meta['k_max'],tau_ctx=.1,
            step=0,tag='action',rng=torch.Generator(device='cpu').manual_seed(12345))
    td.wandb.log=original_log
    rollout=[r for r in records if 'rollout/mse_pred' in r]
    mp=float(np.mean([r['rollout/mse_pred'] for r in rollout]))
    mf=float(np.mean([r['rollout/mse_floor'] for r in rollout]))
    from checkpoint_identity import fingerprints
    summary=dict(checkpoint_fingerprints=fingerprints(MODELS/variant),task=task,split='seen' if task in SEEN_TASK_SET else 'unseen',variant=variant,
        windows=n,test_episodes=len({(m['partition'],m['episode']) for m in data['manifest']}),
        reconstruction=recon,rollout_mse=mp,repeat_mse=mf,
        rollout_psnr=-10*math.log10(max(mp,1e-12)),rollout_delta_psnr=10*math.log10(max(mf,1e-12)/max(mp,1e-12)),
        attribution=next(r for r in records if 'action/action_shuffle_ratio' in r),
        context=8,horizon=16,denoising_steps=8,tau_ctx=.1,
        action_attribution_schedule_steps=1,
        action_attribution_note='original default one-step shortcut d=1; same-task batch/time/zero interventions')
    save(f'offline/{variant}.{task}.json',summary)
    save(f'offline/{variant}.{task}.raw.json',records)
    log('complete offline',variant,task,'deltaPSNR',summary['rollout_delta_psnr'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--variant',required=True)
    a=p.parse_args();evaluate(a.task,a.variant)
