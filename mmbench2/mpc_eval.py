"""Real-environment driver around the released CEM implementation.

The repository releases only an offline planning entrypoint. This adapter adds
reset/step/replan and uses the environment's own normalized info['score'].
Unspecified elite/std/noise choices are recorded, not presented as recovered
paper settings. The paper H/K/population/iterations/rollouts/BC prior are fixed.
"""
import argparse
import gc, hashlib
from mpc_renderer_runtime import provenance
import time
import numpy as np
import torch
from common import *
import plan_cem as pc
from env_wrapper import EnvCollector
from train_dynamics import load_frozen_tokenizer_from_pt_ckpt,make_tau_schedule
from model import temporal_patchify,pack_bottleneck_to_spatial
from interactive import load_dynamics_from_ckpt

def run(task,variant,episodes=3,max_steps=0,smoke=False,episode_indices=None,output_name=None):
    from checkpoint_identity import fingerprints
    identity=fingerprints(MODELS/variant)
    device=torch.device('cuda')
    enc,dec,cfg=load_frozen_tokenizer_from_pt_ckpt(str(MODELS/variant/'tokenizer.pt'),device=device)
    del dec
    dyn,rew,pol,meta=load_dynamics_from_ckpt(str(MODELS/variant/'dynamics.pt'),device=device,
        d_bottleneck=cfg['d_bottleneck'],n_latents=cfg['n_latents'],packing_factor=2)
    if rew is None or pol is None:raise ValueError('MPC requires released reward and BC heads')
    sched=make_tau_schedule(k_max=meta['k_max'],schedule='shortcut',d=.125)
    ad=int(metadata(task)['action_dim']);mask=torch.zeros(16,device=device);mask[:ad]=1
    lang=torch.tensor(metadata(task)['text_embedding'],device=device).view(1,-1)
    rows=[]
    # Match the paper's context corruption; the released offline planning helper
    # hardcodes zero. Keep all CEM code otherwise unchanged and record this override.
    original_sampler=pc.sample_one_timestep_packed
    def sampler(*args,**kw):
        kw['tau_ctx']=.1
        return original_sampler(*args,**kw)
    pc.sample_one_timestep_packed=sampler
    for ep in (range(episodes) if episode_indices is None else episode_indices):
        if not 0 <= ep < episodes:raise ValueError('Episode index is outside the configured seed range')
        seed=731+ep;np.random.seed(seed);torch.manual_seed(seed)
        e=EnvCollector(task,seed=seed)
        obs,info=e.env.reset();total=0.;steps=0;plans=[];t0=time.time();done=False
        generator=torch.Generator(device=device).manual_seed(seed)
        try:
            while not done and (max_steps==0 or steps<max_steps):
                rgb=e._extract_rgb(obs)
                if rgb.shape != (3,224,224) or rgb.dtype != np.uint8:
                    raise ValueError('Invalid renderer RGB shape or dtype')
                if task in {'cup-catch','cup-catch-var1','finger-turn-easy','finger-turn-easy-var1','mw-push'}:
                    vertical_variation=float(np.abs(np.diff(rgb.astype(np.int16),axis=1)).mean())
                    if rgb.std()<1e-6 or vertical_variation<1e-6:
                        raise RuntimeError('Invalid renderer output: black frame or repeated vertical bars')
                x=torch.from_numpy(rgb).to(device).float().view(1,1,3,224,224)/255
                with torch.no_grad():
                    z=enc(temporal_patchify(x,cfg['patch']))[0]
                    z=pack_bottleneck_to_spatial(z,n_spatial=cfg['n_latents']//2,k=2)[0,0]
                    horizon=2 if smoke else 32
                    _,mu=pc.bc_rollout_integrated(pol,dyn,z_start=z,horizon=horizon,act_dim=ad,
                        act_mask_1d=mask,sched=sched,k_max=meta['k_max'],lang_emb=lang,use_amp=True)
                    out=pc.cem_plan(dyn,z_start=z,z_goal=z,horizon=horizon,act_dim=ad,
                        act_mask_1d=mask,sched=sched,k_max=meta['k_max'],lang_emb=lang,
                        cost_kind='reward',n_samples=4 if smoke else 32,n_elites=2 if smoke else 8,
                        n_iters=1 if smoke else 3,init_std=.3,min_std=.05,rollout_batch=64,
                        noise_beta=0.,device=device,use_amp=True,generator=generator,
                        init_mu=mu,n_rollouts_elite=1 if smoke else 2,rew_head=rew,discount=1.)
                if not torch.isfinite(out['best_plan']).all().item():
                    raise FloatingPointError('Non-finite CEM actions; no substitute action executed')
                plans.append(dict(step=steps,predicted_cost=out['best_cost']))
                stride=1 if smoke else 16
                for action in out['best_plan'][:stride]:
                    obs,r,term,trunc,info=e.env.step(action.detach().cpu().numpy().astype(np.float32))
                    total+=float(r);steps+=1;done=bool(term or trunc)
                    if done or (max_steps>0 and steps>=max_steps):break
                log('MPC',variant,task,'ep',ep,'step',steps,'score',info.get('score'),'seconds',round(time.time()-t0,1))
            if 'score' not in info:raise ValueError('No normalized environment score; do not invent normalization')
            row=dict(task=task,variant=variant,episode=ep,seed=seed,steps=steps,
                raw_return=total,normalized_score=float(info['score']),seconds=time.time()-t0,
                completed_episode=done,plans=plans,renderer=provenance(task))
            rows.append(row)
            save(output_name or (f'mpc/{variant}.{task}'+('.smoke' if smoke else '')+'.json'),dict(episodes=rows,checkpoint_fingerprints=identity,
                mean_normalized_score=float(np.mean([x['normalized_score'] for x in rows])),
                protocol=dict(horizon=horizon,replan_stride=stride,population=4 if smoke else 32,
                    iterations=1 if smoke else 3,rollouts_per_candidate=1 if smoke else 2,
                    denoising_steps=8,bc_warm_start=True,elite_count=2 if smoke else 8,
                    init_std=.3,min_std=.05,noise_beta=0.,discount=1.,tau_ctx=.1,
                    context='current observed frame, matching released CEM helper',
                    implementation='new real-environment loop around released plan_cem helpers',
                    renderer=provenance(task),
                    unpublished_settings=['elite_count','init_std','min_std','noise_beta','discount','MPC context length'])))
        finally:
            e.close()
            del e
            gc.collect()
    pc.sample_one_timestep_packed=original_sampler

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--variant',required=True)
    p.add_argument('--episodes',type=int,default=3);p.add_argument('--max_steps',type=int,default=0);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.task,a.variant,a.episodes,a.max_steps,a.smoke)
