import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import random

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import config
import numpy as np
import torch


class ToyDataset(torch.utils.data.Dataset):
    def __len__(self):return 8
    def __getitem__(self,i):return {'x':torch.arange(6,dtype=torch.float32)+i,'i':torch.tensor(i)}


class Contracts(unittest.TestCase):
    def test_public_downloads_are_paired_and_pinned(self):
        r=subprocess.run([sys.executable,str(ROOT/'scripts/download_checkpoints.py'),'--dry-run'],capture_output=True,text=True,check=True)
        jobs=json.loads(r.stdout)
        self.assertEqual({j['filename']for j in jobs},{f'{v}/{s}.pt'for v in ['base','combined']for s in ['tokenizer','dynamics']})
        self.assertEqual(len({j['revision']for j in jobs}),1)
        self.assertTrue(all(len(j['sha256'])==64 for j in jobs))

    def test_online_configuration_is_only_alpha075(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'formal50').mkdir()
            (root/'formal50/snapshot.json').write_text(json.dumps({'snapshot_sha256':'snapshot','collection_protocol_sha256':'protocol'}))
            code="from round_config import activate,VARIANTS; import json; c=activate(); print(json.dumps(dict(variants=list(VARIANTS),config=c.config('wav-cem-alpha075'))))"
            env=dict(os.environ,MMBENCH2_ONLINE_RUN=str(root),PYTHONDONTWRITEBYTECODE='1')
            p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
            result=json.loads(p.stdout);cfg=result['config']
            self.assertEqual(result['variants'],['wav-cem-alpha075'])
            self.assertEqual(cfg['targeted_alpha'],.75)
            self.assertEqual(cfg['selected_episodes_per_task'],50)
            self.assertEqual((cfg['tokenizer_updates'],cfg['dynamics_updates']),(50000,30000))
            self.assertTrue(cfg['idm_frozen']);self.assertFalse(cfg['training_data_changes_during_run'])

    def test_tokenizer_idm_head_train_save_and_reload(self):
        from idm_tokenizer import InverseDynamicsHeadTokenizer
        from idm_hallucination import load_idm_tokenizer_head
        torch.manual_seed(3)
        model=InverseDynamicsHeadTokenizer(n_latents=2,d_bottleneck=4,d_hidden=16)
        before=next(model.parameters()).detach().clone()
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        x=torch.randn(2,1,2,4);y=torch.randn_like(x)
        loss=model(x,y).square().mean();loss.backward();opt.step()
        self.assertFalse(torch.equal(before,next(model.parameters())))
        model.eval()
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'idm.pt';torch.save({'model':model.state_dict(),'step':1},path)
            restored,meta=load_idm_tokenizer_head(str(path),device='cpu',n_latents=2,d_bottleneck=4)
            torch.testing.assert_close(model(x,y),restored(x,y))
            self.assertEqual(meta['step'],1)
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
            with self.assertRaises(ValueError):load_idm_tokenizer_head(str(path),device='cpu',n_latents=3,d_bottleneck=4)

    def test_action_cycle_masks_and_rollout_reset(self):
        from wav_action_cycle import masked_action_error,WAVActionCycleScorer
        requested=torch.zeros(4,3);predicted=torch.tensor([[1.,2.,999.]]).expand(4,-1)
        torch.testing.assert_close(masked_action_error(predicted,requested,torch.tensor([1,1,0])),torch.full((4,),2.5))
        with self.assertRaises(ValueError):masked_action_error(predicted,requested,torch.zeros(3))
        class Head(torch.nn.Module):
            def forward(self,a,b):return (b-a).flatten(2)[:,:,:3]
        scorer=WAVActionCycleScorer(None,None,Head())
        scorer.root_features=torch.zeros(1,1,1,3)
        scorer.bridge=lambda x:x[:,None]
        predictions=torch.ones(4,2,1,3)
        mask=torch.ones(3)
        scorer.begin_rollout(torch.zeros(4,1,3))
        first=scorer.score_transition(predictions,torch.zeros(4,1,3),requested,mask)
        scorer.begin_rollout(torch.zeros(4,1,3))
        repeated=scorer.score_transition(predictions,torch.zeros(4,1,3),requested,mask)
        torch.testing.assert_close(first,repeated)
        torch.testing.assert_close(first,torch.ones(4))

    def test_stream_action_alignment_boundaries_and_alpha(self):
        import stream_data as offline
        import online_stream as online
        from PIL import Image
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'selections').mkdir()
            names=['target0','target1','background']
            (root/'tasks.json').write_text(json.dumps({n:{'action_dim':2,'text_embedding':[0.,1.]}for n in names}))
            n=60;actions=torch.arange(n,dtype=torch.float32)[:,None].expand(-1,16).clone()/1000
            rewards=torch.arange(n,dtype=torch.float32);actions[[0,30]]=float('nan');rewards[[0,30]]=float('nan')
            tile=np.zeros((224,224*n,3),dtype=np.uint8)
            for i in range(n):tile[:,i*224:(i+1)*224]=i
            image=root/'tile.png';Image.fromarray(tile).save(image)
            data=dict(actions=actions,rewards=rewards,spans=[(0,0,30),(1,30,60)],files=[str(image)],bounds=[0,60],n_frames=60)
            for module,part,setting,alpha in [(offline,'active_random','wav-h4-d0125',.5),(online,'active_wav_actioncycle_cem_v1','wav-cem-alpha075',.75)]:
                index={t:{'expert':data,**({part:data}if t!='background'else{})}for t in names}
                torch.save(index,root/'raw-index.pt')
                episodes=[dict(task=t,partition=part,episode=1)for t in names[:2]]
                (root/'selections'/f'{setting}.json').write_text(json.dumps({'episodes':episodes}))
                replacements=dict(RUN=root,REPO=root,ALL_TASKS=names,TARGETS=names[:2])
                if module is online:replacements['config']=lambda setting:{'targeted_alpha':.75,'selected_episodes_per_task':50}
                with patch.multiple(module,**replacements):
                    loader_patch=patch.object(online,'load_online_index',return_value=index)if module is online else patch.object(offline,'DATA',root)
                    with loader_patch:
                        ds=module.RawStreamDataset(setting,'dynamics')
                        self.assertAlmostEqual(sum(ds.weights[:2]),alpha)
                        seg=next(s for s in ds.segments[0]if s['part']==part)
                        self.assertTrue(all(30<=s<=35 for s in seg['starts'].tolist()))
                        ds.current=(seg,*seg['tile_ranges'][0]);ds.remaining=1;ds.rng=random.Random(2)
                        sample=ds[0];start=int(sample['obs'][0,0,0,0])
                        self.assertEqual(tuple(sample['obs'].shape),(25,3,224,224))
                        self.assertTrue(torch.isfinite(sample['act']).all())
                        self.assertAlmostEqual(float(sample['act'][0,0]),(start+1)/1000,places=6)
                        self.assertEqual(float(sample['rew'][0]),start+1)
                        self.assertEqual(float(sample['act'][:,2:].abs().sum()),0.)

    def test_checkpoint_save_keeps_previous_and_immutable_final(self):
        from checkpoint_safe import save_checkpoint
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);w=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.Adam([w]);w.square().backward();opt.step()
            for step in [1,2]:
                payload=dict(model={'w':w.detach().clone()},opt=opt.state_dict(),rms_state={'loss':{}},optimizer_updates=step,experiment_config={'optimizer_updates':2})
                save_checkpoint(payload,root,step)
            self.assertEqual(torch.load(root/'latest.previous.pt',weights_only=False)['optimizer_updates'],1)
            self.assertEqual(torch.load(root/'latest.pt',weights_only=False)['optimizer_updates'],2)
            self.assertTrue((root/'update_000002.pt').exists())

    @unittest.skipUnless(sys.platform=='linux','memfd transport is Linux-specific')
    def test_memfd_prefetch_matches_normal_batches(self):
        from memfd_prefetch import MemfdDataLoader
        expected=list(torch.utils.data.DataLoader(ToyDataset(),batch_size=2,num_workers=0))
        actual=list(MemfdDataLoader(ToyDataset(),batch_size=2,num_workers=2,pin_memory=False))
        self.assertEqual(len(actual),len(expected))
        for a,b in zip(actual,expected):
            for key in a:torch.testing.assert_close(a[key],b[key])

    def test_mpc_episode_cache_resumes_without_duplicates(self):
        from mpc_episodes import run_isolated,save
        from mpc_validation import PROTOCOL,valid_result
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);out=root/'mpc/model.task.json';launched=[]
            def launch(ep):
                launched.append(ep)
                row=dict(task='task',variant='model',episode=ep,seed=731+ep,normalized_score=.2,completed_episode=True,steps=4)
                save(root/'mpc-episode-cache'/f'model.task.ep{ep}.json',dict(episodes=[row],protocol=PROTOCOL,mean_normalized_score=.2))
            run_isolated('task','model',out,launch);self.assertEqual(launched,[0,1,2])
            run_isolated('task','model',out,lambda ep:self.fail('Repeated a completed episode'))
            self.assertTrue(valid_result(json.loads(out.read_text()),'task','model'))

if __name__=='__main__':unittest.main()
