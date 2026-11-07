import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

class ExperimentTests(unittest.TestCase):
    def test_only_requested_offline_methods_and_target_split(self):
        self.assertEqual(set(config.SETTINGS), {'wav-h4-d0125', 'random-pool50', 'ur-norm-pool50', 'uf-norm-pool50', 'us-norm-pool50'})
        self.assertEqual(len(config.SEEN_TASK_SET), 10)
        self.assertEqual(len(config.UNSEEN_TASK_SET), 10)
        self.assertFalse(set(config.SEEN_TASK_SET) & set(config.UNSEEN_TASK_SET))

    def test_metadata_covers_every_seen_and_unseen_task(self):
        from common import metadata
        for task in config.TARGETS:
            row = metadata(task)
            self.assertIsNotNone(row, task)
            self.assertTrue(1 <= row['action_dim'] <= 16, task)
            self.assertTrue(len(row['text_embedding']) > 0, task)
            self.assertTrue(np.isfinite(row['text_embedding']).all(), task)

    def test_same_budget_and_training_only_candidate_pool(self):
        from selection import candidate_pool, choose_episodes
        index = {'task': {part: {'spans': [(i, i * 30, (i + 1) * 30) for i in range(50)]} for part in config.SOURCES}}
        index['task']['test'] = {'spans': [(999, 0, 30)]}
        pool = candidate_pool(index, 'task')
        self.assertEqual(len(pool), 250)
        for i, row in enumerate(pool):
            row.update(wav_h4=i, u_r=250-i, u_f=i, u_s=i)
        for setting in config.SETTINGS:
            method = config.SETTINGS[setting]['method']
            chosen = choose_episodes(pool, method, seed=731)
            self.assertEqual(len(chosen), 50)
            self.assertEqual(len({(r['partition'], r['episode']) for r in chosen}), 50)
            self.assertTrue(all(r['partition'] in config.SOURCES for r in chosen))
            if method != 'random':
                self.assertEqual([r[method] for r in chosen], sorted([r[method] for r in pool], reverse=True)[:50])
            else:
                self.assertEqual(chosen, choose_episodes(pool, method, seed=731))
        index['task'][config.SOURCES[0]]['spans'].pop()
        with self.assertRaises(ValueError): candidate_pool(index, 'task')

    def test_random_selection_needs_no_scores_or_idm(self):
        import selection
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = {'task': {part: {'spans': [(i, 0, 30) for i in range(50)]} for part in config.SOURCES}}
            torch.save(index, root / 'raw-index.pt')
            with patch.multiple(selection, RUN=root, TARGETS=['task']):
                selection.select(['random-pool50'])
            out = json.loads((root / 'selections/random-pool50.json').read_text())
            self.assertEqual(len(out['episodes']), 50)
            self.assertFalse((root / 'scores').exists())

    def test_two_stage_model_update_and_strict_loader_roundtrip(self):
        from model import Encoder, Decoder, Tokenizer, Dynamics, RewardHeadMTP, PolicyHeadMTP
        from model import temporal_patchify, pack_bottleneck_to_spatial, recon_loss_from_mae
        from train_dynamics import dynamics_pretrain_loss, load_frozen_tokenizer_from_pt_ckpt
        from interactive import load_dynamics_from_ckpt
        torch.manual_seed(13)
        shared = dict(d_model=16, n_heads=2, depth=1, n_latents=2, n_patches=4, d_bottleneck=4, time_every=1)
        tokenizer = Tokenizer(Encoder(patch_dim=12, mae_p_min=.5, mae_p_max=.8, **shared), Decoder(d_patch=12, **shared))
        frames = torch.rand(2, 4, 3, 4, 4)
        patches = temporal_patchify(frames, 2)
        before = {k: v.clone() for k, v in tokenizer.state_dict().items()}
        opt = torch.optim.AdamW(tokenizer.parameters(), lr=1e-3)
        pred, mask, keep = tokenizer(patches)
        loss = recon_loss_from_mae(pred, patches, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward(); opt.step()
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in tokenizer.state_dict().items()))
        tokenizer.eval().requires_grad_(False)
        # The frozen inference loader disables MAE; match that inference mode.
        tokenizer.encoder.mae.p_min = tokenizer.encoder.mae.p_max = 0.
        z, _ = tokenizer.encoder(patches)
        packed = pack_bottleneck_to_spatial(z, n_spatial=1, k=2)
        dyn = Dynamics(d_model=16, d_bottleneck=4, d_spatial=8, n_spatial=1, n_register=1, n_agent=1, n_heads=2, depth=1, k_max=4, time_every=1, lang_dim=2)
        rew = RewardHeadMTP(d_model=16, L=2, num_bins=11)
        policy = PolicyHeadMTP(d_model=16, L=2, act_dim_max=16)
        models = [dyn, rew, policy]
        opt = torch.optim.AdamW([p for m in models for p in m.parameters()], lr=1e-3)
        actions = torch.zeros(2, 4, 16); actions[:, 1:, :2] = .3
        act_mask = torch.zeros_like(actions); act_mask[:, :, :2] = 1
        before_dyn = {k: v.clone() for k, v in dyn.state_dict().items()}
        loss, aux = dynamics_pretrain_loss(dyn, z1=packed.detach(), actions=actions, act_mask=act_mask,
            k_max=4, B_self=0, step=0, bootstrap_start=0, lang_emb=torch.zeros(2, 2),
            rew_head=rew, rewards=torch.rand(2, 4), reward_weight=1., policy_head=policy, bc_weight=1.)
        self.assertTrue(torch.isfinite(loss))
        loss.backward(); opt.step()
        self.assertTrue(any(not torch.equal(v, before_dyn[k]) for k, v in dyn.state_dict().items()))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tok_args = dict(H=4, W=4, C=3, patch=2, d_model=16, n_heads=2, depth=1, n_latents=2, d_bottleneck=4, time_every=1)
            torch.save(dict(model=tokenizer.state_dict(), args=tok_args), root / 'tokenizer.pt')
            enc, dec, args = load_frozen_tokenizer_from_pt_ckpt(str(root / 'tokenizer.pt'), device=torch.device('cpu'))
            torch.testing.assert_close(enc(patches)[0], tokenizer.encoder(patches)[0])
            dyn_args = dict(d_model_dyn=16, n_heads=2, dyn_depth=1, n_register=1, n_agent=1, lang_dim=2, k_max=4, time_every=1, reward_L=2, reward_num_bins=11, bc_L=2, bc_act_dim=16)
            torch.save(dict(dynamics=dyn.state_dict(), rew_head=rew.state_dict(), policy_head=policy.state_dict(), args=dyn_args), root / 'dynamics.pt')
            restored, restored_rew, restored_pol, args = load_dynamics_from_ckpt(str(root / 'dynamics.pt'), device=torch.device('cpu'), d_bottleneck=4, n_latents=2, packing_factor=2)
            for a, b in zip([dyn, rew, policy], [restored, restored_rew, restored_pol]):
                for key, value in a.state_dict().items(): torch.testing.assert_close(value, b.state_dict()[key])

    def test_cem_receives_masked_candidate_actions_and_independent_rollouts(self):
        import curiosity_wav as cem
        from wav_action_cycle import WAVActionCycleScorer
        class Head(torch.nn.Module):
            def forward(self, a, b): return (b-a).flatten(2)
        scorer = WAVActionCycleScorer(None, None, Head())
        scorer.root_features = torch.zeros(1, 1, 1, 3)
        scorer.bridge = lambda x: x[:, None]
        def predict(dyn, *, past_packed, actions, **kw):
            return past_packed[:, -1] + actions[:, -1, None, :]
        dyn = torch.nn.Linear(1, 1)
        with patch.object(cem, 'sample_one_timestep_packed', side_effect=predict):
            result = cem.curiosity_mpc_action(dyn, past_packed=torch.zeros(1, 1, 1, 3), past_actions=torch.zeros(1, 1, 3), scorer=scorer, k_max=4, sched={}, act_mask=torch.tensor([1., 1., 0.]), n_candidates=6, horizon=3, n_samples=2, n_elite=2, n_cem_iters=2, action_dim=3, candidate_batch_size=2, device=torch.device('cpu'))
        self.assertEqual(tuple(result['action_sequence'].shape), (3, 3))
        self.assertEqual(float(result['action_sequence'][:, 2].abs().sum()), 0.)
        self.assertLess(float(result['uncertainty']), 1e-10)

    def test_mpc_cache_rejects_different_checkpoint_pair(self):
        from mpc_episodes import run_isolated, save
        from mpc_validation import PROTOCOL
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/'mpc/model.task.json'; launched=[]
            def run(identity):
                def launch(ep):
                    launched.append(ep)
                    row=dict(task='task',variant='model',episode=ep,seed=731+ep,normalized_score=.2,completed_episode=True,steps=4)
                    save(root/'mpc-episode-cache'/f'model.task.ep{ep}.json',dict(episodes=[row],protocol=PROTOCOL,checkpoint_fingerprints=identity,mean_normalized_score=.2))
                run_isolated('task','model',out,launch,checkpoint_fingerprints=identity)
            run({'tokenizer': 'a', 'dynamics': 'a'})
            run({'tokenizer': 'a', 'dynamics': 'a'})
            self.assertEqual(launched, [0,1,2])
            run({'tokenizer': 'a', 'dynamics': 'b'})
            self.assertEqual(launched, [0,1,2,0,1,2])

    def test_changed_evaluation_files_cannot_reuse_bound_run(self):
        from checkpoint_identity import fingerprints, bind
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'tokenizer.pt').write_bytes(b'a'); (root/'dynamics.pt').write_bytes(b'b')
            a = fingerprints(root); bind(root/'evaluation','model',a)
            (root/'dynamics.pt').write_bytes(b'c')
            with self.assertRaises(ValueError): bind(root/'evaluation','model',fingerprints(root))

if __name__ == '__main__':
    unittest.main()
