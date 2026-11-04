"""Small offline integration test; random fixtures are not benchmark results."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from ruamel.yaml import YAML

from datasets.build_pools import build
from datasets.data_selection import (
    SelectionRequest, iter_chunks, load_episode, load_idm, load_world_model,
    predict_idm_action, read_pool_refs, run_selection, _obs_space_from_episode,
)

ROOT = Path(__file__).resolve().parents[1]


def make_fixtures(root):
    rng = np.random.default_rng(4)
    rollout_dir = root / 'stages/update_000001/rollouts'
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir.parent / 'collection.json').write_text(json.dumps({'horizon': 6}))
    for i in range(4):
        path = rollout_dir / f'episode_{i}.npz'
        np.savez(path, state=rng.normal(size=(6, 9, 2)).astype('float32'),
                 action=rng.uniform(-1, 1, (6, 7, 2)).astype('float32'),
                 agentview_image=rng.integers(0, 256, (6, 64, 64, 3, 2), dtype='uint8'),
                 reward=np.zeros(6, dtype='float32'),
                 is_first=np.array([True] + [False] * 5),
                 is_terminal=np.array([False] * 5 + [True]))
        path.with_suffix('.json').write_text(json.dumps({'stable5_success': i % 2 == 0}))
    expert = root / 'expert.hdf5'
    with h5py.File(expert, 'w') as f:
        f.create_dataset('mask/train', data=np.array([b'demo_2', b'demo_0']))
        f.create_dataset('mask/valid', data=np.array([b'demo_1']))
        for i in range(3):
            d = f.create_group(f'data/demo_{i}')
            d.create_dataset('actions', data=rng.normal(size=(6, 7)).astype('float32'))
            d.create_dataset('rewards', data=np.zeros(6, dtype='float32'))
            d.create_dataset('dones', data=np.array([False] * 5 + [True]))
            for k, dim in [('robot0_eef_pos', 3), ('robot0_eef_quat', 4), ('robot0_gripper_qpos', 2)]:
                d.create_dataset('obs/' + k, data=rng.normal(size=(6, dim)).astype('float32'))
            d.create_dataset('obs/agentview_image', data=rng.integers(0, 256, (6, 64, 64, 3), dtype='uint8'))
    pools = root / 'pools'
    summary = build('can', root / 'stages', expert, pools)
    return pools, summary


def tiny_idm_config():
    sys.path.insert(0, str(ROOT / 'IDM'))
    from hydra import compose, initialize_config_dir
    import udrm.resolvers
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'IDM/udrm/cfg')):
        return compose(config_name='train_st_vivit_clam_stm', overrides=[
            'env=robomimic_sailor', 'env.image_obs=True', 'data.seq_len=3',
            'encoder.dim_model=32', 'encoder.dim_feedforward=64', 'decoder.dim_feedforward=64',
            'encoder.n_layers=1', 'decoder.n_layers=1', 'model.fdm.detach_idm_latents=false',
            'model.action_decoder.embedding_dim=32', 'model.action_decoder.net.hidden_dims=[32]'])


class OfflinePipeline(unittest.TestCase):
    def test_pools_wm_training_checkpoints_and_four_selectors(self):
        torch.manual_seed(4)
        device = os.environ.get('WAV_SMOKE_DEVICE', 'cpu')
        from wav.dreamer.wm import WorldModel
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pools, summary = make_fixtures(root)
            self.assertEqual((summary['rollouts'], summary['expert_train'], summary['expert_valid']), (4, 2, 1))
            refs = read_pool_refs(pools / 'sample_pool.jsonl')
            episode = load_episode(refs[0])
            cfg = YAML(typ='safe').load((ROOT / 'wav/configs.yaml').read_text())['defaults']
            cfg.update(device=device, state_only=False, compile=False, precision=32,
                       num_actions=7, action_dim=7, state_dim=9, pred_horizon=2,
                       dyn_deter=32, dyn_hidden=32, dyn_stoch=4, dyn_discrete=4, units=32,
                       batch_length=4, log_openloop_img_pred=False)
            for key in ['encoder', 'decoder']:
                cfg[key].update(cnn_depth=2, mlp_layers=1, mlp_units=32)
            cfg['cont_head']['layers'] = 1
            cfg['reward_head']['layers'] = 1
            config_path = root / 'wm.yaml'
            with config_path.open('w') as f:
                YAML().dump(cfg, f)
            wm = WorldModel(_obs_space_from_episode(episode, False), 0, SimpleNamespace(**cfg)).to(device)
            old = root / 'old.pt'
            torch.save({'_wm.' + k: v for k, v in wm.state_dict().items()}, old)
            before = next(iter(wm.encoder.parameters())).detach().clone()
            batch = next(iter_chunks(episode, 4))
            for _ in range(2):
                _, _, metrics = wm._train(batch)
                self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in metrics.values()))
            self.assertFalse(torch.equal(before, next(iter(wm.encoder.parameters()))))
            new = root / 'new.pt'
            torch.save({'_wm.' + k: v for k, v in wm.state_dict().items()}, new)
            loaded, _ = load_world_model(str(new), episode, device, False, str(config_path))
            for k, v in wm.state_dict().items():
                torch.testing.assert_close(v, loaded.state_dict()[k])
            broken = dict(torch.load(new, weights_only=True, map_location='cpu'))
            broken.pop(next(iter(broken)))
            torch.save(broken, root / 'broken.pt')
            with self.assertRaises(RuntimeError):
                load_world_model(str(root / 'broken.pt'), episode, device, False, str(config_path))

            idm_cfg = tiny_idm_config()
            from udrm.models.clam.space_time_clam_STM import SpaceTimeCLAM
            from udrm.models.mlp_policy import MLPPolicy
            idm = SpaceTimeCLAM(idm_cfg.model, (3, 64, 64), idm_cfg.model.la_dim).to(device)
            decoder = MLPPolicy(idm_cfg.model.action_decoder, idm_cfg.model.la_dim, 7).to(device)
            obs = torch.rand(1, 3, 3, 64, 64, device=device)
            optimizer = torch.optim.Adam(list(idm.parameters()) + list(decoder.parameters()))
            reconstruction = idm(obs, timesteps=torch.arange(3, device=device)[None], states=None)
            action = predict_idm_action(idm, decoder, idm_cfg, obs)
            loss = (reconstruction.reconstructed_obs - obs[:, 1:]).square().mean() + action.square().mean()
            loss.backward()
            optimizer.step()
            idm_path, idm_config_path = root / 'idm.pkl', root / 'idm.yaml'
            torch.save({'model': idm.state_dict(), 'action_decoder': decoder.state_dict(),
                        'input_dim': (3, 64, 64), 'opt': optimizer.state_dict()}, idm_path)
            OmegaConf.save(idm_cfg, idm_config_path)
            idm.eval(); decoder.eval()
            loaded_idm, loaded_decoder, loaded_cfg = load_idm(str(idm_path), str(idm_config_path), device)
            with torch.no_grad():
                torch.testing.assert_close(predict_idm_action(idm, decoder, idm_cfg, obs),
                                           predict_idm_action(loaded_idm, loaded_decoder, loaded_cfg, obs))
            kwargs = dict(device=device, idm_device=device, state_only=False, batch_length=4,
                          wm_config_path=str(config_path), wm_new_ckpt_path=str(new), wm_old_ckpt_path=str(old),
                          wm_ckpt_paths=[str(old), str(new)], wm_ckpt_path=str(new),
                          idm_ckpt_path=str(idm_path), idm_config_path=str(idm_config_path))
            for strategy in ['random', 'progress', 'curiosity', 'idm']:
                with self.subTest(strategy=strategy):
                    result = run_selection(SelectionRequest(pools / 'sample_pool.jsonl',
                        root / f'selected_{strategy}.jsonl', strategy, 2, seed=4, strategy_kwargs=kwargs))
                    self.assertEqual(len(result.selected_refs), 2)
                    self.assertEqual(result.metadata['num_scored'], 4)
                    self.assertNotIn('fallback', result.metadata)
                    self.assertTrue(all(np.isfinite(v) for v in result.scores.values()))
                    ranked = sorted(result.scores, key=result.scores.get, reverse=True)[:2]
                    self.assertEqual([r['episode_id'] for r in result.selected_refs], ranked)
            # One snapshot cannot act as an ensemble, even if the path is repeated.
            kwargs['wm_ckpt_paths'] = [str(new), str(new)]
            result = run_selection(SelectionRequest(pools / 'sample_pool.jsonl',
                root / 'fallback.jsonl', 'curiosity', 2, strategy_kwargs=kwargs))
            self.assertEqual(result.metadata['fallback'], 'random')


if __name__ == '__main__':
    unittest.main()
