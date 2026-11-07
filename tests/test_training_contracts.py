import json
from pathlib import Path
import tempfile
import unittest
import sys

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.episode_io import action_dim, load_rollout, official_demo_split
from datasets.data_selection import predict_idm_action


class TrainingContracts(unittest.TestCase):
    def test_action_loss_updates_inverse_and_decoder_without_fdm(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'IDM'))
        from hydra import compose, initialize_config_dir
        from udrm.models.clam.space_time_clam_STM import SpaceTimeCLAM
        from udrm.trainers.clam_trainer import CLAMTrainer
        from types import SimpleNamespace
        import udrm.resolvers
        config_dir = str(Path(__file__).resolve().parents[1] / 'IDM/udrm/cfg')
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name='train_wav_can_joint', overrides=[
                'encoder.dim_model=32', 'encoder.dim_feedforward=64',
                'decoder.dim_feedforward=64', 'encoder.n_layers=1', 'decoder.n_layers=1'])
        trainer = CLAMTrainer.__new__(CLAMTrainer)
        trainer.cfg, trainer.use_transformer = cfg, True
        trainer.model = SpaceTimeCLAM(cfg.model, input_dim=(3, 64, 64), la_dim=16)
        trainer.action_decoder = torch.nn.Linear(16, 7)
        trainer.action_decoder_loss_fn = torch.nn.MSELoss(reduction='none')
        batch = SimpleNamespace(observations=torch.rand(2, 3, 3, 64, 64),
                                actions=torch.rand(2, 3, 7), timestep=torch.arange(3)[None].expand(2, -1), states=None)
        old_inverse = trainer.model.idm.la_head.weight.detach().clone()
        old_decoder = trainer.action_decoder.weight.detach().clone()
        optimizer = torch.optim.Adam(list(trainer.model.parameters()) + list(trainer.action_decoder.parameters()))
        loss, _ = trainer.compute_action_decoder_loss(batch)
        loss.backward()
        self.assertGreater(float(trainer.model.idm.la_head.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in trainer.model.fdm.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(old_inverse, trainer.model.idm.la_head.weight))
        self.assertFalse(torch.equal(old_decoder, trainer.action_decoder.weight))

    def test_reconstruction_gradient_switch(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'IDM'))
        from hydra import compose, initialize_config_dir
        from udrm.models.clam.space_time_clam_STM import SpaceTimeCLAM
        import udrm.resolvers
        config_dir = str(Path(__file__).resolve().parents[1] / 'IDM/udrm/cfg')
        for fdm_type in ['SpaceTimeFDM', 'SpaceTimeFDM_STM']:
            for detach in [True, False]:
                with self.subTest(fdm_type=fdm_type, detach=detach):
                    torch.manual_seed(7)
                    with initialize_config_dir(version_base=None, config_dir=config_dir):
                        cfg = compose(config_name='train_st_vivit_clam_stm', overrides=[
                            'env=robomimic_sailor', 'env.image_obs=True', 'data.seq_len=3',
                            'encoder.dim_model=32', 'encoder.dim_feedforward=64',
                            'decoder.dim_feedforward=64', 'encoder.n_layers=1',
                            'decoder.n_layers=1', f'model.FDM_TYPE={fdm_type}',
                            f'model.fdm.detach_idm_latents={str(detach).lower()}'])
                    model = SpaceTimeCLAM(cfg.model, input_dim=(3, 64, 64), la_dim=cfg.model.la_dim)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    before_idm = model.idm.la_head.weight.detach().clone()
                    before_fdm = model.fdm.to_recon.weight.detach().clone()
                    obs = torch.rand(1, 3, 3, 64, 64)
                    output = model(obs, timesteps=torch.arange(3)[None], states=None)
                    loss = (output.reconstructed_obs - obs[:, 1:]).square().mean()
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    inverse_grads = [p.grad for p in model.idm.parameters() if p.grad is not None]
                    if detach:
                        self.assertEqual(inverse_grads, [])
                    else:
                        self.assertGreater(float(model.idm.la_head.weight.grad.abs().sum()), 0)
                    self.assertGreater(float(model.fdm.to_recon.weight.grad.abs().sum()), 0)
                    optimizer.step()
                    self.assertEqual(torch.equal(before_idm, model.idm.la_head.weight), detach)
                    self.assertFalse(torch.equal(before_fdm, model.fdm.to_recon.weight))

    def test_action_dim_is_not_prediction_horizon(self):
        self.assertEqual(action_dim(np.zeros((4, 7, 8))), 7)
        self.assertEqual(action_dim(np.zeros((4, 7))), 7)

    def test_masks_override_demo_order_and_reject_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            with h5py.File(Path(tmp) / 'demo.hdf5', 'w') as f:
                for name in ['demo_0', 'demo_1', 'demo_2']:
                    f.create_group('data/' + name)
                f.create_dataset('mask/train', data=np.array([b'demo_2', b'demo_0']))
                f.create_dataset('mask/valid', data=np.array([b'demo_1']))
                self.assertEqual(official_demo_split(f), (['demo_2', 'demo_0'], ['demo_1']))
                f['mask/valid'][0] = b'demo_2'
                with self.assertRaises(ValueError):
                    official_demo_split(f)

    def test_legacy_timeout_correction_does_not_rewrite_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'episode.npz'
            np.savez(path, action=np.zeros((3, 7, 8)), is_terminal=[False, False, True])
            original = path.read_bytes()
            for success in [False, True]:
                path.with_suffix('.json').write_text(json.dumps({'stable5_success': success}))
                ep = load_rollout({'file_path': str(path)})
                self.assertEqual(bool(ep['is_terminal'][-1]), success)
                self.assertEqual(bool(ep['is_truncated'][-1]), not success)
                self.assertEqual(path.read_bytes(), original)

    def test_idm_scoring_decodes_last_transition_only(self):
        from types import SimpleNamespace
        class Inverse(torch.nn.Module):
            def forward(self, obs, timesteps, states):
                assert timesteps.tolist() == [[0, 1, 2], [0, 1, 2]]
                return SimpleNamespace(la=obs)
        model = SimpleNamespace(idm=Inverse())
        cfg = OmegaConf.create({'name': 'st_vivit_clam_stm', 'model': {
            'distributional_la': False, 'concatenate_gripper_state': False}})
        obs = torch.arange(42, dtype=torch.float32).reshape(2, 3, 7)
        result = predict_idm_action(model, torch.nn.Identity(), cfg, obs)
        torch.testing.assert_close(result, obs[:, -1])


if __name__ == '__main__':
    unittest.main()
