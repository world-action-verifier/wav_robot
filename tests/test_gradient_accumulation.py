import copy
from pathlib import Path
import sys
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'IDM'))
from udrm.utils.gradient_accumulation import backward_microbatches


class Accumulation(unittest.TestCase):
    def test_shared_encoder_two_losses_and_uneven_chunks_match_full_batch(self):
        torch.manual_seed(8)
        encoder = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Tanh())
        decoder = torch.nn.Linear(4, 2)
        e2, d2 = copy.deepcopy(encoder), copy.deepcopy(decoder)
        x, z = torch.randn(7, 5), torch.randn(7, 4)
        a, y = torch.randn(11, 5), torch.randn(11, 2)
        criterion = torch.nn.MSELoss()
        full = criterion(encoder(x), z) + .3 * criterion(decoder(encoder(a)), y)
        full.backward()
        class IdentityScaler:
            def scale(self, loss):
                return loss
        first, _ = backward_microbatches(7, 3, lambda s,t: criterion(e2(x[s:t]), z[s:t]), IdentityScaler())
        second, _ = backward_microbatches(11, 4, lambda s,t: .3*criterion(d2(e2(a[s:t])), y[s:t]), IdentityScaler())
        self.assertAlmostEqual(float(full.detach()), first+second, places=6)
        for p,q in zip(list(encoder.parameters())+list(decoder.parameters()), list(e2.parameters())+list(d2.parameters())):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
