"""Tokenizer-latent inverse head from World-Action-Verifier."""
import torch
import torch.nn as nn
from model import MLP


class InverseDynamicsHeadTokenizer(nn.Module):
    """
    Predicts the transition action a_t from a pair of adjacent frozen-tokenizer
    latents (z_t, z_{t+1}), each (B,T,n_latents,d_bottleneck).

    Features are [z_t, z_{t+1}, z_{t+1}-z_t] flattened over the latent grid
    and concatenated -- the explicit delta term gives the head a cheap,
    direct "what changed" signal on top of the two raw codes.
    """

    def __init__(self, *, n_latents: int, d_bottleneck: int, act_dim_max: int = 16,
                 mlp_ratio: float = 4.0, dropout: float = 0.0, d_hidden: int = 0):
        super().__init__()
        d_in = n_latents * d_bottleneck
        # Width defaults to d_in (n_latents*d_bottleneck = 4096 for the released
        # tokenizer), which makes this "lightweight" head ~185M parameters --
        # comparable to the frozen dynamics model itself. --d_hidden decouples
        # the two so the head can be shrunk, and so the tokenizer/backbone arms
        # can be compared at matched capacity rather than matched feature width.
        d_h = int(d_hidden) if int(d_hidden) > 0 else d_in
        self.in_proj = nn.Linear(3 * d_in, d_h)
        self.mlp = MLP(d_h, mlp_ratio=mlp_ratio, dropout=dropout)
        self.out = nn.Linear(d_h, act_dim_max)
        # Small-normal + zero bias: predictions start near 0 (small initial
        # loss) while still letting gradient flow from step 1 -- same
        # convention as PolicyHeadMTP in model.py.
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        B, T = z_t.shape[:2]
        a = z_t.reshape(B, T, -1)
        b = z_tp1.reshape(B, T, -1)
        x = torch.cat([a, b, b - a], dim=-1)
        h = self.in_proj(x)
        h = h + self.mlp(h)
        return torch.tanh(self.out(h))  # (B,T,act_dim_max), matches the [-1,1] action convention
