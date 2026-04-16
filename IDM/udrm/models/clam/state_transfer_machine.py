from typing import Tuple
from pathlib import Path
import sys

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig

from udrm.models.base import BaseModel
from udrm.models.clam.clam import get_vq_cls
from udrm.models.clam.transformer_clam import TransformerCLAM

# from udrm.models.space_time_attn.models import STTransformer
from udrm.models.space_time_attn.models_v2 import STTransformer
from udrm.models.space_time_attn.utils import patchify, unpatchify
from udrm.models.utils.transformer_utils import get_pos_encoding
from udrm.models.utils.utils import CLAMOutput, IDMOutput, compute_perplexity
from udrm.utils.logger import log
import torch.nn.functional as F

class StateTransferMachine(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        output_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pos_enc: str = "learned",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.pos_enc = pos_enc

        self.input_proj = nn.Linear(input_dim, model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pos_embed = get_pos_encoding(pos_enc, embedding_dim=model_dim, max_len=200)
        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, z_seq: torch.Tensor, timesteps: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z_seq: [B, T, D] latent action sequence
            timesteps: [B, T] (optional)
        Returns:
            s_seq: [B, T, output_dim]
        """
        B, T, _ = z_seq.shape
        if timesteps is None:
            timesteps = torch.arange(T, device=z_seq.device).unsqueeze(0).repeat(B, 1)

        x = self.input_proj(z_seq)

        if self.pos_enc == "learned":
            pos = self.pos_embed(timesteps.long())
            x = x + pos
        elif self.pos_enc == "sine":
            pos = self.pos_embed.to(z_seq.device)[timesteps]
            x = x + pos

        causal_mask = torch.triu(torch.ones(T, T, device=z_seq.device), diagonal=1).bool()
        h = self.encoder(x, mask=causal_mask)
        return self.output_proj(h)