from typing import Tuple
from pathlib import Path
import inspect
import os
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


def _resolve_self_attention_block():
    """Use DINOv2 attention block."""
    from dinov2.layers.block import Block as SelfAttentionBlock  # type: ignore

    return SelfAttentionBlock


def _build_self_attention_block(
    block_cls,
    dim: int,
    num_heads: int,
    ffn_ratio: float,
    qkv_bias: bool,
    proj_bias: bool,
    ffn_bias: bool,
    drop: float,
    drop_path: float,
    init_values,
):
    """Create attention block with compatible kwargs across DINO variants."""
    params = inspect.signature(block_cls.__init__).parameters
    kwargs = {}

    if "dim" in params:
        kwargs["dim"] = dim
    if "num_heads" in params:
        kwargs["num_heads"] = num_heads
    if "ffn_ratio" in params:
        kwargs["ffn_ratio"] = ffn_ratio
    elif "mlp_ratio" in params:
        kwargs["mlp_ratio"] = ffn_ratio
    if "qkv_bias" in params:
        kwargs["qkv_bias"] = qkv_bias
    if "proj_bias" in params:
        kwargs["proj_bias"] = proj_bias
    if "ffn_bias" in params:
        kwargs["ffn_bias"] = ffn_bias
    if "drop" in params:
        kwargs["drop"] = drop
    elif "proj_drop" in params:
        kwargs["proj_drop"] = drop
    if "attn_drop" in params:
        kwargs["attn_drop"] = drop
    if "drop_path" in params:
        kwargs["drop_path"] = drop_path
    if "init_values" in params:
        kwargs["init_values"] = init_values
    if "norm_layer" in params:
        kwargs["norm_layer"] = nn.LayerNorm

    return block_cls(**kwargs)


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


class CNNTokenizer(nn.Module):
    def __init__(self, in_chans: int = 6, out_chans: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, out_chans, kernel_size=3, stride=2, padding=1),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 6, 128, 128]
        Returns:
            y: [B, 256, 16, 16]
        """
        return self.net(x)


class SpaceTimeIDM(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: Tuple[int, int, int], la_dim: int):
        super().__init__(cfg=cfg, input_dim=input_dim)
        self.name = "SpaceTimeIDM"

        self.la_dim = la_dim
        C, H, W = input_dim
        # the 0th dimension is the channel dimension
        self.patch_token_dim = C * self.cfg.patch_size**2
        self.model_dim = self.cfg.net.dim_model

        # the output_dim should be the latent dimension
        # the sequence dim is [T * N]
        # make sure H and W are divisible by patch_size
        assert H % self.cfg.patch_size == 0 and W % self.cfg.patch_size == 0
        self.num_patches = (H // self.cfg.patch_size) * (W // self.cfg.patch_size)

        self.input_embed = nn.Linear(self.patch_token_dim, self.model_dim)
        self.encoder = STTransformer(cfg=self.cfg.net)
        self.activation = nn.LeakyReLU(0.2)

        self.action_in = nn.Parameter(torch.randn(1, 1, 1, self.model_dim))

        self.spatial_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )
        self.temporal_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )

        self.la_head = nn.Linear(self.model_dim, self.la_dim)

        #Hayden - VQ
        self.vq = None
        if self.cfg.quantize_la:
            vq_cls = get_vq_cls(self.cfg.vq.name)
            log(f"Using vq {self.cfg.vq.name}", "green")
            self.cfg.vq.kwargs.dim = self.cfg.la_dim
            self.vq = vq_cls(**self.cfg.vq.kwargs)
        else:
            log("Not using vq, continuous latent action space", "red")

    # def forward(
    #     self, observations, timesteps: torch.Tensor, states: torch.Tensor, **kwargs
    # ) -> IDMOutput:
    #     if self.cfg.quantize_la:
    #         vq_cls = get_vq_cls(self.cfg.vq.name)
    #         log(f"Using vq {self.cfg.vq.name}", "green")
    #         self.cfg.vq.kwargs.dim = self.cfg.la_dim
    #         self.vq = vq_cls(**self.cfg.vq.kwargs)
    #     else:
    #         log("Not using vq, continuous latent action space", "red")

    def forward(
        self, observations, timesteps: torch.Tensor, states: torch.Tensor, **kwargs
    ) -> IDMOutput:
        """
        Args:
            observations: [B, T, C, H, W] tensor
            states: [B, T, D] tensor
        """
        B, T, *_ = observations.shape
        if timesteps is None:
            timesteps = torch.arange(T, device=observations.device).unsqueeze(0).repeat(B, 1)

        # need to put channel last for patchify
        observations = observations.permute(0, 1, 3, 4, 2)

        # [B, T, N, E] where N is the number of patches
        # and E is the patch token dimension
        patches = patchify(observations, self.cfg.patch_size)

        # embed the patches
        patches_embed = self.input_embed(patches)
        patches_embed = self.activation(patches_embed)

        # HMM, adding the action token after i embed the patches
        if self.cfg.add_action_token:
            B, T, N, E = patches_embed.shape

            # add a dummy token to represent the latent actions
            action_pad = self.action_in.expand(B, T, 1, self.model_dim)

            # prepend the action token to the patches
            # [B, T, N+1, E]
            patches_embed = torch.cat([action_pad, patches_embed], dim=2)

        # create temporal embeddings using timesteps
        if self.cfg.net.pos_enc == "learned":
            t_pos_embed = self.temporal_pos_embed(timesteps.long())
            t_pos_embed = einops.repeat(t_pos_embed, "B T E -> B T N E", N=N + 1)
        else:
            t_pos_embed = None

        # create spatial embeddings for each patch
        if self.cfg.net.pos_enc == "learned":
            spatial_coord = torch.arange(N + 1).to(patches_embed.device)
            spatial_pos_embed = self.spatial_pos_embed(spatial_coord.long())
            spatial_pos_embed = einops.repeat(
                spatial_pos_embed, "N E -> B T N E", B=B, T=T
            )
        else:
            spatial_pos_embed = None

        pos_embed = spatial_pos_embed + t_pos_embed

        # [B, T, N+1, E]
        z = self.encoder(patches_embed, pos_embed=pos_embed, causal=False)

        # reshape back to [B, T, N+1, E]
        z = z.view(B, T, -1, self.model_dim)

        if self.cfg.add_action_token:
            # the first element is the action token
            la_z = z[:, :, 0]
        else:
            la_z = z.mean(dim=2)

        la = self.la_head(la_z)

        # TODO: clean this up, duplicated with regular CLAM
        #Hayden
        # if self.cfg.quantize_la:
        #     quantized_la, indices, vq_loss = self.vq(la)

        #     if self.cfg.use_quantized_las:  # replace la with quantized_la
        #         la = quantized_la

        #     vq_loss = vq_loss.mean()

        #     vq_outputs = {"indices": indices}
        #     vq_metrics = {
        #         "vq_loss": vq_loss.item(),
        #         "perplexity": compute_perplexity(
        #             indices, self.cfg.vq.kwargs.codebook_size
        #         ),
        #     }
        #     return IDMOutput(
        #         la=la,
        #         quantized_la=quantized_la,
        #         vq_loss=vq_loss,
        #         vq_metrics=vq_metrics,
        #         vq_outputs=vq_outputs,
        #         encoder_out=patches,
        #     )

        # return patches to use in the FDM
        return IDMOutput(la=la, encoder_out=patches)


class DINO_IDM(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: Tuple[int, int, int], la_dim: int):
        super().__init__(cfg=cfg, input_dim=input_dim)
        self.name = "DINO_IDM"
        self.la_dim = la_dim
        # self.la_dim = 9
        C, H, W = input_dim

        self.patch_token_dim = C * self.cfg.patch_size**2
        assert H % self.cfg.patch_size == 0 and W % self.cfg.patch_size == 0
        self.num_patches = (H // self.cfg.patch_size) * (W // self.cfg.patch_size)
        self.model_dim = self.cfg.net.dim_model
        SelfAttentionBlock = _resolve_self_attention_block()

        # Use DINOv3 encoder
        self.use_dino = self.cfg.use_dino
        if self.use_dino:
            REPO_DIR = os.environ.get("DINO_REPO_DIR", "")
            WEIGHT_URL = os.environ.get("DINO_WEIGHT_PATH", "")
            if not REPO_DIR or not WEIGHT_URL:
                raise ValueError(
                    "DINO_IDM requires DINO_REPO_DIR and DINO_WEIGHT_PATH to be set."
                )
            # DINOv3 ViT models pretrained on web images
            self.frozen_dino = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=WEIGHT_URL)
            self.frozen_dino.eval()

        # Use vector quantization
        self.vq = None
        if self.cfg.quantize_la:
            vq_cls = get_vq_cls(self.cfg.vq.name)
            log(f"Using vq {self.cfg.vq.name}", "green")
            self.cfg.vq.kwargs.dim = self.cfg.la_dim
            self.vq = vq_cls(**self.cfg.vq.kwargs)
        else:
            log("Not using vq, continuous latent action space", "red")
        
        # Use CNN encoder as tokenizer
        self.use_cnn_tokenizer = cfg.use_cnn_tokenizer
        if cfg.use_cnn_tokenizer:
            self.tokenizer = CNNTokenizer(6, 256)
        else:
            # Use MLP as tokenizer
            self.tokenizer= nn.Sequential(nn.Linear(self.patch_token_dim, self.model_dim), nn.LeakyReLU(0.2))
        
        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.model_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.model_dim))

        # Use DINOv3 SelfAttentionBlock as backbone
        ffn_ratio = float(self.cfg.net.dim_feedforward) / float(self.model_dim)
        drop = float(getattr(self.cfg.net, "dropout", 0.0))
        qkv_bias = bool(getattr(self.cfg.net, "qkv_bias", True))
        proj_bias = bool(getattr(self.cfg.net, "proj_bias", True))
        ffn_bias = bool(getattr(self.cfg.net, "ffn_bias", True))
        drop_path = float(getattr(self.cfg.net, "drop_path_rate", 0.0))
        init_values = getattr(self.cfg.net, "layerscale_init", None)
        self.blocks = nn.ModuleList(
            [
                _build_self_attention_block(
                    block_cls=SelfAttentionBlock,
                    dim=self.model_dim,
                    num_heads=self.cfg.net.n_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop=0.0,
                    drop_path=0.0,
                    init_values=init_values,
                )
                for _ in range(self.cfg.net.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.model_dim)

        self.spatial_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=300
        )
        self.temporal_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )

        # Action projection head
        self.la_head = nn.Sequential(
            nn.Linear(self.model_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.la_dim),
        )

        # x = torch.randn(1, 3, 224, 224)  # input image
        # with torch.no_grad():
        #     feats = model.forward_features(x)
        #     cls = feats["x_norm_clstoken"]        # [B, D] global representation
        #     patch = feats["x_norm_patchtokens"]   # [B, N, D] patch representation
        #     print(patch.shape)
        #     print(cls.shape)
        self.global_test_step = 0

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            block_params = inspect.signature(blk.forward).parameters
            if "rope_or_rope_list" in block_params:
                x = blk(x, rope_or_rope_list=None)
            else:
                x = blk(x)
        return self.norm(x)

    def forward(
        self, observations, timesteps: torch.Tensor, states: torch.Tensor, **kwargs
    ) -> IDMOutput:
        """
        Args:
            observations: [B, T, C, H, W] tensor
            states: [B, T, D] tensor
        """
        B, T, C, H, W = observations.shape
        # observations = x.view(B, T, C, 224, 224)
        patches = patchify(observations.permute(0, 1, 3, 4, 2), self.cfg.patch_size)

        if self.use_dino:
            x = observations.reshape(B*T, C, H, W).float()  # interpolation expects float
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            observations = x.view(B, T, C, 224, 224)
            with torch.no_grad():
                feats_dino = self.frozen_dino.forward_features(x)
                cls_dino = feats_dino["x_norm_clstoken"]        # [B, D] global representation
                patch_dino = feats_dino["x_norm_patchtokens"]   # [B, N, D] patch representation
            _, N, D = patch_dino.shape
            patch_dino = patch_dino.view(B, T, N, D)
            patch_input = patch_dino # [128, 5, 196, 768]
            input_tokens = patch_input
        # if timesteps is None:
        #     timesteps = torch.arange(T, device=observations.device).unsqueeze(0).repeat(B, 1)
        elif self.use_cnn_tokenizer:
            patch_input = observations
            input_tokens = patch_input
        else:
            # embed the patches
            patches_embed = self.tokenizer(patches) # torch.Size([1, 5, 64, 768/256])
            input_tokens = patches_embed

        # Treat the whole sequence as context
        B, T, N, D = input_tokens.shape
        input_tokens = input_tokens.view(B, T*N, D)

        # if T < 2:
        #     la = torch.zeros((B, T, self.la_dim), device=observations.device, dtype=observations.dtype)
        # else:
        '''
        prev = patch_input[:, :-1]
        curr = patch_input[:, 1:]
        # pair_tokens = torch.cat([prev, curr], dim=2)
        if self.use_cnn_tokenizer:
            cnn_input = torch.cat([prev, curr], dim=2)
            B, T, C, H, W = cnn_input.shape
            # print("cnn_input: ", cnn_input.shape)
            res = self.tokenizer(cnn_input.view(B*T, C, H, W))
            _, C, H, W = res.shape
            res = res.view(B, T, C, H*W).permute(0,1,3,2)
            # print("res: ", res.shape)
        else:
            res = prev - curr
        '''
        # res = patch_input
        # B, T, N, D = res.shape
        cls = self.cls_token.expand(B, 1, D)

        input_tokens = torch.cat([input_tokens, cls], dim=2)
        # N = N + 1

        # t_pos_embed = self.temporal_pos_embed(timesteps.long())
        # t_pos_embed = einops.repeat(t_pos_embed, "B T E -> B T N E", N=N)
        
        B, N, D = input_tokens.shape
        spatial_coord = torch.arange(N).to(input_tokens.device)
        spatial_pos_embed = self.spatial_pos_embed(spatial_coord.long())
        spatial_pos_embed = einops.repeat(
            spatial_pos_embed, "N E -> B N E", B=B, T=T
        )
        res = res + spatial_pos_embed
        
        # if pos_embed is not None:
        #     pos_prev = pos_embed[:, :-1]
        #     pos_curr = pos_embed[:, 1:]
        #     pair_pos = torch.cat([pos_prev, pos_curr], dim=2)
        #     cls_pos = self.cls_pos.expand(B, T - 1, 1, self.model_dim)
        #     pair_pos = torch.cat([cls_pos, pair_pos], dim=2)
        #     pair_tokens = pair_tokens + pair_pos

        res = res.reshape(B * T, N, D)
        z = self._encode_tokens(res)
        self.global_test_step += 1

        cls_out = z[:, 0]
        la_pair = self.la_head(cls_out).view(B, T, self.la_dim)
        # la = torch.zeros((B, T+1, self.la_dim), device=observations.device, dtype=la_pair.dtype)
        # la[:, 1:] = la_pair
        la = la_pair
        # if self.global_test_step % 100 == 0:
        #     print("cls_out.shape:", cls_out.shape)
        #     print("this is cls_out: ", cls_out)
        #     print("this is la: ", la)

        return IDMOutput(la=la, encoder_out=patches)

class SpaceTimeFDM(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int):
        super().__init__(cfg, input_dim)
        self.name = "SpaceTimeFDM"
        C, H, W = input_dim
        self.patch_token_dim = C * self.cfg.patch_size**2

        self.num_patches = (H // self.cfg.patch_size) * (W // self.cfg.patch_size)
        self.seq_len = (self.cfg.seq_len - 1) * self.num_patches
        log(
            f"Patch token dim: {self.patch_token_dim}, Num patches: {self.num_patches}, Sequence length: {self.seq_len}"
        )
        self.model_dim = self.patch_token_dim # self.cfg.net.dim_model

        self.decoder = STTransformer(cfg=self.cfg.net)

        self.patch_embed = nn.Linear(self.patch_token_dim, self.model_dim)
        self.la_embed = nn.Linear(la_dim, self.model_dim)

        self.spatial_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )
        self.temporal_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )

        self.cond_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )

        self.to_recon = nn.Linear(self.model_dim, self.patch_token_dim)

        # also encode the hand position
        if self.cfg.concatenate_gripper_state:
            self.hand_pos_embed = nn.Linear(4, self.model_dim)
            self.input_embed_two = nn.Linear(self.model_dim * 2, self.model_dim)

    def forward(
        self,
        observations,
        idm_output: IDMOutput,
        timesteps: torch.Tensor,
        states: torch.Tensor,
        **kwargs,
    ) -> CLAMOutput:
        """
        Args:
            observations: [B, T, C, H, W] tensor
            idm_output: IDMOutput
        Returns:
            video_recon: [B, T, C, H, W] tensor
        """
        B, T, C, H, W = observations.shape
        # print("observations.shape:", observations.shape)
        # observations.shape: torch.Size([1, 3, 3, 128, 128])
        # print("idm_output.la.shape:", idm_output.la.shape)
        # idm_output.la.shape: torch.Size([1, 3, 16])
        # [B, T-1, la_dim]
        la = idm_output.la[:, 1:].detach() # ([1, 2, 16])
        # la = torch.rand_like(la) # only fdm

        # [B, T, N, E]
        patches = idm_output.encoder_out

        la_embed = self.la_embed(la)
        # add dimension
        la_embed = einops.rearrange(la_embed, "B T E -> B T 1 E")

        # don't feed in the last timestep, need to be causal
        patches_embed = self.patch_embed(patches[:, :-1])

        video_action_patches = la_embed + patches_embed

        # if self.cfg.concatenate_gripper_state:
        #     # extract the hand position and gripper state
        #     # embed the hand position and concatenate with the patches
        #     hand_pos_gripper = extract_state_info(states[:, :-1])
        #     hand_pos_gripper_embed = self.hand_pos_embed(hand_pos_gripper)
        #     hand_pos_gripper_embed = einops.repeat(
        #         hand_pos_gripper_embed, "B T E -> B T N E", N=self.num_patches
        #     )
        #     video_action_patches = self.input_embed_two(
        #         torch.cat([video_action_patches, hand_pos_gripper_embed], dim=-1)
        #     )

        B, T_minus_one, N, E = video_action_patches.shape

        # create temporal embeddings using timesteps
        if self.cfg.net.pos_enc == "learned":
            t_pos_embed = self.temporal_pos_embed(timesteps[:, :-1].long())
            t_pos_embed = einops.repeat(t_pos_embed, "B T E -> B T N E", N=N)
        else:
            t_pos_embed = None

        # create spatial embeddings
        if self.cfg.net.pos_enc == "learned":
            spatial_coord = torch.arange(N).to(video_action_patches.device)
            spatial_pos_embed = self.spatial_pos_embed(spatial_coord.long())
            spatial_pos_embed = einops.repeat(
                spatial_pos_embed, "N E -> B T N E", B=B, T=T - 1
            )
        else:
            spatial_pos_embed = None

        pos_embed = spatial_pos_embed + t_pos_embed

        if self.cfg.net.pos_enc == "learned":
            cond_pos_embed = self.cond_pos_embed(timesteps[:, 1:].long())
            cond_pos_embed = einops.repeat(cond_pos_embed, "B T E -> B T N E", N=N)
        else:
            cond_pos_embed = None

        # [B, (T-1), N, E]
        video_recon = self.decoder(
            video_action_patches,
            pos_embed=pos_embed,
            causal=True,
            cond=la_embed,  # apply cross attention with the learned latent action
            cond_pos_embed=cond_pos_embed,
        )

        video_recon = self.to_recon(video_recon)

        # reshape back to [B, T-1, N, E]
        video_recon = video_recon.view(B, T - 1, -1, self.patch_token_dim)

        # [B, T-1, H, W, C]
        video_recon = unpatchify(video_recon, self.cfg.patch_size, H, W)

        # put channel first
        video_recon = einops.rearrange(video_recon, "B T H W C -> B T C H W")
        return video_recon


class SpaceTimeFDM_STM(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int, use_state_transfer: bool, use_adaln: bool):
        super().__init__(cfg, input_dim)
        self.name = "SpaceTimeFDM"
        C, H, W = input_dim
        self.patch_token_dim = C * self.cfg.patch_size**2

        self.num_patches = (H // self.cfg.patch_size) * (W // self.cfg.patch_size)
        self.seq_len = (self.cfg.seq_len - 1) * self.num_patches
        log(
            f"Patch token dim: {self.patch_token_dim}, Num patches: {self.num_patches}, Sequence length: {self.seq_len}"
        )
        self.model_dim = self.patch_token_dim # self.cfg.net.dim_model

        self.decoder = STTransformer(cfg=self.cfg.net)

        self.patch_embed = nn.Linear(self.patch_token_dim, self.model_dim)

        self.spatial_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )
        self.temporal_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )

        self.cond_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )

        self.to_recon = nn.Linear(self.model_dim, self.patch_token_dim)

        # State transfer machine (causal transformer over latent action sequence)
        self.use_state_transfer = use_state_transfer
        if self.use_state_transfer:
            stm_dim = int(getattr(self.cfg, "stm_dim", self.model_dim))
            stm_out_dim = int(getattr(self.cfg, "stm_out_dim", la_dim))
            stm_layers = int(getattr(self.cfg, "stm_layers", 2))
            stm_heads = int(getattr(self.cfg, "stm_heads", 4))
            stm_ff = int(getattr(self.cfg, "stm_dim_feedforward", 1024))
            stm_dropout = float(getattr(self.cfg, "stm_dropout", 0.1))

            self.state_transfer = StateTransferMachine(
                input_dim=la_dim,
                model_dim=stm_dim,
                output_dim=stm_out_dim,
                n_layers=stm_layers,
                n_heads=stm_heads,
                dim_feedforward=stm_ff,
                dropout=stm_dropout,
                pos_enc=self.cfg.net.pos_enc,
            )
        else:
            self.state_transfer = None
            stm_out_dim = la_dim

        self.use_adaln = use_adaln
        if self.use_adaln:
            self.film = nn.Linear(stm_out_dim, self.model_dim * 2)
        else:
            self.la_embed = nn.Linear(la_dim, self.model_dim)


    def forward(
        self,
        observations,
        idm_output: IDMOutput,
        timesteps: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
        state_seq: torch.Tensor | None = None,
        **kwargs,
    ) -> CLAMOutput:
        """
        Args:
            observations: [B, T, C, H, W] tensor
            idm_output: IDMOutput
        Returns:
            video_recon: [B, T, C, H, W] tensor
        """
        B, T, C, H, W = observations.shape

        # [B, T-1, la_dim]
        la = idm_output.la[:, 1:]
        # print("[FDM] la.shape:", la.shape) # [128, 4, 16]

        # patchify expects channel-last
        observations = observations.permute(0, 1, 3, 4, 2)
        # [B, T, N, E]
        patches = patchify(observations, self.cfg.patch_size)
        # print("[FDM] patches.shape:", patches.shape) # [128, 5, 64, 768]

        # Only use the first observation as the visual anchor.
        # [B, 1, N, E]
        # base_patches_embed = self.patch_embed(patches[:, :1])
        # Switch to using action directly as condition
        base_patches_embed = self.patch_embed(patches[:, :-1])
        base_patches_embed = base_patches_embed.expand(B, T - 1, -1, -1)
        # print("[FDM] base_patches_embed.shape:", base_patches_embed.shape) # [128, 4, 64, 256]

        # Optional: map latent action sequence -> state sequence (causal)
        if self.use_state_transfer and state_seq is None:
            state_seq = self.state_transfer(la, timesteps=timesteps[:, 1:])
            # print("[FDM] state_seq.shape after STM:", state_seq.shape) # [128, 4, 16]
        elif self.use_state_transfer and state_seq is not None:
            pass
        else:
            # Switch to using action directly as state
            state_seq = la

        idm_output.state_seq = state_seq

        if self.use_adaln:
            # Inject action through modulation
            film_params = self.film(state_seq)
            gamma, beta = film_params.chunk(2, dim=-1)
            idm_output.fdm_beta = beta
            gamma = einops.rearrange(gamma, "B T E -> B T 1 E")
            beta = einops.rearrange(beta, "B T E -> B T 1 E")

            video_action_patches = base_patches_embed * (1 + gamma) + beta
            # This can be replaced with patch adaptive normalization
            # print("[FDM] video_action_patches.shape:", video_action_patches.shape) # [128, 4, 64, 256]
        else:
            la_embed = self.la_embed(la)
            # add dimension
            la_embed = einops.rearrange(la_embed, "B T E -> B T 1 E")
            video_action_patches = la_embed + base_patches_embed

        B, T_minus_one, N, E = video_action_patches.shape

        # create spatial embeddings
        if self.cfg.net.pos_enc == "learned":
            spatial_coord = torch.arange(N).to(video_action_patches.device)
            spatial_pos_embed = self.spatial_pos_embed(spatial_coord.long())
            spatial_pos_embed = einops.repeat(spatial_pos_embed, "N E -> BT 1 N E", BT=B * T_minus_one)
        else:
            spatial_pos_embed = None

        # Each prediction uses the same observation anchor, so treat each step independently.
        # [B*(T-1), 1, N, E]
        video_action_patches = einops.rearrange(
            video_action_patches, "B T N E -> (B T) 1 N E"
        )
        # print("[FDM] video_action_patches reshaped for decoder:", video_action_patches.shape) 
        # [512, 1, 64, 256]

        # [B*(T-1), 1, N, E]
        video_recon = self.decoder(
            video_action_patches,
            pos_embed=spatial_pos_embed,
            causal=False,
            cond=None,
            cond_pos_embed=None,
        )

        video_recon_embed = video_recon.view(B, T - 1, -1, self.model_dim)
        idm_output.fdm_features = video_recon_embed

        video_recon = self.to_recon(video_recon)

        # reshape back to [B, T-1, N, E]
        video_recon = video_recon.view(B, T - 1, -1, self.patch_token_dim)

        # [B, T-1, H, W, C]
        video_recon = unpatchify(video_recon, self.cfg.patch_size, H, W)

        # put channel first
        video_recon = einops.rearrange(video_recon, "B T H W C -> B T C H W")
        # print("[FDM] video_recon.shape:", video_recon.shape)  # [128, 4, 3, 128, 128]
        return video_recon


class SpaceTimeCLAM(TransformerCLAM):
    """
    Latent Action Model (LAM) used to distill latent actions
    from history of past video frames. The LAM model employs a
    VAE model to encode video frames into continuous latents.
    Both the encoder and decoder are based on spatial-temporal
    transformers.
    """

    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.name = "ST-ViVit"
        self.la_dim = la_dim
        if cfg.IDM_TYPE == "DINO_IDM":
            self.idm = DINO_IDM(cfg.idm, input_dim=input_dim, la_dim=la_dim)
        else:
            self.idm = SpaceTimeIDM(cfg.idm, input_dim=input_dim, la_dim=la_dim)
        if cfg.FDM_TYPE == "SpaceTimeFDM_STM":
            self.fdm = SpaceTimeFDM_STM(cfg.fdm, input_dim=input_dim, la_dim=la_dim, use_state_transfer=cfg.use_stm, use_adaln=cfg.use_adaln)
        else:
            self.fdm = SpaceTimeFDM(cfg.fdm, input_dim=input_dim, la_dim=la_dim)

    def forward(
        self,
        observations,
        timesteps: torch.Tensor,
        states: torch.Tensor = None,
        **kwargs,
    ) -> CLAMOutput:
        return super().forward(observations, timesteps=timesteps, states=states)

    @torch.no_grad()
    def visualize_dreamer_style_rollout(
        self,
        gt_seq: torch.Tensor,
        idm_len: int | None = None,
    ):
        """
        Dreamer-style rollout for Space-Time CLAM.
        - Slice GT into sliding windows to get latent actions from IDM.
        - Use first action of each window + tail of last window to build full action sequence.
        - Roll out o_{t+1} using only o_0 + s_t (state transfer is causal).
        - Keeps output in [-1, 1] (visualization converts to [0,1] elsewhere).
        """
        device = gt_seq.device
        T_total, C, H, W = gt_seq.shape
        # print("[Rollout] gt_seq.shape:", gt_seq.shape)
        # [Rollout] gt_seq.shape: torch.Size([100, 3, 128, 128])

        # If input length < 2, (o_t -> o_{t+1}) cannot be formed; return an empty tensor.
        if T_total < 2:
            return torch.empty((0, C, H, W), device=device)

        if idm_len is None:
            idm_len = int(getattr(self.cfg.idm.net, "seq_len", 2))
        idm_len = max(2, min(idm_len, T_total))

        # 1) Build sliding windows and get latent actions from IDM
        num_windows = T_total - idm_len + 1
        windows = torch.stack(
            [gt_seq[i : i + idm_len] for i in range(num_windows)], dim=0
        )  # (B, idm_len, C, H, W)
        ts = torch.arange(idm_len, device=device).unsqueeze(0).repeat(num_windows, 1)

        idm_out = self.idm(windows, timesteps=ts, states=None)
        actions_win = idm_out.la[:, 1:]  # (B, idm_len-1, D)
        if self.fdm.state_transfer is not None:
            state_seq = self.fdm.state_transfer(actions_win, timesteps=ts[:, 1:])
        else:
            state_seq = actions_win
        
        # print("[Rollout] actions_win.shape:", actions_win.shape)
        # print("[Rollout] state_seq.shape:", state_seq.shape)
        # [Rollout] actions_win.shape: torch.Size([96, 4, 16])
        # [Rollout] state_seq.shape: torch.Size([96, 4, 16])

        recons = []
        curr_frame = gt_seq[0]
        ts_pair = torch.tensor([[0, 1]], device=device)
        la_dim = actions_win.shape[-1]

        seq_len = state_seq.shape[0]
        for i in range(seq_len-1):
            pair_input = torch.stack([curr_frame, curr_frame], dim=0).unsqueeze(0)

            la_step = torch.zeros((1, 2, la_dim), device=device)
            la_step[:, 1] = actions_win[i, 0]
            idm_step = IDMOutput(la=la_step)

            step_state = state_seq[i : i + 1, :1]
            # print("[Rollout] step_state.shape:", step_state.shape)
            # [Rollout] step_state.shape: torch.Size([1, 1, 16])
            recon = self.fdm(
                pair_input,
                idm_step,
                timesteps=ts_pair,
                states=None,
                state_seq=step_state,
            )
            curr_frame = recon[0, 0].clamp(-1, 1)
            # print("[Rollout] curr_frame.shape:", curr_frame.shape)
            # [Rollout] curr_frame.shape: torch.Size([3, 128, 128])
            recons.append(curr_frame)

        pair_input = curr_frame.unsqueeze(0).expand(idm_len, -1, -1, -1).unsqueeze(0)
        step_state = state_seq[seq_len-1 : seq_len, :]
        recon = self.fdm(
            pair_input,
            idm_step,
            timesteps=torch.tensor([[0, 1, 2, 3]], device=device),
            states=None,
            state_seq=step_state,
        )
        # print("[Rollout] Final step recon shape:", recon.shape)
        # [Rollout] Final step recon shape: torch.Size([1, 4, 3, 128, 128])
        recons_stacked = torch.stack(recons, dim=0)
        recons = torch.cat([recons_stacked, recon[0]], dim=0)
        print(f"Dreamer-style Rollout Done. Frames: {recons.shape}")
        return recons

'''
    #Hayden - ablation experiment for video reconstruction
    @torch.no_grad()
    def rollout_idm_fdm_closed_loop(
        self,
        gt_seq: torch.Tensor,    # (T, C, H, W)
        gt_states: torch.Tensor, # (T, D) or None
        max_steps: int | None = None,
        ):

        use_gt_image: bool = True   # using all GT image?
        use_gt_action: bool = True # using action from GT image?
        use_zero_action: bool = False # reconstruction without latent action


        if use_zero_action and use_gt_action:
            raise ValueError("If use_zero_action=True, it should be use_gt_action=False.")

        # gripper state off
        self.idm.cfg.concatenate_gripper_state = False
        self.fdm.cfg.concatenate_gripper_state = False

        device = gt_seq.device
        T, C, H, W = gt_seq.shape

        # processing max_steps
        if max_steps is None:
            max_steps = T - 1
        else:
            max_steps = min(max_steps, T - 1)

        recons = []

        def sample_la(idm_out):
            la = idm_out.la
            if self.cfg.distributional_la:
                la = self.model.reparameterize(la)
            return la

        # -------------------------
        # 1) Warm-up (step=1): GT 2 frames
        # -------------------------
        if max_steps >= 1:
            pair0 = gt_seq[0:2].unsqueeze(0)                 # (1,2,C,H,W)
            ts0   = torch.tensor([[0, 1]], device=device)     # (1,2)
            state0 = None  # or gt_states[0:2].unsqueeze(0)

            idm_out0 = self.idm(observations=pair0,
                                timesteps=ts0,
                                states=state0)
            la0 = sample_la(idm_out0)

            if use_zero_action:
                la0 = torch.zeros_like(la0)

            idm_step0 = IDMOutput(la=la0, encoder_out=idm_out0.encoder_out)

            recon0 = self.fdm(
                observations=pair0,
                idm_output=idm_step0,
                timesteps=ts0,
                states=state0,
            )
            current_recon = (torch.tanh(recon0[0, -1]) + 1) / 2  # (C,H,W)
            recons.append(current_recon)

        # -------------------------
        # 2) Loop: step=2..max_steps
        # -------------------------
        for step in range(2, max_steps + 1):
            t_prev = step - 1
            t_curr = step
            # use relative timesteps (training uses local 0..T-1 per window)
            ts_pair = torch.tensor([[0, 1]], device=device)
            state_pair = None

            pair_gt = gt_seq[step-2: step].unsqueeze(0)  # (1,2,C,H,W)

            if step == 2:
                # t=1 (GT) + t=2 (Gen)
                prev_img = gt_seq[1]
                curr_img = recons[-1]
            else:
                prev_img = recons[-2]
                curr_img = recons[-1]

            pair_gen = torch.stack([prev_img, curr_img], dim=0).unsqueeze(0)  # (1,2,C,H,W)

            if use_gt_image:
                pair_for_idm = pair_gt
            else:
                pair_for_idm = pair_gen

            idm_out = self.idm(
                observations=pair_for_idm,
                timesteps=ts_pair,
                states=state_pair,
            )

            if use_zero_action:
                la_for_fdm = torch.zeros_like(idm_out.la)
            
            elif use_gt_action:
                idm_out_gt = self.idm(
                    observations=pair_gt,
                    timesteps=ts_pair,
                    states=state_pair,
                )
                la_for_fdm = sample_la(idm_out_gt)
                
            else:
                # just use la from current pair_for_idm
                la_for_fdm = sample_la(idm_out)

            idm_step = IDMOutput(
                la=la_for_fdm,
                encoder_out=idm_out.encoder_out,
            )

            recon_next = self.fdm(
                observations=pair_for_idm,
                idm_output=idm_step,
                timesteps=ts_pair,
                states=state_pair,
            )

            current_recon = (torch.tanh(recon_next[0, -1]) + 1) / 2
            recons.append(current_recon)

        print(f"Action-Conditioned Generation Done. Frames: {len(recons)}")

        if len(recons) > 0:
            recons = torch.stack(recons, dim=0)   # (N, C, H, W)
        else:
            recons = torch.empty((0, C, H, W), device=device)

        return recons
'''
