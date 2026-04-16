from typing import Tuple

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig

from udrm.models.base import BaseModel
from udrm.models.clam.clam import get_vq_cls
from udrm.models.clam.transformer_clam import TransformerCLAM
from udrm.models.space_time_attn.models_v2 import STTransformer
from udrm.models.space_time_attn.utils import patchify, unpatchify
from udrm.models.utils.transformer_utils import get_pos_encoding
from udrm.models.utils.utils import CLAMOutput, IDMOutput, compute_perplexity
from udrm.utils.logger import log

# Import custom NSVQ class
from udrm.models.vqNSVQ import NSVQ 

def extract_state_info(states: torch.Tensor):
    pos_goal = states[:, :, -3:]
    curr_obs_and_prev_obs = states[:, :, :-3]
    curr_obs = curr_obs_and_prev_obs[:, :, : int(curr_obs_and_prev_obs.shape[-1] // 2)]
    hand_pos_gripper = curr_obs[:, :, :4]
    return hand_pos_gripper

class SpaceTimeIDM(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: Tuple[int, int, int], la_dim: int):
        super().__init__(cfg=cfg, input_dim=input_dim)
        self.name = "SpaceTimeIDM"

        self.la_dim = la_dim
        C, H, W = input_dim
        self.patch_token_dim = C * self.cfg.patch_size**2
        self.model_dim = self.cfg.net.dim_model

        assert H % self.cfg.patch_size == 0 and W % self.cfg.patch_size == 0
        self.num_patches = (H // self.cfg.patch_size) * (W // self.cfg.patch_size)

        if self.cfg.concatenate_gripper_state:
            self.hand_pos_embed = nn.Linear(4, self.model_dim)
            self.input_embed_two = nn.Linear(self.model_dim * 2, self.model_dim)

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

        #Hayden - LayerNorm added to prevent codebook collapse
        self.ln_pre_head = nn.LayerNorm(self.model_dim)
        self.la_head = nn.Linear(self.model_dim, self.la_dim)

        # ----------------- VQ Init -----------------
        self.vq = None
        if self.cfg.quantize_la:
            log(f"Initializing NSVQ for LAPA", "green")
            vq_kwargs = dict(self.cfg.vq.kwargs)

            if "codebook_size" in vq_kwargs:
                vq_kwargs["num_embeddings"] = vq_kwargs.pop("codebook_size")
            if "eps" in vq_kwargs:
                vq_kwargs.pop("eps")

            # Quantize based on la_dim
            vq_kwargs["dim"] = self.la_dim        
            vq_kwargs["embedding_dim"] = self.la_dim 
            # image_size/patch_size are ignored in vector-input mode, but kept as fields
            vq_kwargs["image_size"] = 1
            vq_kwargs["patch_size"] = 1
            
            # [Fixed 1] Enable NSVQ vector-input mode flag
            vq_kwargs["is_vector_input"] = True 


            #Hayden
            # [Fixed] Set threshold explicitly to prevent aggressive pruning
            if "discarding_threshold" not in vq_kwargs:
                vq_kwargs["discarding_threshold"] = 0.01

            vq_kwargs["device"] = (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )

            self.vq = NSVQ(**vq_kwargs)
        else:
            log("Not using vq, continuous latent action space", "red")


    def forward(
        self, observations, timesteps: torch.Tensor, states: torch.Tensor, **kwargs
    ) -> IDMOutput:
        
        B, T, *_ = observations.shape
        observations = observations.permute(0, 1, 3, 4, 2)
        patches = patchify(observations, self.cfg.patch_size)
        patches_embed = self.input_embed(patches)
        patches_embed = self.activation(patches_embed)
        
        N_aug = self.num_patches 
        # Keep positional encoding logic unchanged
        if self.cfg.net.pos_enc == "learned":
            t_pos_embed = self.temporal_pos_embed(timesteps.long())
            t_pos_embed = einops.repeat(t_pos_embed, "B T E -> B T N E", N=N_aug)
            
            spatial_coord = torch.arange(N_aug).to(patches_embed.device)
            spatial_pos_embed = self.spatial_pos_embed(spatial_coord.long())
            spatial_pos_embed = einops.repeat(spatial_pos_embed, "N E -> B T N E", B=B, T=T)
            
            pos_embed = spatial_pos_embed + t_pos_embed
        else:
            pos_embed = None

        if self.cfg.concatenate_gripper_state:
            hand_pos_gripper = extract_state_info(states)
            hand_pos_gripper_embed = self.hand_pos_embed(hand_pos_gripper)
            hand_pos_gripper_embed = einops.repeat(
                hand_pos_gripper_embed, "B T E -> B T N E", N=N_aug
            )
            patches_embed = self.input_embed_two(
                torch.cat([patches_embed, hand_pos_gripper_embed], dim=-1)
            )

        # Through encoder
        z = self.encoder(patches_embed, pos_embed=pos_embed, causal=False)
        z = z.view(B, T, -1, self.model_dim)

        # Latent Extraction
        la_z = z.mean(dim=2) # [B, T, model_dim]


        #Hayden #Apply LayerNorm before head
        la_z = self.ln_pre_head(la_z)

        la_continuous = self.la_head(la_z) # [B, T, la_dim]

        # ----------------- VQ Logic Start -----------------
        vq_loss = torch.tensor(0.0, device=observations.device)
        vq_outputs, vq_metrics = {}, {}
        
        pad_la = torch.zeros(B, 1, self.la_dim, device=observations.device)
        # Default: continuous delta (when VQ is not used)
        la_final = torch.cat([pad_la, la_continuous[:, 1:] - la_continuous[:, :-1]], dim=1) if T > 1 else la_continuous

        if self.cfg.quantize_la and self.vq is not None and T > 1:
            
            # Prepare Inputs for NSVQ (Flattened)
            e_current = la_continuous[:, :-1] # t   : [B, T-1, la_dim]
            e_next    = la_continuous[:, 1:]  # t+1 : [B, T-1, la_dim]
            
            flat_current = e_current.reshape(-1, self.la_dim)
            flat_last    = e_next.reshape(-1, self.la_dim)

            # NSVQ Forward (Two Inputs -> Internal Difference)
            quantized_flat, perplexity, _, idx_flat = self.vq(
                input_data_first=flat_current, 
                input_data_last=flat_last,
                codebook_training_only=False #True -> False
            )
            
            quantized_delta = quantized_flat.view(B, T-1, self.la_dim)
            indices = idx_flat.view(B, T-1)

            # Final Output Construction
            la_final = torch.cat([pad_la, quantized_delta], dim=1) 
            
            pad_idx = torch.zeros(B, 1, dtype=torch.long, device=observations.device)
            vq_outputs = {"indices": torch.cat([pad_idx, indices], dim=1)}
            vq_metrics = {"perplexity": perplexity.item()}

        return IDMOutput(
            la=la_final,
            quantized_la=la_final,
            vq_loss=vq_loss,
            vq_metrics=vq_metrics,
            vq_outputs=vq_outputs,
            encoder_out=patches, # raw patches passed to FDM
        )

class SpaceTimeFDM(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int, use_vq: bool = False):
        super().__init__(cfg, input_dim)
        self.name = "SpaceTimeFDM"
        C, H, W = input_dim
        self.patch_token_dim = C * self.cfg.patch_size**2
        self.num_patches = (H // self.cfg.patch_size) * (W // self.cfg.patch_size)
        self.model_dim = self.cfg.net.dim_model

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
        
        B, T, C, H, W = observations.shape

        # Latent Action (Delta)
        la = idm_output.la[:, 1:] # [B, T-1, Dim]
        
        # Previous Frame Patches (Encoder Output)
        patches = idm_output.encoder_out[:, :-1] # [B, T-1, N, Dim]
        
        # [Fixed 2] Apply Stop Gradient (sg)
        # LAPA paper: "apply stop gradient to the patch embedding... to avoid representation collapse"
        patches = patches.detach()

        la_embed = self.la_embed(la)
        la_embed = einops.rearrange(la_embed, "B T E -> B T 1 E")

        patches_embed = self.patch_embed(patches)
        
        # [Fixed 3] Conditioning
        # The paper recommends Cross-Attn, but keep additive conditioning if STTransformer only supports additive
        # Stop Gradient is already applied, so additive conditioning can still avoid collapse
        video_action_patches = la_embed + patches_embed

        if self.cfg.concatenate_gripper_state:
            hand_pos_gripper = extract_state_info(states[:, :-1])
            hand_pos_gripper_embed = self.hand_pos_embed(hand_pos_gripper)
            hand_pos_gripper_embed = einops.repeat(
                hand_pos_gripper_embed, "B T E -> B T N E", N=self.num_patches
            )
            video_action_patches = self.input_embed_two(
                torch.cat([video_action_patches, hand_pos_gripper_embed], dim=-1)
            )

        B, T_minus_one, N, E = video_action_patches.shape

        if self.cfg.net.pos_enc == "learned":
            t_pos_embed = self.temporal_pos_embed(timesteps[:, :-1].long())
            t_pos_embed = einops.repeat(t_pos_embed, "B T E -> B T N E", N=N)
        else:
            t_pos_embed = None

        if self.cfg.net.pos_enc == "learned":
            spatial_coord = torch.arange(N).to(video_action_patches.device)
            spatial_pos_embed = self.spatial_pos_embed(spatial_coord.long())
            spatial_pos_embed = einops.repeat(
                spatial_pos_embed, "N E -> B T N E", B=B, T=T - 1
            )
        else:
            spatial_pos_embed = None

        pos_embed = spatial_pos_embed + t_pos_embed if (
            spatial_pos_embed is not None and t_pos_embed is not None
        ) else None

        if self.cfg.net.pos_enc == "learned":
            cond_pos_embed = self.cond_pos_embed(timesteps[:, 1:].long())
            cond_pos_embed = einops.repeat(cond_pos_embed, "B T E -> B T N E", N=N)
        else:
            cond_pos_embed = None

        video_recon = self.decoder(
            video_action_patches,
            pos_embed=pos_embed,
            causal=True,
            cond=la_embed,
            cond_pos_embed=cond_pos_embed,
        )

        video_recon = self.to_recon(video_recon)
        video_recon = video_recon.view(B, T - 1, -1, self.patch_token_dim)
        video_recon = unpatchify(video_recon, self.cfg.patch_size, H, W)
        video_recon = einops.rearrange(video_recon, "B T H W C -> B T C H W")
        return video_recon

class SpaceTimeCLAM_NSVQ(TransformerCLAM):

    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.name = "ST-CLAM_NSVQ"
        self.la_dim = la_dim
        self.idm = SpaceTimeIDM(cfg.idm, input_dim=input_dim, la_dim=la_dim)
        self.fdm = SpaceTimeFDM(cfg.fdm, input_dim=input_dim, la_dim=la_dim)

    def forward(
        self,
        observations: torch.Tensor,
        timesteps: torch.Tensor,
        states: torch.Tensor = None,
        **kwargs,
    ) -> CLAMOutput:
        
        idm_output = self.idm(
            observations=observations,
            timesteps=timesteps,
            states=states,
        )

        recon = self.fdm(
            observations=observations,
            idm_output=idm_output,
            timesteps=timesteps,
            states=states,
        )

        return CLAMOutput(
            la=idm_output.la,
            reconstructed_obs=recon,
            idm_output=idm_output,
        )

    @torch.no_grad()
    def rollout_idm_fdm_closed_loop(
        self,
        gt_seq: torch.Tensor,    
        gt_states: torch.Tensor, 
        max_steps: int | None = None,
    ):
        use_gt_image = True
        use_gt_action = True
        use_zero_action = False
        
        device = gt_seq.device
        T, C, H, W = gt_seq.shape

        if max_steps is None:
            max_steps = T - 1
        else:
            max_steps = min(max_steps, T - 1)

        recons = []

        def sample_la(idm_out):
            return idm_out.la 

        # step 1: Warmup
        if max_steps >= 1:
            pair0 = gt_seq[0:2].unsqueeze(0)  
            ts0   = torch.tensor([[0, 1]], device=device)  
            state0 = None if gt_states is None else gt_states[0:2].unsqueeze(0)

            idm_out0 = self.idm(pair0, timesteps=ts0, states=state0)
            la0 = sample_la(idm_out0)
            if use_zero_action:
                la0 = torch.zeros_like(la0)

            idm_step0 = IDMOutput(la=la0, encoder_out=idm_out0.encoder_out)
            recon0 = self.fdm(pair0, idm_step0, ts0, state0)
            recons.append((torch.tanh(recon0[0, -1]) + 1) / 2)

        # step 2..max_steps: Closed-loop
        for step in range(2, max_steps + 1):
            t_prev, t_curr = step - 1, step
            ts_pair = torch.tensor([[t_prev, t_curr]], device=device)

            state_pair = None
            if gt_states is not None:
                state_pair = gt_states[step-2: step].unsqueeze(0)

            pair_gt = gt_seq[step-2: step].unsqueeze(0)

            if step == 2:
                prev_img, curr_img = gt_seq[1], recons[-1]
            else:
                prev_img, curr_img = recons[-2], recons[-1]

            pair_gen = torch.stack([prev_img, curr_img], dim=0).unsqueeze(0)
            pair_for_idm = pair_gt if use_gt_image else pair_gen

            idm_out = self.idm(pair_for_idm, timesteps=ts_pair, states=state_pair)

            if use_zero_action:
                la_for_fdm = torch.zeros_like(idm_out.la)
            elif use_gt_action:
                idm_out_gt = self.idm(pair_gt, timesteps=ts_pair, states=state_pair)
                la_for_fdm = sample_la(idm_out_gt)
            else:
                la_for_fdm = sample_la(idm_out)

            idm_step = IDMOutput(la=la_for_fdm, encoder_out=idm_out.encoder_out)
            recon_next = self.fdm(pair_for_idm, idm_step, ts_pair, state_pair)
            recons.append((torch.tanh(recon_next[0, -1]) + 1) / 2)

        if len(recons) > 0:
            recons = torch.stack(recons, dim=0)
        else:
            recons = torch.empty((0, C, H, W), device=device)

        return recons
