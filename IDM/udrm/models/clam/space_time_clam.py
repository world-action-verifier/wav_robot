from typing import Tuple

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
from udrm.models.clam.state_transfer_machine import StateTransferMachine


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
        # the 0th dimension is the channel dimension
        self.patch_token_dim = C * self.cfg.patch_size**2
        self.model_dim = self.cfg.net.dim_model

        # the output_dim should be the latent dimension
        # the sequence dim is [T * N]
        # make sure H and W are divisible by patch_size
        assert H % self.cfg.patch_size == 0 and W % self.cfg.patch_size == 0
        self.num_patches = (H // self.cfg.patch_size) * (W // self.cfg.patch_size)

        # also encode the hand position
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

        # print("[IDM] observations.shape:", observations.shape)
        # [IDM] observations.shape: torch.Size([128, 3, 3, 128, 128])
        # [IDM] patches_embed.shape: torch.Size([128, 3, 64, 256])

        # need to put channel last for patchify
        observations = observations.permute(0, 1, 3, 4, 2)

        # [B, T, N, E] where N is the number of patches
        # and E is the patch token dimension
        patches = patchify(observations, self.cfg.patch_size)

        # embed the patches
        patches_embed = self.input_embed(patches)
        patches_embed = self.activation(patches_embed)
        # print("[IDM] patches_embed.shape:", patches_embed.shape)

        # HMM, adding the action token after i embed the patches
        if self.cfg.add_action_token:
            # print("Adding action token to the patches") #yes
            B, T, N, E = patches_embed.shape

            # add a dummy token to represent the latent actions
            action_pad = self.action_in.expand(B, T, 1, self.model_dim)

            # prepend the action token to the patches
            # [B, T, N+1, E]
            patches_embed = torch.cat([action_pad, patches_embed], dim=2)

        # print("[IDM] patches_embed.shape after action token:", patches_embed.shape)

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

        # if self.cfg.concatenate_gripper_state:
        #     # extract the hand position and gripper state
        #     # embed the hand position and concatenate with the patches
        #     hand_pos_gripper = extract_state_info(states)
        #     hand_pos_gripper_embed = self.hand_pos_embed(hand_pos_gripper)
        #     hand_pos_gripper_embed = einops.repeat(
        #         hand_pos_gripper_embed, "B T E -> B T N E", N=N + 1
        #     )
        #     patches_embed = self.input_embed_two(
        #         torch.cat([patches_embed, hand_pos_gripper_embed], dim=-1)
        #     )

        # [B, T, N+1, E]
        z = self.encoder(patches_embed, pos_embed=pos_embed, causal=False)
        # print("[IDM] z.shape after encoder:", z.shape)
        # [IDM] z.shape after encoder: torch.Size([128, 3, 65, 256])
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
        if self.cfg.quantize_la:
            quantized_la, indices, vq_loss = self.vq(la)

            if self.cfg.use_quantized_las:  # replace la with quantized_la
                la = quantized_la

            vq_loss = vq_loss.mean()

            vq_outputs = {"indices": indices}
            vq_metrics = {
                "vq_loss": vq_loss.item(),
                "perplexity": compute_perplexity(
                    indices, self.cfg.vq.kwargs.codebook_size
                ),
            }
            return IDMOutput(
                la=la,
                quantized_la=quantized_la,
                vq_loss=vq_loss,
                vq_metrics=vq_metrics,
                vq_outputs=vq_outputs,
                encoder_out=patches,
            )

        # return patches to use in the FDM
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
        la = idm_output.la[:, 1:] # ([1, 2, 16])

        # [B, T, N, E]
        patches = idm_output.encoder_out

        la_embed = self.la_embed(la)
        # add dimension
        la_embed = einops.rearrange(la_embed, "B T E -> B T 1 E")

        # don't feed in the last timestep, need to be causal
        patches_embed = self.patch_embed(patches[:, :-1])

        video_action_patches = la_embed + patches_embed

        if self.cfg.concatenate_gripper_state:
            # extract the hand position and gripper state
            # embed the hand position and concatenate with the patches
            hand_pos_gripper = extract_state_info(states[:, :-1])
            hand_pos_gripper_embed = self.hand_pos_embed(hand_pos_gripper)
            hand_pos_gripper_embed = einops.repeat(
                hand_pos_gripper_embed, "B T E -> B T N E", N=self.num_patches
            )
            video_action_patches = self.input_embed_two(
                torch.cat([video_action_patches, hand_pos_gripper_embed], dim=-1)
            )

        B, T_minus_one, N, E = video_action_patches.shape
        # print("timesteps: ", timesteps)
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
        self.idm = SpaceTimeIDM(cfg.idm, input_dim=input_dim, la_dim=la_dim)
        self.fdm = SpaceTimeFDM(cfg.fdm, input_dim=input_dim, la_dim=la_dim)

    def forward(
        self,
        observations,
        timesteps: torch.Tensor,
        states: torch.Tensor = None,
        **kwargs,
    ) -> CLAMOutput:
        return super().forward(observations, timesteps=timesteps, states=states)

    #Hayden - ablation experiment for video reconstruction
    @torch.no_grad()
    def rollout_idm_fdm_closed_loop(
        self,
        gt_seq: torch.Tensor,    # (T, C, H, W)
        gt_states: torch.Tensor, # (T, D) or None
        max_steps: int | None = None,
        ):

        use_gt_image: bool = False   # using all GT image?
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
        # print("timestep: ", ts[0])
        idm_out = self.idm(windows, timesteps=ts, states=None)

        recons = []
        curr_frame = gt_seq[0: idm_len].unsqueeze(0)
        ts_pair = torch.tensor([[0, 1]], device=device)
        # print("idm_out.la: ", idm_out.la.shape)
        for i in range(num_windows):
            patches = patchify(curr_frame.permute(0, 1, 3, 4, 2), self.idm.cfg.patch_size)
            la = idm_out.la[i:i+1, ...]
            idm_step = IDMOutput(la=la, encoder_out=patches)
            # print("curr_frame: ", curr_frame.shape)
            recon = self.fdm(
                curr_frame,
                idm_step,
                timesteps=ts[:1, :],
                states=None,
            )
            # curr_frame = torch.cat([curr_frame[:, 1:,...], gt_seq[idm_len+i: idm_len+i+1].unsqueeze(0)], dim=1)
            curr_frame = torch.cat([recon[:, :1,...], gt_seq[idm_len+i-1: idm_len+i+1].unsqueeze(0)], dim=1)
            # curr_frame = torch.cat([recon[:, :1,...], gt_seq[-3: -1].unsqueeze(0)], dim=1)
            # print("[Rollout] curr_frame.shape:", curr_frame.shape)
            # [Rollout] curr_frame.shape: torch.Size([3, 128, 128])
            recons.append(recon[0, 0, ...])

        recons = torch.stack(recons, dim=0)
        print(f"Dreamer-style Rollout Done. Frames: {recons.shape}")
        return recons
