# NOTE:
# Keep imports lazy to avoid hard failures from optional dependencies
# (e.g., dinov3/torchvision variants) when the selected model doesn't need them.


def get_la_dim(cfg):
    if "hierarchical_clam" in cfg.name:
        if cfg.model.idm.add_continuous_offset:
            # if we add them, the discrete and continuous part have the same dimension
            la_dim = cfg.model.la_dim
        else:
            la_dim = cfg.model.la_dim + cfg.model.idm.discrete_latent_dim
    else:
        la_dim = cfg.model.la_dim

    return la_dim


def get_clam_cls(name):
    if name == "clam":
        from udrm.models.clam.clam import ContinuousLAM

        return ContinuousLAM
    elif name == "transformer_clam":
        from udrm.models.clam.transformer_clam import TransformerCLAM

        return TransformerCLAM
    elif name == "st_vivit_clam":
        from udrm.models.clam.space_time_clam import SpaceTimeCLAM as SpaceTimeCLAM_Base

        return SpaceTimeCLAM_Base
    elif name == "st_vivit_clam_stm":
        from udrm.models.clam.space_time_clam_STM import SpaceTimeCLAM as SpaceTimeCLAM_STM

        return SpaceTimeCLAM_STM
    elif name == "diffusion_clam":
        from udrm.models.clam.diffusion_clam import DiffusionCLAM

        return DiffusionCLAM
    elif name == "nsvq_clam":
        from udrm.models.clam.space_time_clam_NSVQ import SpaceTimeCLAM_NSVQ

        return SpaceTimeCLAM_NSVQ
    elif name == "tssm_clam":
        from udrm.models.clam.space_time_clam_TSSM import SpaceTimeCLAM_TSSM

        return SpaceTimeCLAM_TSSM
    else:
        raise ValueError(f"Unknown CLAM model {name}")


def get_idm_cls(name):
    if name == "mlp":
        from udrm.models.clam.clam import IDM

        return IDM
    elif name == "transformer":
        from udrm.models.clam.transformer_clam import TransformerIDM

        return TransformerIDM
    else:
        raise ValueError(f"Unknown IDM model {name}")
