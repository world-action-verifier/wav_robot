"""Load the frozen IDM architecture from checkpoint tensor shapes."""
import torch


def load_idm_tokenizer_head(ckpt_path: str, *, device, n_latents: int, d_bottleneck: int):
    """Load a trained tokenizer-arm inverse dynamics head (`idm_tokenizer.py`).

    The architecture is recovered from the checkpoint's own tensor shapes, so
    any `--d_hidden` loads without being told which one was used.

    Only the tokenizer arm applies here. It maps a pair of latents
    (z_t, z_{t+1}) to an action, which is exactly what this method has at each
    step. The backbone arm (`idm_backbone.py`) reads contextual features out of
    a Dynamics forward over an observed window, which is a different input
    contract.
    """
    from idm_tokenizer import InverseDynamicsHeadTokenizer

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    d_hidden, d_in3 = sd["in_proj.weight"].shape
    d_in = d_in3 // 3
    expected = n_latents * d_bottleneck
    if d_in != expected:
        raise ValueError(
            f"IDM checkpoint expects flattened latents of {d_in}, but the tokenizer "
            f"gives n_latents*d_bottleneck = {expected}. Checkpoint/tokenizer mismatch."
        )
    head = InverseDynamicsHeadTokenizer(
        n_latents=n_latents, d_bottleneck=d_bottleneck, d_hidden=int(d_hidden),
    )
    head.load_state_dict(sd)
    head.to(device).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    meta = {"step": (ck.get("step") if isinstance(ck, dict) else None),
            "d_hidden": int(d_hidden), "d_in": int(d_in)}
    return head, meta
