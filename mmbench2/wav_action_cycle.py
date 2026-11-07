"""Online action-cycle score. This is distinct from offline state-cycle WAV.

Coverage-aware WM latents MUST be decoded and base-tokenizer encoded before
being passed to the released IDM. No dynamics pass is added by this scorer.
"""
import torch
from model import temporal_patchify
from train_dynamics import decode_packed_to_frames


def masked_action_error(predicted, requested, mask):
    if predicted.shape != requested.shape:
        raise ValueError((predicted.shape, requested.shape))
    if mask is None:
        raise ValueError("An explicit task action mask is required")
    mask = mask.float()
    if mask.ndim == 3:
        mask = mask[:, -1]
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    mask = mask.expand_as(requested)
    if (mask.sum(-1) <= 0).any():
        raise ValueError("Task has no valid action dimensions")
    return ((predicted.float() - requested.float()).square() * mask).sum(-1) / mask.sum(-1)


class WAVActionCycleScorer:
    def __init__(self, decoder, base_encoder, head, *, patch=14,
                 packing_factor=2, image_size=224):
        self.decoder = decoder
        self.base_encoder = base_encoder
        self.head = head
        self.patch = patch
        self.packing_factor = packing_factor
        self.image_size = image_size
        self.root_features = None
        self.previous_features = None
        self.bridge_calls = 0

    @torch.no_grad()
    def encode_image(self, image):
        with torch.autocast(device_type=image.device.type, dtype=torch.bfloat16,
                            enabled=image.is_cuda):
            return self.base_encoder(temporal_patchify(image, self.patch))[0]

    @torch.no_grad()
    def set_root_frame(self, frame):
        """Real current RGB, shape (1,1,3,H,W); no future observation."""
        self.root_features = self.encode_image(frame).detach()
        self.previous_features = None

    @torch.no_grad()
    def bridge(self, packed):
        self.bridge_calls += 1
        with torch.autocast(device_type=packed.device.type, dtype=torch.bfloat16,
                            enabled=packed.is_cuda):
            frames = decode_packed_to_frames(
                self.decoder, z_packed=packed[:, None],
                H=self.image_size, W=self.image_size, C=3,
                patch=self.patch, packing_factor=self.packing_factor)
            return self.encode_image(frames)

    @torch.no_grad()
    def begin_rollout(self, previous):
        # Reset once per CEM iteration, rather than carrying a candidate's
        # terminal imagined state into the next iteration's initial state.
        if self.root_features is None:
            self.previous_features = self.bridge(previous)
        else:
            self.previous_features = self.root_features.expand(len(previous), -1, -1, -1)

    @torch.no_grad()
    def infer(self, previous, following):
        with torch.autocast(device_type=previous.device.type, dtype=torch.bfloat16,
                            enabled=previous.is_cuda):
            return self.head(previous, following)[:, 0].float()

    @torch.no_grad()
    def score_transition(self, predictions, previous, actions, mask):
        if self.previous_features is None:
            self.begin_rollout(previous)
        next_features = self.bridge(predictions.float().mean(1))
        inferred = self.infer(self.previous_features, next_features)
        self.previous_features = next_features.detach()
        score = masked_action_error(inferred, actions, mask)
        if not torch.isfinite(score).all():
            raise FloatingPointError("Non-finite WAV action-cycle score")
        return score

    @torch.no_grad()
    def score_independent(self, predictions, previous, actions, mask):
        """Logging a real step must not reuse a terminal CEM cache."""
        self.begin_rollout(previous)
        return self.score_transition(predictions, previous, actions, mask)
