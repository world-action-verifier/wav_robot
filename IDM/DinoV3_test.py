import torch
import os

REPO_DIR = os.environ.get("DINO_REPO_DIR", "")

WEIGHT_URL = os.environ.get("DINO_WEIGHT_PATH", "")
if not REPO_DIR or not WEIGHT_URL:
    raise ValueError("Please set DINO_REPO_DIR and DINO_WEIGHT_PATH environment variables.")
# DINOv3 ViT models pretrained on web images
model = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=WEIGHT_URL)
model.eval()

x = torch.randn(1, 3, 224, 224)  # input image
with torch.no_grad():
    feats = model.forward_features(x)
    cls = feats["x_norm_clstoken"]        # [B, D] global representation
    patch = feats["x_norm_patchtokens"]   # [B, N, D] patch representation
    print(patch.shape)
    print(cls.shape)