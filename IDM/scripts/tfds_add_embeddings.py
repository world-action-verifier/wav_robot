#!/usr/bin/env python3
"""
Add visual embeddings to a TFDS dataset and save as a new TFDS dataset.
The output dataset name becomes: {ds_name}_emb_{model_name} (or override).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./IDM/data", help="Root containing tensorflow_datasets/")
    parser.add_argument("--dataset-name", required=True, help="TFDS group name")
    parser.add_argument("--ds-name", required=True, help="TFDS dataset name")
    parser.add_argument("--dataset-variant", default=None)
    parser.add_argument("--model-name", default="dinov3", help="Embedding model name label")
    parser.add_argument("--output-ds-name", default=None, help="Override output dataset name")
    parser.add_argument("--image-key", default="images", help="Key to read images from")
    parser.add_argument("--image-size", type=int, default=224, help="Resize images before encoding")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda or cpu (auto if omitted)")
    parser.add_argument("--overwrite", action="store_true")

    # DINOv3 options (local repo + weights)
    parser.add_argument("--dino-repo-dir", default=None, help="Local DINOv3 repo path")
    parser.add_argument("--dino-weights", default=None, help="Local DINOv3 checkpoint path")
    parser.add_argument("--dino-arch", default="dinov3_vitb16", help="Torch hub entry name")
    parser.add_argument("--dino-mean", default="0.485,0.456,0.406")
    parser.add_argument("--dino-std", default="0.229,0.224,0.225")
    return parser.parse_args()


def get_dataset_dir(base_dir: Path, group_name: str, ds_name: str, variant: Optional[str]) -> Path:
    if variant:
        return base_dir / "tensorflow_datasets" / group_name / ds_name / variant
    return base_dir / "tensorflow_datasets" / group_name / ds_name


def parse_float_list(value: str, length: int) -> Tuple[float, ...]:
    items = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(items) != length:
        raise ValueError(f"Expected {length} floats, got {len(items)} from: {value}")
    return tuple(items)


def ensure_hwc(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (T,H,W,C), got {frames.shape}")
    if frames.shape[-1] in (1, 3, 4):
        return frames
    if frames.shape[1] in (1, 3, 4):
        return np.transpose(frames, (0, 2, 3, 1))
    return frames


class DinoV3Embedder:
    def __init__(
        self,
        repo_dir: str,
        weights: str,
        arch: str,
        device: torch.device,
        image_size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        batch_size: int,
    ):
        if not repo_dir or not weights:
            raise ValueError("DINOv3 requires --dino-repo-dir and --dino-weights")
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.model = torch.hub.load(repo_dir, arch, source="local", weights=weights)
        self.model.eval().to(device)

    @torch.no_grad()
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        frames = ensure_hwc(frames)
        x = torch.from_numpy(frames).to(self.device).float() / 255.0
        x = x.permute(0, 3, 1, 2)
        if self.image_size > 0 and (x.shape[-1] != self.image_size or x.shape[-2] != self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std

        outputs = []
        for i in range(0, x.shape[0], self.batch_size):
            feats = self.model.forward_features(x[i : i + self.batch_size])
            cls = feats["x_norm_clstoken"]
            outputs.append(cls.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


def make_spec_from_episode(ep: Dict[str, np.ndarray]) -> Dict[str, tf.TensorSpec]:
    spec = {}
    for key, value in ep.items():
        if not isinstance(value, np.ndarray):
            continue
        shape = list(value.shape)
        if len(shape) >= 1:
            shape[0] = None
        spec[key] = tf.TensorSpec(shape=tuple(shape), dtype=tf.dtypes.as_dtype(value.dtype))
    return spec


def embed_episode(
    episode: Dict[str, np.ndarray],
    image_key: str,
    embedder,
) -> Dict[str, np.ndarray]:
    if image_key not in episode:
        raise KeyError(f"Missing image key '{image_key}' in episode")
    frames = episode[image_key]
    embeddings = embedder(frames)
    out = dict(episode)
    out["embeddings"] = embeddings
    return out


def generator_with_first(
    iterator: Iterator[Dict[str, np.ndarray]],
    first: Dict[str, np.ndarray],
    embedder,
    image_key: str,
) -> Iterator[Dict[str, np.ndarray]]:
    yield embed_episode(first, image_key=image_key, embedder=embedder)
    for ep in iterator:
        yield embed_episode(ep, image_key=image_key, embedder=embedder)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.data_dir)
    in_dir = get_dataset_dir(base_dir, args.dataset_name, args.ds_name, args.dataset_variant)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input dataset not found: {in_dir}")

    model_name = args.model_name
    out_ds_name = args.output_ds_name or f"{args.ds_name}_emb_{model_name}"
    out_dir = get_dataset_dir(base_dir, args.dataset_name, out_ds_name, args.dataset_variant)

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset exists: {out_dir} (use --overwrite)")
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if model_name.lower() == "dinov3":
        mean = parse_float_list(args.dino_mean, 3)
        std = parse_float_list(args.dino_std, 3)
        embedder = DinoV3Embedder(
            repo_dir=args.dino_repo_dir,
            weights=args.dino_weights,
            arch=args.dino_arch,
            device=device,
            image_size=args.image_size,
            mean=mean,
            std=std,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    ds = tf.data.experimental.load(str(in_dir))
    it = ds.as_numpy_iterator()
    first = next(it, None)
    if first is None:
        raise ValueError("Empty dataset.")

    first_emb = embed_episode(first, image_key=args.image_key, embedder=embedder)
    spec = make_spec_from_episode(first_emb)

    def gen():
        yield first_emb
        for ep in it:
            yield embed_episode(ep, image_key=args.image_key, embedder=embedder)

    wrapped = tqdm(gen(), desc="Embedding episodes", unit="ep")

    out_ds = tf.data.Dataset.from_generator(lambda: wrapped, output_signature=spec)
    if hasattr(tf.data.Dataset, "save"):
        out_ds.save(str(out_dir))
    else:
        tf.data.experimental.save(out_ds, str(out_dir))
    print(f"Saved dataset to: {out_dir}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
