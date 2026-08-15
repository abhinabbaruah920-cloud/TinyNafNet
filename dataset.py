"""Paired .npy restoration dataset with deterministic optional validation split."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from utils import image_files, load_array


@dataclass(frozen=True)
class Pair:
    degraded: Path
    gt: Path
    key: str


def build_pairs(degraded_dir: str | Path, gt_dir: str | Path) -> list[Pair]:
    degraded_root = Path(degraded_dir)
    gt_root = Path(gt_dir)
    if not degraded_root.exists() or not gt_root.exists():
        return []

    degraded = {
        p.relative_to(degraded_root).with_suffix("").as_posix(): p
        for p in image_files(degraded_root)
    }
    gt = {
        p.relative_to(gt_root).with_suffix("").as_posix(): p
        for p in image_files(gt_root)
    }
    keys = sorted(set(degraded) & set(gt))
    return [Pair(degraded[k], gt[k], k) for k in keys]


def split_pairs(pairs: list[Pair], val_fraction: float, seed: int) -> tuple[list[Pair], list[Pair]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if val_fraction == 0 or len(pairs) < 2:
        return pairs, []
    n_val = max(1, int(round(len(pairs) * val_fraction)))
    n_val = min(n_val, len(pairs) - 1)
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    val_ids = set(indices[:n_val])
    train = [p for i, p in enumerate(pairs) if i not in val_ids]
    val = [p for i, p in enumerate(pairs) if i in val_ids]
    return train, val


class PairedRestorationDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        pairs: list[Pair],
        patch_size: int | None = 256,
        scale: int = 2,
        augment: bool = False,
        value_scale: float | None = None,
    ) -> None:
        self.pairs = pairs
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        self.value_scale = value_scale
        if not pairs:
            raise RuntimeError("Dataset contains no paired files")

    @staticmethod
    def _crop_pair(lr: Tensor, gt: Tensor, patch_size: int, scale: int) -> tuple[Tensor, Tensor]:
        lr_patch = patch_size // scale
        if lr.shape[-2] < lr_patch or lr.shape[-1] < lr_patch:
            raise ValueError(
                f"Patch {lr_patch} is larger than degraded image {tuple(lr.shape[-2:])}"
            )
        top = random.randint(0, lr.shape[-2] - lr_patch)
        left = random.randint(0, lr.shape[-1] - lr_patch)
        lr = lr[:, top:top + lr_patch, left:left + lr_patch]
        gt_top, gt_left = top * scale, left * scale
        gt = gt[:, gt_top:gt_top + patch_size, gt_left:gt_left + patch_size]
        return lr, gt

    @staticmethod
    def _geometry(lr: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
        if random.random() < 0.5:
            lr, gt = torch.flip(lr, [-1]), torch.flip(gt, [-1])
        if random.random() < 0.5:
            lr, gt = torch.flip(lr, [-2]), torch.flip(gt, [-2])
        k = random.randint(0, 3)
        if k:
            lr = torch.rot90(lr, k, [-2, -1])
            gt = torch.rot90(gt, k, [-2, -1])
        return lr, gt

    @staticmethod
    def _blur(x: Tensor, sigma: float) -> Tensor:
        radius = max(1, int(round(3 * sigma)))
        coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel2d = kernel[:, None] * kernel[None, :]
        kernel2d = kernel2d[None, None]
        return F.conv2d(x.unsqueeze(0), kernel2d, padding=radius).squeeze(0)

    @classmethod
    def _degradation_aug(cls, x: Tensor) -> Tensor:
        if random.random() < 0.25:
            contrast = random.uniform(0.90, 1.10)
            brightness = random.uniform(-0.05, 0.05)
            x = (x * contrast + brightness).clamp(0, 1)
        if random.random() < 0.20:
            x = (x + torch.randn_like(x) * random.uniform(0.003, 0.03)).clamp(0, 1)
        if random.random() < 0.20:
            x = cls._blur(x, random.uniform(0.4, 1.0)).clamp(0, 1)
        return x

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        pair = self.pairs[index]
        degraded, _, _ = load_array(pair.degraded, self.value_scale)
        gt, _, _ = load_array(pair.gt, self.value_scale)

        expected = (degraded.shape[-2] * self.scale, degraded.shape[-1] * self.scale)
        if tuple(gt.shape[-2:]) != expected:
            raise ValueError(
                f"Size mismatch for {pair.key}: degraded={tuple(degraded.shape[-2:])}, "
                f"gt={tuple(gt.shape[-2:])}, scale={self.scale}"
            )

        if self.patch_size is not None:
            degraded, gt = self._crop_pair(degraded, gt, self.patch_size, self.scale)
        if self.augment:
            degraded, gt = self._geometry(degraded, gt)
            degraded = self._degradation_aug(degraded)

        return {"input": degraded, "target": gt, "name": pair.key}
