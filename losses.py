"""Restoration losses: L1 + SSIM + optional LPIPS."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def ssim_map(x: Tensor, y: Tensor, window: int = 11, sigma: float = 1.5) -> Tensor:
    coords = torch.arange(window, device=x.device, dtype=x.dtype) - (window - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    pad = window // 2
    mu_x = F.conv2d(x, kernel, padding=pad)
    mu_y = F.conv2d(y, kernel, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x.pow(2), mu_y.pow(2), mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, kernel, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-8)


def ssim(x: Tensor, y: Tensor) -> Tensor:
    return ssim_map(x.float(), y.float()).mean().clamp(0, 1)


class RestorationLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.2,
        perceptual_weight: float = 0.0,
        gradient_weight: float = 0.05,
    ) -> None:
        super().__init__()

        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.gradient_weight = gradient_weight

        self.gradient_loss = GradientLoss()

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:

        l1 = F.l1_loss(pred, target)

        ssim_value = ssim(pred, target)
        ssim_loss = 1.0 - ssim_value

        gradient = self.gradient_loss(pred, target)

        total = (
            self.l1_weight * l1
            + self.ssim_weight * ssim_loss
            + self.gradient_weight * gradient
        )

        return total


class GradientLoss(nn.Module):
    """Sobel-gradient L1 loss for fine structural preservation."""

    def __init__(self) -> None:
        super().__init__()

        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_y = F.conv2d(pred, self.sobel_y, padding=1)

        target_x = F.conv2d(target, self.sobel_x, padding=1)
        target_y = F.conv2d(target, self.sobel_y, padding=1)

        return (
            F.l1_loss(pred_x, target_x)
            + F.l1_loss(pred_y, target_y)
        )