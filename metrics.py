"""Image restoration metrics: PSNR, SSIM and LPIPS."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None


def psnr(
    prediction: Tensor,
    target: Tensor,
    data_range: float = 1.0,
) -> float:
    """Compute PSNR for tensors in [0, 1]."""
    prediction = prediction.float()
    target = target.float()

    mse = torch.mean(
        (prediction - target) ** 2
    ).item()

    if mse <= 1e-12:
        return float("inf")

    return float(
        10.0 * np.log10(
            (data_range ** 2) / mse
        )
    )


def ssim(
    prediction: Tensor,
    target: Tensor,
) -> float:
    """Compute grayscale SSIM."""

    if structural_similarity is None:
        raise ImportError(
            "scikit-image is required for SSIM. "
            "Install it with: pip install scikit-image"
        )

    prediction_np = (
        prediction.detach()
        .float()
        .squeeze()
        .cpu()
        .numpy()
    )

    target_np = (
        target.detach()
        .float()
        .squeeze()
        .cpu()
        .numpy()
    )

    return float(
        structural_similarity(
            prediction_np,
            target_np,
            data_range=1.0,
        )
    )


class LPIPSLoss:
    """
    LPIPS metric wrapper.

    LPIPS expects 3-channel images in [-1, 1].
    Grayscale images are replicated to 3 channels.
    """

    def __init__(
        self,
        net: str = "alex",
        device: Optional[torch.device] = None,
    ) -> None:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS is not installed. "
                "Run: pip install lpips"
            ) from exc

        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model = lpips.LPIPS(
            net=net,
        ).to(self.device)

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def __call__(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> float:
        """
        Calculate mean LPIPS over a batch.

        Input:
            [N,1,H,W] tensors in [0,1].
        """

        prediction = prediction.float().clamp(
            0.0,
            1.0,
        )

        target = target.float().clamp(
            0.0,
            1.0,
        )

        # LPIPS expects 3-channel images.
        if prediction.shape[1] == 1:
            prediction = prediction.repeat(
                1,
                3,
                1,
                1,
            )

        if target.shape[1] == 1:
            target = target.repeat(
                1,
                3,
                1,
                1,
            )

        # Convert [0,1] -> [-1,1].
        prediction = (
            prediction * 2.0 - 1.0
        )

        target = (
            target * 2.0 - 1.0
        )

        prediction = prediction.to(
            self.device,
            non_blocking=True,
        )

        target = target.to(
            self.device,
            non_blocking=True,
        )

        value = self.model(
            prediction,
            target,
        )

        return float(
            value.mean().item()
        )