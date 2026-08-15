"""Shared utility functions for data IO, checkpoints and reproducibility."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".npy",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_files(folder: Path) -> list[Path]:
    """Return supported files directly inside a folder."""
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _npy_to_tensor(
    path: Path,
    value_scale: float | None = None,
) -> tuple[torch.Tensor, str, int]:
    """
    Load a NumPy array as [1,H,W] float32.

    Important:
    For the KLA dataset, float32 .npy inputs are kept in their original
    numerical range. We do NOT normalize each image by its own maximum.
    """

    arr = np.load(path)

    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Invalid NumPy data in {path}")

    # Remove dimensions of size 1.
    arr = np.squeeze(arr)

    # Expected grayscale formats:
    # H x W
    # 1 x H x W
    # H x W x 1
    if arr.ndim == 2:
        pass

    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"Unsupported grayscale array shape {arr.shape} "
                f"for {path}"
            )

    else:
        raise ValueError(
            f"Unsupported NumPy shape {arr.shape} for {path}"
        )

    original_dtype = arr.dtype

    # Convert to float32 without changing the actual values.
    arr = arr.astype(np.float32, copy=False)

    # ------------------------------------------------------------
    # Explicit scale, when the user knows the dataset's range.
    # ------------------------------------------------------------
    if value_scale is not None:
        if value_scale <= 0:
            raise ValueError("value_scale must be > 0")

        arr = arr / float(value_scale)

    # ------------------------------------------------------------
    # Integer arrays can safely be converted according to dtype.
    # Float arrays are deliberately NOT normalized automatically.
    # ------------------------------------------------------------
    elif np.issubdtype(original_dtype, np.integer):
        if original_dtype == np.uint8:
            arr = arr / 255.0

        elif original_dtype == np.uint16:
            arr = arr / 65535.0

        elif original_dtype == np.int16:
            # Conservative handling for signed 16-bit data.
            info = np.iinfo(original_dtype)
            arr = (arr - info.min) / float(info.max - info.min)

        elif original_dtype == np.int32:
            info = np.iinfo(original_dtype)
            arr = (arr - info.min) / float(info.max - info.min)

        elif original_dtype == np.uint32:
            arr = arr / float(np.iinfo(original_dtype).max)

        else:
            info = np.iinfo(original_dtype)
            arr = (arr - info.min) / float(info.max - info.min)

    tensor = torch.from_numpy(
        np.ascontiguousarray(arr)
    ).unsqueeze(0)

    # Do NOT clamp .npy inputs here.
    #
    # Your actual KLA NoisyLR data contains values such as:
    # min ~= -0.0026
    # max ~= 1.3258
    #
    # Those values should reach the model unchanged.

    return tensor, "L", 32


def _image_to_tensor(
    path: Path,
) -> tuple[torch.Tensor, str, int]:
    """Load PNG/JPEG/TIFF as [1,H,W] float32 in [0,1]."""

    with Image.open(path) as image:
        # Convert multi-channel images to grayscale.
        if image.mode not in {"L", "I", "I;16", "F"}:
            image = image.convert("L")

        arr = np.asarray(image)
        mode = image.mode

    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.dtype == np.uint8:
        scale = 255.0
        bits = 8

    elif arr.dtype == np.uint16:
        scale = 65535.0
        bits = 16

    elif np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = float(info.max)
        bits = arr.dtype.itemsize * 8

    elif np.issubdtype(arr.dtype, np.floating):
        scale = 1.0
        bits = 32

    else:
        raise ValueError(
            f"Unsupported image dtype {arr.dtype} for {path}"
        )

    arr = arr.astype(np.float32) / scale

    tensor = torch.from_numpy(
        np.ascontiguousarray(arr)
    ).unsqueeze(0)

    return tensor.clamp(0.0, 1.0), mode, bits


def load_array(
    path: Path,
    value_scale: float | None = None,
) -> tuple[torch.Tensor, str, int]:
    """
    Load .npy or standard image files.

    Returns:
        tensor: [1,H,W] float32
        mode: source mode
        bits: effective bit depth
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".npy":
        return _npy_to_tensor(path, value_scale)

    if suffix in {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }:
        return _image_to_tensor(path)

    raise ValueError(
        f"Unsupported file extension '{suffix}' for {path}"
    )


def save_array_or_image(
    tensor: torch.Tensor,
    path: Path,
    bits: int = 32,
) -> None:
    """
    Save a tensor to .npy or a standard image format.

    Output is clipped to [0,1] because the ground-truth domain is [0,1].
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = (
        tensor.detach()
        .float()
        .squeeze()
        .cpu()
        .numpy()
    )

    suffix = path.suffix.lower()

    if suffix == ".npy":
        # Model output is expected to represent normalized GT values.
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
        np.save(path, x)
        return

    x = np.clip(x, 0.0, 1.0)

    if bits > 8:
        arr = np.round(x * 65535.0).astype(np.uint16)
        Image.fromarray(arr, mode="I;16").save(path)
    else:
        arr = np.round(x * 255.0).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(path)


def save_image(
    tensor: torch.Tensor,
    path: Path,
    bits: int = 8,
) -> None:
    """Backward-compatible image-saving wrapper."""
    save_array_or_image(tensor, path, bits)


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Save a dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def count_parameters(
    model: torch.nn.Module,
) -> int:
    """Count trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def parameter_millions(
    model: torch.nn.Module,
) -> float:
    """Return trainable parameter count in millions."""
    return count_parameters(model) / 1e6


def resolve_device(
    requested: str,
) -> torch.device:
    """Resolve auto/cpu/cuda device selection."""

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def torch_load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load a checkpoint safely."""
    return torch.load(
        path,
        map_location=device,
        weights_only=False,
    )


class AverageMeter:
    """Running average accumulator."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(
        self,
        value: float,
        count: int = 1,
    ) -> None:
        self.total += value * count
        self.count += count

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)