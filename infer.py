"""Fast batch inference for TinyNAFNet.

Supports:
    - .npy
    - PNG / JPEG / TIFF
    - grayscale 2x super-resolution
    - FP16/BF16 CUDA inference
    - batched inference
    - channels_last
    - CUDA warm-up
    - optional torch.compile
    - forward-time throughput
    - end-to-end throughput
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import ModelConfig, build_model
from utils import (
    image_files,
    load_array,
    resolve_device,
    save_array_or_image,
)


class InferenceDataset(Dataset[dict[str, object]]):
    """Load individual inference samples."""

    def __init__(
        self,
        root: Path,
        value_scale: float | None = None,
    ) -> None:
        self.root = Path(root)
        self.files = image_files(self.root)
        self.value_scale = value_scale

        if not self.files:
            raise RuntimeError(
                f"No supported files found in {self.root}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        path = self.files[index]

        tensor, _, bits = load_array(
            path,
            self.value_scale,
        )

        return {
            "input": tensor,
            "path": str(path),
            "bits": bits,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast TinyNAFNet inference."
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input folder.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output folder.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=28,
        help="Model width if not stored in checkpoint.",
    )

    parser.add_argument(
        "--middle-blocks",
        type=int,
        default=2,
        help="Middle NAFBlocks if not stored in checkpoint.",
    )

    parser.add_argument(
        "--scale",
        type=int,
        choices=(1, 2, 4),
        default=2,
        help="Upscaling factor if not stored in checkpoint.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device.",
    )

    parser.add_argument(
        "--value-scale",
        type=float,
        default=None,
        help=(
            "Optional explicit divisor for .npy values. "
            "Do not use this for your current float32 KLA data."
        ),
    )

    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default="fp16",
        help="CUDA autocast precision.",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile(mode='max-autotune').",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of CUDA warm-up iterations.",
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    fallback_width: int,
    fallback_middle_blocks: int,
    fallback_scale: int,
) -> torch.nn.Module:
    """Load TinyNAFNet and recover configuration from checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state = checkpoint.get(
        "model",
        checkpoint,
    )

    saved_cfg = checkpoint.get(
        "model_config",
        None,
    )

    if saved_cfg is not None:
        width = int(
            saved_cfg.get(
                "width",
                fallback_width,
            )
        )

        middle_blocks = int(
            saved_cfg.get(
                "middle_blocks",
                fallback_middle_blocks,
            )
        )

        scale = int(
            saved_cfg.get(
                "scale",
                fallback_scale,
            )
        )

        print(
            "Using model configuration from checkpoint:"
        )

        print(
            f"  width={width}"
        )

        print(
            f"  middle_blocks={middle_blocks}"
        )

        print(
            f"  scale={scale}"
        )

    else:
        width = fallback_width
        middle_blocks = fallback_middle_blocks
        scale = fallback_scale

        print(
            "Checkpoint does not contain model_config."
        )

        print(
            "Using command-line configuration:"
        )

        print(
            f"  width={width}"
        )

        print(
            f"  middle_blocks={middle_blocks}"
        )

        print(
            f"  scale={scale}"
        )

    model_cfg = ModelConfig(
        width=width,
        middle_blocks=middle_blocks,
        scale=scale,
    )

    model = build_model(
        model_cfg
    ).to(device)

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    return model


def get_autocast_context(
    device: torch.device,
    amp_dtype: torch.dtype,
):
    """Return CUDA autocast or a no-op context."""

    if device.type != "cuda":
        return nullcontext()

    return torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=True,
    )


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than 0"
        )

    if args.workers < 0:
        raise ValueError(
            "--workers cannot be negative"
        )

    if args.warmup < 0:
        raise ValueError(
            "--warmup cannot be negative"
        )

    device = resolve_device(
        args.device
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA build: {torch.version.cuda}"
        )

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision(
            "high"
        )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = load_model(
        checkpoint_path=args.weights,
        device=device,
        fallback_width=args.width,
        fallback_middle_blocks=args.middle_blocks,
        fallback_scale=args.scale,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,} "
        f"({parameter_count / 1e6:.3f}M)"
    )

    # -------------------------------------------------------------
    # Channels-last
    # -------------------------------------------------------------

    use_channels_last = (
        device.type == "cuda"
    )

    if use_channels_last:
        model = model.to(
            memory_format=torch.channels_last
        )

    # -------------------------------------------------------------
    # Optional torch.compile
    # -------------------------------------------------------------

    if args.compile and device.type == "cuda":
        try:
            model = torch.compile(
                model,
                mode="reduce-overhead",
            )
            print("torch.compile enabled: reduce-overhead")
        except Exception as exc:
            print(f"torch.compile unavailable: {exc}")
            print("Falling back to eager CUDA inference.")

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    dataset = InferenceDataset(
        root=args.input,
        value_scale=args.value_scale,
    )

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }

    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(
        **loader_kwargs
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    amp_dtype = (
        torch.float16
        if args.amp_dtype == "fp16"
        else torch.bfloat16
    )

    # -------------------------------------------------------------
    # Warm-up
    #
    # We obtain one batch first so warm-up uses the actual input
    # dimensions instead of assuming 128x128.
    # -------------------------------------------------------------

    warmup_batch = None

    if device.type == "cuda" and args.warmup > 0:
        print(
            f"Running {args.warmup} CUDA warm-up iterations..."
        )

        warmup_iterator = iter(loader)

        try:
            warmup_batch = next(
                warmup_iterator
            )
        except StopIteration:
            raise RuntimeError(
                f"No supported files found in {args.input}"
            )

        warmup_input = warmup_batch["input"].to(
            device,
            non_blocking=True,
        )

        if use_channels_last:
            warmup_input = warmup_input.to(
                memory_format=torch.channels_last
            )

        # We only need enough iterations to warm up kernels.
        with torch.inference_mode():
            for _ in range(args.warmup):
                with get_autocast_context(
                    device,
                    amp_dtype,
                ):
                    _ = model(
                        warmup_input
                    )

        torch.cuda.synchronize()

        del warmup_input

    # -------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------

    forward_time = 0.0
    end_to_end_start = time.perf_counter()

    image_count = 0

    print()

    with torch.inference_mode():

        for batch in tqdm(
            loader,
            desc="Inference",
        ):
            # -----------------------------------------------------
            # CPU → GPU
            # -----------------------------------------------------

            x = batch["input"].to(
                device,
                non_blocking=True,
            )

            if use_channels_last:
                x = x.to(
                    memory_format=torch.channels_last
                )

            # -----------------------------------------------------
            # GPU forward timing
            # -----------------------------------------------------

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            with get_autocast_context(
                device,
                amp_dtype,
            ):
                y = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            forward_time += (
                time.perf_counter()
                - start
            )

            # -----------------------------------------------------
            # Clamp output to GT domain
            # -----------------------------------------------------

            y = y.float().clamp(
                0.0,
                1.0,
            )

            # -----------------------------------------------------
            # Save outputs
            # -----------------------------------------------------

            paths = batch["path"]
            bits = batch["bits"]

            for index, path_string in enumerate(
                paths
            ):
                path = Path(
                    path_string
                )

                # Preserve directory structure relative to input.
                rel = path.relative_to(
                    args.input
                )

                output_path = (
                    args.output / rel
                )

                extension = (
                    path.suffix.lower()
                )

                if extension not in {
                    ".npy",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tif",
                    ".tiff",
                }:
                    output_path = (
                        output_path.with_suffix(
                            ".npy"
                        )
                    )

                save_array_or_image(
                    y[index],
                    output_path,
                    int(bits[index]),
                )

            image_count += (
                x.shape[0]
            )

    end_to_end_time = (
        time.perf_counter()
        - end_to_end_start
    )

    # -------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------

    forward_images_per_sec = (
        image_count
        / max(
            forward_time,
            1e-9,
        )
    )

    forward_ms_per_image = (
        1000.0
        * forward_time
        / max(
            image_count,
            1,
        )
    )

    end_to_end_images_per_sec = (
        image_count
        / max(
            end_to_end_time,
            1e-9,
        )
    )

    end_to_end_ms_per_image = (
        1000.0
        * end_to_end_time
        / max(
            image_count,
            1,
        )
    )

    print()
    print("=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)

    print(
        f"Images:                 {image_count}"
    )

    print(
        f"Forward time:           {forward_time:.4f} s"
    )

    print(
        f"Forward images/sec:     "
        f"{forward_images_per_sec:.3f}"
    )

    print(
        f"Forward ms/image:       "
        f"{forward_ms_per_image:.3f}"
    )

    print(
        f"End-to-end time:        "
        f"{end_to_end_time:.4f} s"
    )

    print(
        f"End-to-end images/sec:  "
        f"{end_to_end_images_per_sec:.3f}"
    )

    print(
        f"End-to-end ms/image:    "
        f"{end_to_end_ms_per_image:.3f}"
    )

    print(
        f"Batch size:             {args.batch_size}"
    )

    print(
        f"AMP dtype:              {args.amp_dtype}"
    )

    print(
        f"Channels-last:          {use_channels_last}"
    )

    print(
        f"Compiled:               "
        f"{bool(args.compile and device.type == 'cuda')}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()