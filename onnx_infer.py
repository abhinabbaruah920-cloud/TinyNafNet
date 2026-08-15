"""Production ONNX Runtime inference for TinyNAFNet.

Target:
    KLA semiconductor restoration
    Input : 128x128 grayscale .npy
    Output: 256x256 grayscale .npy

Designed for CUDA GPUs and large-batch inference.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import image_files, load_array, save_array_or_image


SUPPORTED_INPUT_EXTENSIONS = {
    ".npy",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}


class InferenceDataset(Dataset[dict[str, object]]):
    """Folder dataset for inference."""

    def __init__(
        self,
        root: Path,
        value_scale: float | None = None,
    ) -> None:
        self.root = Path(root)
        self.value_scale = value_scale

        self.files = [
            p
            for p in image_files(self.root)
            if p.suffix.lower()
            in SUPPORTED_INPUT_EXTENSIONS
        ]

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
        description="Fast TinyNAFNet ONNX inference."
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="ONNX model.",
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
        "--warmup",
        type=int,
        default=10,
        help="Warm-up iterations.",
    )

    parser.add_argument(
        "--value-scale",
        type=float,
        default=None,
        help="Optional explicit .npy scale.",
    )

    parser.add_argument(
        "--no-io-binding",
        action="store_true",
        help="Disable CUDA I/O binding.",
    )

    return parser.parse_args()


def create_session(
    model_path: Path,
) -> ort.InferenceSession:
    """Create optimized CUDA ONNX Runtime session."""

    providers = ort.get_available_providers()

    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable.\n"
            f"Available providers: {providers}"
        )

    session_options = ort.SessionOptions()

    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ],
    )


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
        )

    session = create_session(
        args.model
    )

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    input_type = input_meta.type

    if "float16" in input_type:
        input_dtype = np.float16
    elif "float" in input_type:
        input_dtype = np.float32
    else:
        raise RuntimeError(
            f"Unsupported ONNX input type: {input_type}"
        )

    print(
        f"Model: {args.model}"
    )

    print(
        "Providers:",
        session.get_providers(),
    )

    print(
        f"Input:  {input_meta.name} "
        f"{input_meta.shape} "
        f"{input_type}"
    )

    print(
        f"Output: {output_meta.name} "
        f"{output_meta.shape}"
    )

    dataset = InferenceDataset(
        args.input,
        args.value_scale,
    )

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
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

    iterator = iter(loader)

    try:
        first_batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(
            "Input directory is empty."
        ) from exc

    def make_numpy_batch(
        batch: dict[str, object],
    ) -> np.ndarray:
        return (
            batch["input"]
            .numpy()
            .astype(
                input_dtype,
                copy=False,
            )
        )

    first_x = make_numpy_batch(
        first_batch
    )

    use_io_binding = (
        not args.no_io_binding
    )

    def run_inference(
        x: np.ndarray,
    ) -> np.ndarray:
        """Run one ONNX batch."""

        if not use_io_binding:
            return session.run(
                [output_meta.name],
                {
                    input_meta.name: x,
                },
            )[0]

        ort_input = (
            ort.OrtValue.ortvalue_from_numpy(
                x,
                "cuda",
                0,
            )
        )

        binding = session.io_binding()

        binding.bind_ortvalue_input(
            input_meta.name,
            ort_input,
        )

        binding.bind_output(
            output_meta.name,
            "cuda",
            0,
        )

        session.run_with_iobinding(
            binding
        )

        return binding.copy_outputs_to_cpu()[0]

    # -------------------------------------------------------------
    # Warm-up
    # -------------------------------------------------------------

    print(
        f"Warm-up iterations: {args.warmup}"
    )

    for _ in range(args.warmup):
        _ = run_inference(
            first_x
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # -------------------------------------------------------------
    # Timed inference
    # -------------------------------------------------------------

    total_images = 0
    forward_time = 0.0

    start_total = time.perf_counter()

    batches = [first_batch]

    for batch in iterator:
        batches.append(batch)

    for batch in tqdm(
        batches,
        desc="ONNX inference",
    ):
        x = make_numpy_batch(batch)

        start = time.perf_counter()

        y = run_inference(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        forward_time += (
            time.perf_counter()
            - start
        )

        y = np.clip(
            y,
            0.0,
            1.0,
        )

        paths = batch["path"]
        bits = batch["bits"]

        for index, path_string in enumerate(paths):
            path = Path(path_string)

            relative = path.relative_to(
                args.input
            )

            output_path = (
                args.output / relative
            )

            if path.suffix.lower() not in {
                ".npy",
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
            }:
                output_path = (
                    output_path.with_suffix(".npy")
                )

            save_array_or_image(
                torch.from_numpy(
                    y[index]
                ),
                output_path,
                int(bits[index]),
            )

        total_images += x.shape[0]

    total_time = (
        time.perf_counter()
        - start_total
    )

    forward_images_per_sec = (
        total_images
        / max(
            forward_time,
            1e-9,
        )
    )

    forward_ms = (
        1000.0
        * forward_time
        / max(
            total_images,
            1,
        )
    )

    end_to_end_images_per_sec = (
        total_images
        / max(
            total_time,
            1e-9,
        )
    )

    end_to_end_ms = (
        1000.0
        * total_time
        / max(
            total_images,
            1,
        )
    )

    print()
    print("=" * 64)
    print("FINAL ONNX INFERENCE RESULTS")
    print("=" * 64)
    print(
        f"Images:                 {total_images}"
    )
    print(
        f"Input dtype:            {input_type}"
    )
    print(
        f"Batch size:             {args.batch_size}"
    )
    print(
        f"I/O binding:            {use_io_binding}"
    )
    print()
    print(
        f"Forward time:           "
        f"{forward_time:.4f} s"
    )
    print(
        f"Forward images/sec:     "
        f"{forward_images_per_sec:.3f}"
    )
    print(
        f"Forward ms/image:       "
        f"{forward_ms:.3f}"
    )
    print()
    print(
        f"End-to-end time:        "
        f"{total_time:.4f} s"
    )
    print(
        f"End-to-end images/sec:  "
        f"{end_to_end_images_per_sec:.3f}"
    )
    print(
        f"End-to-end ms/image:    "
        f"{end_to_end_ms:.3f}"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()