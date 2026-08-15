"""Export TinyNAFNet to TorchScript and ONNX.

Designed for the KLA 2x grayscale restoration task:

    Input : [N, 1, 128, 128]
    Output: [N, 1, 256, 256]

The architecture configuration is read from the checkpoint whenever
available, preventing width/scale mismatches during export.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from model import ModelConfig, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TinyNAFNet to TorchScript and ONNX."
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/best.pt"),
        help="Path to TinyNAFNet checkpoint.",
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("outputs/tiny_nafnet.onnx"),
        help="Output ONNX file.",
    )

    parser.add_argument(
        "--onnx-fp16",
        type=Path,
        default=None,
        help=(
            "Optional FP16 ONNX output. "
            "Requires onnx and onnxconverter-common."
        ),
    )

    parser.add_argument(
        "--torchscript",
        type=Path,
        default=None,
        help="Optional TorchScript output.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Input height.",
    )

    parser.add_argument(
        "--width-px",
        type=int,
        default=128,
        help="Input width.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Export batch size.",
    )

    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Allow dynamic batch dimension in ONNX.",
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version.",
    )

    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Export device.",
    )

    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load model and recover architecture configuration."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model",
        checkpoint,
    )

    saved_config = checkpoint.get(
        "model_config",
        None,
    )

    if saved_config is None:
        print(
            "Warning: checkpoint does not contain "
            "'model_config'. Using defaults."
        )

        cfg = ModelConfig(
            width=28,
            middle_blocks=2,
            scale=2,
        )

    else:
        cfg = ModelConfig(
            width=int(
                saved_config.get(
                    "width",
                    28,
                )
            ),
            enc_blocks=tuple(
                saved_config.get(
                    "enc_blocks",
                    (1, 1, 1, 1),
                )
            ),
            dec_blocks=tuple(
                saved_config.get(
                    "dec_blocks",
                    (1, 1, 1, 1),
                )
            ),
            middle_blocks=int(
                saved_config.get(
                    "middle_blocks",
                    2,
                )
            ),
            scale=int(
                saved_config.get(
                    "scale",
                    2,
                )
            ),
            dw_expand=int(
                saved_config.get(
                    "dw_expand",
                    2,
                )
            ),
            ffn_expand=int(
                saved_config.get(
                    "ffn_expand",
                    2,
                )
            ),
        )

    model = build_model(cfg).to(device)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    config_dict = {
        "width": cfg.width,
        "enc_blocks": cfg.enc_blocks,
        "dec_blocks": cfg.dec_blocks,
        "middle_blocks": cfg.middle_blocks,
        "scale": cfg.scale,
        "dw_expand": cfg.dw_expand,
        "ffn_expand": cfg.ffn_expand,
    }

    return model, config_dict


def count_parameters(
    model: torch.nn.Module,
) -> int:
    """Count trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    dummy_input: torch.Tensor,
    opset: int,
    dynamic_batch: bool,
) -> None:
    """Export model to ONNX."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dynamic_batch:
        dynamic_axes = {
            "input": {
                0: "batch",
            },
            "output": {
                0: "batch",
            },
        }

    else:
        dynamic_axes = None

    print()
    print("Exporting ONNX...")
    print(f"  Output: {output_path}")
    print(f"  Opset : {opset}")
    print(
        f"  Input : {tuple(dummy_input.shape)}"
    )

    with torch.inference_mode():

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    print(
        f"ONNX export complete: "
        f"{output_path}"
    )


def export_torchscript(
    model: torch.nn.Module,
    output_path: Path,
    dummy_input: torch.Tensor,
) -> None:
    """Export model to TorchScript."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Exporting TorchScript...")

    with torch.inference_mode():
        traced = torch.jit.trace(
            model,
            dummy_input,
            strict=True,
        )

        traced = torch.jit.freeze(
            traced
        )

        traced.save(
            str(output_path)
        )

    print(
        f"TorchScript export complete: "
        f"{output_path}"
    )


def validate_onnx(
    onnx_path: Path,
    dummy_input: torch.Tensor,
    pytorch_output: torch.Tensor,
) -> None:
    """Validate exported ONNX output against PyTorch."""

    try:
        import onnx
    except ImportError:
        print(
            "onnx package not installed. "
            "Skipping ONNX structural validation."
        )
        return

    print()
    print("Validating ONNX model...")

    model = onnx.load(
        str(onnx_path)
    )

    onnx.checker.check_model(
        model
    )

    print(
        "ONNX structural validation passed."
    )

    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "onnxruntime not installed. "
            "Skipping numerical validation."
        )
        return

    providers = [
        "CPUExecutionProvider",
    ]

    if "CUDAExecutionProvider" in (
        ort.get_available_providers()
    ):
        providers.insert(
            0,
            "CUDAExecutionProvider",
        )

    session = ort.InferenceSession(
        str(onnx_path),
        providers=providers,
    )

    input_name = (
        session.get_inputs()[0].name
    )

    # ONNX Runtime accepts NumPy arrays.
    input_numpy = (
        dummy_input
        .detach()
        .cpu()
        .numpy()
    )

    output = session.run(
        None,
        {
            input_name: input_numpy,
        },
    )[0]

    pytorch_numpy = (
        pytorch_output
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    if output.shape != pytorch_numpy.shape:
        raise RuntimeError(
            "ONNX output shape mismatch: "
            f"ONNX={output.shape}, "
            f"PyTorch={pytorch_numpy.shape}"
        )

    max_error = float(
        abs(
            output.astype("float32")
            - pytorch_numpy.astype("float32")
        ).max()
    )

    mean_error = float(
        abs(
            output.astype("float32")
            - pytorch_numpy.astype("float32")
        ).mean()
    )

    print(
        f"ONNX output shape: {output.shape}"
    )

    print(
        f"Maximum absolute error: "
        f"{max_error:.8e}"
    )

    print(
        f"Mean absolute error: "
        f"{mean_error:.8e}"
    )

    print(
        f"Execution providers: "
        f"{session.get_providers()}"
    )


def convert_to_fp16(
    input_path: Path,
    output_path: Path,
) -> None:
    """Convert an ONNX model to FP16."""

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "Install onnx before FP16 conversion."
        ) from exc

    try:
        from onnxconverter_common import float16
    except ImportError as exc:
        raise RuntimeError(
            "Install onnxconverter-common before "
            "using --onnx-fp16."
        ) from exc

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Converting ONNX to FP16...")

    model = onnx.load(
        str(input_path)
    )

    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=False,
    )

    onnx.save(
        fp16_model,
        str(output_path),
    )

    print(
        f"FP16 ONNX export complete: "
        f"{output_path}"
    )


def main() -> None:
    args = parse_args()

    if args.height <= 0 or args.width_px <= 0:
        raise ValueError(
            "Input dimensions must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size must be positive."
        )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        device = torch.device(
            "cuda"
        )

    else:
        device = torch.device(
            "cpu"
        )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA build: "
            f"{torch.version.cuda}"
        )

    print(
        f"Device: {device}"
    )

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------

    model, config = load_model_from_checkpoint(
        args.weights,
        device,
    )

    parameter_count = count_parameters(
        model
    )

    print()
    print(
        f"Model parameters: "
        f"{parameter_count:,} "
        f"({parameter_count / 1e6:.3f}M)"
    )

    print(
        "Model configuration:"
    )

    for key, value in config.items():
        print(
            f"  {key}: {value}"
        )

    # -------------------------------------------------------------
    # Dummy input
    # -------------------------------------------------------------

    dummy_input = torch.randn(
        args.batch_size,
        1,
        args.height,
        args.width_px,
        device=device,
        dtype=torch.float32,
    )

    if device.type == "cuda":
        model = model.to(
            memory_format=torch.channels_last
        )

        dummy_input = dummy_input.to(
            memory_format=torch.channels_last
        )

    # -------------------------------------------------------------
    # PyTorch sanity check
    # -------------------------------------------------------------

    print()
    print("Running PyTorch sanity check...")

    with torch.inference_mode():
        pytorch_output = model(
            dummy_input
        )

    expected_height = (
        args.height
        * config["scale"]
    )

    expected_width = (
        args.width_px
        * config["scale"]
    )

    expected_shape = (
        args.batch_size,
        1,
        expected_height,
        expected_width,
    )

    print(
        f"PyTorch input:  "
        f"{tuple(dummy_input.shape)}"
    )

    print(
        f"PyTorch output: "
        f"{tuple(pytorch_output.shape)}"
    )

    if tuple(pytorch_output.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected output shape. "
            f"Expected {expected_shape}, "
            f"got {tuple(pytorch_output.shape)}"
        )

    print(
        "PyTorch shape check passed."
    )

    # -------------------------------------------------------------
    # ONNX
    # -------------------------------------------------------------

    export_onnx(
        model=model,
        output_path=args.onnx,
        dummy_input=dummy_input,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
    )

    # -------------------------------------------------------------
    # Validate ONNX
    # -------------------------------------------------------------

    validate_onnx(
        onnx_path=args.onnx,
        dummy_input=dummy_input,
        pytorch_output=pytorch_output,
    )

    # -------------------------------------------------------------
    # Optional TorchScript
    # -------------------------------------------------------------

    if args.torchscript is not None:
        export_torchscript(
            model=model,
            output_path=args.torchscript,
            dummy_input=dummy_input,
        )

    # -------------------------------------------------------------
    # Optional FP16 ONNX
    # -------------------------------------------------------------

    if args.onnx_fp16 is not None:
        convert_to_fp16(
            input_path=args.onnx,
            output_path=args.onnx_fp16,
        )

    print()
    print("=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)

    print(
        f"ONNX: {args.onnx}"
    )

    if args.torchscript is not None:
        print(
            f"TorchScript: "
            f"{args.torchscript}"
        )

    if args.onnx_fp16 is not None:
        print(
            f"ONNX FP16: "
            f"{args.onnx_fp16}"
        )

    print(
        f"Input shape:  "
        f"{tuple(dummy_input.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(pytorch_output.shape)}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()