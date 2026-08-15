"""Convert KLA .npy grayscale arrays to PNG images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def convert_file(
    input_path: Path,
    output_path: Path,
    auto_range: bool = False,
) -> None:
    """Convert one .npy array to an 8-bit grayscale PNG."""

    array = np.load(input_path)

    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale array, "
            f"got shape {array.shape} in {input_path}"
        )

    array = array.astype(np.float32)

    if auto_range:
        minimum = float(array.min())
        maximum = float(array.max())

        if maximum > minimum:
            array = (
                array - minimum
            ) / (
                maximum - minimum
            )
        else:
            array = np.zeros_like(array)
    else:
        # KLA GT values are in [0, 1].
        # For NoisyLR, clip only for visualization.
        array = np.clip(
            array,
            0.0,
            1.0,
        )

    array = (
        array * 255.0
    ).round().astype(np.uint8)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        array,
        mode="L",
    ).save(output_path)


def convert_folder(
    input_dir: Path,
    output_dir: Path,
    auto_range: bool,
) -> None:
    """Convert all .npy files in a folder."""

    files = sorted(
        input_dir.glob("*.npy")
    )

    if not files:
        raise RuntimeError(
            f"No .npy files found in {input_dir}"
        )

    for index, input_path in enumerate(
        files,
        start=1,
    ):
        output_path = (
            output_dir
            / f"{input_path.stem}.png"
        )

        convert_file(
            input_path,
            output_path,
            auto_range,
        )

        print(
            f"[{index}/{len(files)}] "
            f"{input_path.name} -> "
            f"{output_path.name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert .npy grayscale files to PNG."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input .npy file or folder.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG file or folder.",
    )

    parser.add_argument(
        "--auto-range",
        action="store_true",
        help=(
            "Normalize each array using its own min/max "
            "for visualization."
        ),
    )

    args = parser.parse_args()

    if args.input.is_file():
        if args.input.suffix.lower() != ".npy":
            raise ValueError(
                "Input file must be .npy"
            )

        output = args.output

        if output.suffix.lower() != ".png":
            output = output.with_suffix(
                ".png"
            )

        convert_file(
            args.input,
            output,
            args.auto_range,
        )

        print(
            f"Saved: {output}"
        )

    elif args.input.is_dir():
        convert_folder(
            args.input,
            args.output,
            args.auto_range,
        )

    else:
        raise FileNotFoundError(
            args.input
        )


if __name__ == "__main__":
    main()