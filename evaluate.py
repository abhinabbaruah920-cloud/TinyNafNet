"""Evaluate TinyNAFNet using PSNR, SSIM and LPIPS."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from metrics import LPIPSLoss, psnr, ssim
from model import ModelConfig, build_model
from utils import image_files, load_array


class EvaluationDataset(Dataset[dict[str, object]]):
    """Paired evaluation dataset."""

    def __init__(
        self,
        degraded_dir: Path,
        gt_dir: Path,
        files: list[tuple[Path, Path]],
    ) -> None:
        self.degraded_dir = Path(
            degraded_dir
        )
        self.gt_dir = Path(
            gt_dir
        )
        self.pairs = files

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        degraded_path, gt_path = self.pairs[index]

        degraded, _, _ = load_array(
            degraded_path
        )

        target, _, _ = load_array(
            gt_path
        )

        return {
            "input": degraded,
            "target": target,
            "name": degraded_path.name,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TinyNAFNet with "
            "PSNR, SSIM and LPIPS."
        )
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/best.pt"),
    )

    parser.add_argument(
        "--degraded",
        type=Path,
        default=Path(
            "datasets/train/NoisyLR"
        ),
    )

    parser.add_argument(
        "--gt",
        type=Path,
        default=Path(
            "datasets/train/GT"
        ),
    )

    parser.add_argument(
        "--val-degraded",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--val-gt",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.10,
        help=(
            "Validation fraction when explicit "
            "validation folders are unavailable."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )

    parser.add_argument(
        "--lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default="alex",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        return torch.device("cuda")

    if requested == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def create_pairs(
    degraded_dir: Path,
    gt_dir: Path,
) -> list[tuple[Path, Path]]:
    """Match degraded and GT files by filename stem."""

    degraded = {
        path.stem: path
        for path in image_files(
            degraded_dir
        )
    }

    gt = {
        path.stem: path
        for path in image_files(
            gt_dir
        )
    }

    names = sorted(
        set(degraded) & set(gt)
    )

    if not names:
        raise RuntimeError(
            f"No paired files found:\n"
            f"degraded={degraded_dir}\n"
            f"gt={gt_dir}"
        )

    return [
        (
            degraded[name],
            gt[name],
        )
        for name in names
    ]


def split_pairs(
    pairs: list[tuple[Path, Path]],
    fraction: float,
    seed: int,
) -> tuple[
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
]:
    """Create deterministic train/validation split."""

    if not 0.0 < fraction < 1.0:
        raise ValueError(
            "val-split must be between 0 and 1."
        )

    pairs = list(pairs)

    rng = random.Random(
        seed
    )

    rng.shuffle(
        pairs
    )

    val_count = max(
        1,
        int(
            round(
                len(pairs)
                * fraction
            )
        ),
    )

    val_pairs = pairs[
        :val_count
    ]

    train_pairs = pairs[
        val_count:
    ]

    return train_pairs, val_pairs


def load_model(
    weights: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load model using checkpoint configuration."""

    checkpoint = torch.load(
        weights,
        map_location=device,
        weights_only=False,
    )

    state = checkpoint.get(
        "model",
        checkpoint,
    )

    saved_config = checkpoint.get(
        "model_config",
        {},
    )

    config = ModelConfig(
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

    model = build_model(
        config
    ).to(device)

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    return model


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lpips_metric: LPIPSLoss,
) -> tuple[float, float, float]:
    """Calculate PSNR, SSIM and LPIPS."""

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    for batch in tqdm(
        loader,
        desc="Evaluating",
    ):
        inputs = batch["input"].to(
            device,
            non_blocking=True,
        )

        targets = batch["target"].to(
            device,
            non_blocking=True,
        )

        if device.type == "cuda":
            inputs = inputs.to(
                memory_format=torch.channels_last
            )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            predictions = model(
                inputs
            )

        predictions = predictions.float().clamp(
            0.0,
            1.0,
        )

        targets = targets.float().clamp(
            0.0,
            1.0,
        )

        # ---------------------------------------------------------
        # PSNR / SSIM
        # ---------------------------------------------------------

        for prediction, target in zip(
            predictions,
            targets,
        ):
            prediction_1 = (
                prediction.unsqueeze(0)
            )

            target_1 = (
                target.unsqueeze(0)
            )

            psnr_values.append(
                psnr(
                    prediction_1,
                    target_1,
                )
            )

            ssim_values.append(
                ssim(
                    prediction_1,
                    target_1,
                )
            )

        # ---------------------------------------------------------
        # LPIPS
        # ---------------------------------------------------------

        lpips_values.append(
            lpips_metric(
                predictions,
                targets,
            )
        )

    mean_psnr = sum(
        psnr_values
    ) / max(
        len(psnr_values),
        1,
    )

    mean_ssim = sum(
        ssim_values
    ) / max(
        len(ssim_values),
        1,
    )

    mean_lpips = sum(
        lpips_values
    ) / max(
        len(lpips_values),
        1,
    )

    return (
        mean_psnr,
        mean_ssim,
        mean_lpips,
    )


def main() -> None:
    args = parse_args()

    torch.manual_seed(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # -------------------------------------------------------------
    # Determine evaluation pairs
    # -------------------------------------------------------------

    explicit_validation = (
        args.val_degraded is not None
        and args.val_gt is not None
    )

    if explicit_validation:
        pairs = create_pairs(
            args.val_degraded,
            args.val_gt,
        )

        print(
            f"Validation files: "
            f"{len(pairs)}"
        )

    else:
        all_pairs = create_pairs(
            args.degraded,
            args.gt,
        )

        _, pairs = split_pairs(
            all_pairs,
            args.val_split,
            args.seed,
        )

        print(
            f"Full paired dataset: "
            f"{len(all_pairs)}"
        )

        print(
            f"Validation split: "
            f"{len(pairs)}"
        )

    dataset = EvaluationDataset(
        args.degraded,
        args.gt,
        pairs,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = load_model(
        args.weights,
        device,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Parameters: "
        f"{parameter_count:,} "
        f"({parameter_count / 1e6:.3f}M)"
    )

    checkpoint = torch.load(
        args.weights,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint.get(
        "model_config",
        {},
    )

    print(
        f"Model config: "
        f"width={config.get('width', 28)}, "
        f"middle={config.get('middle_blocks', 2)}, "
        f"scale={config.get('scale', 2)}"
    )

    # -------------------------------------------------------------
    # LPIPS
    # -------------------------------------------------------------

    print(
        f"Loading LPIPS network: "
        f"{args.lpips_net}"
    )

    lpips_metric = LPIPSLoss(
        net=args.lpips_net,
        device=device,
    )

    # -------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------

    mean_psnr, mean_ssim, mean_lpips = evaluate(
        model=model,
        loader=loader,
        device=device,
        lpips_metric=lpips_metric,
    )

    print()
    print("=" * 60)
    print("RESTORATION QUALITY")
    print("=" * 60)

    print(
        f"PSNR :  {mean_psnr:.4f} dB"
    )

    print(
        f"SSIM :  {mean_ssim:.6f}"
    )

    print(
        f"LPIPS:  {mean_lpips:.6f}"
    )

    print("=" * 60)

    print()
    print("Metric direction:")
    print("PSNR  -> higher is better")
    print("SSIM  -> higher is better")
    print("LPIPS -> lower is better")


if __name__ == "__main__":
    main()