"""Train Tiny NAFNet on paired KLA .npy data."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import PairedRestorationDataset, build_pairs, split_pairs
from losses import RestorationLoss
from model import ModelConfig, build_model
from trainer import Trainer
from utils import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-degraded", type=Path, default=Path("datasets/train/NoisyLR"))
    p.add_argument("--train-gt", type=Path, default=Path("datasets/train/GT"))
    p.add_argument("--val-degraded", type=Path, default=Path("datasets/val/NoisyLR"))
    p.add_argument("--val-gt", type=Path, default=Path("datasets/val/GT"))
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("weights"))
    p.add_argument("--log-dir", type=Path, default=Path("logs"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--l1-weight", type=float, default=1.0)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--perceptual-weight", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--scale", type=int, choices=(1, 2, 4), default=1)
    p.add_argument("--width", type=int, default=28)
    p.add_argument("--middle-blocks", type=int, default=2)
    p.add_argument("--value-scale", type=float, default=None, help="Numeric divisor for .npy values; default auto.")
    p.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="fp16")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument(
        "--gradient-weight",
        type=float,
        default=0.02,
    )
    return p.parse_args()


def setup_distributed() -> tuple[bool, int, int, torch.device]:
    if "RANK" not in os.environ:
        return False, 0, 1, resolve_device("auto")
    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, rank, int(os.environ["WORLD_SIZE"]), torch.device("cuda", local_rank)


def main() -> None:
    args = parse_args()
    distributed, rank, _, device = setup_distributed()
    seed_everything(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    cfg = ModelConfig(width=args.width, middle_blocks=args.middle_blocks, scale=args.scale)
    model = build_model(cfg).to(device)
    if rank == 0:
        print(f"TinyNAFNet parameters: {model.parameter_count():,} ({model.parameter_count()/1e6:.3f}M)")

    all_pairs = build_pairs(args.train_degraded, args.train_gt)
    if not all_pairs:
        raise RuntimeError(f"No paired files found in {args.train_degraded} and {args.train_gt}. For .npy, stems/relative paths must match.")
    train_pairs, auto_val_pairs = split_pairs(all_pairs, args.val_split, args.seed)

    explicit_val_pairs = build_pairs(args.val_degraded, args.val_gt)
    val_pairs = explicit_val_pairs if explicit_val_pairs else auto_val_pairs
    if rank == 0:
        print(f"Paired train files: {len(train_pairs)}")
        print(f"Validation files: {len(val_pairs)}" + (" (auto split)" if not explicit_val_pairs and val_pairs else ""))

    train_ds = PairedRestorationDataset(train_pairs, args.patch_size, args.scale, True, args.value_scale)
    val_ds = PairedRestorationDataset(val_pairs, None, args.scale, False, args.value_scale) if val_pairs else None
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed and val_ds else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    if distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )
    criterion = RestorationLoss(
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        gradient_weight=args.gradient_weight,
        perceptual_weight=args.perceptual_weight,
    ).to(device)
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, criterion, device,
                      args.output_dir, args.log_dir, args.epochs, args.grad_clip, args.patience, args.min_delta,
                      amp=True, amp_dtype=amp_dtype, rank=rank)
    if args.resume:
        trainer.load(args.resume)
    trainer.fit()
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
