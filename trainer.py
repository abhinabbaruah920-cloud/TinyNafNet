"""Production training loop for TinyNAFNet.

Features:
    - AMP / FP16 / BF16
    - Gradient clipping
    - Cosine learning-rate scheduling
    - Resume training
    - Best-checkpoint selection by PSNR
    - Optional validation
    - Early stopping
    - TensorBoard logging
    - DDP-compatible structure
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import psnr, ssim
from utils import AverageMeter


class Trainer:
    """Training and validation manager for TinyNAFNet."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        device: torch.device,
        output_dir: Path,
        log_dir: Path,
        epochs: int,
        grad_clip: float,
        patience: int,
        min_delta: float,
        amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        rank: int = 0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.device = device

        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)

        self.epochs = epochs
        self.grad_clip = grad_clip
        self.patience = patience
        self.min_delta = min_delta

        self.rank = rank

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------
        # Resume / monitoring state
        # ---------------------------------------------------------

        self.start_epoch = 0

        # Best model is selected by PSNR.
        self.best_psnr = -float("inf")
        self.best_ssim = -float("inf")

        # Number of consecutive epochs without meaningful PSNR gain.
        self.bad_epochs = 0

        # ---------------------------------------------------------
        # AMP
        # ---------------------------------------------------------

        self.amp_enabled = (
            amp and device.type == "cuda"
        )

        self.amp_dtype = amp_dtype

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.amp_enabled
                and amp_dtype == torch.float16
            ),
        )

        # ---------------------------------------------------------
        # TensorBoard
        # ---------------------------------------------------------

        self.writer = None

        if self.rank == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(
                    str(self.log_dir)
                )
            except ImportError:
                print(
                    "Warning: TensorBoard is not installed. "
                    "TensorBoard logging is disabled."
                )

    # =============================================================
    # Model helpers
    # =============================================================

    def unwrap(self) -> nn.Module:
        """Return the underlying model, unwrapping DDP if necessary."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _model_config(self) -> dict[str, Any] | None:
        """Extract model configuration when available."""
        model = self.unwrap()

        cfg = getattr(model, "cfg", None)

        if cfg is None:
            return None

        result: dict[str, Any] = {}

        for name in (
            "width",
            "enc_blocks",
            "dec_blocks",
            "middle_blocks",
            "scale",
            "dw_expand",
            "ffn_expand",
        ):
            if hasattr(cfg, name):
                value = getattr(cfg, name)

                if isinstance(value, tuple):
                    value = list(value)

                result[name] = value

        return result or None

    # =============================================================
    # AMP helper
    # =============================================================

    def _autocast_context(self):
        """Return the appropriate autocast context."""
        return torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        )

    # =============================================================
    # Training
    # =============================================================

    def train_epoch(self, epoch: int) -> float:
        """Run one complete training epoch."""

        self.model.train()

        meter = AverageMeter()

        if self.rank == 0:
            iterator = tqdm(
                self.train_loader,
                desc=(
                    f"Epoch {epoch + 1}/{self.epochs}"
                ),
                leave=False,
            )
        else:
            iterator = self.train_loader

        for batch in iterator:
            inp = batch["input"].to(
                self.device,
                non_blocking=True,
            )

            tgt = batch["target"].to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # -----------------------------------------------------
            # Forward + loss
            # -----------------------------------------------------

            with self._autocast_context():
                pred = self.model(inp)
                loss = self.criterion(
                    pred,
                    tgt,
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss detected: "
                    f"{loss.detach().item()}"
                )

            # -----------------------------------------------------
            # Backward
            # -----------------------------------------------------

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()

                # Unscale before gradient clipping.
                self.scaler.unscale_(
                    self.optimizer
                )

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip,
                    )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            else:
                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip,
                    )

                self.optimizer.step()

            meter.update(
                float(loss.detach().item()),
                inp.shape[0],
            )

        return meter.average

    # =============================================================
    # Validation
    # =============================================================

    @torch.inference_mode()
    def validate(self) -> dict[str, float]:
        """Evaluate loss, PSNR and SSIM."""

        if self.val_loader is None:
            return {}

        self.model.eval()

        loss_meter = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()

        for batch in self.val_loader:
            inp = batch["input"].to(
                self.device,
                non_blocking=True,
            )

            tgt = batch["target"].to(
                self.device,
                non_blocking=True,
            )

            with self._autocast_context():
                pred = self.model(inp)

                loss = self.criterion(
                    pred,
                    tgt,
                )

            # Metrics should be evaluated in float32.
            pred_metric = pred.float().clamp(
                0.0,
                1.0,
            )

            tgt_metric = tgt.float().clamp(
                0.0,
                1.0,
            )

            batch_size = inp.shape[0]

            loss_meter.update(
                float(loss.detach().item()),
                batch_size,
            )

            for prediction, target in zip(
                pred_metric,
                tgt_metric,
            ):
                prediction = prediction.unsqueeze(0)
                target = target.unsqueeze(0)

                psnr_meter.update(
                    float(
                        psnr(
                            prediction,
                            target,
                        )
                    ),
                    1,
                )

                ssim_meter.update(
                    float(
                        ssim(
                            prediction,
                            target,
                        )
                    ),
                    1,
                )

        return {
            "loss": loss_meter.average,
            "psnr": psnr_meter.average,
            "ssim": ssim_meter.average,
        }

    # =============================================================
    # Checkpoint saving
    # =============================================================

    def save(
        self,
        epoch: int,
        best: bool = False,
    ) -> None:
        """Save the latest and optionally best checkpoint."""

        if self.rank != 0:
            return

        state: dict[str, Any] = {
            "epoch": epoch,

            "model": self.unwrap().state_dict(),

            "optimizer": self.optimizer.state_dict(),

            "scheduler": self.scheduler.state_dict(),

            "scaler": self.scaler.state_dict(),

            "best_psnr": self.best_psnr,

            "best_ssim": self.best_ssim,

            "bad_epochs": self.bad_epochs,
        }

        model_config = self._model_config()

        if model_config is not None:
            state["model_config"] = model_config

        # Always save the latest completed epoch.
        torch.save(
            state,
            self.output_dir / "last.pt",
        )

        # Save best PSNR model separately.
        if best:
            torch.save(
                state,
                self.output_dir / "best.pt",
            )

    # =============================================================
    # Resume
    # =============================================================

    def load(
        self,
        path: Path,
    ) -> None:
        """Resume training from checkpoint."""

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.unwrap().load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        if "scaler" in checkpoint:
            self.scaler.load_state_dict(
                checkpoint["scaler"]
            )

        self.start_epoch = (
            int(
                checkpoint.get(
                    "epoch",
                    -1,
                )
            )
            + 1
        )

        self.best_psnr = float(
            checkpoint.get(
                "best_psnr",
                -float("inf"),
            )
        )

        self.best_ssim = float(
            checkpoint.get(
                "best_ssim",
                -float("inf"),
            )
        )

        self.bad_epochs = int(
            checkpoint.get(
                "bad_epochs",
                0,
            )
        )

        if self.rank == 0:
            print(
                f"Resumed checkpoint: {path}"
            )

            print(
                f"Starting epoch: "
                f"{self.start_epoch + 1}"
            )

            print(
                f"Best PSNR so far: "
                f"{self.best_psnr:.4f}"
            )

            print(
                f"Best SSIM so far: "
                f"{self.best_ssim:.5f}"
            )

    # =============================================================
    # Main training loop
    # =============================================================

    def fit(self) -> None:
        """Train until completion or early stopping."""

        for epoch in range(
            self.start_epoch,
            self.epochs,
        ):
            # -----------------------------------------------------
            # DDP sampler
            # -----------------------------------------------------

            sampler = getattr(
                self.train_loader,
                "sampler",
                None,
            )

            if hasattr(
                sampler,
                "set_epoch",
            ):
                sampler.set_epoch(epoch)

            # -----------------------------------------------------
            # IMPORTANT:
            # Read LR BEFORE scheduler.step().
            # -----------------------------------------------------

            current_lr = float(
                self.optimizer.param_groups[0]["lr"]
            )

            start = time.perf_counter()

            # -----------------------------------------------------
            # Train
            # -----------------------------------------------------

            train_loss = self.train_epoch(
                epoch
            )

            # -----------------------------------------------------
            # Validate
            # -----------------------------------------------------

            val = self.validate()

            elapsed = (
                time.perf_counter()
                - start
            )

            # -----------------------------------------------------
            # IMPORTANT:
            #
            # scheduler.step() happens EXACTLY ONCE here.
            #
            # There must NOT be another scheduler.step()
            # inside train_epoch().
            # -----------------------------------------------------

            self.scheduler.step()

            if self.rank != 0:
                continue

            # =====================================================
            # Validation enabled
            # =====================================================

            if val:
                current_psnr = val["psnr"]
                current_ssim = val["ssim"]

                print(
                    f"Epoch {epoch + 1}: "
                    f"train={train_loss:.6f} "
                    f"val={val['loss']:.6f} "
                    f"PSNR={current_psnr:.3f} "
                    f"SSIM={current_ssim:.5f} "
                    f"lr={current_lr:.3e} "
                    f"time={elapsed:.2f}s"
                )

                # -------------------------------------------------
                # Best-model selection is based on PSNR.
                # -------------------------------------------------

                improved = (
                    current_psnr
                    > self.best_psnr
                    + self.min_delta
                )

                if improved:
                    self.best_psnr = current_psnr
                    self.best_ssim = max(
                        self.best_ssim,
                        current_ssim,
                    )

                    self.bad_epochs = 0

                    self.save(
                        epoch,
                        best=True,
                    )

                    print(
                        f"  New best PSNR: "
                        f"{self.best_psnr:.4f} dB"
                    )

                else:
                    self.bad_epochs += 1

                    # Save latest epoch even when it isn't best.
                    self.save(
                        epoch,
                        best=False,
                    )

                # -------------------------------------------------
                # TensorBoard
                # -------------------------------------------------

                if self.writer is not None:
                    self.writer.add_scalar(
                        "train/loss",
                        train_loss,
                        epoch,
                    )

                    self.writer.add_scalar(
                        "val/loss",
                        val["loss"],
                        epoch,
                    )

                    self.writer.add_scalar(
                        "val/psnr",
                        current_psnr,
                        epoch,
                    )

                    self.writer.add_scalar(
                        "val/ssim",
                        current_ssim,
                        epoch,
                    )

                    self.writer.add_scalar(
                        "train/lr",
                        current_lr,
                        epoch,
                    )

                    self.writer.add_scalar(
                        "train/epoch_time",
                        elapsed,
                        epoch,
                    )

                # -------------------------------------------------
                # Early stopping
                # -------------------------------------------------

                if (
                    self.bad_epochs
                    >= self.patience
                ):
                    print(
                        "Early stopping."
                    )
                    print(
                        f"Best PSNR: "
                        f"{self.best_psnr:.4f} dB"
                    )
                    print(
                        f"Best SSIM: "
                        f"{self.best_ssim:.5f}"
                    )
                    break

            # =====================================================
            # Validation disabled
            # =====================================================

            else:
                print(
                    f"Epoch {epoch + 1}: "
                    f"train={train_loss:.6f} "
                    f"lr={current_lr:.3e} "
                    f"time={elapsed:.2f}s "
                    f"(validation disabled)"
                )

                self.save(
                    epoch,
                    best=False,
                )

                if self.writer is not None:
                    self.writer.add_scalar(
                        "train/loss",
                        train_loss,
                        epoch,
                    )

                    self.writer.add_scalar(
                        "train/lr",
                        current_lr,
                        epoch,
                    )

                    self.writer.add_scalar(
                        "train/epoch_time",
                        elapsed,
                        epoch,
                    )

        # ---------------------------------------------------------
        # Close TensorBoard writer
        # ---------------------------------------------------------

        if self.writer is not None:
            self.writer.close()

        if self.rank == 0:
            print()
            print("Training finished.")

            if self.val_loader is not None:
                print(
                    f"Best PSNR: "
                    f"{self.best_psnr:.4f} dB"
                )

                print(
                    f"Best SSIM: "
                    f"{self.best_ssim:.5f}"
                )