"""Tiny NAFNet for single-channel semiconductor image restoration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from naf_blocks import NAFBlock


@dataclass(frozen=True)
class ModelConfig:
    width: int = 28
    enc_blocks: tuple[int, ...] = (1, 1, 1, 1)
    dec_blocks: tuple[int, ...] = (1, 1, 1, 1)
    middle_blocks: int = 2
    scale: int = 2
    dw_expand: int = 2
    ffn_expand: int = 2


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 2, 2, 0, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 1, 1, 0, bias=True)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        return self.shuffle(self.conv(x))


class TinyNAFNet(nn.Module):
    """NAFNet-style encoder/decoder optimized for grayscale restoration.

    For scale=1 the network predicts a residual at the input resolution.
    For scale=2/4 the low-resolution input is processed and residual output
    is upscaled with PixelShuffle over a bicubic baseline.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.scale not in (1, 2, 4):
            raise ValueError("scale must be 1, 2, or 4")

        self.cfg = cfg
        self.scale = cfg.scale
        self.intro = nn.Conv2d(1, cfg.width, 3, 1, 1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        chans: list[int] = []
        ch = cfg.width
        for block_count in cfg.enc_blocks:
            chans.append(ch)
            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(ch, cfg.dw_expand, cfg.ffn_expand)
                        for _ in range(block_count)
                    ]
                )
            )
            self.downs.append(Downsample(ch, ch * 2))
            ch *= 2

        self.middle = nn.Sequential(
            *[
                NAFBlock(ch, cfg.dw_expand, cfg.ffn_expand)
                for _ in range(cfg.middle_blocks)
            ]
        )

        for idx, block_count in enumerate(cfg.dec_blocks):
            in_ch = ch
            out_ch = chans[-(idx + 1)]
            self.ups.append(Upsample(in_ch, out_ch))
            ch = out_ch
            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(ch, cfg.dw_expand, cfg.ffn_expand)
                        for _ in range(block_count)
                    ]
                )
            )

        if cfg.scale == 1:
            self.hr_up = nn.Identity()
        else:
            modules: list[nn.Module] = []
            remaining = cfg.scale
            ch_in = cfg.width
            while remaining > 1:
                modules.append(nn.Conv2d(ch_in, ch_in * 4, 1, 1, 0, bias=True))
                modules.append(nn.PixelShuffle(2))
                remaining //= 2
            self.hr_up = nn.Sequential(*modules)

        self.ending = nn.Conv2d(cfg.width, 1, 3, 1, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        inp = x
        x = self.intro(x)
        skips: list[Tensor] = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for up, dec in zip(self.ups, self.decoders):
            x = up(x)
            skip = skips.pop()
            x = x + skip
            x = dec(x)

        x = self.hr_up(x)
        residual = self.ending(x)

        if self.scale == 1:
            base = inp
        else:
            base = F.interpolate(
                inp,
                scale_factor=self.scale,
                mode="bicubic",
                align_corners=False,
            )
        return base + residual

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: ModelConfig | None = None) -> TinyNAFNet:
    return TinyNAFNet(cfg or ModelConfig())
