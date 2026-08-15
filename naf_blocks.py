"""Lightweight NAFNet building blocks.

The block design follows the official NAFNet principles: LayerNorm2d,
SimpleGate, depthwise 3x3 convolution, simplified channel attention,
and residual scaling parameters.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Compact NAFBlock with depthwise spatial mixing."""

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1, 1, 0, bias=True)
        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            3,
            1,
            1,
            groups=dw_channels,
            bias=True,
        )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1, 1, 0, bias=True),
        )
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, 1, 1, 0, bias=True)
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, ffn_channels, 1, 1, 0, bias=True)
        self.ffn2 = nn.Conv2d(ffn_channels // 2, channels, 1, 1, 0, bias=True)
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp: Tensor) -> Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv2(x)
        x = self.drop1(x)
        y = inp + x * self.beta

        x = self.ffn1(self.norm2(y))
        x = self.sg(x)
        x = self.ffn2(x)
        x = self.drop2(x)
        return y + x * self.gamma
