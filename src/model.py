# -*- coding: utf-8 -*-
"""Siamese U-Net for bi-temporal change detection.

Architecture
------------
- ONE shared-weight encoder (the Siamese branch) processes image A and image B.
- At every encoder scale the two feature maps are FUSED by concatenation
  (change information lives in the difference/comparison of the two streams).
- A standard U-Net decoder with skip connections consumes the fused features
  and outputs a single-channel change-probability map (logits).

Input : two (B, 3, S, S) tensors  (image at t1, image at t2)
Output:      (B, 1, S, S) logits; change pixels -> logits > 0 (sigmoid > 0.5)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SiameseUNet(nn.Module):
    """Shared-encoder Siamese U-Net with configurable fusion (concat, diff, diff_cat)."""

    def __init__(self, in_ch: int = 3, base: int = 32, out_ch: int = 1,
                 fusion: str = "concat"):
        super().__init__()
        self.fusion = fusion
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16

        # ---- shared (Siamese) encoder branch ----
        self.enc1 = DoubleConv(in_ch, c1)   # 1/1
        self.enc2 = DoubleConv(c1, c2)      # 1/2
        self.enc3 = DoubleConv(c2, c3)      # 1/4
        self.enc4 = DoubleConv(c3, c4)      # 1/8
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c4, c5)  # 1/16

        # ---- decoder on fused features ----
        # Channel multiplier for fusion:
        # "concat"   : 2 * c (A, B)
        # "diff"     : 1 * c (|A - B|)
        # "diff_cat" : 3 * c (A, B, |A - B|)
        m = 3 if fusion == "diff_cat" else (1 if fusion == "diff" else 2)

        self.up4 = nn.ConvTranspose2d(c5 * m, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 + c4 * m, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3 * m, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2 * m, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1 * m, c1)
        self.head = nn.Conv2d(c1, out_ch, kernel_size=1)

    def encode(self, x):
        """Shared encoder -> per-scale feature maps."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b

    def fuse(self, fa, fb):
        if self.fusion == "diff":
            return torch.abs(fa - fb)
        elif self.fusion == "diff_cat":
            return torch.cat([fa, fb, torch.abs(fa - fb)], dim=1)
        else:
            return torch.cat([fa, fb], dim=1)

    def forward(self, xa, xb):
        if xa.shape[-1] % 16 or xa.shape[-2] % 16:
            raise ValueError(
                f"Input size {tuple(xa.shape[-2:])} must be a multiple of 16 "
                "(4 pooling stages). Use IMAGE_SIZE=256 or another multiple of 16."
            )
        e1a, e2a, e3a, e4a, ba = self.encode(xa)
        e1b, e2b, e3b, e4b, bb = self.encode(xb)

        # fuse each scale by concatenation
        f1, f2, f3, f4, f5 = (self.fuse(e1a, e1b), self.fuse(e2a, e2b),
                              self.fuse(e3a, e3b), self.fuse(e4a, e4b),
                              self.fuse(ba, bb))

        d4 = self.up4(f5)                       # 1/16 -> 1/8
        d4 = self.dec4(torch.cat([d4, f4], dim=1))
        d3 = self.up3(d4)                       # 1/8 -> 1/4
        d3 = self.dec3(torch.cat([d3, f3], dim=1))
        d2 = self.up2(d3)                       # 1/4 -> 1/2
        d2 = self.dec2(torch.cat([d2, f2], dim=1))
        d1 = self.up1(d2)                       # 1/2 -> 1/1
        d1 = self.dec1(torch.cat([d1, f1], dim=1))
        return self.head(d1)                    # raw logits (no sigmoid here)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
