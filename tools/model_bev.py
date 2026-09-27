#!/usr/bin/env python3
"""A small BEV segmentation network, constrained to what a 2019 Jetson can run.

Every operator here exists in TensorRT 8.2 at ONNX opset 11: Conv, BatchNorm, ReLU,
MaxPool, ConvTranspose, Concat. No bilinear resize (align_corners handling differs
between exporters), no attention, no deformable anything, no custom CUDA. The point is
a model that survives export and INT8 calibration unchanged, not one that wins a
leaderboard.

Roughly 1.9M parameters at width 32, which fits the Nano's 4 GB with room for the
input pipeline.
"""
import torch
import torch.nn as nn

N_IN = 5
N_CLASSES = 8


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class BEVUNet(nn.Module):
    def __init__(self, width: int = 32, n_in: int = N_IN, n_classes: int = N_CLASSES):
        super().__init__()
        w = width
        self.enc1 = block(n_in, w)
        self.enc2 = block(w, w * 2)
        self.enc3 = block(w * 2, w * 4)
        self.enc4 = block(w * 4, w * 8)
        self.pool = nn.MaxPool2d(2)
        # ConvTranspose rather than Upsample: one operator, no resize semantics to
        # disagree about between torch, ONNX and TensorRT.
        self.up3 = nn.ConvTranspose2d(w * 8, w * 4, 2, stride=2)
        self.dec3 = block(w * 8, w * 4)
        self.up2 = nn.ConvTranspose2d(w * 4, w * 2, 2, stride=2)
        self.dec2 = block(w * 4, w * 2)
        self.up1 = nn.ConvTranspose2d(w * 2, w, 2, stride=2)
        self.dec1 = block(w * 2, w)
        self.head = nn.Conv2d(w, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


if __name__ == "__main__":
    m = BEVUNet()
    n = sum(p.numel() for p in m.parameters())
    x = torch.zeros(1, N_IN, 400, 400)
    with torch.no_grad():
        y = m(x)
    print(f"BEVUNet width=32: {n/1e6:.2f}M params")
    print(f"  {tuple(x.shape)} -> {tuple(y.shape)}")
    ops = {type(mod).__name__ for mod in m.modules()}
    print(f"  operator types: {sorted(ops - {'BEVUNet', 'Sequential'})}")
