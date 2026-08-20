"""
src/unet_model.py
Lightweight U-Net architecture for binary skin lesion segmentation.

Designed for fast training on Google Colab T4 GPU or local execution.
Default base channel width = 32 (~7.7M parameters), configurable.

Architecture:
  Encoder: 4 downsampling stages with DoubleConv + MaxPool
  Bottleneck: DoubleConv
  Decoder: 4 upsampling stages with ConvTranspose2d / Bilinear + DoubleConv + Skip Connections
  Head: 1x1 Conv producing 1-channel raw logits
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv with skip connection"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle potential padding differences
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y > 0 or diff_x > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 convolution head producing target channel logits"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for binary lesion segmentation.

    Parameters
    ----------
    in_channels : int, default=3 (RGB image)
    out_channels : int, default=1 (binary mask logits)
    base_channels : int, default=32
    bilinear : bool, default=False (uses ConvTranspose2d)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        b = base_channels
        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)        # 32 -> 64
        self.down2 = Down(b * 2, b * 4)    # 64 -> 128
        self.down3 = Down(b * 4, b * 8)    # 128 -> 256
        factor = 2 if bilinear else 1
        self.down4 = Down(b * 8, b * 16 // factor)  # 256 -> 512 (or 256)

        self.up1 = Up(b * 16, b * 8 // factor, bilinear)  # 512 -> 256
        self.up2 = Up(b * 8, b * 4 // factor, bilinear)   # 256 -> 128
        self.up3 = Up(b * 4, b * 2 // factor, bilinear)   # 128 -> 64
        self.up4 = Up(b * 2, b, bilinear)                 # 64 -> 32
        self.outc = OutConv(b, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Run forward pass and return thresholded binary mask [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()


def get_unet(
    in_channels: int = 3,
    out_channels: int = 1,
    base_channels: int = 32,
    bilinear: bool = False,
) -> UNet:
    """Factory helper to instantiate U-Net."""
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        bilinear=bilinear,
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    model = get_unet(base_channels=32)
    stats = count_parameters(model)
    print(f"U-Net (base=32): {stats['total']:,} total parameters ({stats['trainable']:,} trainable)")
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape} -> Output shape: {y.shape}")
