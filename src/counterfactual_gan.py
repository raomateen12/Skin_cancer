"""
src/counterfactual_gan.py
==========================
Counterfactual Skin Lesion Explanation Generator using Progressive Exaggeration
(Singla et al., ICLR 2020: "Explanation by Progressive Exaggeration").

Architecture:
1. Encoder: Image (3, 128, 128) -> Latent code z (128-dim) + spatial identity skip
2. Generator: (z, target_melanoma_prob) -> Counterfactual image (3, 128, 128)
3. Discriminator: PatchGAN 70x70 realism discriminator
4. Frozen Guidance: Pretrained EfficientNet-B0 classifier providing melanoma probability
   consistency loss L_cls = ||P_mel(G(z, t)) - t||^2.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

# ImageNet normalization constants for classifier guidance
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
MELANOMA_CLASS_IDX = 4  # 'mel' class index in HAM10000 7-class schema


# ── 1. Encoder ────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Convolution + InstanceNorm + LeakyReLU block."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual Block with InstanceNorm."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CounterfactualEncoder(nn.Module):
    """
    Encodes 128x128x3 lesion image into a compact latent vector z (latent_dim=128)
    plus spatial bottleneck features for high structural preservation.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 128, base_channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Initial conv (128x128)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Downsampling: 128 -> 64 -> 32 -> 16 -> 8
        self.down1 = ConvBlock(base_channels, base_channels * 2, stride=2)       # 64x64
        self.down2 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)   # 32x32
        self.down3 = ConvBlock(base_channels * 4, base_channels * 4, stride=2)   # 16x16
        self.down4 = ConvBlock(base_channels * 4, base_channels * 4, stride=2)   # 8x8

        # Residual bottleneck at 8x8
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

        # Latent projection: Global Avg Pool (8x8) -> FC -> latent_dim
        self.fc_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 4, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            latent_z: [B, latent_dim]
            bottleneck_feat: [B, base_channels*4, 8, 8]
        """
        feat0 = self.in_conv(x)
        feat1 = self.down1(feat0)
        feat2 = self.down2(feat1)
        feat3 = self.down3(feat2)
        feat4 = self.down4(feat3)
        bottleneck = self.res_blocks(feat4)
        z = self.fc_latent(bottleneck)
        return z, bottleneck


# ── 2. Generator ──────────────────────────────────────────────────────────────

class DeconvBlock(nn.Module):
    """Upsample (TransposeConv or Bilinear + Conv) + InstanceNorm + ReLU block."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CounterfactualGenerator(nn.Module):
    """
    Takes encoded latent z (or bottleneck features) and a target melanoma probability
    t in [0, 1], generating a modified 128x128 lesion image.
    """
    def __init__(self, out_channels: int = 3, latent_dim: int = 128, base_channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        # Condition projection MLP: target_prob scalar (1-dim) -> condition embedding (64-dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Combine bottleneck features [B, 256, 8, 8] and condition [B, 256, 1, 1]
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(base_channels * 4 * 2, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
        )

        # Upsampling: 8 -> 16 -> 32 -> 64 -> 128
        self.up1 = DeconvBlock(base_channels * 4, base_channels * 4, stride=2)   # 16x16
        self.up2 = DeconvBlock(base_channels * 4, base_channels * 4, stride=2)   # 32x32
        self.up3 = DeconvBlock(base_channels * 4, base_channels * 2, stride=2)   # 64x64
        self.up4 = DeconvBlock(base_channels * 2, base_channels, stride=2)       # 128x128

        # Output projection: 128x128 -> RGB with Tanh [-1, 1]
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward(self, bottleneck: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
        """
        bottleneck: [B, base_channels*4, 8, 8]
        target_prob: [B, 1] scalar in [0, 1]
        returns: [B, 3, 128, 128] in range [-1, 1]
        """
        B, _, H, W = bottleneck.shape
        cond_emb = self.cond_mlp(target_prob).view(B, -1, 1, 1).expand(-1, -1, H, W)
        fused = torch.cat([bottleneck, cond_emb], dim=1)
        x = self.fuse_conv(fused)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        out = self.out_conv(x)
        return out


# ── 3. Discriminator (PatchGAN) ───────────────────────────────────────────────

class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN realism discriminator.
    Classifies 128x128 image patches as real or synthetic.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            # Layer 1: 128 -> 64 (No norm on input layer)
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 64 -> 32
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 32 -> 16
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 16 -> 15 (stride 1)
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Output patch predictions: 1 channel
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── 4. Full Counterfactual GAN Module & Guidance Loss ─────────────────────────

class CounterfactualGAN(nn.Module):
    """
    Combined Counterfactual GAN Wrapper encapsulating Encoder, Generator, and Discriminator.
    """
    def __init__(self, latent_dim: int = 128, base_channels: int = 64):
        super().__init__()
        self.encoder = CounterfactualEncoder(in_channels=3, latent_dim=latent_dim, base_channels=base_channels)
        self.generator = CounterfactualGenerator(out_channels=3, latent_dim=latent_dim, base_channels=base_channels)
        self.discriminator = PatchDiscriminator(in_channels=3, base_channels=base_channels)

    def generate_counterfactual(self, x: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
        """Forward pass for generating counterfactuals."""
        _, bottleneck = self.encoder(x)
        return self.generator(bottleneck, target_prob)

    def get_parameter_count(self) -> dict[str, int]:
        """Compute parameter counts across all sub-networks."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        total = enc_params + gen_params + disc_params
        return {
            "encoder": enc_params,
            "generator": gen_params,
            "discriminator": disc_params,
            "total": total,
        }


# ── 5. Classifier Consistency & Loss Utilities ────────────────────────────────

def get_frozen_classifier(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load and freeze EfficientNet-B0 guidance model."""
    from src.model import get_efficientnet_b0
    classifier = get_efficientnet_b0(num_classes=7)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    classifier.load_state_dict(state)
    classifier.to(device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier


def get_melanoma_probability(
    images_128_tanh: torch.Tensor,
    classifier: nn.Module,
    mel_class_idx: int = MELANOMA_CLASS_IDX,
) -> torch.Tensor:
    """
    Differentiably compute melanoma prediction probability P(mel) from 128x128 [-1, 1] images.
    Rescales to [0, 1], upsamples to 224x224, applies ImageNet normalization, and passes
    through frozen classifier with Softmax.
    """
    device = images_128_tanh.device

    # 1. Un-normalize from [-1, 1] to [0, 1]
    img_01 = (images_128_tanh + 1.0) / 2.0

    # 2. Differentiable bilinear upsample to 224x224 (EfficientNet-B0 expected resolution)
    img_224 = F.interpolate(img_01, size=(224, 224), mode="bilinear", align_corners=False)

    # 3. Apply ImageNet mean/std normalization
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    img_norm = (img_224 - mean) / std

    # 4. Forward through frozen classifier
    logits = classifier(img_norm)
    probs = F.softmax(logits, dim=1)
    mel_prob = probs[:, mel_class_idx:mel_class_idx+1]  # [B, 1]
    return mel_prob


class CounterfactualLoss(nn.Module):
    """
    Total Loss for Counterfactual GAN:
      L_G = lambda_adv * L_adv + lambda_recon * L_recon + lambda_cls * L_cls + lambda_id * L_id
    """
    def __init__(
        self,
        classifier: nn.Module,
        lambda_adv: float = 1.0,
        lambda_recon: float = 10.0,
        lambda_cls: float = 20.0,
        lambda_id: float = 5.0,
    ):
        super().__init__()
        self.classifier = classifier
        self.lambda_adv = lambda_adv
        self.lambda_recon = lambda_recon
        self.lambda_cls = lambda_cls
        self.lambda_id = lambda_id
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def compute_generator_loss(
        self,
        real_img: torch.Tensor,
        orig_prob: torch.Tensor,
        target_prob: torch.Tensor,
        fake_target_img: torch.Tensor,
        recon_img: torch.Tensor,
        pred_fake_disc: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all Generator loss terms:
          - Adversarial Loss (LSGAN): (D(G(x, t)) - 1)^2
          - Self-Reconstruction Loss: ||G(x, orig_prob) - x||_1
          - Classifier Consistency Loss: ||P_mel(G(x, t)) - t||^2
          - Identity Preservation Loss: ||G(x, t) - x||_1 (subtle structural anchor)
        """
        # 1. Adversarial loss (fool discriminator)
        loss_adv = self.mse_loss(pred_fake_disc, torch.ones_like(pred_fake_disc))

        # 2. Self-reconstruction loss (when conditioned on original prob, match input)
        loss_recon = self.l1_loss(recon_img, real_img)

        # 3. Classifier consistency loss (generated image must match target probability)
        pred_mel_prob = get_melanoma_probability(fake_target_img, self.classifier)
        loss_cls = self.mse_loss(pred_mel_prob, target_prob)

        # 4. Identity structural preservation loss
        loss_id = self.l1_loss(fake_target_img, real_img)

        # Total weighted generator loss
        loss_g = (
            self.lambda_adv * loss_adv
            + self.lambda_recon * loss_recon
            + self.lambda_cls * loss_cls
            + self.lambda_id * loss_id
        )

        return {
            "loss_g": loss_g,
            "loss_adv": loss_adv,
            "loss_recon": loss_recon,
            "loss_cls": loss_cls,
            "loss_id": loss_id,
            "pred_mel_prob": pred_mel_prob,
        }

    def compute_discriminator_loss(
        self,
        pred_real_disc: torch.Tensor,
        pred_fake_disc: torch.Tensor,
    ) -> torch.Tensor:
        """LSGAN Discriminator loss: 0.5 * [(D(real)-1)^2 + D(fake)^2]."""
        loss_d_real = self.mse_loss(pred_real_disc, torch.ones_like(pred_real_disc))
        loss_d_fake = self.mse_loss(pred_fake_disc, torch.zeros_like(pred_fake_disc))
        loss_d = 0.5 * (loss_d_real + loss_d_fake)
        return loss_d
