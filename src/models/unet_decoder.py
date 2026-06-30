"""
unet_decoder.py

U-Net Decoder for Lesion Segmentation

Author: Stroke CDSS Project
"""

import torch
import torch.nn as nn


# ============================================================
# Decoder Block
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = nn.Sequential(

            nn.Conv2d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        x,
        skip
    ):

        x = self.upsample(x)

        # Concatenate skip connection
        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.conv(x)

        return x


# ============================================================
# U-Net Decoder
# ============================================================

class UNetDecoder(nn.Module):

    """
    Expected Swin feature maps:

    f1 : [B, 96, 56, 56]
    f2 : [B, 192, 28, 28]
    f3 : [B, 384, 14, 14]
    f4 : [B, 768, 7, 7]
    """

    def __init__(self):

        super().__init__()

        # 7x7 -> 14x14
        self.dec4 = DecoderBlock(
            in_channels=768,
            skip_channels=384,
            out_channels=384
        )

        # 14x14 -> 28x28
        self.dec3 = DecoderBlock(
            in_channels=384,
            skip_channels=192,
            out_channels=192
        )

        # 28x28 -> 56x56
        self.dec2 = DecoderBlock(
            in_channels=192,
            skip_channels=96,
            out_channels=96
        )

        # 56x56 -> 112x112
        self.final_up1 = nn.ConvTranspose2d(
            96,
            64,
            kernel_size=2,
            stride=2
        )

        # 112x112 -> 224x224
        self.final_up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        # Segmentation head
        self.segmentation_head = nn.Sequential(

            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                1,
                kernel_size=1
            )

            # No sigmoid here
            # BCEWithLogitsLoss will handle it
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        features
    ):

        f1, f2, f3, f4 = features

        x = self.dec4(
            f4,
            f3
        )

        x = self.dec3(
            x,
            f2
        )

        x = self.dec2(
            x,
            f1
        )

        x = self.final_up1(
            x
        )

        x = self.final_up2(
            x
        )

        mask = self.segmentation_head(
            x
        )

        return mask


# ============================================================
# Unit Test
# ============================================================

if __name__ == "__main__":

    model = UNetDecoder()

    f1 = torch.randn(
        2, 96, 56, 56
    )

    f2 = torch.randn(
        2, 192, 28, 28
    )

    f3 = torch.randn(
        2, 384, 14, 14
    )

    f4 = torch.randn(
        2, 768, 7, 7
    )

    pred_mask = model(
        [f1, f2, f3, f4]
    )

    print(
        "Output Shape:",
        pred_mask.shape
    )