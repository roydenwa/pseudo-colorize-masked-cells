import torch
import torch.nn.functional as F

from torch import nn
from einops.layers.torch import Rearrange
from vit_pytorch.mobile_vit import MobileViTBlock


def neck(
    in_channels,
    out_channels,
    depth=2,
    kernel_size=3,
    patch_size=(16, 16),
    mlp_dim=1024,
    num_backbone_patch_rows=48,
    num_backbone_patch_cols=48,
):
    return nn.Sequential(
        Rearrange(
            "b (h w) d -> b d h w", h=num_backbone_patch_cols, w=num_backbone_patch_rows
        ),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        MobileViTBlock(
            dim=out_channels,
            depth=depth,
            channel=out_channels,
            kernel_size=kernel_size,
            patch_size=patch_size,
            mlp_dim=mlp_dim,
        ),
        nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        ),
        nn.LayerNorm([out_channels, num_backbone_patch_cols, num_backbone_patch_rows]),
        nn.Upsample(scale_factor=2, mode="bilinear"),
        MobileViTBlock(
            dim=out_channels,
            depth=depth,
            channel=out_channels,
            kernel_size=kernel_size,
            patch_size=patch_size,
            mlp_dim=mlp_dim,
        ),
        nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        ),
        nn.LayerNorm(
            [out_channels, 2 * num_backbone_patch_cols, 2 * num_backbone_patch_rows]
        ),
    )


def upsample_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="bilinear"),
    )


class ContextBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_conv_3x3 = nn.Conv2d(
            in_channels=1, out_channels=10, kernel_size=3, padding="same"
        )
        self.b_conv_3x3 = nn.Conv2d(
            in_channels=1, out_channels=10, kernel_size=3, padding="same"
        )
        self.r_conv_1x1 = nn.Conv2d(
            in_channels=10, out_channels=1, kernel_size=1, padding="same"
        )
        self.b_conv_1x1 = nn.Conv2d(
            in_channels=10, out_channels=1, kernel_size=1, padding="same"
        )
        self.hwc_to_chw = Rearrange("b h w c -> b c h w")

    def forward(self, img):
        img = self.hwc_to_chw(img)
        r, g, b = torch.split(img, 1, dim=1)
        r = self.r_conv_3x3(r)
        r = self.r_conv_1x1(r)
        r = r * g + g  # MAC
        r = F.sigmoid(r)
        b = self.b_conv_3x3(b)
        b = self.b_conv_1x1(b)
        b = b * g + g
        b = F.sigmoid(b)

        return torch.concat((r, g, b), dim=1)
