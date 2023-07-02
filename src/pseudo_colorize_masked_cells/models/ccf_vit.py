import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from vit_pytorch.vit import Transformer
from vit_pytorch.mobile_vit import MobileViTBlock
from torchmetrics.functional.classification import binary_jaccard_index

from .vit_backbone import ViTBackbone


class CCFViT(pl.LightningModule):
    def __init__(
        self,
        image_size=384,
        patch_size_backbone=8,
        dim_backbone=512,
        depth_backbone=6,
        heads_backbone=8,
        mlp_dim_backbone=1024,
        lr=1e-4,
        rel_dims=False,
        aux_pcolor=False,
        decoder_dim=512,
        decoder_depth=3,
        decoder_heads=8,
        decoder_mlp_dim=1024,
        backbone_eval=False,
        heads_add_3x3_convs=False,
        neck_dual_conv=False,
        neck_dim=256,
        head_channels=256,
        heads_add_activation=False,
    ):
        super().__init__()
        self.context_block = ContextBlock()
        self.backbone = ViTBackbone(
            image_size=image_size,
            patch_size=patch_size_backbone,
            num_classes=1,
            dim=dim_backbone,
            depth=depth_backbone,
            heads=heads_backbone,
            mlp_dim=mlp_dim_backbone,
        )
        self.neck = neck(
            in_channels=dim_backbone, out_channels=neck_dim, dual_conv=neck_dual_conv
        )
        self.heatmap_head = nn.Sequential(
            # upsample_block(256, 256), # TODO: Make num upsample blocks dynamic based on patch_size_backbone
            upsample_block(neck_dim, head_channels, heads_add_3x3_convs),
            upsample_block(head_channels, head_channels, heads_add_3x3_convs),
            nn.Conv2d(head_channels, 1, kernel_size=1),
        )
        self.hw_head = nn.Sequential(
            # upsample_block(256, 256),
            upsample_block(neck_dim, head_channels, heads_add_3x3_convs),
            upsample_block(head_channels, head_channels, heads_add_3x3_convs),
            nn.Conv2d(head_channels, 2, kernel_size=1),
        )
        self.lr = lr
        self.image_size = image_size
        self.rel_dims = rel_dims
        self.chw_to_hwc = Rearrange("b c h w -> b h w c")

        self.aux_pcolor = aux_pcolor
        self.backbone_eval = backbone_eval

        self.to_img = Rearrange(
            "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
            h=image_size // patch_size_backbone,
            w=image_size // patch_size_backbone,
            p1=patch_size_backbone,
            p2=patch_size_backbone,
        )
        if aux_pcolor:
            self.aux_decoder = Transformer(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                mlp_dim=decoder_mlp_dim,
                dim_head=decoder_dim // decoder_heads,
            )
            pixel_values_per_patch = (
                patch_size_backbone**2 * 3
            )  # pcolor img w/ 3 channels
            self.to_pixels_aux = nn.Linear(decoder_dim, pixel_values_per_patch)

        self.heads_add_activation = heads_add_activation

    def forward(self, x):
        if self.backbone_eval:
            with torch.no_grad():
                self.backbone.eval()
                x = self.context_block(x)
                tokens, _, _ = self.backbone(x)
        else:
            x = self.context_block(x)
            tokens, _, _ = self.backbone(x)

        _tokens = rearrange(tokens, "b (h w) d -> b h w d", h=48, w=48)
        x = self.neck(_tokens)
        heatmap = self.heatmap_head(x)
        hw = self.hw_head(x)

        if self.heads_add_activation:
            heatmap = F.sigmoid(heatmap)
            hw = F.relu(hw)

        if not self.aux_pcolor:
            return heatmap, self.chw_to_hwc(hw)
        else:
            pcolor_img = self.aux_decoder(tokens)
            pcolor_img = self.to_pixels_aux(pcolor_img)
            return heatmap, self.chw_to_hwc(hw), self.to_img(pcolor_img)

    def training_step(self, batch, batch_idx):
        if self.aux_pcolor:
            pred_heatmap, pred_dims, pcolor_img = self.forward(batch["img"])
        else:
            pred_heatmap, pred_dims = self.forward(batch["img"])

        heatmap_loss = F.huber_loss(
            input=pred_heatmap[:, 0, ...], target=batch["heatmap"]
        )
        heatmap_miou = binary_jaccard_index(
            preds=(pred_heatmap[:, 0, ...] > 0.75).float(),
            target=(batch["heatmap"] > 0.75).float(),
        )
        dims_masked = pred_dims * batch["centroid_dim_mask"]

        if self.rel_dims:
            height_loss = F.huber_loss(
                input=dims_masked[..., 0],
                target=batch["centroid_dim_blocks"][..., 0] / self.image_size,
            )
            width_loss = F.huber_loss(
                input=dims_masked[..., 1],
                target=batch["centroid_dim_blocks"][..., 1] / self.image_size,
            )
        else:
            height_loss = F.huber_loss(
                input=dims_masked[..., 0], target=batch["centroid_dim_blocks"][..., 0]
            )
            width_loss = F.huber_loss(
                input=dims_masked[..., 1], target=batch["centroid_dim_blocks"][..., 1]
            )

        loss = heatmap_loss + 0.5 * height_loss + 0.5 * width_loss

        if self.aux_pcolor:
            pcolor_loss = F.mse_loss(input=pcolor_img, target=batch["pcolor_img"])
            loss += pcolor_loss
            self.log("train_pcolor_loss", pcolor_loss, sync_dist=True)

        self.log("train_heatmap_loss", heatmap_loss, sync_dist=True)
        self.log("train_height_loss", height_loss, sync_dist=True)
        self.log("train_width_loss", width_loss, sync_dist=True)
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_miou", heatmap_miou, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.aux_pcolor:
            pred_heatmap, pred_dims, pcolor_img = self.forward(batch["img"])
        else:
            pred_heatmap, pred_dims = self.forward(batch["img"])

        heatmap_loss = F.huber_loss(
            input=pred_heatmap[:, 0, ...], target=batch["heatmap"]
        )
        heatmap_miou = binary_jaccard_index(
            preds=(pred_heatmap[:, 0, ...] > 0.75).float(),
            target=(batch["heatmap"] > 0.75).float(),
        )
        dims_masked = pred_dims * batch["centroid_dim_mask"]

        if self.rel_dims:
            height_loss = F.huber_loss(
                input=dims_masked[..., 0],
                target=batch["centroid_dim_blocks"][..., 0] / self.image_size,
            )
            width_loss = F.huber_loss(
                input=dims_masked[..., 1],
                target=batch["centroid_dim_blocks"][..., 1] / self.image_size,
            )
        else:
            height_loss = F.huber_loss(
                input=dims_masked[..., 0], target=batch["centroid_dim_blocks"][..., 0]
            )
            width_loss = F.huber_loss(
                input=dims_masked[..., 1], target=batch["centroid_dim_blocks"][..., 1]
            )

        loss = heatmap_loss + 0.5 * height_loss + 0.5 * width_loss

        if self.aux_pcolor:
            pcolor_loss = F.mse_loss(input=pcolor_img, target=batch["pcolor_img"])
            loss += pcolor_loss
            self.log("val_pcolor_loss", pcolor_loss, sync_dist=True)

        self.log("val_heatmap_loss", heatmap_loss, sync_dist=True)
        self.log("val_height_loss", height_loss, sync_dist=True)
        self.log("val_width_loss", width_loss, sync_dist=True)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_miou", heatmap_miou, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=50,
                    eta_min=1e-6,
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }


def neck(
    in_channels,
    out_channels,
    depth=2,
    kernel_size=3,
    patch_size=(16, 16),
    mlp_dim=1024,
    num_backbone_patch_rows=48,
    num_backbone_patch_cols=48,
    dual_conv=False,
):
    if dual_conv:
        return nn.Sequential(
            Rearrange("b patch_row patch_col c -> b c patch_row patch_col"),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
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
                [out_channels, num_backbone_patch_cols, num_backbone_patch_rows]
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.LayerNorm(
                [out_channels, num_backbone_patch_cols, num_backbone_patch_rows]
            ),
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
    else:
        return nn.Sequential(
            Rearrange("b patch_row patch_col c -> b c patch_row patch_col"),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
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
                [out_channels, num_backbone_patch_cols, num_backbone_patch_rows]
            ),
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


def upsample_block(in_channels, out_channels, add_3x3_convs: bool = False):
    if add_3x3_convs:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
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
