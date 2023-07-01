import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from einops import repeat
from vit_pytorch import MAE
from torchmetrics.functional import structural_similarity_index_measure

from .ccf_vit import ContextBlock


class PcolorMAE(MAE, pl.LightningModule):
    def __init__(self, *args, lr, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_block = ContextBlock()
        self.lr = lr

    def forward(self, img, pcolor_img):
        device = img.device
        img = self.context_block(img)

        # get patches
        patches = self.to_patch(img)
        pcolor_patches = self.to_patch(pcolor_img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)

        if self.encoder.pool == "cls":
            tokens = tokens + self.encoder.pos_embedding[:, 1 : (num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens = tokens + self.encoder.pos_embedding.to(device)

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(
            unmasked_indices
        )

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.zeros(
            batch, num_patches, self.decoder_dim, device=device
        )
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # get the patches to be masked for the final reconstruction loss
        masked_pcolor_patches = pcolor_patches[batch_range, masked_indices]

        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_pcolor_patches)
        ssim_score = structural_similarity_index_measure(
            pred_pixel_values.view(
                -1,
                3,
                int(self.encoder.image_size * (1 - self.masking_ratio)),
                int(self.encoder.image_size * (1 - self.masking_ratio)),
            ),
            masked_pcolor_patches.view(
                -1,
                3,
                int(self.encoder.image_size * (1 - self.masking_ratio)),
                int(self.encoder.image_size * (1 - self.masking_ratio)),
            ),
        )

        return recon_loss, ssim_score

    def training_step(self, batch, batch_idx):
        loss, ssim_score = self.forward(
            batch["img"], batch["pcolor_img"].view(-1, 3, 384, 384)
        )

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_ssim", ssim_score, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, ssim_score = self.forward(
            batch["img"], batch["pcolor_img"].view(-1, 3, 384, 384)
        )

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_ssim", ssim_score, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=75,
                    eta_min=1e-6,
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }
