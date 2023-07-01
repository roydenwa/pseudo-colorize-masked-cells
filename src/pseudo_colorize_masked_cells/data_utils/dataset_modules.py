import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from glob import glob
from skimage import io
from torch.utils.data import Dataset, DataLoader, Subset

from .preprocess_microscopy_imgs import (
    mask2bboxes,
    bboxes2heatmap,
    bboxes2centroid_dim_blocks,
    pseudo_colorize_img,
)


class N2DHSIM(Dataset):
    def __init__(self, dataset_folder: str):
        img_paths = sorted(glob(dataset_folder + "/imgs/*.tif"))
        mask_paths = sorted(glob(dataset_folder + "/masks/*.tif"))
        self.data = []

        for img_path, mask_path in zip(img_paths, mask_paths):
            self.data.append([img_path, mask_path])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        bboxes = mask2bboxes(mask)
        heatmap = bboxes2heatmap(bboxes, img.shape)
        centroid_dim_blocks = bboxes2centroid_dim_blocks(bboxes, img.shape)

        pcolor_img = pseudo_colorize_img(img, plt.cm.nipy_spectral)
        img = torch.tensor(np.dstack((img, img, img)), dtype=torch.float32)
        heatmap = torch.tensor(heatmap, dtype=torch.float32)
        centroid_dim_blocks = torch.tensor(centroid_dim_blocks, dtype=torch.float32)
        centroid_dim_mask = torch.where(centroid_dim_blocks > 0, 1.0, 0.0)

        return {
            "img": img,
            "pcolor_img": pcolor_img,
            "heatmap": heatmap,
            "centroid_dim_blocks": centroid_dim_blocks,
            "centroid_dim_mask": centroid_dim_mask,
        }


class N2DHSIMDataModule(pl.LightningDataModule):
    def __init__(
        self, dataset_folder, batch_size, num_dataloader_workers=8, pin_memory=True
    ):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str):
        if stage == "fit":
            n2dh_sim_ds = N2DHSIM(self.dataset_folder)
            self.train_split = Subset(n2dh_sim_ds, torch.arange(0, 1719, 1))
            self.val_split = Subset(n2dh_sim_ds, torch.arange(1719, 1719 + 430, 1))
        elif stage == "predict":
            n2dh_sim_ds = N2DHSIM(self.dataset_folder)
            self.test_split = Subset(n2dh_sim_ds, torch.arange(1719, 1719 + 430, 1))

    def train_dataloader(self):
        return DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_split,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_split,
            batch_size=1,
            num_workers=self.num_dataloader_workers,
        )
