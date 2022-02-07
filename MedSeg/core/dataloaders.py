import pytorch_lightning as pl
import glob
import os
import config
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
)
import download_spleen_data
from monai.data import CacheDataset, list_data_collate, decollate_batch, pad_list_data_collate
import torch


class SpleenImageDataLoader(pl.LightningDataModule):
    """
    Data Loader for Spleen Medical Dataset
    ------------------------------------------
    If not downloaded previously, run the
    download_spleen_data.py file
    first
    """

    def __init__(self, data_dir: str, num_val_samples: int = 10):
        super().__init__()
        # data_dir : same as DATA_DIR of config.py
        self.data_dir = data_dir
        self.num_val_samples = num_val_samples
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

    def prepare_data(self):
        download_spleen_data.download_spleen_data()

    def setup(self, stage=None):

        # set up the correct data path
        train_images = sorted(
            glob.glob(
                os.path.join(self.data_dir, "Task09_Spleen", "imagesTr", "*.nii.gz")
            )
        )
        train_labels = sorted(
            glob.glob(
                os.path.join(self.data_dir, "Task09_Spleen", "labelsTr", "*.nii.gz")
            )
        )
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]

        self.train_files, self.val_files = (
            data_dicts[: -self.num_val_samples],
            data_dicts[-self.num_val_samples :],
        )

    def train_dataloader(self):
        set_determinism(seed=config.RNG_SEED)

        x = CacheDataset(
            data=self.train_files,
            transform=self.train_transforms,
            cache_rate=1.0,
            num_workers=config.NUM_WORKERS,
        )

        return torch.utils.data.DataLoader(
            dataset = x,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            collate_fn=pad_list_data_collate,          
        )

    def val_dataloader(self):
        set_determinism(seed=config.RNG_SEED)

        x = CacheDataset(
            data=self.val_files,
            transform=self.val_transforms,
            cache_rate=1.0,
            num_workers=config.NUM_WORKERS,
        )

        return torch.utils.data.DataLoader(
            dataset = x,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            collate_fn=pad_list_data_collate,          
        )