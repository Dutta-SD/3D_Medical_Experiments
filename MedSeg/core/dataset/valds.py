import torch
from monai.apps import DecathlonDataset

import config
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
)
from transforms.custom_transform import ConvertToMultiChannelBasedOnBratsClassesd

DATA_ROOT_DIR = str(config.DATA_DIR)


_val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(
            keys=["image", "label"], roi_size=[224, 224, 144], random_size=False
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ]
)

_val_ds = DecathlonDataset(
    root_dir=DATA_ROOT_DIR,
    task="Task01_BrainTumour",
    transform=_val_transform,
    section="validation",
    # download=True,
    cache_rate=0.0,
    num_workers=4,
)
VAL_DATALOADER = torch.utils.data.DataLoader(
    _val_ds,
    batch_size=config.VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
)
