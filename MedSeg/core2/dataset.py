# Data for BRATS data
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
from custom_transform import ConvertToMultiChannelBasedOnBratsClassesd

_train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
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
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        EnsureTyped(keys=["image", "label"]),
    ]
)
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
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ]
)
# Set root data directory
_root_dir = str(config.DATA_DIR)

# Data loaders
# here we don't cache any data in case out of memory issue
_train_ds = DecathlonDataset(
    root_dir=_root_dir,
    task="Task01_BrainTumour",
    transform=_train_transform,
    section="training",
    # download=True,
    cache_rate=0.0,
    num_workers=4,
)
train_loader = torch.utils.data.DataLoader(
    _train_ds, batch_size=1, shuffle=True, num_workers=4
)

_val_ds = DecathlonDataset(
    root_dir=_root_dir,
    task="Task01_BrainTumour",
    transform=_val_transform,
    section="validation",
    # download=True,
    cache_rate=0.0,
    num_workers=4,
)
val_loader = torch.utils.data.DataLoader(
    _val_ds, batch_size=1, shuffle=False, num_workers=4
)
