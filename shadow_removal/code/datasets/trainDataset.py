from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, PILToTensor, ConvertImageDtype
import config
import glob
import os
from PIL import Image
import torch


_train_transforms_default = Compose([PILToTensor(), ConvertImageDtype(torch.float32)])


class TrainDataISTD(Dataset):
    def __init__(
        self,
        transforms=_train_transforms_default,
        load_mask=False,
        load_shadow_free=False,
    ):
        super(TrainDataISTD, self).__init__()
        self.transforms = transforms
        # Load Shadow Images
        self._shadow_images = glob.glob(
            str(config.DATA_DIR / "ISTD_Dataset/train/train_A/*")
        )
        self._shadow_free_images = None
        self._shadow_masks = None
        self.load_mask = load_mask
        self.load_shadow_free = load_shadow_free

        if load_mask:
            self._shadow_masks = glob.glob(
                str(config.DATA_DIR / "ISTD_Dataset/train/train_B/*")
            )
        if load_shadow_free:
            self._shadow_images = glob.glob(
                str(config.DATA_DIR / "ISTD_Dataset/train/train_C/*")
            )

    def __len__(self):
        # return len(self._shadow_images)
        return 64

    def __getitem__(self, idx):
        img_dict = {}
        img_dict["shadow_image"] = Image.open(self._shadow_images[idx])
        if self.load_mask:
            img_dict["shadow_mask_image"] = Image.open(self._shadow_masks[idx])
        if self.load_shadow_free:
            img_dict["shadow_free_image"] = Image.open(self._shadow_free_images[idx])

        if self.transforms is not None:
            img_dict = {key: self.transforms(value) for key, value in img_dict.items()}

        return img_dict
