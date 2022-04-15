import torch
import torch.nn.functional as F
from torchvision.transforms.functional import convert_image_dtype
import numpy as np
import config
from einops import repeat


@torch.no_grad()
def np2GPUTensor(np_array: np.array) -> torch.Tensor:
    """Convert 2D np array to float 3D GPU Tensor"""
    cpu_array = np.expand_dims(np_array, axis=0).astype(np.float32)
    return torch.from_numpy(cpu_array).to(config.DEVICE)


@torch.jit.script
def convert_shadow_mask_single2MultiChannel(
    shadow_mask: torch.Tensor,
    img_height: int = 480,
    img_width: int = 640,
    n_channels: int = 3,
    kernel_size: int = -1,
) -> torch.Tensor:
    """
    shadow_mask : Expected [1, H, W] shaped Tensor
    """
    with torch.no_grad():
        base_tensor = torch.zeros(
            n_channels, img_height, img_width, device=config.DEVICE
        )
        normalised_mask = convert_image_dtype(shadow_mask)
        final_mask = 1 - normalised_mask

        if kernel_size != -1:
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=config.DEVICE)
            final_mask = F.conv2d(final_mask, kernel, padding="same")

        # Expected Output: torch.Size([1, 4, 480, 640]); Final Image Shape -- [4, H, W]
        return torch.cat((base_tensor, final_mask), dim=0).unsqueeze(0)


@torch.jit.script
def repeat_for_batch(img_tensor, batch_size: int = config.BATCH_SIZE):
    """NOTE: img tensor should have fake batch diimension of 1"""
    return torch.repeat_interleave(img_tensor, batch_size, 0)


@torch.jit.script
def apply_shadow_mask_2_batch(
    shadow_mask: torch.Tensor,
    batch_shadow_image: torch.Tensor,
    batch_size: int = config.BATCH_SIZE,
    alpha: float = 0.90,
) -> torch.Tensor:
    """
    img_mask = shadow_mask * image
    final_image = alpha * image + beta * img_mask

    NOTE: :param shadow_mask: Expected shape is of [1, 4, H, W]
    NOTE: :param batch_shadow_image: Expected shape is of [BS, 3, H, W]
    """

    with torch.no_grad():
        # Repeat for batch
        shp = batch_shadow_image.shape
        # Repeat the Shadow
        batch_shadow_mask = repeat_for_batch(shadow_mask, batch_size=batch_size)
        # Fake Alpha channels for all images
        batch_alpha_channel = torch.ones(
            batch_size,
            1,
            shp[2],
            shp[3],
            device=config.DEVICE,
        )
        # Add alpha channel
        batch_img_with_alpha = torch.cat(
            [batch_shadow_image, batch_alpha_channel], dim=1
        )

        # Blend
        batch_img_mask = batch_img_with_alpha * batch_shadow_mask
        batch_final_img_batch = (
            alpha * batch_img_with_alpha + (1 - alpha) * batch_img_mask
        )

        return batch_final_img_batch
