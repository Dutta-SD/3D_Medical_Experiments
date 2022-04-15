import torch
import torch.nn.functional as F
from torchvision.transforms import ConvertImageDtype
import numpy as np
import config


@torch.no_grad()
def np2GPUTensor(np_array: np.array) -> torch.cuda.Tensor:
    """Convert 2D np array to float 3D GPU Tensor"""
    cpu_array = np.expand_dims(np_array, axis=0)
    return torch.from_numpy(cpu_array).to(config.DEVICE)


@torch.jit.script
def convert_shadow_mask_single2MultiChannel(
    shadow_mask: torch.cuda.Tensor,
    img_height: int = 480,
    img_width: int = 640,
    n_channels: int = 3,
    **kwargs: dict,
) -> torch.cuda.Tensor:
    with torch.no_grad():
        base_tensor = torch.zeros(
            n_channels, img_height, img_width, device=config.DEVICE
        )
        normalised_mask = ConvertImageDtype(torch.float32)(shadow_mask)
        final_mask = 1 - normalised_mask

        if "kernel_size" in kwargs:
            kernel = torch.ones(
                1, kwargs["kernel_size"], kwargs["kernel_size"], device=config.DEVICE
            )
            final_mask = F.conv2d(final_mask, kernel, padding="same")

        # Expected Output: torch.Size([1, 4, 480, 640])
        return torch.cat([base_tensor, final_mask]).unsqueeze(0)


def apply_shadow_mask_2_batch(
    img_batch: torch.cuda.Tensor,
    shadow_batch: torch.cuda.Tensor,
) -> torch.cuda.Tensor:
    
