# Training Utils
import torch
import torchvision as tv
import matplotlib.pyplot as plt
import utils as uts


def save_model(model: torch.nn.Module, path_to_save: str) -> None:
    """
    Preferred to use save / load_state_dict
    See: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """

    model.eval()

    torch.save(model.state_dict(), path=path_to_save)


def save_batch_as_image(img_batch: torch.TensorType, path_to_save: str) -> None:
    """
    Saves a batch as a grid of images in .png format
    """

    grid_img = tv.utils.make_grid(img_batch, normalize=True)
    plt.imsave(path_to_save, grid_img.permute(1, 2, 0))


def train_one_batch():
    pass
