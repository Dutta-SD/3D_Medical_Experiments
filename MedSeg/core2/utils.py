# Utils
import torch
import numpy as np
import matplotlib.pyplot as plt
import config


def viz_one_image(
    img_batch: dict,
    slice_num: int = 60,
    img_save_name: str = "Viz_4.png",
):
    # Each image has 4 3d images
    # Each label has 3 3d labels

    # if img_batch["image"].shape != 4 or img_batch["label"] != 4:
    #     raise AttributeError(
    #         f"Expected 4 dimensional tensors, got tensors"
    #         f"of shape{img_batch['image'].shape} and {img_batch['label'].shape}"
    #     )

    # if img_batch["image"].shape[0] != 4 or img_batch["label"][0] != 3:
    #     raise Exception("Bad Shape of input tensors")

    # Image Plot
    plt.figure(figsize=(24, 12))
    plt.axis("off")

    fig, axs = plt.subplots(2, 4, constrained_layout=True)

    # images
    for i in range(4):

        axs[0, i].imshow(img_batch["image"][0][i, :, :,
                                               slice_num].detach().cpu(),
                         cmap="gray")
        axs[0, i].axes.get_xaxis().set_visible(False)
        axs[0, i].axes.get_yaxis().set_visible(False)
        axs[0, i].set_title(f"Image Channel {i+1}", fontdict={"fontsize": 10})

    # Labels
    for i in range(3):
        axs[1, i].imshow(img_batch["label"][0][i, :, :,
                                               slice_num].detach().cpu())
        axs[1, i].axes.get_xaxis().set_visible(False)
        axs[1, i].axes.get_yaxis().set_visible(False)
        axs[1, i].set_title(f"Label Channel {i+1}", fontdict={"fontsize": 10})

    fig.delaxes(axs[1][3])
    fig.suptitle(f"MRI Image Slice Number: {slice_num}", fontsize=16)

    plt.savefig(config.LOG_DIR / img_save_name, dpi=300, bbox_inches="tight")


# # test
# from dataset import val_loader as loader

# batch = None
# with torch.no_grad():
#     for d in loader:
#         batch = d
#         break

#     one_tensor = batch
#     # viz_one_image(one_tensor, img_save_name="Viz_3_full.png", slice_num=100)
#     print(batch["image"].shape)
