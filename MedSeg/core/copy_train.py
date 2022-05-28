import matplotlib.pyplot as plt
import torch.optim as optim
from monai.losses.dice import DiceLoss
from torch.nn import Module

import config
from components import MedSegModel
from dataset import train_loader
from components.componentLogger import get_logger

# ******************************************************
LOG = get_logger()

# Model
model = MedSegModel().to(config.DEVICE)

# Optimizer
optimizer = optim.Adam(model.parameters())

# loss criterion
criterion = DiceLoss(smooth_nr=0,
                     smooth_dr=1e-5,
                     squared_pred=True,
                     to_onehot_y=False,
                     sigmoid=True).to(config.DEVICE)

# Loss List
loss_list = []
# *******************************************************


def get_image_and_labels_from_batch(batch: dict):
    # Data -- Oth and 1st modality
    x1 = batch["image"][:, 0, :, :, :].to(config.DEVICE).unsqueeze(1)
    x2 = batch["image"][:, 1, :, :, :].to(config.DEVICE).unsqueeze(1)

    # Log
    LOG.info(f"Shape of Inputs: {x1.shape} & {x2.shape}")

    # Labels -- 0th prediction
    y_true = batch["label"][:, 0, :, :, :].to(config.DEVICE).unsqueeze(1)

    # Log
    LOG.info(f"Shape of Output is: {y_true.shape}")
    return x1, x2, y_true


def train_one_epoch(model, optimizer, dloader):
    """
    Trains Model for One epoch

    Args:
        model (torch.nn.Module) - The model to be trained
        optimizer (torch.optim) - The optimizer for the model
        dloader (torch.utils.data.Dataloader) - The data loader for the model
    """
    model.train()
    for idx, batch in enumerate(dloader):
        optimizer.zero_grad()

        x1, x2, y_true = get_image_and_labels_from_batch(batch)
        y_pred = model(x1, x2)

        LOG.info(f"Shape of predicted output from model is {y_pred.shape}")

        loss = criterion(y_pred, y_true)

        # Train
        loss.backward()
        optimizer.step()

    return loss


if __name__ == "__main__":
    for epoch in range(config.NUM_TRAIN_EPOCHS):
        print(f"EPOCH: {epoch}")
        loss_list += [
            train_one_epoch(model, optimizer, train_loader).detach().item()
        ]
        print(f"Current Train Loss: {loss_list[-1]}")

    # plt.plot(loss_list)
