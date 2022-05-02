import matplotlib.pyplot as plt
import torch.optim as optim
from monai.losses.dice import DiceLoss
from torch.nn import Module

import config
from components import MedSegModel
from dataset import train_loader
from components.componentLogger import get_logger

LOG = get_logger()

# Model
model: Module = MedSegModel().to(config.DEVICE)

# Optimizer
optimizer: optim.Optimizer = optim.Adam(model.parameters())

# loss criterion
criterion = DiceLoss(
    to_onehot_y=True,
).to(config.DEVICE)

# Loss List
loss_list = []


def train_one_epoch(model: Module, optimizer, dloader):
    """
    Trains Model for One epoch

    Args:
        model (torch.nn.Module) - The model to be trained
        optimizer (torch.optim) - The optimizer for the model
        dloader (torch.utils.data.Dataloader) - The data loader for the model
    """
    model.train()
    for idx, batch in enumerate(dloader):
        # Just use 2 modalities
        x1, x2 = batch["image"][0][0], batch["image"][0][1]
        # Adds 2 fake dimensions
        x1, x2 = x1.unsqueeze(0).unsqueeze(0), x2.unsqueeze(0).unsqueeze(0)
        LOG.debug(f"Shape of Inputs: {x1.shape} & {x2.shape}")
        # Just work on a single Label
        y_true = batch["label"][0][0]
        optimizer.zero_grad()
        y_pred = model(x1, x2)
        loss = criterion(y_pred, y_true)

        # Train
        loss.backward()
        optimizer.step()

    return loss


if __name__ == "__main__":
    for epoch in range(config.NUM_TRAIN_EPOCHS):
        print(f"EPOCH: {epoch}")
        loss_list += [train_one_epoch(model, optimizer, train_loader).detach().item()]
        print(f"Current Train Loss: {loss_list[-1]}")

    plt.plot(loss_list)
