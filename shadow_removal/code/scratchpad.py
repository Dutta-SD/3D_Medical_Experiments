# To test various classes
# from dataloaders.testLoaders import TestDataISTD

# x = TestDataISTD()

# print(x[0]['shadow_image'].shape)
from components.Critic.Critic import Critic
import torch
import config

x, y = torch.randn(8, 3, 480, 640).to(config.DEVICE), torch.randn(8, 3, 480, 640).to(
    config.DEVICE
)
model = Critic().to(config.DEVICE)
print(model(x, y).mean())
