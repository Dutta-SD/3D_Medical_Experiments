import torch
from torch.nn import Module

class ActorUnet(Module):
    "UNet AutoEncoder"

    def __init__(self, **kwargs):
        self._encoder = 