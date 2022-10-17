import torch

class DropoutLayer(torch.nn.Module):

    def __init__(self, p: float = 0.5):
        super().__init__()

    def forward(self, x):
        pass

    @classmethod
    def create_instance(cls):
        pass
