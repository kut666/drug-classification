"""
This module is designed for to create a neural network model.
"""

from torch import nn


class ClassificationNetwork(nn.Module):
    """
    This class is designed for to create a neural network model.
    """

    def __init__(self, input_shape: int, num_classes: int):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, num_classes), nn.LogSoftmax(dim=-1)
        )

    def forward(self, inp):
        out = self.model(inp)
        return out
