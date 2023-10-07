"""
This module is designed for to create a dataset from a csv file.
"""
import pandas as pd
import torch
from encoder import Encoder
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    This class is designed for to create a dataset from a csv file.
    """

    def __init__(
        self, x_data: pd.DataFrame, y_data: pd.DataFrame, encoder: Encoder, mode: str
    ):
        self.x_data = torch.tensor(
            encoder.encoder_data(x_data, mode), dtype=torch.float32
        )
        self.mode = mode
        if self.mode != "test":
            self.y_data = torch.tensor(encoder.encoder_target(y_data), dtype=torch.int64)

    def __len__(self):
        return self.x_data.size(0)

    def __getitem__(self, idx: int):
        if self.mode != "test":
            return self.x_data[idx, :], self.y_data[idx]
        return self.x_data[idx, :]
