from torch.utils.data import Dataset, DataLoader, random_split
from encoder import Encoder
import torch
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, X_data: pd.DataFrame, Y_data: pd.DataFrame, encoder: Encoder, mode:str):
        self.x = torch.tensor(encoder.encoder_data(X_data, mode), dtype=torch.float32)  
        self.mode= mode
        if self.mode != "test":
            self.y = torch.tensor(encoder.encoder_target(Y_data), dtype=torch.int64)
            

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx: int):
        if self.mode != "test":
            return self.x[idx, :], self.y[idx]
        else:
            return self.x[idx, :]