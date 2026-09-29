import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class myDataset(Dataset):
    def __init__(self, x_data, y_data, x_mean=None, x_std=None, y_mean=None, y_std=None, log_transform=False):
        self.x_data = x_data
        self.y_data = y_data

        # Apply log transformation if required
        if log_transform:
            self.x_data = np.log(self.x_data)

        # Normalize x if mean and std are provided
        if x_mean is not None and x_std is not None:
            self.x_data = (self.x_data - x_mean) / x_std

        # Normalize y if mean and std are provided
        if y_mean is not None and y_std is not None:
            self.y_data = (self.y_data - y_mean) / y_std
        
        self.y_data = torch.tensor(self.y_data).float()
        self.x_data = torch.tensor(self.x_data).float()

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.y_data[idx], self.x_data[idx]
