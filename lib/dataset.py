from typing import Any, Tuple
from torch.utils.data import Dataset
import torch
import math

def createModularAdditionData(module: int):
    return torch.Tensor([[i, j, (i+j) % module] for i in range(module) for j in range(module)]).to(dtype=torch.int)

def randomSplit(n: int, fractions: Tuple[float, float, float], device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    factor = n/(fractions[0]+fractions[1]+fractions[2])
    indexes = torch.randperm(n, device=device)

    train_length = math.ceil(fractions[0]*factor)
    val_length = math.floor(fractions[1]*factor)

    return indexes[0:train_length],  indexes[train_length : (train_length + val_length)], indexes[(train_length + val_length):]

class CustomDataset(Dataset):
    def __init__(self, data, indexes):
        super().__init__()
        self.data = data
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index) -> Any:
        a, b, label = self.data[self.indexes[index]]
        return a, b, label