import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import (
    Dataset,
    DataLoader, 
    TensorDataset
)
from torch.nn.utils.rnn import pack_padded_sequence
import json
import os
import pickle
import random

# from ipdb import set_trace

class MyDataset(Dataset):
    def __init__(self, df):
        self.data = df.values
        self.X = self.data[:, :-1]  # features
        self.y = self.data[:, -1]   # target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.Tensor(self.X[index]), torch.Tensor([self.y[index]])


class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(out)
        out = self.output_layer(out)
        return out