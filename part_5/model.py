import torch.nn as nn
import torch.optim as optim
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import random
from db import DB


class MyLSTM(nn.Module):
    def __init__(self, max_length, vocab_size, embed_dim, hidden_dim, output_dim, num_layers):
        super(MyLSTM, self).__init__()
   
    def forward(self, x, pad_len):
        return y_pred

class MyMLP(torch.nn.Module):
    def __init__(self, max_length, vocab_size, embed_dim, hidden_dim, output_dim, num_layers):
        super(MyMLP, self).__init__()

        self.layer = torch.nn.Sequential(  
                                            torch.nn.Linear(embed_dim, hidden_dim),
                                            torch.nn.Softplus(),
                                            torch.nn.Dropout(0.2),
                                            torch.nn.Linear(hidden_dim, output_dim),
                                            torch.nn.Softplus(),
                                            torch.nn.Dropout(0.2),
                                            torch.nn.Linear(output_dim, 1),
                                        )
        

    def forward(self, x, pad_len):
        return y_pred

with open('conditions.json', 'r') as c:
    conditions = json.load(c)

