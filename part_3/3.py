import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 1. Seed settings

# 2. Download data and select data
def download_data(ticker, start_date, end_date):
    return df


# 3. Data normalization
def min_max_scaler(data):
    return scaled_data, min_val, max_val


# 4. Create sequence
def create_sequences(data, seq_length):


# 5. Defining datasets and dataloaders
class StockDataset(Dataset):
    def __init__(self, sequences, labels):

    def __len__(self):
        return 

    def __getitem__(self, idx):
        return 



# 6. RNN Model definition
class RNN(nn.Module):

# 7. Loss function and optimizer settings

# 8. Model training
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}')

train_model(model, train_loader, criterion, optimizer)


