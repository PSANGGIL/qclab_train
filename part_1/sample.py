import torch
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from glob import glob
from time import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DB(Dataset):

    def __init__(self, filename, precision=32):
        
        if precision==32:
            self.precision = torch.float32
        elif precision==64:
            self.precision = torch.float64        
    
        self.data = pd.read_csv(filename)
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        smiles          = self.data.iloc[idx, 1]
        singlet_energy  = self.data.iloc[idx, 2]
        triplet_energy  = self.data.iloc[idx, 3]
        
        return smiles,\
                torch.tensor(singlet_energy , dtype=self.precision),\
                torch.tensor(triplet_energy , dtype=self.precision)

if __name__ == "__main__":
    data_path = "../data/excitated_energy/test_db.csv"
    
    db = DB ( data_path, precision=64)[0]
    print((db))
