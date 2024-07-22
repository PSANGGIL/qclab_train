import torch
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from glob import glob
from time import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DB(Dataset):
    def __init__(self, filename)

    def __len__(self):
        
        return #length of database

    def __getitem__(self, idx):
        
        return #smiles, singlet_energy, triplet_energy
