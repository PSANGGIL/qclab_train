import torch
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
from glob import glob
from time import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DB(Dataset):
    def __init__(self, filename, emb_dim = 8, precision=64):

        if precision==16:
            self.precision = torch.half
        elif precision==32:
            self.precision = torch.float32
        elif precision==64:
            self.precision = torch.float64
        
        self.filename = filename
        st = time()
        f=pd.read_csv(self.filename)
        print(time()-st, 'loading csv file')
        st = time()
        
        self.num_mol = len(f['smi'])
        self.smi = f['smi']	
        self.s_inp = np.array(f['s_energy(ev)'])
        self.t_inp = np.array(f['t_energy(ev)'])

    def __len__(self):
        return self.num_mol

    def __getitem__(self, idx):
        smiles = self.smi[idx]
        singlet_energy =(self.s_inp[idx])
        triplet_energy =(self.t_inp[idx])

        return smiles, \
                torch.as_tensor((singlet_energy)).type(self.precision), \
                torch.as_tensor((triplet_energy)).type(self.precision)

if __name__ == "__main__":
    db = DB("../data/excitated_energy/test_db.csv")
    print(db.__len__())
    print(db[0])
