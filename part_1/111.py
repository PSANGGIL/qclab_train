import torch
import pandas as pd
# import numpy as np
# from tqdm import tqdm, trange
# from glob import glob
# from time import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DB(Dataset):
    def __init__(self, filename):
        self.data = pd.read_csv(filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        row = self.data.iloc[idx] 
        smiles = row['smi']
        singlet_energy = torch.tensor(row['s_energy(ev)'], dtype=torch.double)
        triplet_energy = torch.tensor(row['t_energy(ev)'], dtype=torch.double)
        return smiles, singlet_energy, triplet_energy

if __name__ == "__main__":
    
    dataset = DB("../data/excitated_energy/test_db.csv")
    print(f"the length of the database: {len(dataset)}")
    
    #print(dataset[0])
    #exit(-1)
    #for smiles, singlet_energy, triplet_energy in dataset:
    #    
    #    print(f'SMILES: {smiles}')
    #    print(f'Singlet Energy: {singlet_energy}')
    #    print(f'Triplet Energy: {triplet_energy}')
    #    break


    data_loader = DataLoader(dataset, batch_size=2)

    print(f"the length of the database: {len(dataset)}")

    for smiles, singlet_energy, triplet_energy in data_loader:
        print(f'SMILES: {smiles}')
        print(f'Singlet Energy: {singlet_energy}')
        print(f'Triplet Energy: {triplet_energy}')
        break
