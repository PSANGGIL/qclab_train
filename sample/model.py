import torch
import os
import torch
from torch import nn
from torch.nn.functional import relu, elu
import numpy as np
from torch.utils.data import DataLoader
import math
import time
from db3 import DB
from tqdm import tqdm
from torch import optim


class MyModel(torch.nn.Module):
    def __init__(self, D_in, D_hid1, D_hid2, D_out):
        super(MyModel, self).__init__()
        self.linear1  = torch.nn.Linear(D_in, D_hid1)
        self.linear2  = torch.nn.Linear(D_hid1, D_hid2)
        self.linear3  = torch.nn.Linear(D_hid2, D_out)
        
        self.dropout  = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.relu = nn.ReLU() 
        self.elu  = nn.ELU() 
        self.sp   = nn.Softplus()
        
        self.layer = torch.nn.Sequential(   torch.nn.Linear(D_in, D_hid1),
                                            torch.nn.Softplus(),
                                            torch.nn.Dropout(0.2),
                                            torch.nn.Linear(D_hid1, D_hid2),
                                            torch.nn.Softplus(),
                                            torch.nn.Dropout(0.3),
                                            torch.nn.Linear(D_hid2, D_out)
                                        )

    def forward(self, x):
        y_pred = self.layer(x)
        return y_pred


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torh.decice('cpu')
emb_dim  = 128
hid_dim  = 64
hid_dim2 = 16

batch_size = 1000

train_dataset = DB('../data/train_db.csv', emb_dim = emb_dim)
valid_dataset = DB('../data/valid_db.csv', emb_dim = emb_dim)    
test_dataset  = DB('../data/test_db.csv' , emb_dim = emb_dim)    

model = MyModel(emb_dim, hid_dim, hid_dim2, 2)
model = model.to(device)
model.parameters()

loss_fn = torch.nn.MSELoss()
loss_fn = loss_fn.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

PATH = './weight/'

train_dataloader = DataLoader(train_dataset, batch_size = batch_size,   shuffle=True, drop_last=True)
train_dataloader = DataLoader(test_dataset, batch_size = batch_size,   shuffle=True, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size,   shuffle=False, drop_last=True)
test_dataloader  = DataLoader(test_dataset , batch_size = batch_size,   shuffle=False, drop_last=True)

train_min = []
valid_min = []
test_min  = []
singlet_loss = []
triplet_loss = []
sp = torch.nn.Softplus()
for epoch in tqdm(range(100)):
    train_loss = 0.0
    valid_loss = 0.0
    test_loss  = 0.0
    s_loss = []
    t_loss = []
    model.train()
    for batch_idx, samples in (enumerate(train_dataloader)):
        optimizer.zero_grad()
        with torch.no_grad():
            x = samples[0]
            s_energy = samples[1]
            t_energy = samples[2]
            x = torch.torch.linalg.norm(x, dim = 1).to(device)#.requires_grad_()
        y = torch.stack([s_energy, t_energy], dim = 1).to(device)#.requires_grad_()
        
        y_pred = sp(model(x))
        loss = torch.sum(torch.abs(y - y_pred)) / (batch_size)
        
        loss.backward()
        optimizer.step()
        
        train_loss+=loss.item()

    if (epoch%1==0):
        print(epoch, 'train_loss:', train_loss / len(train_dataloader))
        train_min.append(train_loss / len(train_dataloader))

    model.eval()
    with torch.no_grad():
        for batch_idx, samples in (enumerate(valid_dataloader)):
            with torch.no_grad():
                x = samples[0]
                s_energy = samples[1]
                t_energy = samples[2]
                x = torch.torch.linalg.norm(x, dim = 1).to(device)#.requires_grad_()
            y = torch.stack([s_energy, t_energy], dim = 1).to(device)#.requires_grad_()
            y_pred = sp(model(x))
            loss = torch.sum(torch.abs(y - y_pred)) / (batch_size)

            valid_loss+=loss.item()
        if (epoch%1==0):
            print(epoch, 'valid_loss:', valid_loss / len(valid_dataloader))
            valid_min.append(valid_loss / len(valid_dataloader))
        
        for batch_idx, samples in (enumerate(test_dataloader)):
            with torch.no_grad():
                x = samples[0]
                s_energy = samples[1]
                t_energy = samples[2]
                x = torch.torch.linalg.norm(x, dim = 1).to(device)#.requires_grad_()
            y = torch.stack([s_energy, t_energy], dim = 1).to(device)#.requires_grad_()
            y_pred = sp(model(x))
            loss = torch.sum(torch.abs(y - y_pred)) / (batch_size)
            
            test_loss+=loss.item()
            s_loss.append(torch.sum((y[:,0] - y_pred[:,0]))/len(test_dataloader))
            t_loss.append(torch.sum((y[:,1] - y_pred[:,1]))/len(test_dataloader))
        if (epoch%1==0):
            print(epoch, 'test_loss :', test_loss / len(test_dataloader))
            test_min.append(test_loss / len(test_dataloader))
    singlet_loss.append(np.sum(s_loss))    
    triplet_loss.append(np.sum(s_loss))
    print(11111111111)
    exit(-1)
    torch.save(model.state_dict(), PATH + str(epoch) + '_' + str((test_loss) / len(test_dataloader))+'.pt') 

print(len(singlet_loss))
print((singlet_loss))
print('s_error:',min(np.abs(singlet_loss)).item() / batch_size)
print('t_error:',min(np.abs(triplet_loss)).item() / batch_size)
print('train_mini:', np.min(train_min))
print('valid_mini:', np.min(valid_min))
print('test_mini:' , np.min(test_min ))

print('s-s_min', singlet_loss[singlet_loss.index(min((singlet_loss)).item())] / batch_size)
print('t-s_min', triplet_loss[singlet_loss.index(min((singlet_loss)).item())] / batch_size)
print('s-t_min', singlet_loss[triplet_loss.index(min((triplet_loss)).item())] / batch_size)
print('t-t_min', triplet_loss[triplet_loss.index(min((triplet_loss)).item())] / batch_size)


