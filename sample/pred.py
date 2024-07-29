# Code in file nn/two_layer_net_module.py
import torch
import os
import torch
from torch import nn
from torch.nn.functional import relu, elu
#import pytorch_lightning as pl
#from   pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import numpy as np
from torch.utils.data import DataLoader
import math
import time
from db3 import DB
from tqdm import tqdm
from torch import optim


class MyModel(torch.nn.Module):
    def __init__(self, D_in, D_hid1, D_hid2, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and 
        assign them as
        member variables.
        """
        super(MyModel, self).__init__()
#        self.linear1 = torch.nn.Linear(D_in, H)
#        self.linear2 = torch.nn.Linear(H, D_out)


        #self.readout = nn.Sequential(
        #                            nn.Linear(D_in, H),
#				    nn.ReLU(),
#				    nn.Linear(H, 2),
#                                    )

        self.linear1  = torch.nn.Linear(D_in, D_hid1)
        self.linear2  = torch.nn.Linear(D_hid1, D_hid2)
        self.linear3  = torch.nn.Linear(D_hid2, D_out)
        
        self.dropout  = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.relu = nn.ReLU() 
        self.elu  = nn.ELU() 
        self.sp   = nn.Softplus()
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        #h_relu = self.linear1(x).clamp(min=0)
        #y_pred = self.linear2(h_relu)
        y_pred = self.linear1(x)
        y_pred = self.sp(y_pred)
        y_pred = self.dropout(y_pred)
        
        y_pred = self.linear2(y_pred)
        y_pred = self.sp(y_pred)
        y_pred = self.dropout2(y_pred)
        
        y_pred = self.linear3(y_pred)
        #h_relu = self.linear1(x).clamp(min=0)
        #y_pred = self.linear2(h_relu)

        return y_pred
device = torch.device('cpu')

emb_dim  = 128
hid_dim  = 64
hid_dim2 = 16

batch_size = 1000

test_dataset  = DB('../data/full_db.csv' , emb_dim = emb_dim)    

PATH = 'model_state_dict.pt'

model = MyModel(emb_dim, hid_dim, hid_dim2, 2)

model.load_state_dict(torch.load(PATH))
model.eval()

model = model.to(device)
model.parameters()

loss_fn = torch.nn.MSELoss()
loss_fn = loss_fn.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

test_dataloader  = DataLoader(test_dataset , batch_size = batch_size,   shuffle=False, drop_last=True)
#train_dataloader = DataLoader(train_db, batch_size = args.batch_size,   shuffle=True,  num_workers=num_workers,  drop_last=True, persistent_workers=False)
#valid_dataloader = DataLoader(train_db, batch_size = args.batch_size*2, shuffle=False, num_workers=num_workers,                  persistent_workers=False)
test_min  = []
singlet_loss = []
triplet_loss = []
sp = torch.nn.Softplus()
test_loss  = 0.0
s_loss = []
t_loss = []
for batch_idx, samples in (enumerate(test_dataloader)):
    with torch.no_grad():
        x = samples[0]
        s_energy = samples[1]
        t_energy = samples[2]
        x = torch.torch.linalg.norm(x, dim = 1).to(device)#.requires_grad_()
    y = torch.stack([s_energy, t_energy], dim = 1).to(device)#.requires_grad_()
    #y_pred = model(x)
    y_pred = sp(model(x))
            
    loss = loss_fn(y, y_pred)

            
    print('sss ',torch.sum(y[:,0]-y_pred[:,0])/(batch_size))
    print('ttt ',torch.sum(y[:,1]-y_pred[:,1])/(batch_size))
    
    print('s_max ',torch.max(abs(y[:,0]-y_pred[:,0])))
    print('t_max ',torch.max(abs(y[:,1]-y_pred[:,1])))
    print('s_min ',torch.min(abs(y[:,0]-y_pred[:,0])))
    print('t_min ',torch.min(abs(y[:,1]-y_pred[:,1])))
    test_loss+=loss.item()
    s_loss.append((y[:,0]-y_pred[:,0]).detach().numpy())
    t_loss.append((y[:,1]-y_pred[:,1]).detach().numpy())
    exit(-1)

np.save('./loss/e_97/s_loss.npy', s_loss)
np.save('./loss/e_97/t_loss.npy', t_loss)
 
print(test_loss / len(test_dataloader))

