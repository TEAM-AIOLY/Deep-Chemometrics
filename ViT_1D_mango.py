import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
import torch.utils.data as data_utils
from data.load_dataset_atonce import MangoDataset

from net.base_net import ViT_1D
from utils.misc import data_augmentation
from net.chemtools.metrics import ccc, r2_score,RMSEP
from utils.training import train
from utils.testing import test


            
            
data_path= "./data/dataset/Mango/mango_dm_full_outlier_removed2.mat",  
data = sp.io.loadmat(data_path)
print(data.keys())

params = {
    "dataset_type": "VisNIR",
    "data_path": data_path,  
    "y_labels": ['dm_mango'],
    "batch_size": 1024,
    "num_epochs": 2000,
    "model_name": "_ViT1D_Mango",
    "seed": 42,
    "spec_dims": None,
    "mean": None,
    "std": None,
    
    "slope": 0.1,
    "offset": 0.1,
    "noise": 0.0005,
    "shift": 0.1,
    
     "LR": 0.01, "WD": 0.0015, "PS": 10, "DE": 64, "TL": 8, "HDS": 8, "MLP": 64,
     "DP" : 0.5,
     "ID" : 'optim'
}


Ycal = data["DM_cal"]
Ytest = data["DM_test"]
Xcal = data["SP_all_train"]
Xtest = data["SP_all_test"]
print("X and Y training set")
print(Ycal.shape)
print(Xcal.shape)

print("X and Y testing set")
print(Ytest.shape)
print(Xtest.shape)

## Spliting the train set 
x_cal, x_val, y_cal, y_val = train_test_split(Xcal, Ycal, test_size=0.20, shuffle=True, random_state=42) 

## The wavelenghts for the XX axis when we plot the spectra
x_scale=data['wave'].astype(np.float32).reshape(-1,1)

## Check for dimensions
print('Data set dimensions ----------------------------')
print('Full Train set dims X Y = {}\t{}'.format(Xcal.shape, Ycal.shape))
print('Calibration set dims X Y = {}\t{}'.format(x_cal.shape, y_cal.shape))
print('val set dims X Y = {}\t{}'.format(x_val.shape, y_val.shape))
print('Test set dims X Y = {}\t{}'.format(Xtest.shape, Ytest.shape))
print('wavelengths number = {}'.format(np.shape(x_scale)))


mean = np.mean(x_cal, axis=0)
std = np.std(x_cal, axis=0)

mean = torch.tensor(mean)
std = torch.tensor(std)


cal = MangoDataset(x_cal,y_cal, transform=data_augmentation(slope=params['slope'], offset=params['offset'], noise=params['noise'], shift=params['shift']))
cal_loader = data_utils.DataLoader(cal, batch_size=1024, shuffle=True)


val = data_utils.TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
val_loader = data_utils.DataLoader(val, batch_size=1024, shuffle=True)

test_dt = data_utils.TensorDataset(torch.Tensor(Xtest), torch.Tensor(Ytest))
test_loader = data_utils.DataLoader(test_dt, batch_size=1024, shuffle=True)

spec_dims = x_cal.shape[1]
params['spec_dims']=spec_dims

model=ViT_1D(mean = params['mean'], std = params['std'], seq_len = params['spec_dims'], patch_size = params['PS'], 
                    dim_embed = params['DE'], trans_layers = params['TL'], heads = params['HDS'], mlp_dim = params['MLP'], out_dims = 1,dropout= params['DP'])
optimizer = optim.Adam(model.parameters(), lr=params['LR'], weight_decay=params['WD'])


print(sum(p.numel() for p in model.parameters()))

criterion = nn.MSELoss(reduction='none')

base_path =os.path.dirname(params['data_path'])+f"/model_benchmark/Mango_dm/{params['model_name']}/run_{params['ID']}"
os.makedirs(base_path, exist_ok=True)
print(base_path)
    

train_losses, val_losses,val_r2_scores,best_model_path,best_epoch = train(model, optimizer, criterion, cal_loader, val_loader, 
                                num_epochs=params['num_epochs'],save_path=base_path)