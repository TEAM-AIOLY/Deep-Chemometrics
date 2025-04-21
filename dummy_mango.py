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

root = os.getcwd()

data_path= root+"/data/dataset/Mango/mango_dm_full_outlier_removed2.mat" 
data = sp.io.loadmat(data_path)

params = {
    "dataset_type": "VisNIR",
    "data_path": data_path,  
    "y_labels": ['dm_mango'],
    "batch_size": 512,
    "num_epochs": 2,
    "model_name": "_ViT1D_Mango",
    "seed": 42,
    "spec_dims": None,
    "mean": None,
    "std": None,
    "slope": 0.,
    "offset": 0.,
    "noise": 0.00005,
    "shift": 0.,
    
     "LR": 0.01, "WD": 0.0015, "PS": 10, "DE": 64, "TL": 8, "HDS": 8, "MLP": 64,
     "DP" : 0.5,
     "ID" : 'dummy'
}


Ycal = data["DM_cal"]
Ytest = data["DM_test"]
Xcal = data["Sp_cal"]
Xtest = data["Sp_test"]
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

mean = np.mean(x_cal, axis=0)
std = np.std(x_cal, axis=0)

mean = torch.tensor(mean)
std = torch.tensor(std)

params['mean'] =mean
params['std'] =std

cal = MangoDataset(x_cal,y_cal, transform=data_augmentation(slope=params['slope'], offset=params['offset'], noise=params['noise'], shift=params['shift']))
cal_loader = data_utils.DataLoader(cal, batch_size=params['batch_size'], shuffle=True)


val = data_utils.TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
val_loader = data_utils.DataLoader(val, batch_size=params['batch_size'], shuffle=True)

test_dt = data_utils.TensorDataset(torch.Tensor(Xtest), torch.Tensor(Ytest))
test_loader = data_utils.DataLoader(test_dt, batch_size=params['batch_size'], shuffle=True)

spec_dims = x_cal.shape[1]
params['spec_dims']=spec_dims

model=ViT_1D(mean = params['mean'], std = params['std'], seq_len = params['spec_dims'], patch_size = params['PS'], 
                    dim_embed = params['DE'], trans_layers = params['TL'], heads = params['HDS'], mlp_dim = params['MLP'], out_dims = 1,dropout= params['DP'])
optimizer = optim.Adam(model.parameters(), lr=params['LR'], weight_decay=params['WD'])

criterion = nn.MSELoss(reduction='none')

base_path =os.path.dirname(params['data_path'])+f"/model_benchmark/Mango_dm/{params['model_name']}/run_{params['ID']}"
os.makedirs(base_path, exist_ok=True)


train_losses, val_losses,val_r2_scores,best_model_path,best_epoch = train(model, optimizer, criterion, cal_loader, val_loader, 
                                num_epochs=params['num_epochs'],save_path=base_path)


tl = torch.stack(train_losses).numpy()
vl = torch.stack(val_losses).numpy()
r2= np.array(val_r2_scores)

# Plotting Training and Validation Losses
for i,y in enumerate(params['y_labels']):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(tl[:,i], label=f'Training Loss',color='tab:blue')
    ax1.plot(vl[:,i], label=f'Validation Loss',color='tab:orange')
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.9), fancybox=True, shadow=True, fontsize=12)

    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylabel('R2 Score', color='tab:green')
    ax2.plot(r2[:,i], label=f'R2 Score', linestyle='--',color='tab:green')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.75), fancybox=True, shadow=True, fontsize=12)

    ax1.grid(True)
    plt.title(f'Training performances for:  {y}')
    # plt.show(block=False)
    
    pdf_path =base_path+ f"/RMSE_{params['dataset_type']}.pdf"
    plt.savefig(pdf_path, format='pdf')
    

    y_pred,Y=test(model,best_model_path,val_loader)
    CCC =ccc(y_pred,Y)
    r2=r2_score(y_pred, Y)
    rmsep=RMSEP(y_pred, Y)
                
    metrics_dict = {
    'dataset_type': params['dataset_type'],
    'data_path': params['data_path'], 
    'mean': params['mean'],
    'std': params['std'],
    'y_labels': params['y_labels'],
    'num_epochs': params['num_epochs'],
    "batch_size": params['batch_size'],
    "seed": params['seed'] ,
    "LR": params['LR'] ,
    "WD": params['WD'] ,
    "slope":params['slope'],
    "offset":params['offset'],
    "noise": params['noise'],
    "shift":params['shift'],
    "CCC": CCC,
    "r2": r2,
    "rmsep":rmsep,
    "N parameters" : sum(p.numel() for p in model.parameters()),
    "model_name": params["model_name"],
    "best epoch": best_epoch
    
    }
    
        
    with open((base_path+'/metrics.text'), 'w') as f:
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")


    fig, ax = plt.subplots()
    
    # Scatter plot of X vs Y
    plt.scatter(Y,y_pred,edgecolors='k',alpha=0.5)
    
    # Plot of the 45 degree line
    plt.plot([Y.min()-1,Y.max()+1],[Y.min()-1,Y.max()+1],'r')
    
    plt.text(0.05, 0.95, f'CCC: {CCC:.2f}\nR²: {r2:.2f}', 
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                color='red', fontweight='bold', fontfamily='serif')
    
    ax.set_xlabel('Real Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Predicted vs Real Values for dry matter')
    
    plt.tight_layout()
    plt.grid()
    # fig.show(block=False)
    
    pdf_path = base_path +f"/predicted_vs_observed_{params['y_labels']}.pdf"
    plt.savefig(pdf_path, format='pdf')
    

    fig, ax = plt.subplots()
    hexbin = ax.hexbin(Y.squeeze().numpy(), y_pred.squeeze().numpy(), gridsize=50, cmap='viridis', mincnt=1)
    cb = fig.colorbar(hexbin, ax=ax, orientation='vertical')
    cb.set_label('Density')
    lims = [np.min([Y.numpy(), y_pred.numpy()]), np.max([Y.numpy(), y_pred.numpy()])]
    ax.plot(lims, lims, 'k-', label= params['y_labels'])  
    
    plt.text(0.05, 0.95, f'CCC: {CCC:.2f}\nR²: {r2:.2f}', 
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                color='red', fontweight='bold', fontfamily='serif')
    
    ax.set_xlabel('Real Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Predicted vs Real Values for dry matter')
    
        
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=True, ncol=5, fontsize=12) #labelcolor =default_colors[target_index]
    plt.tight_layout()
    plt.grid()
    # fig.show()
    
    hexbin_pdf_path = base_path+ f"/fig_hexbin.pdf"
    plt.savefig(hexbin_pdf_path, format='pdf')


    