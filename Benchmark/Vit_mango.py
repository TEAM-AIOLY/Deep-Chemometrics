
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split
import json

import torch 
from torch import nn
from torch import optim
import torch.utils.data as data_utils

from src import utils
from src.net import  ViT_1D
from src.training.training import Trainer
from src.utils.misc import TrainerConfig
from sklearn.model_selection import train_test_split


#########################################################################################################
# Read configuration file as a list of dict from json file
#########################################################################################################
def parse_args(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        # Load parameters from JSON file
        with open(file_path, 'r') as f:
            params_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON file: {e}")

    return params_dict
#########################################################################################################
#########################################################################################################


#########################################################################################################
#Get list of Config dicts
#########################################################################################################
config_root= "data/dataset/Mango/config/_ViT1D_Mango_jz(dm_mango).json"
root = os.getcwd()
config_path =os.path.join(root,config_root)


try:
        params_dict = parse_args(config_path)
        print(f"{len(params_dict)} parameters sets")
except Exception as e:
        print(f"Error during execution: {e}")

#########################################################################################################
#########################################################################################################


#########################################################################################################
# Get data and split cal-val
#########################################################################################################
data = sp.io.loadmat("./data/dataset/Mango/mango_dm_full_outlier_removed2.mat")


Ycal = data["DM_cal"]
Ytest = data["DM_test"]
Xcal = data["Sp_cal"]
Xtest = data["Sp_test"]


## Spliting the train set 
x_cal, x_val, y_cal, y_val = train_test_split(Xcal, Ycal, test_size=0.20, shuffle=True, random_state=42) 
x_scale=data['wave'].astype(np.float32).reshape(-1,1)


mean = np.mean(x_cal, axis=0)
std = np.std(x_cal, axis=0)

model_name ="Vit-1D_Mango"
spec_dims = x_cal.shape[1]
#########################################################################################################
#########################################################################################################



for idx,params in enumerate(params_dict): 
 if idx==0:
    config = TrainerConfig(model_name = model_name)
    config.update_config(batch_size=params['batch_size'],learning_rate=params['LR'],num_epochs=params['num_epochs'],classification=False) 
 
    cal = data_utils.TensorDataset(torch.Tensor(x_cal), torch.Tensor(y_cal))
    cal_loader = data_utils.DataLoader(cal, batch_size=config.batch_size, shuffle=True)

    val = data_utils.TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
    val_loader = data_utils.DataLoader(val, batch_size=config.batch_size, shuffle=True)

    test_dt = data_utils.TensorDataset(torch.Tensor(Xtest), torch.Tensor(Ytest))
    test_loader = data_utils.DataLoader(test_dt, batch_size=config.batch_size, shuffle=True)
    
    params['spec_dims']=spec_dims
    params['mean'] = mean
    params['std'] = std
    
        
    model=ViT_1D(mean = params['mean'], std = params['std'], seq_len = params['spec_dims'], patch_size = params['PS'], 
                    dim_embed = params['DE'], trans_layers = params['TL'], heads = params['HDS'], mlp_dim = params['MLP'], out_dims = len(params['y_labels']) )
    
    nb_train_params =sum(p.numel() for p in model.parameters())
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.003/2)
    criterion = nn.MSELoss(reduction='none')
    
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, train_loader=cal_loader, val_loader=val_loader, config=config)
    
    local_run=f"benchmark/Mango_dm/{params['model_name']}/run_{params['ID']}"
    base_path =os.path.join(root,local_run)
    os.makedirs(base_path, exist_ok=True)
    config.update_config(save_path=base_path)

    train_losses, val_losses,val_r2_scores, final_path,best_epoch = trainer.train()
    
    perf,y_pred =utils.test(model, final_path, test_loader,config)
    perf['r2'] = [r if 0 <= r <= 1 else 0 for r in perf['r2']]



# #############################
# #plot
# ##############################


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
        pdf_path =base_path+ f"/RMSE_{params['dataset_type']}.pdf"
        plt.savefig(pdf_path, format='pdf')
        plt.close('all')


    fig, ax = plt.subplots()
        
    # Scatter plot of X vs Y
    plt.scatter(Ytest,y_pred,edgecolors='k',alpha=0.5)
    
    # Plot of the 45 degree line
    plt.plot([Ytest.min()-1,y_pred.max()+1],[Ytest.min()-1,y_pred.max()+1],'r')
    
    plt.text(0.05, 0.95, f"CCC: {perf['ccc'][0]:.2f}\nR²: {perf['r2'][0]:.2f} \n RMSEP :{perf['rmsep'][0]:.3f}", 
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                color='red', fontweight='bold', fontfamily='serif')
    
    ax.set_xlabel('Real Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Predicted vs Real Values for dry matter')
    
    plt.tight_layout()
    plt.grid()
    
    
    pdf_path = base_path +f"/predicted_vs_observed_{params['y_labels']}.pdf"
    plt.savefig(pdf_path, format='pdf')
    plt.close('all')
    
    fig, ax = plt.subplots()
    hexbin = ax.hexbin(Ytest, y_pred, gridsize=50, cmap='viridis', mincnt=1)
    cb = fig.colorbar(hexbin, ax=ax, orientation='vertical')
    cb.set_label('Density')
    lims = [np.min([Ytest, y_pred]), np.max([Ytest, y_pred])]
    ax.plot(lims, lims, 'k-', label= params['y_labels'])  
    
    plt.text(0.05, 0.95, f"CCC: {perf['ccc'][0]:.2f}\nR²: {perf['r2'][0]:.2f} \n RMSEP :{perf['rmsep'][0]:.3f}", 
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                color='red', fontweight='bold', fontfamily='serif')
    
    ax.set_xlabel('Real Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Predicted vs Real Values for dry matter')
    
        
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=True, ncol=5, fontsize=12) 
    plt.tight_layout()
    plt.grid()
    hexbin_pdf_path = base_path+ f"/fig_hexbin.pdf"
    plt.savefig(hexbin_pdf_path, format='pdf')
    plt.close('all')    
    
    
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
    "CCC": perf['ccc'],
    "r2": perf['r2'],
    "rmsep":perf['rmsep'],
    "N parameters" : nb_train_params,
    "model_name": params["model_name"],
    "best epoch": best_epoch,
    "Run_ID": f"run_{params['ID']}"
    }
    
        
    with open((base_path+'/metrics.text'), 'w') as f:
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")