import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import json
import torch
from torch import nn, optim
import torch.utils.data as data_utils

from src import utils
from src.net import ViT_1D
from src.training.training import Trainer
from src.utils.misc import TrainerConfig
from src.utils.Loaddatasets import DatasetLoader

import time

configs = [
    "data/dataset/ossl/config/_ViT1D_ossl_jz(oc_usda).json",
    "data/dataset/mango/config/_ViT1D_Mango_jz(dm_mango).json",
    "data/dataset/wheat/config/_ViT1D_Wheat_jz(Wheat_dt).json"
]

config_root = configs[1]
root = os.getcwd()
config_path = os.path.join(root, config_root)



try:
    params_dicts = DatasetLoader.parse_args(config_path)
    print(f"{len(params_dicts)} parameter sets")
except Exception as e:
    print(f"Error during execution: {e}")
    sys.exit(1)

start = time.time()
data = DatasetLoader.get_data(config_path)
end=time.time()
print(f"Data loaded in {end - start:.2f} seconds")
mean = np.mean(data["x_cal"], axis=0)
std = np.std(data["x_cal"], axis=0)

#########################################################################################################
# Main loop over configs
#########################################################################################################
for idx, params in enumerate(params_dicts):
    print(f"\n=== Running config {idx+1}/{len(params_dicts)}: {params.get('model_name', 'unnamed')} ===")

    batch_size = params.get('batch_size', 128)

    # Example: create DataLoaders from the already loaded data
    cal_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(data["x_cal"], dtype=torch.float32),
            torch.tensor(data["y_cal"], dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=True
    )
    val_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(data["x_val"], dtype=torch.float32),
            torch.tensor(data["y_val"], dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=False
    )
    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(data["x_test"], dtype=torch.float32),
            torch.tensor(data["y_test"], dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=False
    )

    # Set up config for training
    config = TrainerConfig(model_name=params['model_name'])
    config.update_config(
        batch_size=params['batch_size'],
        learning_rate=params['LR'],
        num_epochs=20,#params['num_epochs']
        classification=False
    )

    params['spec_dims'] = spec_dims = data["x_cal"].shape[1]
    params['mean'] = mean
    params['std'] = std

    # Model
    model = ViT_1D(
        mean=params['mean'],
        std=params['std'],
        seq_len=params['spec_dims'],
        patch_size=params['PS'],
        dim_embed=params['DE'],
        trans_layers=params['TL'],
        heads=params['HDS'],
        mlp_dim=params['MLP'],
        out_dims=len(params['y_labels'])
    )

    nb_train_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=params.get('WD', 0.003/2))
    criterion = nn.MSELoss(reduction='none')

    # Training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=cal_loader,
        val_loader=val_loader,
        config=config
    )

    local_run=f"Benchmark/{params['dataset_type']}/{params['model_name']}/run_{params['ID']}"
    base_path =os.path.join(root,local_run)
    os.makedirs(base_path, exist_ok=True)
    config.update_config(save_path=base_path)

    train_losses, val_losses,val_r2_scores, final_path,best_epoch = trainer.train()
    
    perf,y_pred =utils.test(model, final_path, test_loader,config)
    
    
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
    plt.scatter(data["y_test"],y_pred,edgecolors='k',alpha=0.5)
    
    # Plot of the 45 degree line
    plt.plot([data["y_test"].min()-1,y_pred.max()+1],[data["y_test"].min()-1,y_pred.max()+1],'r')
    
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
    hexbin = ax.hexbin(data["y_test"], y_pred, gridsize=50, cmap='viridis', mincnt=1)
    cb = fig.colorbar(hexbin, ax=ax, orientation='vertical')
    cb.set_label('Density')
    lims = [np.min([data["y_test"], y_pred]), np.max([data["y_test"], y_pred])]
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
    'y_labels': params['y_labels'],
    'num_epochs': params['num_epochs'],
    "batch_size": params['batch_size'],
    "seed": params['seed'] ,
    "LR": params['LR'] ,
    "WD": params['WD'] ,
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