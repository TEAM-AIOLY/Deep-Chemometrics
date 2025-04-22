import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
import pandas as pd 

import torch
from torch import nn, optim
import torch.utils.data as data_utils
from data.load_dataset_atonce import MangoDataset

from net.base_net import ViT_1D
from utils.misc import data_augmentation
from net.chemtools.metrics import ccc, r2_score,RMSEP
from utils.training import train
from utils.testing import test


default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
root = os.getcwd()

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


if __name__ == "__main__":
    config_root= "data/dataset/Mango/config/_ViT1D_Mango_jz(dm_mango).json"
    config_path =os.path.join(root,config_root)
    try:
        params_dict = parse_args(config_path)
        print(f"{len(params_dict)} parameters sets")
    except Exception as e:
        print(f"Error during execution: {e}")
          
    for idx,params in enumerate(params_dict): 
           
        augmentation = data_augmentation(slope=params['slope'], offset=params['offset'], noise=params['noise'], shift=params['shift'])
        data_path=os.path.join(root,params['data_path'])
        data = sp.io.loadmat(data_path)
       
        Ycal = data["DM_cal"]
        Ytest = data["DM_test"]
        Xcal = data["Sp_cal"]
        Xtest = data["Sp_test"]
        
        x_cal, x_val, y_cal, y_val = train_test_split(Xcal, Ycal, test_size=0.20, shuffle=True, random_state=42) 
        x_scale=data['wave'].astype(np.float32).reshape(-1,1)

        
       
        
        mean = np.mean(x_cal, axis=0)
        std = np.std(x_cal, axis=0)
        
        mean = torch.tensor(mean)
        std = torch.tensor(std)

        params['mean'] = mean
        params['std'] = std
        
        
        cal = MangoDataset(x_cal,y_cal, transform=data_augmentation(slope=params['slope'], offset=params['offset'], noise=params['noise'], shift=params['shift']))
        cal_loader = data_utils.DataLoader(cal, batch_size=1024, shuffle=True)


        val = data_utils.TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
        val_loader = data_utils.DataLoader(val, batch_size=1024, shuffle=True)

        test_dt = data_utils.TensorDataset(torch.Tensor(Xtest), torch.Tensor(Ytest))
        test_loader = data_utils.DataLoader(test_dt, batch_size=1024, shuffle=True)
        
        spec_dims = x_cal.shape[1]
        params['spec_dims']=spec_dims
        
        model=ViT_1D(mean = params['mean'], std = params['std'], seq_len = params['spec_dims'], patch_size = params['PS'], 
                    dim_embed = params['DE'], trans_layers = params['TL'], heads = params['HDS'], mlp_dim = params['MLP'], out_dims = len(params['y_labels']) )
        
        nb_train_params =sum(p.numel() for p in model.parameters())
        
        optimizer = optim.Adam(model.parameters(), lr=params['LR'], weight_decay=params['WD'])
        criterion = nn.MSELoss(reduction='none')
        
        base_path =os.path.dirname(data_path)+f"/model_benchmark/Mango_dm/{params['model_name']}/run_{params['ID']}"
        os.makedirs(base_path, exist_ok=True)
        print(base_path)
        
        
        train_losses, val_losses,val_r2_scores,best_model_path,best_epoch = train(model, optimizer, criterion, cal_loader, val_loader, 
                                        num_epochs=params['num_epochs'],save_path=base_path)

        tl = torch.stack(train_losses).numpy()
        vl = torch.stack(val_losses).numpy()
        r2= np.array(val_r2_scores)
        
        epochs = np.arange(1, len(tl) + 1)
        df_losses = pd.DataFrame({
            'epoch': epochs
        })
        loss_csv_path = os.path.join(base_path, 'epoch_losses.csv')
        df_losses.to_csv(loss_csv_path, index=False)        
        
        

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
            
        y_pred,Y=test(model,best_model_path,val_loader)
        
        CCC =ccc(y_pred,Y)
        r2=r2_score(y_pred, Y)
        rmsep=RMSEP(y_pred, Y)
                  
        metrics_dict = {
        'dataset_type': params['dataset_type'],
        'data_path': data_path, 
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
        "N parameters" : nb_train_params,
        "model_name": params["model_name"],
        "best epoch": best_epoch,
        "Run_ID": f"run_{params['ID']}"
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
        
        
        pdf_path = base_path +f"/predicted_vs_observed_{params['y_labels']}.pdf"
        plt.savefig(pdf_path, format='pdf')
        plt.close('all')
        
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
                    fancybox=True, shadow=True, ncol=5, fontsize=12) 
        plt.tight_layout()
        plt.grid()
        hexbin_pdf_path = base_path+ f"/fig_hexbin.pdf"
        plt.savefig(hexbin_pdf_path, format='pdf')
        plt.close('all')    
        
        x_cal, x_val, y_cal, y_val    
        del augmentation
        del cal
        del Xcal, Ycal, Xtest, Ytest,x_val, y_val,x_cal,y_cal
        del cal_loader, val_loader, test_loader
        del mean, std
        del model, optimizer, criterion
        del train_losses, val_losses, val_r2_scores, best_model_path

        # Optional: Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


