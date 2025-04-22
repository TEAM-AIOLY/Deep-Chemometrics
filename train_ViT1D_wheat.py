import os
import numpy as np
import scipy as sp
import pandas as pd
import json
import matplotlib.pyplot as plt

import torch 
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.utils.data as data_utils

from net.chemtools.PLS import PLS
from net.base_net import ViT_1D
from utils.training import train
from utils.testing import test
from utils.misc import data_augmentation




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
    config_root= "data/dataset/Wheat_dt/config/_ViT1D_Wheat_jz(Wheat_dt).json"
    config_path =os.path.join(root,config_root)
    try:
        params_dict = parse_args(config_path)
        print(f"{len(params_dict)} parameters sets")
    except Exception as e:
        print(f"Error during execution: {e}")
          
    for idx,params in enumerate(params_dict): 
        root_path=os.path.join(root,params['data_path'])
        
        # Initialize dictionaries to store DataFrames for each keyword
        keywords = ["cal", "test", "val"]

        # Read data and extract X and y for each dataset
        data = {
            key: (
                np.array(pd.read_csv(f"{root_path}/Wheat_{key}.csv").iloc[:, 0:-1]),  # X
                np.array(pd.read_csv(f"{root_path}/Wheat_{key}.csv").iloc[:, -1] )  # y
            )
            for key in keywords
        }

        # Access X and y for train, test, and val
        X_cal, y_cal= data["cal"]
        X_test, y_test = data["test"]
        X_val, y_val = data["val"]

        X_cal =X_cal
        X_val =X_val
        X_test =X_test
        

        mean = np.mean(X_cal, axis=0)
        std = np.std(X_cal, axis=0)
        
        mean = torch.tensor(mean)
        std = torch.tensor(std)

        params['mean'] = mean
        params['std'] = std
        

        # # Convert np.array to Dataloader 

        cal = data_utils.TensorDataset(torch.Tensor(X_cal), torch.Tensor(y_cal))
        cal_loader = data_utils.DataLoader(cal, batch_size=params["batch_size"], shuffle=True)

        val = data_utils.TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        val_loader = data_utils.DataLoader(val, batch_size=params["batch_size"], shuffle=True)

        test_dt = data_utils.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        test_loader = data_utils.DataLoader(test_dt, batch_size=params["batch_size"], shuffle=True)
        
        spec_dims = X_cal.shape[1]
        params['spec_dims']=spec_dims
        
        model=ViT_1D(mean = params['mean'], std = params['std'], seq_len = params['spec_dims'], patch_size = params['PS'], 
                    dim_embed = params['DE'], trans_layers = params['TL'], heads = params['HDS'], mlp_dim = params['MLP'], out_dims = params['nb_classes'] )
        
        nb_train_params =sum(p.numel() for p in model.parameters())
        
        optimizer = optim.Adam(model.parameters(), lr=params['LR'], weight_decay=params['WD'])
        criterion = nn.MSELoss(reduction='none')
        
        base_path =os.path.dirname(root_path)+f"/model_benchmark/Wheat_dt/{params['model_name']}/run_{params['ID']}"
        os.makedirs(base_path, exist_ok=True)
        print(base_path)
        
        
        train_losses, val_losses,val_r2_scores,best_model_path,best_epoch = train(model, optimizer, criterion, cal_loader, val_loader, 
                                        num_epochs=params['num_epochs'],save_path=base_path)

        # tl = torch.stack(train_losses).numpy()
        # vl = torch.stack(val_losses).numpy()
        # f1= np.array(val_r2_scores)
        
        # print(tl.shape())
        
        # y_pred,Y=test(model,best_model_path,val_loader)
       
                  
        # metrics_dict = {
        # 'dataset_type': params['dataset_type'],
        # 'data_path': root_path, 
        # 'mean': params['mean'],
        # 'std': params['std'],
        # 'nb_classes': params['nb_classes'],
        # 'num_epochs': params['num_epochs'],
        # "batch_size": params['batch_size'],
        # "seed": params['seed'] ,
        # "LR": params['LR'] ,
        # "WD": params['WD'] ,
        # "f1": f1,
        # "N parameters" : nb_train_params,
        # "model_name": params["model_name"],
        # "best epoch": best_epoch,
        # "Run_ID": f"run_{params['ID']}"
        # }
        
            
        # with open((base_path+'/metrics.text'), 'w') as f:
        #     for key, value in metrics_dict.items():
        #         f.write(f"{key}: {value}\n")



        # fig, ax = plt.subplots()
        
        # # Scatter plot of X vs Y
        # plt.scatter(Y,y_pred,edgecolors='k',alpha=0.5)
        
        # # Plot of the 45 degree line
        # plt.plot([Y.min()-1,Y.max()+1],[Y.min()-1,Y.max()+1],'r')
        
        # plt.text(0.05, 0.95, f'CCC: {CCC:.2f}\nR²: {r2:.2f}', 
        #             transform=plt.gca().transAxes, fontsize=12,
        #             verticalalignment='top', horizontalalignment='left',
        #             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
        #             color='red', fontweight='bold', fontfamily='serif')
        
        # ax.set_xlabel('Real Values')
        # ax.set_ylabel('Predicted Values')
        # ax.set_title(f'Predicted vs Real Values for dry matter')
        
        # plt.tight_layout()
        # plt.grid()
        
        
        # pdf_path = base_path +f"/predicted_vs_observed_{params['y_labels']}.pdf"
        # plt.savefig(pdf_path, format='pdf')
        # plt.close('all')
        
        # fig, ax = plt.subplots()
        # hexbin = ax.hexbin(Y.squeeze().numpy(), y_pred.squeeze().numpy(), gridsize=50, cmap='viridis', mincnt=1)
        # cb = fig.colorbar(hexbin, ax=ax, orientation='vertical')
        # cb.set_label('Density')
        # lims = [np.min([Y.numpy(), y_pred.numpy()]), np.max([Y.numpy(), y_pred.numpy()])]
        # ax.plot(lims, lims, 'k-', label= params['y_labels'])  
        
        # plt.text(0.05, 0.95, f'CCC: {CCC:.2f}\nR²: {r2:.2f}', 
        #             transform=plt.gca().transAxes, fontsize=12,
        #             verticalalignment='top', horizontalalignment='left',
        #             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
        #             color='red', fontweight='bold', fontfamily='serif')
        
        # ax.set_xlabel('Real Values')
        # ax.set_ylabel('Predicted Values')
        # ax.set_title(f'Predicted vs Real Values for dry matter')
        
            
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        #             fancybox=True, shadow=True, ncol=5, fontsize=12) 
        # plt.tight_layout()
        # plt.grid()
        # hexbin_pdf_path = base_path+ f"/fig_hexbin.pdf"
        # plt.savefig(hexbin_pdf_path, format='pdf')
        # plt.close('all')    
        
        

        # del cal
        # del X_cal, y_cal, X_test, y_test,X_val, y_val,X_cal,y_cal
        # del cal_loader, val_loader, test_loader
        # del mean, std
        # del model, optimizer, criterion
        # del train_losses, val_losses, val_r2_scores, best_model_path

        # # Optional: Clear PyTorch cache
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

