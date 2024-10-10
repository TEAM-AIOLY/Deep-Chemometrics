import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from net.base_net import ResNet18_1D
from data.load_dataset import SoilSpectralDataSet
from utils.training import train
from utils.testing import test
from utils.misc import data_augmentation
from net.chemtools.metrics import ccc, r2_score,RMSEP
import json

import matplotlib.pyplot as plt

import pickle

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['font.family'] = 'Times New Roman'



if __name__ == "__main__":
    seed=42

    data_path ="./data/dataset/ossl/ossl_all_L1_v1.2.csv"
     
    param_file = os.path.dirname(data_path)+ '/optuna_params/optuna_ResNet18_params_mir.json'
    
    with open(param_file,'r') as f:  
        params_dict = json.load(f)
        
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    LR = 0.0035
    WD = 0.0055
    
    slope= 0.2
    offset= 0.2
    noise=0.002
    shift=0.05

    DP=0.5
    IP=8
    
    for i,param_set in enumerate(params_dict):
      
        params = param_set
        if len(params['y_labels'])>1:
            continue
        

        # Apply augmentation to the dataset using the suggested parameters
        augmentation = data_augmentation(slope=slope, offset=offset, noise=noise, shift=shift)
        spectral_data = SoilSpectralDataSet(data_path=params['data_path'], dataset_type=params['dataset_type'], y_labels=params['y_labels'],preprocessing=None)
        
        dataset_size = len(spectral_data)
        test_size = int(0.2 * dataset_size)
        train_size = dataset_size - test_size
        cal_size = int(0.75 * train_size)
        val_size = train_size - cal_size      
    

        train_dataset, test_dataset = random_split(spectral_data, [train_size, test_size], 
                                                    generator=torch.Generator().manual_seed(params['seed']))
        cal_dataset, val_dataset = random_split(train_dataset, [cal_size, val_size], 
                                                generator=torch.Generator().manual_seed(params['seed']))
        
        cal_dataset.dataset.preprocessing=augmentation

        # Create data loaders
        cal_loader = DataLoader(cal_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)
        test_loader= DataLoader(test_dataset,batch_size=params['batch_size'], shuffle=False, num_workers=0)
        
        params['spec_dims'] = spectral_data.spec_dims
               
        labs = '_'.join(params['y_labels'])    
        
        save_path = os.path.dirname(data_path) + f"/models/{params['dataset_type']}/{params['model_name']}/{labs}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        base_path = os.path.dirname(data_path) + f"/figures/{params['dataset_type']}/{params['model_name']}/{labs}"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
            
        mean = torch.zeros(params['spec_dims'])
        std = torch.zeros(params['spec_dims'])
        
        for inputs, _ in cal_loader:
            mean += inputs.sum(dim=0)

        mean /= len(cal_loader.dataset)

        # Calculate std over the training dataset
        for inputs, _ in cal_loader:
            std += ((inputs - mean) ** 2).sum(dim=0)

        std = torch.sqrt(std / len(cal_loader.dataset))
        params['mean'] = mean
        params['std'] = std


        # Model, optimizer, and training
        model = ResNet18_1D(params['spec_dims'], mean=params['mean'], std=params['std'], out_dims=len(params['y_labels']))
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD,dropout=DP, inplanes=IP)
        criterion = nn.MSELoss(reduction='none')

        train_losses, val_losses, val_r2_scores , final_save_path = train(model, optimizer, criterion, cal_loader, val_loader, 
                                        num_epochs=500, early_stop=False, plot_fig=False,save_path=save_path)


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
            
            pdf_path =base_path+ f"/RMSE_{params['dataset_type']}_{labs}.pdf"
            plt.savefig(pdf_path, format='pdf')

            pickle_path = base_path + f"/RMSE_{['dataset_type']}_{labs}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(fig, f)
        


            y_pred,Y=test(model,final_save_path,val_loader)
            
            CCC =ccc(y_pred,Y)
            r2=r2_score(y_pred, Y)
            rmsep=RMSEP(y_pred, Y)
            
            params_path = os.path.dirname(params['data_path'])+f"/models/metrics_test/{params['dataset_type']}/{params['model_name']}/{labs}"
            if not os.path.exists(params_path):
                    os.makedirs(params_path)
                    
            metrics_dict = {
            'dataset_type': params['dataset_type'],
            'data_path': params['data_path'], 
            'mean': params['mean'],
            'std': params['std'],
            'y_labels': params['y_labels'],
            'num_epochs': params['num_epochs'],
            "batch_size": params['batch_size'],
            "seed": params['seed'] ,
            "LR": LR, 
            "WD":WD,
            "slope":slope,
            "offset":offset,
            "noise": noise,
            "shift":shift,
            "CCC": CCC,
            "r2": r2,
            "rmsep":rmsep
            }
            
                
            with open((params_path+'/metrics.text'), 'w') as f:
                for key, value in metrics_dict.items():
                    f.write(f"{key}: {value}\n")

        
        print("CCC: %5.5f, R2: %5.5f, RMSEP: %5.5f"%(CCC, r2, rmsep))
    
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
        ax.set_title(f'Predicted vs Real Values for {labs} (log x + 1)')
        
        plt.tight_layout()
        plt.grid()
        # fig.show(block=False)
        
        pdf_path = base_path +f"/predicted_vs_observed_{params['dataset_type']}_{labs}.pdf"
        plt.savefig(pdf_path, format='pdf')

        pickle_path = base_path + f"/predicted_vs_observed_{params['dataset_type']}_{labs}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(fig, f)
        

        
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
        ax.set_title(f'Predicted vs Real Values for {labs} (log x + 1)')
        
        
        all_labels = ["oc_usda.c729_w.pct","na.ext_usda.a726_cmolc.kg","clay.tot_usda.a334_w.pct", "k.ext_usda.a725_cmolc.kg", "ph.h2o_usda.a268_index" ]  #
        if len(params['y_labels'])==1:
            target_index = all_labels.index(params['y_labels'][0])
        else:
            target_index=len(params['y_labels'])
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=5, fontsize=12,labelcolor =default_colors[target_index])
        plt.tight_layout()
        plt.grid()
        # fig.show()
        
        hexbin_pdf_path = base_path+ f"/fig_hexbin_{labs}.pdf"
        plt.savefig(hexbin_pdf_path, format='pdf')

        # Save the figure object using pickle
        hexbin_pickle_path = base_path+ f"/fig_hexbin_{labs}.pkl"
        with open(hexbin_pickle_path, 'wb') as f:
            pickle.dump(fig, f)