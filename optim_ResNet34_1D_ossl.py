
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from net.base_net import ResNet101_1D
from data.load_dataset import SoilSpectralDataSet
from utils.training import train
from utils.misc import data_augmentation

import optuna

import json




def objective(trial, params):
    # Hyperparameter suggestions
    LR = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    WD = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    # Data augmentation parameters to be optimized
    # slope = trial.suggest_float("slope", 0.0, 0.3)
    # offset = trial.suggest_float("offset", 0.0, 0.3)
    # noise = trial.suggest_float("noise", 0.0, 0.3)
    # shift = trial.suggest_float("shift", 0.0, 0.3)
    
    # DP =trial.suggest_float("dropout", 0.0, 0.75)
    # IP =trial.suggest_int("inplanes", 4, 32)
    
    slope = 0.1, offset = 0.1, noise = 0.1, shift = 0.1
    DP=0.5
    IP=8

    # Apply augmentation to the dataset using the suggested parameters
    augmentation = data_augmentation(slope=slope, offset=offset, noise=noise, shift=shift)
    spectral_data = SoilSpectralDataSet(data_path=params['data_path'], dataset_type=params['dataset_type'], 
                                         y_labels=params['y_labels'], preprocessing=augmentation)

    # Dataset split (same logic as before)
    dataset_size = len(spectral_data)
    test_size = int(0.2 * dataset_size)
    train_val_size = dataset_size - test_size
    train_size = int(0.75 * train_val_size)
    val_size = train_val_size - train_size

    train_val_dataset, _ = random_split(spectral_data, [train_val_size, test_size], 
                                                   generator=torch.Generator().manual_seed(params['seed']))
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(params['seed']))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)


    # Model, optimizer, and training
    model = ResNet101_1D(mean=params['mean'], std=params['std'], out_dims=len(params['y_labels']),dropout=DP, inplanes=IP)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss(reduction='none')

    _, _, val_r2_scores = train(model, optimizer, criterion, train_loader, val_loader, 
                                     num_epochs=params['num_epochs'], early_stop=False, plot_fig=False,save_path=None)

   
   

    return val_r2_scores



if __name__ == "__main__":
    
    data_path ="./data/dataset/ossl/ossl_all_L1_v1.2.csv"
     
    param_file = os.path.dirname(data_path)+ '/optuna_params/optuna_ResNet101_params_mir.json'
    
    with open(param_file,'r') as f:  
        params_dict = json.load(f)
    
    for param_set in params_dict:
        params = param_set

        torch.manual_seed(params['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(params['seed'])
            torch.cuda.manual_seed_all(params['seed'])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        

        spectral_data = SoilSpectralDataSet(data_path=params['data_path'], dataset_type=params['dataset_type'], 
                                            y_labels=params['y_labels'], preprocessing=None)
        params['spec_dims'] = spectral_data.spec_dims
        
        labs = '_'.join(params['y_labels'])  # Create a string from labels for the save path
        mean = torch.zeros(params['spec_dims'])
        std = torch.zeros(params['spec_dims'])
        train_loader = DataLoader(spectral_data, batch_size=params['batch_size'], shuffle=True, num_workers=0)
        for inputs, _ in train_loader:
            mean += inputs.sum(dim=0)

        mean /= len(train_loader.dataset)

        # Calculate std over the training dataset
        for inputs, _ in train_loader:
            std += ((inputs - mean) ** 2).sum(dim=0)

        std = torch.sqrt(std / len(train_loader.dataset))
        params['mean'] = mean
        params['std'] = std

        params_dict = {
            'dataset_type': params['dataset_type'],
            'data_path': params['data_path'],  # Add this line to include data_path
            'spec_dims': params['spec_dims'],
            'mean': params['mean'],
            'std': params['std'],
            'y_labels': params['y_labels'],
            'num_epochs': params['num_epochs'],
            "batch_size": params['batch_size'],
            "seed": params['seed']  
        }
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, params_dict), n_trials=100)
        
        best_trial = study.best_trial
        best_params = best_trial.params

        print("Best hyperparameters: ", study.best_params)
        
        params_path = os.path.dirname(params['data_path'])+f"/optimize_models/{params['dataset_type']}/{params['model_name']}/{labs}"
        if not os.path.exists(params_path):
                os.makedirs(params_path)
        
            
        with open((params_path+'/best_params.text'), 'w') as f:
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")








