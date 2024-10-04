import numpy as np
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from net.base_net import CuiNet
from data.load_dataset import SoilSpectralDataSet
from utils.training import train
from utils.testing import test
from utils.misc import data_augmentation
from net.chemtools.metrics import ccc, r2_score, RMSEP
import matplotlib.pyplot as plt
import optuna





def objective(trial, params):
    # Hyperparameter suggestions
    LR = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    WD = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    # Data augmentation parameters to be optimized
    slope = trial.suggest_float("slope", 0.0, 0.3)
    offset = trial.suggest_float("offset", 0.0, 0.3)
    noise = trial.suggest_float("noise", 0.0, 0.3)
    shift = trial.suggest_float("shift", 0.0, 0.3)

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

    train_val_dataset, test_dataset = random_split(spectral_data, [train_val_size, test_size], 
                                                   generator=torch.Generator().manual_seed(params['seed']))
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(params['seed']))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)
    

    # Model, optimizer, and training
    model = CuiNet(params['spec_dims'], mean=params['mean'], std=params['std'], out_dims=len(params['y_labels']))
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss(reduction='none')

    _, _, _, final_save_path = train(model, optimizer, criterion, train_loader, val_loader, 
                                     num_epochs=params['num_epochs'], early_stop=False, plot_fig=False,save_path=params['save_path'])

    # Evaluate model performance
    y_pred, Y = test(model, final_save_path, test_loader)
    r2 = r2_score(y_pred, Y)
    rmsep = RMSEP(y_pred, Y)
    ccc_value = ccc(y_pred, Y)

    return r2



if __name__ == "__main__":

    params = {
        "dataset_type": "mir",
        "data_path": "./data/dataset/ossl_all_L1_v1.2.csv",
        "y_labels": ["oc_usda.c729_w.pct"],
        "batch_size": 1024,
        "num_epochs": 1000,
        "model_name": "_CuiNet_OSSL_",
        "seed": 42,
        "spec_dims": None,  # Will be set after loading dataset
        "mean": None,       # To be computed from the dataset
        "std": None,        # To be computed from the dataset

    }
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
    save_path = os.path.dirname(params['data_path']) + f'/models/{params["dataset_type"]}/{params["model_name"]}/{labs}'
    params['save_path']=save_path

# Ensure save_path directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    

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
        "save_path": params['save_path'],
        "batch_size": params['batch_size'],
        "seed": params['seed']  
    }
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, params_dict), n_trials=100)
    
    best_trial = study.best_trial
    best_params = best_trial.params

    print("Best hyperparameters: ", study.best_params)
    
    params_path = os.path.join(os.path.dirname(params['data_path']), 'optimize_models', params['dataset_type'], params['model_name'])
    if not os.path.exists(params_path):
        os.makedirs(params_path)
        
    with open(os.path.join(params_path, 'best_params.txt'), 'w') as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")








