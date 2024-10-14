
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from net.base_net import DeepSpectraCNN
from data.load_dataset import SoilSpectralDataSet
from utils.training import train
from utils.misc import data_augmentation

import optuna

import json




def objective(trial, params):
    # Hyperparameter suggestions
    LR = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    WD = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    # Data augmentation parameters to be optimized
    slope = trial.suggest_float("slope", 0.0, 0.3)
    offset = trial.suggest_float("offset", 0.0, 0.3)
    noise = trial.suggest_float("noise", 0.0, 0.3)
    shift = trial.suggest_float("shift", 0.0, 0.3)
    # 
    DP=0.5
                 

    # Apply augmentation to the dataset using the suggested parameters
    augmentation = data_augmentation(slope=slope, offset=offset, noise=noise, shift=shift)
    spectral_data = SoilSpectralDataSet(data_path=params['data_path'], dataset_type=params['dataset_type'], y_labels=params['y_labels'],preprocessing=None)
    
    dataset_size = len(spectral_data)
    test_size = int(0.2 * dataset_size)
    train_size = dataset_size - test_size
    cal_size = int(0.75 * train_size)
    val_size = train_size - cal_size      
    

    train_dataset, _ = random_split(spectral_data, [train_size, test_size], 
                                                   generator=torch.Generator().manual_seed(params['seed']))
    cal_dataset, val_dataset = random_split(train_dataset, [cal_size, val_size], 
                                              generator=torch.Generator().manual_seed(params['seed']))
    
    cal_dataset.dataset.preprocessing=augmentation

    # Create data loaders
    cal_loader = DataLoader(cal_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)


    # Model, optimizer, and training
    model=DeepSpectraCNN(input_dim=params['spec_dims'],mean = params['mean'], std = params['std'],dropout=DP,out_dims=len(params['y_labels']))
    
   
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss(reduction='none')

    _, _, val_r2_scores = train(model, optimizer, criterion, cal_loader, val_loader, 
                                     num_epochs=300, early_stop=False, plot_fig=False,save_path=None)

    criteria=None
    flattened = [item for sublist in val_r2_scores for item in sublist]
    
    if len(flattened) == 1:
        # If there's only one element after flattening, return that element as a float
        criteria= float(flattened[0])
    else:
        # If there are multiple elements, return the mean of the values
        criteria= float(sum(flattened) / len(flattened))
   

    return criteria



if __name__ == "__main__":
    
    data_path ="./data/dataset/ossl/ossl_all_L1_v1.2.csv"
     
    param_file = os.path.dirname(data_path)+ '/optuna_params/optuna_DeepSpectra_params_mir.json'
    
    with open(param_file,'r') as f:  
        params_dict = json.load(f)
    
    for i,param_set in enumerate( params_dict):
        params = param_set
        if i<3:
            continue
        

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
        study.optimize(lambda trial: objective(trial, params_dict), n_trials=30)
        
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
        
        # best_trial = study.best_trial
        # best_value = best_trial.value
        # if best_value < 1:
        #     best_params = best_trial.params
        # else:
        #     sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
        #     second_best_trial = None
        #     for trial in sorted_trials:
        #         if trial.value is not None and trial.value < 1:
        #             second_best_trial = trial
        #             best_params = second_best_trial.params
        #             break

        # print("Best hyperparameters: ", study.best_params)
        
        params_path = os.path.dirname(params['data_path'])+f"/optimize_models/{params['dataset_type']}/{params['model_name']}/{labs}"
        if not os.path.exists(params_path):
                os.makedirs(params_path)
        
            
        # with open((params_path+'/best_params.text'), 'w') as f:
        #     for key, value in best_params.items():
        #         f.write(f"{key}: {value}\n")

            
        with open((params_path+'/params.text'), 'w') as f:
            for tr in sorted_trials:
                for key, value in (tr.params).items():
                    f.write(f"{key}: {value}\n")
                f.write(f"res={tr.value}\n")
                f.write(f"\n")








