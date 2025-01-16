import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from net.base_net import ViT_1D
from data.load_dataset import SoilSpectralDataSet
from utils.training import train
from utils.misc import data_augmentation

import optuna

import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization script.")
    parser.add_argument('--config_file', type=str, required=True, help='Path to the JSON configuration file.')

    args = parser.parse_args()

    # Load parameters from JSON file
    with open(args.config_file, 'r') as f:
        params_dict = json.load(f)

    # Convert y_labels to a list
    params_dict['y_labels'] = params_dict['y_labels'].split(',')

    return params_dict



def objective(trial, params):
    # Hyperparameter suggestions using bounds from params
    LR = trial.suggest_float("lr", params['lr_bounds'][0], params['lr_bounds'][1], log=True)
    WD = trial.suggest_float("weight_decay", params['weight_decay_bounds'][0], params['weight_decay_bounds'][1], log=True)

    # Data augmentation parameters using bounds
    slope = trial.suggest_float("slope", params['slope_bounds'][0], params['slope_bounds'][1])
    offset = trial.suggest_float("offset", params['offset_bounds'][0], params['offset_bounds'][1])
    noise = trial.suggest_float("noise", params['noise_bounds'][0], params['noise_bounds'][1])
    shift = trial.suggest_float("shift", params['shift_bounds'][0], params['shift_bounds'][1])
    
    PS = trial.suggest_int("patch_size", params['patch_size_bounds'][0], params['patch_size_bounds'][1])
    DE = trial.suggest_int("dim_embed", params['dim_embed_bounds'][0], params['dim_embed_bounds'][1])
    TL = trial.suggest_int("trans_layers", params['trans_layers_bounds'][0], params['trans_layers_bounds'][1])
    HDS = trial.suggest_int("heads", params['heads_bounds'][0], params['heads_bounds'][1])
    MLP = trial.suggest_int("mlp_dim", params['mlp_dim_bounds'][0], params['mlp_dim_bounds'][1])
    

    # Apply augmentation to the dataset using the suggested parameters
    augmentation = data_augmentation(slope=slope, offset=offset, noise=noise, shift=shift)
    spectral_data = SoilSpectralDataSet(data_path=params['data_path'], dataset_type=params['dataset_type'], y_labels=params['y_labels'], preprocessing=None)
    
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
    cal_loader = DataLoader(cal_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=20)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=20)


    # Model, optimizer, and training
    model=ViT_1D(mean = params['mean'], std = params['std'], seq_len = params['spec_dims'], patch_size = PS, 
                 dim_embed = DE, trans_layers = TL, heads = HDS, mlp_dim = MLP, out_dims = len(params['y_labels']) )
   
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss(reduction='none')

    _, _, val_r2_scores = train(model, optimizer, criterion, cal_loader, val_loader, 
                                     num_epochs=params['num_epochs'], early_stop=False, plot_fig=False,save_path=None)
   
    r2_mean_targets = [sum(r2_epoch) / len(r2_epoch) for r2_epoch in val_r2_scores]

    criteria = max(r2_mean_targets)

    return criteria

def main():

    params_dict = parse_args()
    
    # Set manual seed for reproducibility
    torch.manual_seed(params_dict['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params_dict['seed'])
        torch.cuda.manual_seed_all(params_dict['seed'])

    spectral_data = SoilSpectralDataSet(data_path=params_dict['data_path'], dataset_type=params_dict['dataset_type'], 
                                        y_labels=params_dict['y_labels'], preprocessing=None)

    params_dict['spec_dims'] = spectral_data.spec_dims
    
    train_loader = DataLoader(spectral_data, batch_size=params_dict['batch_size'], shuffle=True, num_workers=0)
    mean = torch.zeros(params_dict['spec_dims'])
    std = torch.zeros(params_dict['spec_dims'])

    for inputs, _ in train_loader:
        mean += inputs.sum(dim=0)

    mean /= len(train_loader.dataset)

    for inputs, _ in train_loader:
        std += ((inputs - mean) ** 2).sum(dim=0)

    std = torch.sqrt(std / len(train_loader.dataset))
    params_dict['mean'] = mean
    params_dict['std'] = std

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, params_dict), n_trials=100)
    
    best_trial = study.best_trial

         
    params_path = os.path.dirname(params_dict['data_path'])+f"/optimize_models/{params_dict['dataset_type']}/{params_dict['model_name']}"
    if not os.path.exists(params_path):
            os.makedirs(params_path)
            
        
    with open(f"{params_path}/{params_dict['id']}.json", 'w') as f:
        json.dump(best_trial.params, f, indent=4)
        
    
if __name__ == "__main__":
    main()






