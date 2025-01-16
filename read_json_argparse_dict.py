import os
import argparse
import json

import torch
from torch import nn, optim

from data.load_dataset import SoilSpectralDataSet
from utils.misc import data_augmentation
from torch.utils.data import DataLoader, random_split
from net.base_net import ViT_1D


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization script.")
    parser.add_argument('--config_file', type=str, required=True, help='Path to the JSON configuration file.')

    args = parser.parse_args()
    file_path=args.config_file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        # Load parameters from JSON file
        with open(args.config_file, 'r') as f:
            params_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON file: {e}")

    return params_dict


if __name__ == "__main__":
    # param_file_path ="./data/dataset/ossl/config/_ViT1D_(oc_us).json"
    try:
        params_dict = parse_args()
        print(len(params_dict))  # This will print the parameters read from the config file
    except Exception as e:
        print(f"Error during execution: {e}")
        

    seed=42
    NUM_WORKERS =0
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
    params =params_dict[0]   
        
    augmentation = data_augmentation(slope=params['slope'], offset=params['offset'], noise=params['noise'], shift=params['shift'])
    spectral_data = SoilSpectralDataSet(data_path=params['data_path'], dataset_type=params['dataset_type'], y_labels=params['y_labels'],preprocessing=None)
    params['spec_dims'] = spectral_data.spec_dims
    
    dataset_size = len(spectral_data)
    test_size = int(0.2 * dataset_size)
    train_size = dataset_size - test_size
    cal_size = int(0.75 * train_size)
    val_size = train_size - cal_size      
    
    train_dataset, test_dataset = random_split(spectral_data, [train_size, test_size], 
                                                generator=torch.Generator().manual_seed(seed))
    cal_dataset, val_dataset = random_split(train_dataset, [cal_size, val_size], 
                                            generator=torch.Generator().manual_seed(seed))
    
    cal_dataset.dataset.preprocessing=augmentation

    # Create data loaders
    cal_loader = DataLoader(cal_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=NUM_WORKERS)
    test_loader= DataLoader(test_dataset,batch_size=params['batch_size'], shuffle=False, num_workers=NUM_WORKERS)
    
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
    
    model=ViT_1D(mean = params['mean'], std = params['std'], seq_len = params['spec_dims'], patch_size = params['PS'], 
                dim_embed = params['DE'], trans_layers = params['TL'], heads = params['HDS'], mlp_dim = params['MLP'], out_dims = len(params['y_labels']) )
    
    optimizer = optim.Adam(model.parameters(), lr=params['LR'], weight_decay=params['WD'])
    criterion = nn.MSELoss(reduction='none')


    base_path =os.path.dirname(params['data_path'])+f"/model_benchmark/{params['network']}/run_{params['ID']}"
    os.makedirs(base_path, exist_ok=True)
    print(base_path)
        
        
    
