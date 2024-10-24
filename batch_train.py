import argparse
import json
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from net.base_net import ViT_1D
from data.load_dataset import SoilSpectralDataSet
from utils.training import train
from utils.testing import test
from utils.misc import data_augmentation
from net.chemtools.metrics import ccc, r2_score,RMSEP


def load_config(file_path, index):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as file:
            config_list = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON file: {e}")
    
    if index < 0 or index >= len(config_list):
        raise IndexError(f"Index {index} is out of range for configuration list with {len(config_list)} entries.")
    
    config =config_list[index]
    for key in config:
        if isinstance(config[key], float):
            config[key] = np.float32(config[key])
        elif isinstance(config[key], np.longdouble): 
            config[key] = np.float32(config[key])
    return config

def get_args():
 
    parser = argparse.ArgumentParser(description="Read one configuration from the JSON file.")
    
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the JSON configuration file containing a list of parameter sets.')
    parser.add_argument('--config_index', type=int, required=True,
                        help='Index of the configuration to read from the JSON file.')
    args = parser.parse_args()
    
    # Load and return the configuration dictionary based on the arguments
    config = load_config(args.config_path, args.config_index)
    
    return config

if __name__ == "__main__":
   
    seed=42
    params = get_args()
  
  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    
    
    augmentation = data_augmentation(slope=params['slope'], offset=params['offset'], noise=params['noise'], shift=params['shift'])
    spectral_data = SoilSpectralDataSet(data_path=params['data_path'], dataset_type=params['dataset_type'], y_labels=params['y_labels'],preprocessing=None)
    
    dataset_size = len(spectral_data)
    test_size = int(0.2 * dataset_size)
    train_size = dataset_size - test_size
    cal_size = int(0.75 * train_size)
    val_size = train_size - cal_size      
    
    print(dataset_size)
    print(cal_size)
    print(val_size)


    # train_dataset, test_dataset = random_split(spectral_data, [train_size, test_size], 
    #                                             generator=torch.Generator().manual_seed(params['seed']))
    # cal_dataset, val_dataset = random_split(train_dataset, [cal_size, val_size], 
    #                                         generator=torch.Generator().manual_seed(params['seed']))
    
    # cal_dataset.dataset.preprocessing=augmentation

    # # Create data loaders
    # cal_loader = DataLoader(cal_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    # val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)
    # test_loader= DataLoader(test_dataset,batch_size=params['batch_size'], shuffle=False, num_workers=0)