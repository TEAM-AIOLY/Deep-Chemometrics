import json
import torch
import os
from torch.utils.data import DataLoader
from data.load_dataset import SoilSpectralDataSet
from utils.training import train
from utils.testing import test
from utils.misc import data_augmentation
from net.base_net import CuiNet
from net.chemtools.metrics import ccc, r2_score, RMSEP
import optuna


def load_params_from_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    

if __name__ == "__main__":
    # Load parameters from the file
    data_path ="./data/dataset/ossl/ossl_all_L1_v1.2.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path {data_path} does not exist.")
    
    
    param_file =os.path.dirname('data_path')+'\optuna_params.json'
    params_list = load_params_from_file('params.json')