
import sys
import os
import pandas as pd 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split
import json
from scipy.signal import savgol_filter

import torch 
from torch import nn
from torch import optim
import torch.utils.data as data_utils

from src import utils
from src.net import  ViT_1D
from src.training.training import Trainer
from src.utils.misc import TrainerConfig
from src.utils.misc import snv


#########################################################################################################
# Read configuration file as a list of dict from json file
#########################################################################################################
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
#########################################################################################################
#########################################################################################################


#########################################################################################################
#Get list of Config dicts
#########################################################################################################
config_root= "data/dataset/wheat/config/_ViT1D_Wheat_jz(Wheat_dt).json"
root = os.getcwd()
config_path =os.path.join(root,config_root)


try:
        params_dict = parse_args(config_path)
        print(f"{len(params_dict)} parameters sets")
except Exception as e:
        print(f"Error during execution: {e}")

#########################################################################################################
#########################################################################################################


#########################################################################################################
# Get data and split cal-val
#########################################################################################################
cal1 = pd.read_csv('../data/dataset/Wheat_dt/DT_train-1.csv', header=None)
cal2 = pd.read_csv('../data/dataset/Wheat_dt/DT_train-2.csv', header=None)
cal3 = pd.read_csv('../data/dataset/Wheat_dt/DT_train-3.csv', header=None)
cal4 = pd.read_csv('../data/dataset/Wheat_dt/DT_train-4.csv', header=None)
cal5 = pd.read_csv('../data/dataset/Wheat_dt/DT_train-5.csv', header=None)
## validation
val1 = pd.read_csv('../data/dataset/Wheat_dt/DT_val-1.csv', header=None)
val2 = pd.read_csv('../data/dataset/Wheat_dt/DT_val-2.csv', header=None)
## test
pre1 = pd.read_csv('../data/dataset/Wheat_dt/DT_test-1.csv', header=None)
pre2 = pd.read_csv('../data/dataset/Wheat_dt/DT_test-2.csv', header=None)
pre3 = pd.read_csv('../data/dataset/Wheat_dt/DT_test-3.csv', header=None)




## Concatenate input variables, X
cal_features = np.concatenate((cal1.iloc[:, 0:-1],cal2.iloc[:, 0:-1],cal3.iloc[:, 0:-1],cal4.iloc[:, 0:-1],cal5.iloc[:, 0:-1]),axis=0)
val_features = np.concatenate((val1.iloc[:, 0:-1],val2.iloc[:, 0:-1]),axis = 0)
pre_features = np.concatenate((pre1.iloc[:, 0:-1],pre2.iloc[:, 0:-1],pre3.iloc[:, 0:-1]),axis = 0)

## Concatenate the target variable or lables, Y
cal_labels = np.concatenate((cal1.iloc[:, -1],cal2.iloc[:, -1],cal3.iloc[:, -1],cal4.iloc[:, -1],cal5.iloc[:, -1]),axis = 0)
val_labels = np.concatenate((val1.iloc[:, -1],val2.iloc[:, -1]),axis=0)
pre_labels = np.concatenate((pre1.iloc[:, -1],pre2.iloc[:, -1],pre3.iloc[:, -1]),axis = 0)

## Settings for the smooth derivatives using a Savitsky-Golay filter
w = 13 ## Sav.Gol window size
p = 2  ## Sav.Gol polynomial degree

## Perform data augmentation in the feature space by combining different types of typical chemometric spectral pre-processings
## [spectra, SNV, 1st Deriv., 2nd Deriv., 1st Deriv. SNV, 2nd Deriv. SNV]
x_cal = np.concatenate((cal_features, snv(cal_features),savgol_filter(cal_features, w, polyorder = p, deriv=1),\
                                savgol_filter(cal_features, w, polyorder = p, deriv=2),savgol_filter(snv(cal_features), w, polyorder = p, deriv=1),\
                                savgol_filter(snv(cal_features), w, polyorder = p, deriv=2)),axis = 1)
x_val = np.concatenate((val_features, snv(val_features),savgol_filter(val_features, w, polyorder = p, deriv=1),\
                                savgol_filter(val_features, w, polyorder = p, deriv=2),savgol_filter(snv(val_features), w, polyorder = p, deriv=1),\
                                savgol_filter(snv(val_features), w, polyorder = p, deriv=2)),axis =1)
x_test= np.concatenate((pre_features, snv(pre_features),savgol_filter(pre_features, w, polyorder = p, deriv=1),\
                                savgol_filter(pre_features, w, polyorder = p, deriv=2),savgol_filter(snv(pre_features), w, polyorder = p, deriv=1),\
                                savgol_filter(snv(pre_features), w, polyorder = p, deriv=2)),axis =1)


mean = np.mean(cal_features, axis=0)
std = np.std(cal_features, axis=0)

model_name ="ViT-1D_Wheat"
spec_dims = cal_features.shape[-1]
#########################################################################################################
#########################################################################################################
