import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
import torch.utils.data as data_utils
from data.load_dataset_atonce import MangoDataset

from net.base_net import ViT_1D
from utils.misc import data_augmentation
from net.chemtools.metrics import ccc, r2_score,RMSEP
from utils.training import train
from utils.testing import test





data_path= "./data/dataset/Mango/mango_dm_full_outlier_removed2.mat",  
y_labels=['dm_mango']

data = sp.io.loadmat(data_path)
print(data.keys())

Ycal = data["DM_cal"]
Ytest = data["DM_test"]
Xcal = data["SP_all_train"]
Xtest = data["SP_all_test"]
print("X and Y training set")
print(Ycal.shape)
print(Xcal.shape)

print("X and Y testing set")
print(Ytest.shape)
print(Xtest.shape)

## Spliting the train set 
x_cal, x_val, y_cal, y_val = train_test_split(Xcal, Ycal, test_size=0.20, shuffle=True, random_state=42) 

## The wavelenghts for the XX axis when we plot the spectra
x_scale=data['wave'].astype(np.float32).reshape(-1,1)

## Check for dimensions
print('Data set dimensions ----------------------------')
print('Full Train set dims X Y = {}\t{}'.format(Xcal.shape, Ycal.shape))
print('Calibration set dims X Y = {}\t{}'.format(x_cal.shape, y_cal.shape))
print('val set dims X Y = {}\t{}'.format(x_val.shape, y_val.shape))
print('Test set dims X Y = {}\t{}'.format(Xtest.shape, Ytest.shape))
print('wavelengths number = {}'.format(np.shape(x_scale)))