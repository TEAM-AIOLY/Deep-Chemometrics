import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from net.base_net import CuiNet
from data.load_dataset import SoilSpectralDataSet
from utils.training import train
from utils.testing import test
from utils.misc import data_augmentation
from net.chemtools.metrics import ccc, r2_score,RMSEP

import matplotlib.pyplot as plt


default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['font.family'] = 'Times New Roman'


if __name__ == "__main__":
    seed=42
    batch_size=1024
    slope= 0.2
    offset= 0.2
    noise=0.002
    shift=0.05
    all_labels = ["oc_usda.c729_w.pct","na.ext_usda.a726_cmolc.kg","clay.tot_usda.a334_w.pct", "k.ext_usda.a725_cmolc.kg", "ph.h2o_usda.a268_index" ]  #

    
    data_path ="./data/dataset/ossl/ossl_all_L1_v1.2.csv"
        
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    augmentation = data_augmentation(slope=slope, offset=offset, noise=noise, shift=shift)
    spectral_data = SoilSpectralDataSet(data_path=data_path, dataset_type="mir", y_labels=all_labels[0],preprocessing=None)
    
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
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader= DataLoader(test_dataset,batch_size=batch_size, shuffle=False, num_workers=0)
    