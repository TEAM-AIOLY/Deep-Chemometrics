import numpy as np
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from net.base_net import CuiNet, ViT_1D,ResNet18_1D,DeepSpectraCNN
from data.load_dataset import SoilSpectralDataSet
from utils.training import train
from utils.testing import test 



if __name__ == "__main__":

################################### SEEDING ###################################
    # Set seed for reproducibility (for dataset splitting)
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
###############################################################################
    
################################# SET DEVICE ##################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
###############################################################################
    
############################# DEFINE TRAINING PARAMS ##########################
    num_epochs = 1000
    BATCH = 1024
    LR = 0.0001
    save_interval = 50  # Save model every 10 epochs
    
    y_labels = ["oc_usda.c729_w.pct"]  #, "na.ext_usda.a726_cmolc.kg","clay.tot_usda.a334_w.pct", "k.ext_usda.a725_cmolc.kg", "ph.h2o_usda.a268_index"
                    
    dataset_type = "mir"
    data_path ="./data/dataset/ossl_all_L1_v1.2.csv"
   