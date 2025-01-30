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