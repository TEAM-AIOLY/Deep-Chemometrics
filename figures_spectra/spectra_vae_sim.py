
import os
import sys
import scipy as sp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
from src.utils.dataset_loader import DatasetLoader


dataset= {"data_path": "data/dataset/Wheat_dt/",
    "dataset_type": "wheat"}


data = DatasetLoader.load(dataset)

x_cal = data["x_cal"]   
y_cal = data["y_cal"]
x_val = data["x_val"]
y_val = data["y_val"]
x_test = data["x_test"]
y_test = data["y_test"]
