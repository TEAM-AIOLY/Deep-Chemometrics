import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter,detrend
from net.chemtools.PLS import PLS,PLSDA, LDA

from sklearn.model_selection import train_test_split



data_path='./data/dataset/3rd_data_sampling_and_microbial_data/'                                                                    
ref_path ='./data/dataset/3rd_data_sampling_and_microbial_data/MIX MICROBE dis.csv'

data = pd.read_csv(data_path, sep=',', header=0)
wv =np.array(data.keys()[1:]).astype(float)

spectral_data = np.array(data.iloc[:, 1:])
T_ref = np.array(data.iloc[:, 0]).astype(str)

ref= pd.read_csv(ref_path, sep=',', header=0)
ref_data = np.array(ref.iloc[:, 1:])
mixrobe_data = ref_data[:,1:]
ferti =ref_data[:,0]