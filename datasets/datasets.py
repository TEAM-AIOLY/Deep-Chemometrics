import sys
import os
import pandas as pd
import scipy
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from utils. augmentations import emsc, emsa


class DatasetWithIdx(TensorDataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # Use PyTorch tensors
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label, idx

class WheatDataset(TensorDataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32)  # Use PyTorch tensors
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        # wavelengths of the dataset
        pas = (1645-975)/200
        self.wavelengths = torch.arange(200)*pas+975
        # params needed for EMSA
        self.X = None
        self.A = None
        self.order = None
        self.coefs = None
        self.residues = None
        self.coefs_std = None
        # compute parameters for transform if needed
        if transform == "emsa_2":
            self.order = int(transform[transform.find("_")+1])
            _, self.coefs, self.residues, self.X, self.A= emsc(
                self.data, self.wavelengths,
                order=self.order, reference=torch.mean(self.data,axis=0),
                return_coefs=True)
            self.coefs_std = self.coefs.std(axis=0)
        
    def __len__(self):
        return len(self.data)
    
    def apply_emsa(self, idx):
        coefs = self.coefs[idx,:]
        residues = self.residues[idx]
        return emsa(coefs, residues, self.coefs_std, self.X)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform == "emsa_2":
            sample = self.apply_emsa(idx)

        return sample, label, idx
        
def loadDataset(dataset:str,cwd:str):
    precision = torch.float64 # float64 to make sure cov mat is PSD, float32 is used during training
    fraction = 1 if "fraction" not in dataset else dataset["fraction"]
    if dataset["name"] == "wheat" :
        cal1 = pd.read_csv(os.path.join(cwd, 'datasets/classif/wheat/DT_train-1.csv'),header=None)
        cal2 = pd.read_csv(os.path.join(cwd, 'datasets/classif/wheat/DT_train-2.csv'),header=None)

        cal3 = pd.read_csv(os.path.join(cwd, 'datasets/classif/wheat/DT_train-3.csv'),header=None)
        cal4 = pd.read_csv(os.path.join(cwd, 'datasets/classif/wheat/DT_train-4.csv'),header=None)
        cal5 = pd.read_csv(os.path.join(cwd, 'datasets/classif/wheat/DT_train-5.csv'),header=None)
        ## validation
        val1 = pd.read_csv(os.path.join(cwd, 'datasets/classif/wheat/DT_val-1.csv'),header=None)
        val2 = pd.read_csv(os.path.join(cwd, 'datasets/classif/wheat/DT_val-1.csv'),header=None)
        ## test
        pre1 = pd.read_csv(os.path.join(cwd, 'datasets/classif/wheat/DT_test-1.csv'),header=None)
        pre2 = pd.read_csv(os.path.join(cwd, 'datasets/classif/wheat/DT_test-2.csv'),header=None)
        pre3 = pd.read_csv(os.path.join(cwd, 'datasets/classif/wheat/DT_test-3.csv'),header=None)

        ## Concatenate input variables, X
        cal_features = np.concatenate((cal1.iloc[:, 0:-1],cal2.iloc[:, 0:-1],cal3.iloc[:, 0:-1],cal4.iloc[:, 0:-1],cal5.iloc[:, 0:-1]),axis=0)
        val_features = np.concatenate((val1.iloc[:, 0:-1],val2.iloc[:, 0:-1]),axis = 0)
        pre_features = np.concatenate((pre1.iloc[:, 0:-1],pre2.iloc[:, 0:-1],pre3.iloc[:, 0:-1]),axis = 0)

        ## Concatenate the target variable or lables, Y
        cal_labels = np.concatenate((cal1.iloc[:, -1],cal2.iloc[:, -1],cal3.iloc[:, -1],cal4.iloc[:, -1],cal5.iloc[:, -1]),axis = 0)
        val_labels = np.concatenate((val1.iloc[:, -1],val2.iloc[:, -1]),axis=0)
        pre_labels = np.concatenate((pre1.iloc[:, -1],pre2.iloc[:, -1],pre3.iloc[:, -1]),axis = 0)

        ## One-hot encoding for Y 
        y_cal = np.eye(30)[cal_labels]
        y_val = np.eye(30)[val_labels]
        y_test = np.eye(30)[pre_labels]

        # transform numpy tensors to torch tensors
        cal_features = torch.tensor(cal_features[:int(fraction*len(cal_features))], dtype=precision) 
        val_features = torch.tensor(val_features, dtype=precision)
        pre_features = torch.tensor(pre_features, dtype=precision)
        y_cal = torch.tensor(y_cal[:int(fraction*len(cal_features))], dtype=precision)
        y_val = torch.tensor(y_val, dtype=precision)
        y_test = torch.tensor(y_test, dtype=precision)

        # calculate mean and std per column
        mean = torch.mean(cal_features, axis=0)
        std = torch.std(cal_features, axis=0)

        return cal_features, val_features, pre_features, y_cal, y_val, y_test, mean, std, 30
    
    elif dataset["name"] == "mango":
        mat = scipy.io.loadmat(os.path.join(cwd, 'datasets/regression/mango/mango_dm_full_outlier_removed2.mat'))
        x_cal = mat['Sp_cal']
        x_test = mat['Sp_test']
        y_cal = mat['DM_cal']
        y_test = mat['DM_test']
        x_cal, x_val, y_train, y_val = train_test_split(x_cal,y_cal, test_size=0.25, random_state=42) # seed fixed to ensure fair comparison 
        x_cal = torch.tensor(x_cal[:int(fraction*len(x_cal))], dtype=precision)
        x_val = torch.tensor(x_val, dtype=precision)
        x_test = torch.tensor(x_test, dtype=precision)
        y_cal = torch.tensor(y_train[:int(fraction*len(x_cal))], dtype=precision)
        y_val = torch.tensor(y_val, dtype=precision)
        y_test = torch.tensor(y_test, dtype=precision)
        mean = torch.mean(x_cal, axis=0)
        std = torch.std(x_cal, axis=0)
        return x_cal, x_val, x_test, y_train, y_val, y_test, mean, std, 1
    
    elif dataset["name"] == "corn":
        mat_corns = scipy.io.loadmat(os.path.join(cwd, 'datasets/regression/corn/corn.mat'))
        if dataset["target"] == "moisture":
            y = np.array(mat_corns['propvals']['data'])[0,0][:,0]
        elif dataset["target"] == "oil":
            y = np.array(mat_corns['propvals']['data'])[0,0][:,1]
        elif dataset["target"] == "protein":
            y = np.array(mat_corns['propvals']['data'])[0,0][:,2]
        elif dataset["target"] == "starch_values":
            y = np.array(mat_corns['propvals']['data'])[0,0][:,3]
        else : 
            sys.exit("for corn dataset precise target in config file : moisture, oil, protein or starch_values")
        m5spec = np.array(mat_corns['m5spec']['data'])[0,0]
        mp5spec = np.array(mat_corns['mp5spec']['data'])[0,0]
        mp6spec = np.array(mat_corns['mp6spec']['data'])[0,0]
        features = np.concatenate([m5spec,mp5spec,mp6spec], axis=0)
        y = np.expand_dims(np.tile(y,3),axis=1)
        x_cal_val, x_test, y_train_val, y_test = train_test_split(features, y, test_size=0.15, random_state=42) # seed fixed to ensure fair comparison 
        x_cal, x_val, y_train, y_val = train_test_split(x_cal_val, y_train_val, test_size=0.2, random_state=42) # seed fixed to ensure fair comparison 
        x_cal = torch.tensor(x_cal[:int(fraction*len(x_cal))], dtype=precision)
        x_val = torch.tensor(x_val, dtype=precision)
        x_test = torch.tensor(x_test, dtype=precision)
        y_cal = torch.tensor(y_train, dtype=precision)
        y_val = torch.tensor(y_val, dtype=precision)
        y_test = torch.tensor(y_test[:int(fraction*len(x_cal))], dtype=precision)
        mean = torch.mean(x_cal, axis=0)
        std = torch.std(x_cal, axis=0)
        return x_cal, x_val, x_test, y_train, y_val, y_test, mean, std, 1
    
    else : 
        sys.exit("for now only wheat, mango and corn datasets are implemented, for corn dataset precise the target : moisture, oil, protein or starch_values")


