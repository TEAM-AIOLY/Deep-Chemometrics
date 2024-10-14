import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from net.base_net import CuiNet
from data.load_dataset import SoilSpectralDataSet
from data.load_dataset_atonce import  SpectralDataset
from utils.training import train
from utils.testing import test
from utils.misc import data_augmentation
from net.chemtools.metrics import ccc, r2_score,RMSEP

import matplotlib.pyplot as plt


default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['font.family'] = 'Times New Roman'

y_labels = ["oc_usda.c729_w.pct","na.ext_usda.a726_cmolc.kg","clay.tot_usda.a334_w.pct", "k.ext_usda.a725_cmolc.kg", "ph.h2o_usda.a268_index" ]  #   
data_path ="./data/dataset/ossl/ossl_all_L1_v1.2.csv"

dataset = SpectralDataset(data_path, y_labels, "mir")
spec_dims=dataset.spec_dims


wavelength = dataset.get_spectral_dimensions()
num_samples = 200 
X_train=dataset.X_train
X_val=dataset.X_val
Y_train=dataset.Y_train
Y_val=dataset.Y_val

train_size = len(dataset.X_train)
train_sample_indices = np.random.choice(train_size, min(num_samples, train_size), replace=False)

val_size = len(dataset.X_val)
val_sample_indices = np.random.choice(val_size, min(num_samples, val_size), replace=False)

X_train_sample = X_train[train_sample_indices].numpy()
Y_train_sample = Y_train[train_sample_indices].numpy()

X_val_sample = X_val[val_sample_indices].numpy()
Y_val_sample = Y_val[val_sample_indices].numpy()
    
epsilon = 1e-10
Y_train_log = torch.log(torch.where(Y_train > 0, Y_train, torch.tensor(epsilon, dtype=Y_train.dtype)))
Y_val_log = torch.log(torch.where(Y_val > 0, Y_val, torch.tensor(epsilon, dtype=Y_val.dtype)))
    
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


plt.figure(figsize=(16, 8))
for i in range(min(num_samples, train_size)):
    plt.plot(wavelength,X_train_sample[i], alpha=0.5, label=f'Sample {i+1}' if i < 10 else "")
    plt.title("X_train")
    plt.xlabel("wavelenght (nm)")
    plt.ylabel("Pseudo absorbance")
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=True, ncol=5, labelcolor=default_colors) 
plt.tight_layout()
plt.show(block=False)

base_path = "D:/fig_ossl/fig_database/"
if not os.path.exists(base_path):
            os.makedirs(base_path)
            
save_path = base_path+'/sample_spec.pdf'
plt.savefig(save_path, format='pdf')


plt.figure(figsize=(16, 8))
for i in range(min(num_samples, val_size)):
    _=plt.plot(wavelength, X_val_sample[i], alpha=0.5, label=f'Sample {i+1}' if i < 10 else "")
    plt.title(" X_val")
    plt.xlabel("Wavelength nm")
    plt.ylabel("Pseudo absorbance")
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=5, labelcolor=default_colors)
plt.tight_layout()
plt.show(block=False)





for j in range(len(y_labels)):
    plt.figure(figsize=(16, 8))
    _=plt.hist(Y_train_log[:, j], bins=100, alpha=0.5, label=y_labels[j],color=default_colors[j])
    plt.title("Histogram of Sampled Training Targets (Y_train)")
    plt.xlabel("y value")
    plt.ylabel("Frequency")
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=5, labelcolor=default_colors)
    plt.tight_layout()
    plt.show(block=False)
    
    save_path = base_path+f"/hist_{y_labels[j]}.pdf"
    plt.savefig(save_path, format='pdf')

for j in range(len(y_labels)):
    plt.figure(figsize=(16, 8))
    _=plt.hist((Y_val_log[:, j]), bins=100, alpha=0.5, label=y_labels[j]);
    plt.title("Histogram of Sampled Validation Targets (Y_val)")
    plt.xlabel("y value")
    plt.ylabel("Frequency")
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    fancybox=True, shadow=True, ncol=5, labelcolor=default_colors)

    plt.tight_layout()
    plt.show(block=False)

# if __name__ == "__main__":
#     seed=42
#     batch_size=1024
#     slope= 0.2
#     offset= 0.2
#     noise=0.002
#     shift=0.05
#     all_labels = ["oc_usda.c729_w.pct","na.ext_usda.a726_cmolc.kg","clay.tot_usda.a334_w.pct", "k.ext_usda.a725_cmolc.kg", "ph.h2o_usda.a268_index" ]  #

    
#     data_path ="./data/dataset/ossl/ossl_all_L1_v1.2.csv"
        
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


#     augmentation = data_augmentation(slope=slope, offset=offset, noise=noise, shift=shift)
#     spectral_data = SoilSpectralDataSet(data_path=data_path, dataset_type="mir", y_labels=all_labels[0],preprocessing=None)
    
#     dataset_size = len(spectral_data)
#     test_size = int(0.2 * dataset_size)
#     train_size = dataset_size - test_size
#     cal_size = int(0.75 * train_size)
#     val_size = train_size - cal_size      


#     train_dataset, test_dataset = random_split(spectral_data, [train_size, test_size], 
#                                                 generator=torch.Generator().manual_seed(seed))
#     cal_dataset, val_dataset = random_split(train_dataset, [cal_size, val_size], 
#                                             generator=torch.Generator().manual_seed(seed))
    
#     cal_dataset.dataset.preprocessing=augmentation

#     # Create data loaders
#     cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#     test_loader= DataLoader(test_dataset,batch_size=batch_size, shuffle=False, num_workers=0)
    