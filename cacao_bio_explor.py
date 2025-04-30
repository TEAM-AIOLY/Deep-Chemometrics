import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data_path = './data/dataset/3rd_data_sampling_and_microbial_data/y_bio.csv'
data = pd.read_csv(data_path, sep=';', header=0, lineterminator='\n', skip_blank_lines=True)

Treatments = data.iloc[1:16, 0]
Treatments=np.array(Treatments).astype(str)

headers = ['Girth_final','Girth_increment','height_final','height_increment','Nb_leaves','roor_length', 'Fresh_weight_leaves',
           'Fresh_weight_stem','Fresh_weight_root','Total_fresh_weight', 'Dry_weight_leaves','Dry_weight_stem','Dry_weight_root','Total_dry_weight']


X =data.iloc[1:16,1:]
X = X.replace(',', '.')
X=np.array(X).astype(float)
print(X)