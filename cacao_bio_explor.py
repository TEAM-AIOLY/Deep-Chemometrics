import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# data_path = './data/dataset/3rd_data_sampling_and_microbial_data/y_bio.csv'
data_path = './data/dataset/3rd_data_sampling_and_microbial_data/bio_rep_data.csv'
data = pd.read_csv(data_path, sep=';', header=0, lineterminator='\n', skip_blank_lines=True)

col_labs = [f"T{i}" for i in range(1, 16)]  # Generate T1 to T15
Treatments = np.array([label for label in col_labs for _ in range(4)])  # Repeat each label 4 times

headers = ['Girth_final','Girth_increment','height_final','height_increment','Nb_leaves','roor_length', 'Fresh_weight_leaves',
           'Fresh_weight_stem','Fresh_weight_root','Total_fresh_weight', 'Dry_weight_leaves','Dry_weight_stem','Dry_weight_root','Total_dry_weight']


X =data.iloc[2:62,2:]
X = X.map(lambda x: str(x).replace(',', '.'))
X=np.array(X).astype(float)

