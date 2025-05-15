import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,detrend
from net.chemtools.PLS import PLS,MLDA,MLDA_v2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import distinctipy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.lines import Line2D
import random

import random
random.seed(42)

data_path='./data/dataset/3rd_data_sampling_and_microbial_data/spec_data-rep.csv'                                                                    
y_microbe_path ='./data/dataset/3rd_data_sampling_and_microbial_data/y_mircob_rep.csv'

figure_root = os.path.join(os.path.dirname(data_path), 'new_discri_figures')
if not os.path.exists(figure_root):
    os.makedirs(figure_root)
    
two_d_folder = os.path.join(figure_root, '2D_plots')
three_d_folder = os.path.join(figure_root, '3D_plots')
os.makedirs(two_d_folder, exist_ok=True)
os.makedirs(three_d_folder, exist_ok=True)
microbe_2d_folder = os.path.join(two_d_folder, 'microbe_labels')
os.makedirs(microbe_2d_folder, exist_ok=True)
microbe_3d_folder = os.path.join(three_d_folder, 'microbe_labels')
os.makedirs(microbe_3d_folder, exist_ok=True)

############################################################################################
############################################################################################
data = pd.read_csv(data_path, sep=';', header=0, lineterminator='\n', skip_blank_lines=True)
data = data.dropna(how='all')
first_col = data.iloc[1:1984, 0]
wv =np.array(first_col.str.replace(',', '.')).astype(float)

# Generate a single column array with repeated column names
column_labels = [f"T{i}" for i in range(1, 16)]  # Generate T1 to T15
repeated_labels = np.array([label for label in column_labels for _ in range(4)])  # Repeat each label 4 times

sepctral_data = data.iloc[1:1984, 1:]
sepctral_data = sepctral_data.map(lambda x: str(x).replace(',', '.'))  # Replace commas with dots
X = np.array(sepctral_data).astype(float)
X=X.T

Y_data= pd.read_csv(y_microbe_path,sep=';',header=0)
Y_data=Y_data.map(lambda x: str(x).replace(',', '.'))
Y_ferti = np.array(Y_data.iloc[1, 1:]).astype(float)

Y_microbe = np.array(Y_data.iloc[2:,1:]).T.astype(float)

label_names = ['B1', 'B2', 'B3', 'B4']
default_colors = distinctipy.get_colors(15)



# Settings for the smooth derivatives using a Savitsky-Golay filter
w = 21 ## Sav.Gol window size
p = 2  ## Sav.Gol polynomial degree
d=1 ## Sav.Gol derive  order

X = np.log1p(X) 
X = savgol_filter(X, window_length=w, polyorder=p,deriv=d, axis=1)
X= detrend(X, axis=1)


n_samples = len(X)
# Create mask where every 4th sample is for test
indices = np.arange(n_samples)
test_mask = (indices + 1) % 4 == 0  # +1 makes index 3, 7, 11... into test
# Apply mask
X_train, X_test = X[~test_mask], X[test_mask]
y_train, y_test = Y_microbe[~test_mask], Y_microbe[test_mask]
repeated_labels_train, repeated_labels_test = repeated_labels[~test_mask], repeated_labels[test_mask]

n_components = 20      

plsr = PLS(ncomp=n_components)

plsr.fit(X_train, y_train)  # Fit PLSDA for the current class
T_train = plsr.transform(X_train)
scores_test =plsr.transform(X_test)

mlda =MLDA()
mlda.fit(T_train,y_train)
preds =mlda.transform(scores_test)
classes =mlda.predict(scores_test)
proj =mlda.projections

W =(plsr.W)
DV = (W@proj.T).T 
DV = np.array(DV)


plt.figure()
for i in range(DV.shape[0]):
    plt.plot(wv,DV[i,:])
plt.title('Discriminant Vectors (DV) in spectral space')
plt.xlabel("wavelength (nm)")
plt.grid(True)
plt.savefig(os.path.join(figure_root, 'plsda_loadings.png'), dpi=900)  
# plt.show()



ferti_norm = (Y_ferti - Y_ferti.min()) / (Y_ferti.max() - Y_ferti.min() + 1e-6)
ferti_norm_test = ferti_norm[test_mask]
sizes = 30 + 300 * ferti_norm_test ** 1.5

unique_ferti = np.unique(Y_ferti)
ferti_legend_elements = []
for val in unique_ferti:
    norm_val = (val - Y_ferti.min()) / (Y_ferti.max() - Y_ferti.min() + 1e-6)
    size = 30 + 300 * norm_val ** 1.5
    ferti_legend_elements.append(Line2D([0], [0], marker='o', color='gray', label=f'Ferti: {val:.1f}',
        markerfacecolor='gray', markersize=np.sqrt(size), markeredgecolor='black', linewidth=0))



axis_pairs = [(0, 1), (1, 2), (2, 3)]

for label_idx, label_name in enumerate(label_names):
    presence = y_test[:, label_idx].astype(int)
    colors = ['tab:red' if val == 1 else 'gray' for val in presence]
    
    label_folder_2d = os.path.join(microbe_2d_folder, label_name)
    os.makedirs(label_folder_2d, exist_ok=True)
    
    for pc_x, pc_y in axis_pairs:
        if max(pc_x, pc_y) >= preds.shape[1]:
            continue  # skip if the axis exceeds available components

        fig, ax = plt.subplots()
        for i in range(len(preds)):
            ax.scatter(preds[i, pc_x], preds[i, pc_y], color=colors[i],
                       s=sizes[i], edgecolor='black', alpha=0.85)
            ax.annotate(f'T{i+1}',
                        xy=(preds[i, pc_x], preds[i, pc_y]), xytext=(5, 2),
                        textcoords='offset points',
                        fontsize=9)
         
        ax.set_xlabel(f"DV{pc_x + 1}")
        ax.set_ylabel(f"DV{pc_y + 1}")
        ax.set_title(f"DV{pc_x + 1} vs DV{pc_y + 1} - {label_name} Presence\nPoint size = Fertilization level")
        ax.grid(True)
        ax.legend(handles=ferti_legend_elements, title="Fertilization Size",
                   bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1)
        plt.tight_layout()
        save_path = os.path.join(label_folder_2d, f'mlda_microbe_{label_name}_DV{pc_x+1}_vs_DV{pc_y+1}.png')
        plt.savefig(save_path, dpi=900)
        plt.close()