import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter,detrend
import distinctipy
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.lines import Line2D
import random

random.seed(42)
data_path='./data/dataset/3rd_data_sampling_and_microbial_data/data 3rd sampling 450-950nm.csv'

figure_root = os.path.join(os.path.dirname(data_path), 'new_spectral_figures')
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


data = pd.read_csv(data_path, sep=',', header=0)

wv =(np.array(data.keys()[1:]).astype(float))
spectral_data = (np.array(data.iloc[:, 1:]))



Treatments = np.array([f"T{i}" for i in range(1, 16)])  # Generate T1 to T15

microbe_path = './data/dataset/3rd_data_sampling_and_microbial_data/MIX MICROBE dis.csv'
data_microbe = pd.read_csv(microbe_path, sep=',', header=0, lineterminator='\n', skip_blank_lines=True)
Y = np.array(data_microbe.iloc[0:17, 1:])
y_micro = Y[:, 1:]
y_ferti = Y[:, 0].astype(float)
label_names = ['B1', 'B2', 'B3', 'B4']
default_colors = distinctipy.get_colors(15)

# X = np.log1p(spectral_data) 

# Settings for the smooth derivatives using a Savitsky-Golay filter
w = 37 ## Sav.Gol window size
p = 2  ## Sav.Gol polynomial degree
d=1 ## Sav.Gol derive  order

X = savgol_filter(spectral_data, window_length=w, polyorder=p,deriv=d, axis=1)
X= detrend(X, axis=1)


n_comp = 6
pca = PCA(n_components=n_comp)
pca_scores = pca.fit_transform(X)
pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)


# === 1. Loading plots ===
plt.figure(figsize=(12, 6))
for i in range(pca_loadings.shape[1]):
    plt.plot(wv,pca_loadings[:, i], label=f'PC{i+1}',color=default_colors[i])

plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("Principal Component Loadings")
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., title="PC's")
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(figure_root, 'pca_spectral_loadings.png'), dpi=900)  # 
plt.close()

fig, axes = plt.subplots(n_comp, 1, figsize=(12, 2.5 * n_comp), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(wv,pca_loadings[:, i],color=default_colors[i])
    ax.set_title(f'PC{i+1} Loadings')
    ax.set_ylabel('Intensity')
    ax.grid(True)
plt.tight_layout()
plt.xlabel("Wavelength (nm)")
# plt.show()
plt.savefig(os.path.join(figure_root, 'pca_spectral_loadings_subplot.png'), dpi=900)  # 
plt.close()



# === Colors and Fertilization Sizes ===

ferti_norm = (y_ferti - y_ferti.min()) / (y_ferti.max() - y_ferti.min() + 1e-6)
sizes = 30 + 300 * ferti_norm ** 1.5  # emphasize effect

unique_ferti = np.unique(y_ferti)
ferti_legend_elements = []
for val in unique_ferti:
    norm_val = (val - y_ferti.min()) / (y_ferti.max() - y_ferti.min() + 1e-6)
    size = 30 + 300 * norm_val ** 1.5
    ferti_legend_elements.append(Line2D([0], [0], marker='o', color='gray', label=f'Ferti: {val:.1f}',
        markerfacecolor='gray', markersize=np.sqrt(size), markeredgecolor='black', linewidth=0))




pc_pairs = [(0, 1), (2, 3), (4, 5)]

for i, j in pc_pairs:
    fig, ax = plt.subplots()
    ax.set_title(f'PC{i + 1} vs PC{j + 1}')
    for k, treatment in enumerate(np.unique(Treatments)):
        idx = Treatments == treatment
        for x, y, label, ferti_size in zip(pca_scores[idx, i], pca_scores[idx, j], Treatments[idx], sizes[idx]):
            ax.scatter(x, y, color=default_colors[k], s=ferti_size, edgecolor='black', alpha=0.85)
            ax.annotate(label,
                        xy=(x, y), xytext=(5, 2),  # offset in points (x, y)
                        textcoords='offset points',
                        fontsize=9)
            
    ax.set_xlabel(f"PC{i + 1} ({pca.explained_variance_ratio_[i]*100:.1f}%)")
    ax.set_ylabel(f"PC{j + 1} ({pca.explained_variance_ratio_[j]*100:.1f}%)")
    ax.grid(True)

    ax.legend(handles=ferti_legend_elements, title="Fertilization Size",
              bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
    plt.tight_layout()
    # plt.show()
    save_subfolder = os.path.join(two_d_folder, f"PC{i+1}_vs_PC{j+1}")
    os.makedirs(save_subfolder, exist_ok=True)
    plt.savefig(os.path.join(save_subfolder, f'PC_{i+1}_vs_PC_{j+1}.png'), dpi=900)  # Save as PNG with high DPI
    plt.close()


for label_idx, label_name in enumerate(label_names):
    presence = y_micro[:, label_idx].astype(int)
    colors = ['tab:red' if val == 1 else 'gray' for val in presence]
    
    label_folder_2d = os.path.join(microbe_2d_folder, label_name)
    os.makedirs(label_folder_2d, exist_ok=True)
    
    for pc_x, pc_y in pc_pairs:
        fig, ax = plt.subplots()
        for i in range(len(pca_scores)):
            ax.scatter(pca_scores[i, pc_x], pca_scores[i, pc_y], color=colors[i],
                        s=sizes[i], edgecolor='black', alpha=0.85)
            ax.annotate(f'T{i+1}',
                        xy=(pca_scores[i, pc_x], pca_scores[i, pc_y]), xytext=(5, 2),  # offset in points (x, y)
                        textcoords='offset points',
                        fontsize=9)
         
        ax.set_xlabel(f"PC{pc_x + 1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc_y + 1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)")
        ax.set_title(f"PC{pc_x + 1} vs PC{pc_y + 1} - {label_name} Presence\nPoint size = Fertilization level")
        ax.grid(True)
        ax.legend(handles=ferti_legend_elements, title="Fertilization Size",
                   bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1)
        plt.tight_layout()
        # plt.show()
        save_path = os.path.join(label_folder_2d, f'pca_microbe_{label_name}_PC{pc_x+1}_vs_PC{pc_y+1}.png')
        plt.savefig(save_path, dpi=900)
        plt.close()
        
        
triplets = [(0, 1, 2), (3, 4, 5)]
for pc1, pc2, pc3 in triplets:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k, treatment in enumerate(np.unique(Treatments)):
        idx = Treatments == treatment
        ax.scatter(pca_scores[idx, pc1], pca_scores[idx, pc2], pca_scores[idx, pc3],
                   label=treatment, color=default_colors[k], s=sizes[idx], edgecolor='black', alpha=0.85)
        xrange = pca_scores[:, pc1].ptp()
        yrange = pca_scores[:, pc2].ptp()
        zrange = pca_scores[:, pc3].ptp()

        offset_ratio = 0.015  # you can tune this value
        offset = np.array([offset_ratio * r for r in (xrange, yrange, zrange)])

        for x, y, z, label in zip(pca_scores[idx, pc1], pca_scores[idx, pc2], pca_scores[idx, pc3], Treatments[idx]):
            ax.text(x + offset[0], y + offset[1], z + offset[2], label,
                    fontsize=9, ha='left', va='bottom')

            
            
    ax.set_xlabel(f"PC{pc1 + 1} ({pca.explained_variance_ratio_[pc1]*100:.1f}%)")
    ax.set_ylabel(f"PC{pc2 + 1} ({pca.explained_variance_ratio_[pc2]*100:.1f}%)")
    ax.set_zlabel(f"PC{pc3 + 1} ({pca.explained_variance_ratio_[pc3]*100:.1f}%)")
    ax.set_title(f'3D PCA Plot: PC{pc1+1} vs PC{pc2+1} vs PC{pc3+1}')
    ferti_legend = ax.legend(handles=ferti_legend_elements, title="Fertilization Size",
                             bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
    ax.add_artist(ferti_legend)
    plt.tight_layout()
    # plt.show()
    save_subfolder = os.path.join(three_d_folder, f"PC{pc1+1}_vs_PC{pc2+1}_vs_PC{pc3+1}")
    os.makedirs(save_subfolder, exist_ok=True)
    plt.savefig(os.path.join(save_subfolder, f'3D_PC{pc1+1}_vs_PC{pc2+1}_vs_PC{pc3+1}.png'), dpi=900)  # Save as PNG with high DPI
    plt.close()
    
    
    
for label_idx, label_name in enumerate(label_names):
    presence = y_micro[:, label_idx].astype(int)
    colors = ['tab:red' if val == 1 else 'gray' for val in presence]
    label_folder_3d = os.path.join(microbe_3d_folder, label_name)
    os.makedirs(label_folder_3d, exist_ok=True)
    
    for pc1, pc2, pc3 in triplets:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(pca_scores)):
            ax.scatter(pca_scores[i, pc1], pca_scores[i, pc2], pca_scores[i, pc3],
                       color=colors[i], s=sizes[i], edgecolor='black', alpha=0.85)
           
           
            xrange = pca_scores[:, pc1].ptp()
            yrange = pca_scores[:, pc2].ptp()
            zrange = pca_scores[:, pc3].ptp()
            offset = np.array([0.015 * xrange, 0.015 * yrange, 0.015 * zrange])  # ~1% offset

            for i in range(len(pca_scores)):
                x, y, z = pca_scores[i, pc1], pca_scores[i, pc2], pca_scores[i, pc3]
                ax.scatter(x, y, z, color=colors[i], s=sizes[i], edgecolor='black', alpha=0.85)

                ax.text(x + offset[0], y + offset[1], z + offset[2], f'T{i+1}', fontsize=9, ha='left', va='bottom')
            
            
            
        ax.set_xlabel(f"PC{pc1 + 1} ({pca.explained_variance_ratio_[pc1]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc2 + 1} ({pca.explained_variance_ratio_[pc2]*100:.1f}%)")
        ax.set_zlabel(f"PC{pc3 + 1} ({pca.explained_variance_ratio_[pc3]*100:.1f}%)")
        ax.set_title(f"3D PCA: PC{pc1+1} vs PC{pc2+1} vs PC{pc3+1} - {label_name} Presence")
        ferti_legend = ax.legend(handles=ferti_legend_elements, title="Fertilization Size",
                                 bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
        ax.add_artist(ferti_legend)
        plt.tight_layout()
        # plt.show()
        save_path = os.path.join(label_folder_3d, f'pca_microbe_{label_name}_PC{pc1+1}_vs_PC{pc2+1}_vs_PC{pc3+1}.png')
        plt.savefig(save_path, dpi=900)
        plt.close()