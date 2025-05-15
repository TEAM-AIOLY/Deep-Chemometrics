import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import distinctipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import random

random.seed(42)

# === Load and preprocess data ===
data_path = './data/dataset/3rd_data_sampling_and_microbial_data/y_bio.csv'

    
    
data = pd.read_csv(data_path, sep=';', header=0, lineterminator='\n', skip_blank_lines=True)
Treatments = np.array([f"T{i}" for i in range(1, 16)])  # Generate T1 to T15

headers = ['Girth_final','Girth_increment','height_final','height_increment','Nb_leaves','roor_length',
           'Fresh_weight_leaves','Fresh_weight_stem','Fresh_weight_root','Total_fresh_weight',
           'Dry_weight_leaves','Dry_weight_stem','Dry_weight_root','Total_dry_weight']

X = data.iloc[1:17, 1:]
X = X.map(lambda x: str(x).replace(',', '.'))
X = np.array(X).astype(float)
X = StandardScaler().fit_transform(X)

microbe_path = './data/dataset/3rd_data_sampling_and_microbial_data/MIX MICROBE dis.csv'
data_microbe = pd.read_csv(microbe_path, sep=',', header=0, lineterminator='\n', skip_blank_lines=True)
Y = np.array(data_microbe.iloc[0:17, 1:])
y_micro = Y[:, 1:]
y_ferti = Y[:, 0].astype(float)
label_names = ['B1', 'B2', 'B3', 'B4']


figure_root = os.path.join(os.path.dirname(data_path), 'new_figures')
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


# === PCA ===
n_comp = 6
pca = PCA(n_components=n_comp)
pca_scores = pca.fit_transform(X)
pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# === Colors and Fertilization Sizes ===
default_colors = distinctipy.get_colors(15)
ferti_norm = (y_ferti - y_ferti.min()) / (y_ferti.max() - y_ferti.min() + 1e-6)
sizes = 30 + 300 * ferti_norm ** 1.5  # emphasize effect

# === 1. Loading plots ===
plt.figure(figsize=(12, 6))
for i in range(pca_loadings.shape[1]):
    plt.plot(pca_loadings[:, i], marker='o', label=f'PC{i+1}', color=default_colors[i])
plt.xticks(np.arange(len(headers)), labels=headers, rotation=45, ha='right')
plt.xlabel("Variables")
plt.ylabel("Magnitude")
plt.title("Principal Component Loadings")
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., title="PC's")
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(figure_root, 'pca_loadings.png'), dpi=900)  # Save as PNG with high DPI
plt.close()

fig, axes = plt.subplots(n_comp, 1, figsize=(12, 2.5 * n_comp), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(pca_loadings[:, i], marker='o', color=default_colors[i])
    ax.set_title(f'PC{i+1} Loadings')
    ax.set_ylabel('Magnitude')
    ax.grid(True)
axes[-1].set_xticks(np.arange(len(headers)))
axes[-1].set_xticklabels(headers, rotation=45, ha='right')
axes[-1].set_xlabel("Variables")
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(figure_root, 'pca_loadings_subplot.png'), dpi=900)  # Save as PNG with high DPI
plt.close()


# ===   Prepare Fertilization Size Legend ===
unique_ferti = np.unique(y_ferti)
ferti_legend_elements = []
for val in unique_ferti:
    norm_val = (val - y_ferti.min()) / (y_ferti.max() - y_ferti.min() + 1e-6)
    size = 30 + 300 * norm_val ** 1.5
    ferti_legend_elements.append(Line2D([0], [0], marker='o', color='gray', label=f'Ferti: {val:.1f}',
        markerfacecolor='gray', markersize=np.sqrt(size), markeredgecolor='black', linewidth=0))

# === 2. 2D PCA Plots by PC Pairs ===
pc_pairs = [(0, 1), (2, 3), (4, 5)]

for i, j in pc_pairs:
    plt.figure(figsize=(8, 6))
    plt.title(f'PC{i + 1} vs PC{j + 1}')
    for k, treatment in enumerate(np.unique(Treatments)):
        idx = Treatments == treatment
        for x, y, label, ferti_size in zip(pca_scores[idx, i], pca_scores[idx, j], Treatments[idx], sizes[idx]):
            plt.scatter(x, y, color=default_colors[k], s=ferti_size, edgecolor='black', alpha=0.85)
            plt.text(x + 0.1, y, label, fontsize=9)
    plt.xlabel(f"PC{i + 1} ({pca.explained_variance_ratio_[i]*100:.1f}%)")
    plt.ylabel(f"PC{j + 1} ({pca.explained_variance_ratio_[j]*100:.1f}%)")
    plt.grid(True)
    plt.legend(handles=ferti_legend_elements, title="Fertilization Size",
           bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
    plt.tight_layout()
    # plt.show()
    save_subfolder = os.path.join(two_d_folder, f"PC{i+1}_vs_PC{j+1}")
    os.makedirs(save_subfolder, exist_ok=True)
    plt.savefig(os.path.join(save_subfolder, f'PC_{i+1}_vs_PC_{j+1}.png'), dpi=900)  # Save as PNG with high DPI
    plt.close()

# === 3. 2D PCA with Microbe Labels ===
# for label_idx, label_name in enumerate(label_names):
#     presence = y_micro[:, label_idx].astype(int)
#     colors = ['tab:red' if val == 1 else 'gray' for val in presence]

#     for pc_x, pc_y in pc_pairs:
#         plt.figure(figsize=(8, 6))
#         for i in range(len(pca_scores)):
#             plt.scatter(pca_scores[i, pc_x], pca_scores[i, pc_y], color=colors[i],
#                         s=sizes[i], edgecolor='black', alpha=0.85)
#             plt.text(pca_scores[i, pc_x] + 0.1, pca_scores[i, pc_y], f'T{i+1}', fontsize=7)
#         plt.xlabel(f"PC{pc_x + 1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)")
#         plt.ylabel(f"PC{pc_y + 1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)")
#         plt.title(f"PC{pc_x + 1} vs PC{pc_y + 1} - {label_name} Presence\nPoint size = Fertilization level")
#         plt.grid(True)
#         plt.legend(handles=ferti_legend_elements, title="Fertilization Size",
#            bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1)
#         plt.tight_layout()
#         # plt.show()
#         save_path_2d_microbe = os.path.join(microbe_2d_folder, 'pca_microbe_labels.png')
#         plt.tight_layout()
#         plt.savefig(save_path_2d_microbe, dpi=900)  # Save as PNG with high DPI
#         plt.close()

for label_idx, label_name in enumerate(label_names):
    presence = y_micro[:, label_idx].astype(int)
    colors = ['tab:red' if val == 1 else 'gray' for val in presence]
    
    label_folder_2d = os.path.join(microbe_2d_folder, label_name)
    os.makedirs(label_folder_2d, exist_ok=True)

    for pc_x, pc_y in pc_pairs:
        plt.figure(figsize=(8, 6))
        for i in range(len(pca_scores)):
            plt.scatter(pca_scores[i, pc_x], pca_scores[i, pc_y], color=colors[i],
                        s=sizes[i], edgecolor='black', alpha=0.85)
            plt.text(pca_scores[i, pc_x] + 0.1, pca_scores[i, pc_y], f'T{i+1}', fontsize=7)
        plt.xlabel(f"PC{pc_x + 1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)")
        plt.ylabel(f"PC{pc_y + 1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)")
        plt.title(f"PC{pc_x + 1} vs PC{pc_y + 1} - {label_name} Presence\nPoint size = Fertilization level")
        plt.grid(True)
        plt.legend(handles=ferti_legend_elements, title="Fertilization Size",
                   bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1)
        plt.tight_layout()
        save_path = os.path.join(label_folder_2d, f'pca_microbe_{label_name}_PC{pc_x+1}_vs_PC{pc_y+1}.png')
        plt.savefig(save_path, dpi=900)
        plt.close()

# === 5. 3D PCA Plot ===
triplets = [(0, 1, 2), (3, 4, 5)]
for pc1, pc2, pc3 in triplets:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for k, treatment in enumerate(np.unique(Treatments)):
        idx = Treatments == treatment
        ax.scatter(pca_scores[idx, pc1], pca_scores[idx, pc2], pca_scores[idx, pc3],
                   label=treatment, color=default_colors[k], s=sizes[idx], edgecolor='black', alpha=0.85)
        for x, y, z, label in zip(pca_scores[idx, pc1], pca_scores[idx, pc2], pca_scores[idx, pc3], Treatments[idx]):
            ax.text(x, y, z + 0.2, label, fontsize=9)
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
    
# === 6. 3D PCA with Microbe Labels ===
# for label_idx, label_name in enumerate(label_names):
#     presence = y_micro[:, label_idx].astype(int)
#     colors = ['tab:red' if val == 1 else 'gray' for val in presence]

#     for pc1, pc2, pc3 in triplets:
#         fig = plt.figure(figsize=(10, 8))
#         ax = fig.add_subplot(111, projection='3d')
#         for i in range(len(pca_scores)):
#             ax.scatter(pca_scores[i, pc1], pca_scores[i, pc2], pca_scores[i, pc3],
#                        color=colors[i], s=sizes[i], edgecolor='black', alpha=0.85)
#             ax.text(pca_scores[i, pc1], pca_scores[i, pc2], pca_scores[i, pc3] + 0.2,
#                     f'T{i+1}', fontsize=7)
#         ax.set_xlabel(f"PC{pc1 + 1} ({pca.explained_variance_ratio_[pc1]*100:.1f}%)")
#         ax.set_ylabel(f"PC{pc2 + 1} ({pca.explained_variance_ratio_[pc2]*100:.1f}%)")
#         ax.set_zlabel(f"PC{pc3 + 1} ({pca.explained_variance_ratio_[pc3]*100:.1f}%)")
#         ax.set_title(f'3D PCA - {label_name} Presence\nPoint size = Fertilization')
#         ax.legend(handles=ferti_legend_elements, title="Fertilization Size",
#                   bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
#         plt.tight_layout()
#         # plt.show()
#         save_path_3d_microbe = os.path.join(microbe_3d_folder, 'pca_microbe_labels_3d.png')
#         plt.tight_layout()
#         plt.savefig(save_path_3d_microbe, dpi=900)  # Save as PNG with high DPI
#         plt.close()
        
for label_idx, label_name in enumerate(label_names):
    presence = y_micro[:, label_idx].astype(int)
    colors = ['tab:red' if val == 1 else 'gray' for val in presence]
    label_folder_3d = os.path.join(microbe_3d_folder, label_name)
    os.makedirs(label_folder_3d, exist_ok=True)

    for pc1, pc2, pc3 in triplets:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(pca_scores)):
            ax.scatter(pca_scores[i, pc1], pca_scores[i, pc2], pca_scores[i, pc3],
                       color=colors[i], s=sizes[i], edgecolor='black', alpha=0.85)
            ax.text(pca_scores[i, pc1], pca_scores[i, pc2], pca_scores[i, pc3] + 0.2, f'T{i+1}', fontsize=8)
        ax.set_xlabel(f"PC{pc1 + 1} ({pca.explained_variance_ratio_[pc1]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc2 + 1} ({pca.explained_variance_ratio_[pc2]*100:.1f}%)")
        ax.set_zlabel(f"PC{pc3 + 1} ({pca.explained_variance_ratio_[pc3]*100:.1f}%)")
        ax.set_title(f"3D PCA: PC{pc1+1} vs PC{pc2+1} vs PC{pc3+1} - {label_name} Presence")
        ferti_legend = ax.legend(handles=ferti_legend_elements, title="Fertilization Size",
                                 bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
        ax.add_artist(ferti_legend)
        plt.tight_layout()
        save_path = os.path.join(label_folder_3d, f'pca_microbe_{label_name}_PC{pc1+1}_vs_PC{pc2+1}_vs_PC{pc3+1}.png')
        plt.savefig(save_path, dpi=900)
        plt.close()

triplets = [(0, 1, 2), (3, 4, 5)]
for pc1, pc2, pc3 in triplets:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot samples
    for k, treatment in enumerate(np.unique(Treatments)):
        idx = Treatments == treatment
        ax.scatter(pca_scores[idx, pc1], pca_scores[idx, pc2], pca_scores[idx, pc3],
                   label=treatment, color=default_colors[k], s=sizes[idx], edgecolor='black', alpha=0.85)
        for x, y, z, label in zip(pca_scores[idx, pc1], pca_scores[idx, pc2], pca_scores[idx, pc3], Treatments[idx]):
            ax.text(x, y, z + 0.2, label, fontsize=9)
    # Plot variable vectors (arrows) - Scale arrows by 30 for visibility
    arrow_scale = 30.0  # Scaling factor for arrows
    text_offset = 3.5  # Offset for the text from the tip of the arrows

    for i, (var, x, y, z) in enumerate(zip(headers, pca_loadings[:, pc1], pca_loadings[:, pc2], pca_loadings[:, pc3])):
        # Plot arrow using quiver (scaled by a factor for visibility)
        ax.quiver(0, 0, 0, x*arrow_scale, y*arrow_scale, z*arrow_scale, color='black', alpha=0.6, length=0.1, normalize=False)
        
        # Place text at the tip of the arrow, slightly offset to avoid overlap
        ax.text(x*text_offset , y*text_offset , z*text_offset , var, 
                ha='center', va='center', fontsize=8, color='black')

    ax.set_xlabel(f"PC{pc1 + 1} ({pca.explained_variance_ratio_[pc1]*100:.1f}%)")
    ax.set_ylabel(f"PC{pc2 + 1} ({pca.explained_variance_ratio_[pc2]*100:.1f}%)")
    ax.set_zlabel(f"PC{pc3 + 1} ({pca.explained_variance_ratio_[pc3]*100:.1f}%)")
    ax.set_title(f'3D PCA Plot: PC{pc1+1} vs PC{pc2+1} vs PC{pc3+1}')

    ferti_legend = ax.legend(handles=ferti_legend_elements, title="Fertilization Size",
                             bbox_to_anchor=(1.05, 1), loc='upper left', handletextpad=1.2, labelspacing=1.0)
    ax.add_artist(ferti_legend)

    plt.tight_layout()
    # plt.show()
    save_path_3d_arrows = os.path.join(three_d_folder, 'pca_with_arrows.png')
    plt.tight_layout()
    plt.savefig(save_path_3d_arrows, dpi=900)  # Save as PNG with high DPI
    plt.close()
  
        
    