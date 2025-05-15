import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import distinctipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

data_path = './data/dataset/3rd_data_sampling_and_microbial_data/y_bio.csv'
data = pd.read_csv(data_path, sep=';', header=0, lineterminator='\n', skip_blank_lines=True)
Treatments = np.array([f"T{i}" for i in range(1, 16)])  # Generate T1 to T15
headers = ['Girth_final','Girth_increment','height_final','height_increment','Nb_leaves','roor_length', 'Fresh_weight_leaves',
           'Fresh_weight_stem','Fresh_weight_root','Total_fresh_weight', 'Dry_weight_leaves','Dry_weight_stem','Dry_weight_root','Total_dry_weight']


X =data.iloc[1:17,1:]
X = X.map(lambda x: str(x).replace(',', '.'))
X=np.array(X).astype(float)
X = StandardScaler().fit_transform(X)


microbe_path  ='./data/dataset/3rd_data_sampling_and_microbial_data/MIX MICROBE dis.csv'
data_microbe = pd.read_csv(microbe_path, sep=',', header=0, lineterminator='\n', skip_blank_lines=True)
Y=np.array(data_microbe.iloc[0:17,1:])

y_micro =Y[:,1:]
y_ferti =Y[:,0]
label_names = ['B1', 'B2', 'B3', 'B4']

n_comp= 6

pca =PCA(n_components=n_comp)
pca.fit(X)
pca_scores = pca.transform(X)
pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)


default_colors = distinctipy.get_colors(15)

plt.figure(figsize=(12, 6))
for i in range(pca_loadings.shape[1]):
    plt.plot(pca_loadings[:, i], marker='o', label=f'PC{i+1}', color=default_colors[i])

# Annotate x-axis with variable names
plt.xticks(ticks=np.arange(len(headers)), labels=headers, rotation=45, ha='right')
plt.xlabel("Variables")
plt.ylabel("Magnitude")
plt.title("Principal Component Loadings")
plt.grid(True)
# Put the legend on the right as a vertical column
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., title="PC's")
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend
plt.show()


fig, axes = plt.subplots(n_comp, 1, figsize=(12, 2.5 * n_comp), sharex=True)
for i in range(n_comp):
    ax = axes[i]
    ax.plot(pca_loadings[:, i], marker='o', color=default_colors[i])
    ax.set_title(f'PC{i+1} Loadings')
    ax.set_ylabel('Magnitude')
    ax.grid(True)

# Set shared x-axis labels
axes[-1].set_xticks(np.arange(len(headers)))
axes[-1].set_xticklabels(headers, rotation=45, ha='right')
axes[-1].set_xlabel("Variables")

plt.tight_layout()
plt.show()
    
 

pc_pairs = [(0, 1), (2, 3), (4, 5)]  # PC1 vs PC2, PC3 vs PC4, PC5 vs PC6

for i in range(0, n_comp, 2):  # Step by 2 (PC1 vs PC2, PC3 vs PC4, etc.)
    j = i + 1  # Get the next component for pairing (PC2, PC4, etc.)
    
    # Create a new figure for each pair
    plt.figure(figsize=(8, 6))
    plt.title(f'PC{i + 1} vs PC{j + 1}')
    
    # Normalize and exaggerate fertilization size effect
    ferti_norm = (y_ferti.astype(float) - y_ferti.min()) / (y_ferti.max() - y_ferti.min() + 1e-6)
    sizes = 30 + 300 * ferti_norm**1.5  # Exaggerate with power scaling
    
    # Scatter plot for each treatment with the corresponding color
    for k, treatment in enumerate(np.unique(Treatments)):
        idx = Treatments == treatment
        for x, y, label, ferti_size in zip(pca_scores[idx, i], pca_scores[idx, j], Treatments[idx], sizes[idx]):
            plt.scatter(x, y, color=default_colors[k], s=ferti_size, edgecolor='black', alpha=0.85)
            plt.text(x + 0.1, y, label, fontsize=9)

    # Label the axes with the explained variance ratio
    plt.xlabel(f"PC{i + 1} ({pca.explained_variance_ratio_[i] * 100:.1f}%)")
    plt.ylabel(f"PC{j + 1} ({pca.explained_variance_ratio_[j] * 100:.1f}%)")
    plt.grid(True)

    # Add the legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1)

    # Show the plot
    plt.tight_layout()
    plt.show()

for label_idx, label_name in enumerate(label_names):
    binary_labels = y_micro[:, label_idx].astype(int)
    
    for pc_x, pc_y in pc_pairs:
        presence = y_micro[:, label_idx]
        colors = ['gray' if val == 0 else 'tab:red' for val in presence]

        # Normalize and exaggerate fertilization size effect
        ferti_norm = (y_ferti.astype(float) - y_ferti.min()) / (y_ferti.max() - y_ferti.min() + 1e-6)
        sizes = 30 + 300 * ferti_norm**1.5  # Exaggerate with power scaling

        # Create a new figure for each pair of components
        plt.figure(figsize=(8, 6))

        # Scatter plot
        for i in range(len(pca_scores)):
            plt.scatter(
                pca_scores[i, pc_x],
                pca_scores[i, pc_y],
                color=colors[i],
                s=sizes[i],
                edgecolor='black',
                alpha=0.85
            )
            plt.text(
                pca_scores[i, pc_x] + 0.1,
                pca_scores[i, pc_y],
                f'T{i+1}',
                fontsize=7
            )

        # Label the axes with the explained variance ratio
        plt.xlabel(f"PC{pc_x + 1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)")
        plt.ylabel(f"PC{pc_y + 1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)")
        plt.title(f"PC{pc_x + 1} vs PC{pc_y + 1} - {label_names[label_idx]} Presence\nPoint size = Fertilization level")
        plt.grid(True)
        plt.tight_layout()

        # Show the plot
        plt.show()
        
        
ferti_norm = (y_ferti.astype(float) - y_ferti.min()) / (y_ferti.max() - y_ferti.min() + 1e-6)
sizes = 30 + 300 * ferti_norm**1.5

# Prepare fertilization size legend using Line2D
unique_ferti = np.unique(y_ferti)
ferti_legend_elements = []
for val in unique_ferti:
    norm_val = (val - y_ferti.min()) / (y_ferti.max() - y_ferti.min() + 1e-6)
    size = 30 + 300 * norm_val**1.5
    ferti_legend_elements.append(
        Line2D([0], [0], marker='o', color='gray', label=f'Ferti: {val}',
               markerfacecolor='gray', markersize=np.sqrt(size), markeredgecolor='black', linewidth=0)
    )

triplets = [(0, 1, 2), (3, 4, 5)]  # PC1–3 and PC4–6

for triplet in triplets:
    pc1, pc2, pc3 = triplet
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for k, treatment in enumerate(np.unique(Treatments)):
        idx = Treatments == treatment
        xs, ys, zs = pca_scores[idx, pc1], pca_scores[idx, pc2], pca_scores[idx, pc3]
        ax.scatter(xs, ys, zs, label=treatment, color=default_colors[k], s=sizes[idx], edgecolor='black', alpha=0.85)

        for x, y, z, label in zip(xs, ys, zs, Treatments[idx]):
            ax.text(x, y, z + 0.2, label, fontsize=9)

    ax.set_xlabel(f"PC{pc1 + 1} ({pca.explained_variance_ratio_[pc1] * 100:.1f}%)")
    ax.set_ylabel(f"PC{pc2 + 1} ({pca.explained_variance_ratio_[pc2] * 100:.1f}%)")
    ax.set_zlabel(f"PC{pc3 + 1} ({pca.explained_variance_ratio_[pc3] * 100:.1f}%)")
    ax.set_title(f'3D PCA Plot: PC{pc1 + 1} vs PC{pc2 + 1} vs PC{pc3 + 1}')

    # Only fertilization size legend
    ferti_legend = ax.legend(
    handles=ferti_legend_elements,
    title="Fertilization Size",
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    handletextpad=1.2,     # space between handle and text
    labelspacing=1.0       # vertical space between entries
)
    ax.add_artist(ferti_legend)

    plt.tight_layout()
    plt.show()
    
    
for label_idx, label_name in enumerate(label_names):
    presence = y_micro[:, label_idx].astype(int)
    colors = ['tab:red' if val == 1 else 'gray' for val in presence]

    for triplet in triplets:
        pc1, pc2, pc3 = triplet
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(pca_scores)):
            ax.scatter(
                pca_scores[i, pc1],
                pca_scores[i, pc2],
                pca_scores[i, pc3],
                color=colors[i],
                s=sizes[i],
                edgecolor='black',
                alpha=0.85
            )
            ax.text(
                pca_scores[i, pc1],
                pca_scores[i, pc2],
                pca_scores[i, pc3] + 0.2,
                f'T{i + 1}',
                fontsize=8
            )

        ax.set_xlabel(f"PC{pc1 + 1} ({pca.explained_variance_ratio_[pc1] * 100:.1f}%)")
        ax.set_ylabel(f"PC{pc2 + 1} ({pca.explained_variance_ratio_[pc2] * 100:.1f}%)")
        ax.set_zlabel(f"PC{pc3 + 1} ({pca.explained_variance_ratio_[pc3] * 100:.1f}%)")
        ax.set_title(f'3D PCA: PC{pc1 + 1} vs PC{pc2 + 1} vs PC{pc3 + 1}\n{label_name} Presence – Size = Fertilization')

        # Only fertilization size legend
        ferti_legend = ax.legend(
            handles=ferti_legend_elements,
            title="Fertilization Size",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            handletextpad=1.2,
            labelspacing=0.7
        )
        ax.add_artist(ferti_legend)

        plt.tight_layout()
        plt.show()

















############################
   
    
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# for k, treatment in enumerate(np.unique(Treatments)):
#     idx = Treatments == treatment
#     xs, ys, zs = pca_scores[idx, 0], pca_scores[idx, 1], pca_scores[idx, 2]
#     ax.scatter(xs, ys, zs, label=treatment, color=default_colors[k], s=60)
    
#     # Add text labels above each point
#     for x, y, z, label in zip(xs, ys, zs, Treatments[idx]):
#         ax.text(x, y, z + 0.2, label, fontsize=9)

# ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
# ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
# ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)")
# ax.set_title('3D Plot of PC1, PC2, and PC3')
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

# plt.tight_layout()
# plt.show()


# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# for k, treatment in enumerate(np.unique(Treatments)):
#     idx = Treatments == treatment
#     xs, ys, zs = pca_scores[idx, 3], pca_scores[idx, 4], pca_scores[idx, 5]
#     ax.scatter(xs, ys, zs, label=treatment, color=default_colors[k], s=60)
    
#     # Add text labels above each point
#     for x, y, z, label in zip(xs, ys, zs, Treatments[idx]):
#         ax.text(x, y, z + 0.2, label, fontsize=9)

# ax.set_xlabel(f"PC4 ({pca.explained_variance_ratio_[3] * 100:.1f}%)")
# ax.set_ylabel(f"PC5 ({pca.explained_variance_ratio_[4] * 100:.1f}%)")
# ax.set_zlabel(f"PC6 ({pca.explained_variance_ratio_[5] * 100:.1f}%)")
# ax.set_title('3D Plot of PC4, PC5, and PC6')
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

# plt.tight_layout()
# plt.show()