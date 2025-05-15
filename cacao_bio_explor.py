import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import distinctipy
from mpl_toolkits.mplot3d import Axes3D

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
X = StandardScaler().fit_transform(X)



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


# plt.figure(figsize=(8, 6))

# for i, treatment in enumerate(np.unique(Treatments)):
#     idx = Treatments == treatment
#     plt.scatter(pca_scores[idx, 0], pca_scores[idx, 1], label=treatment,color=default_colors[i])

# plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
# plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
# plt.title("PCA Scores by Treatment")
# plt.grid(True)
# Legend as a vertical column on the right
# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1)
# plt.tight_layout()
# plt.show()



for i in range(0, n_comp, 2):  # Step by 2 (PC1 vs PC2, PC3 vs PC4, etc.)
    j = i + 1  # Get the next component for pairing (PC2, PC4, etc.)
    
    # Create a new figure for each pair
    plt.figure(figsize=(8, 6))
    plt.title(f'PC{i + 1} vs PC{j + 1}')
    
    # Scatter plot for each treatment with the corresponding color
    for k, treatment in enumerate(np.unique(Treatments)):
        idx = Treatments == treatment
        plt.scatter(pca_scores[idx, i], pca_scores[idx, j], label=treatment, color=default_colors[k])

    # Label the axes with the explained variance ratio
    plt.xlabel(f"PC{i + 1} ({pca.explained_variance_ratio_[i] * 100:.1f}%)")
    plt.ylabel(f"PC{j + 1} ({pca.explained_variance_ratio_[j] * 100:.1f}%)")
    plt.grid(True)

    # Add the legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1)

    # Show the plot
    plt.tight_layout()
    plt.show()
    
    
    
    
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each treatment with the corresponding color
for k, treatment in enumerate(np.unique(Treatments)):
    idx = Treatments == treatment
    ax.scatter(pca_scores[idx, 0], pca_scores[idx, 1], pca_scores[idx, 2], 
               label=treatment, color=default_colors[k], s=60)

# Labels for the axes
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)")

# Add title
ax.set_title('3D Plot of PC1, PC2, and PC3')

# Add legend
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

# Show the plot
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each treatment with the corresponding color
for k, treatment in enumerate(np.unique(Treatments)):
    idx = Treatments == treatment
    ax.scatter(pca_scores[idx, 3], pca_scores[idx, 4], pca_scores[idx, 5], 
               label=treatment, color=default_colors[k], s=60)

# Labels for the axes
ax.set_xlabel(f"PC4 ({pca.explained_variance_ratio_[3] * 100:.1f}%)")
ax.set_ylabel(f"PC5 ({pca.explained_variance_ratio_[4] * 100:.1f}%)")
ax.set_zlabel(f"PC6 ({pca.explained_variance_ratio_[5] * 100:.1f}%)")

# Add title
ax.set_title('3D Plot of PC4, PC5, and PC6')

# Add legend
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

# Show the plot
plt.tight_layout()
plt.show()