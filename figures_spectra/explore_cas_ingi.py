import os
import h5py
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from distinctipy import get_colors


import tensorly as tl
from tensorly.decomposition import parafac

# --- Paths ---
data_path = "C:/00_aioly/GitHub/Deep-chemometrics/figures_spectra/CAS_phI_trialII_Ivy.mat"
design_mat_path = "C:/00_aioly/GitHub/Deep-chemometrics/figures_spectra/design_mat_Ivy.mat"

# --- Result folder ---
result_folder = "PCA_results"
base_path = os.path.join(os.path.dirname(design_mat_path), result_folder)
os.makedirs(base_path, exist_ok=True)

# --- Load data ---
arrays = {}
f = h5py.File(data_path, "r")
for k, v in f.items():
    arrays[k] = np.array(v)
f.close()

design_mat = scipy.io.loadmat(design_mat_path)["design_mat"]
spectral_data = arrays["sp_CAS2_D2D6"][31:1452, :].T

metadata = pd.DataFrame(design_mat, columns=["variété","plant","date"])
ref = arrays["hplc_msort"].T
labels = pd.DataFrame(ref, columns=["variété","plant","date","mad","asiat","madAc","asiatAc","TTP"])

# --- Step 1: Raw spectra ---
fig = plt.figure()
plt.plot(spectral_data.T)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbance")
plt.title("Spectra of CAS")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "01_Spectra_of_CAS.pdf"))
plt.close(fig)

# --- Step 2: PCA ---
n_comp = 6
pca = PCA(n_components=n_comp)
pca.fit(spectral_data)
pca_scores = pca.transform(spectral_data)
pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# PCA Loadings
fig = plt.figure()
for i in range(pca_loadings.shape[1]):
    plt.plot(pca_loadings[:, i], label=f'PC{i+1}')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Loading Value')
plt.title('PCA Loadings')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_path, "02_PCA_Loadings.pdf"))
plt.close(fig)

# --- Step 3: PCA Scores by categorical factors ---
metadata["plant_unique"] = metadata.groupby(["variété", "plant"]).ngroup()
factors = ["variété", "plant_unique", "date"]
pc_pairs = [(i, i+1) for i in range(n_comp-1)]

for factor in factors:
    for pcx, pcy in pc_pairs:
        fig = plt.figure(figsize=(7,6))
        sns.scatterplot(
            x=pca_scores[:, pcx],
            y=pca_scores[:, pcy],
            hue=metadata[factor].astype(str),
            palette="tab10",
            alpha=0.8
        )
        plt.xlabel(f"PC{pcx+1} ({pca.explained_variance_ratio_[pcx]*100:.1f}%)")
        plt.ylabel(f"PC{pcy+1} ({pca.explained_variance_ratio_[pcy]*100:.1f}%)")
        title = f"PCA Scores_PC{pcx+1}_vs_PC{pcy+1}_by_{factor}"
        plt.title(title.replace("_", " "))
        plt.legend(title=factor, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f"03_{title}.pdf"))
        plt.close(fig)

# --- Step 4: Average PCA Scores ---
scores_df = pd.DataFrame(pca_scores, columns=[f"PC{i+1}" for i in range(n_comp)])
scores_df = pd.concat([scores_df, metadata[["variété","plant","date","plant_unique"]]], axis=1)

avg_scores = scores_df.groupby(["plant_unique", "date"]).mean().reset_index()
plant_info = metadata.drop_duplicates(subset=["plant_unique", "date"]).set_index(["plant_unique", "date"])
avg_scores = avg_scores.join(
    plant_info[["variété"]].rename(columns={"variété": "variété_info"}),
    on=["plant_unique", "date"]
)

for factor in factors:
    unique_vals = avg_scores[factor].unique()
    colors = get_colors(len(unique_vals))
    palette = {str(val): '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
               for val, (r,g,b) in zip(unique_vals, colors)}
    
    for pcx, pcy in pc_pairs:
        fig = plt.figure(figsize=(7,6))
        sns.scatterplot(
            x=avg_scores[f"PC{pcx+1}"],
            y=avg_scores[f"PC{pcy+1}"],
            hue=avg_scores[factor].astype(str),
            palette=palette,
            s=150,
            alpha=0.8
        )
        plt.xlabel(f"PC{pcx+1} ({pca.explained_variance_ratio_[pcx]*100:.1f}%)")
        plt.ylabel(f"PC{pcy+1} ({pca.explained_variance_ratio_[pcy]*100:.1f}%)")
        title = f"Averaged_PCA_Scores_PC{pcx+1}_vs_PC{pcy+1}_by_{factor}"
        plt.title(title.replace("_", " "))
        plt.legend(title=factor, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f"04_{title}.pdf"))
        plt.close(fig)

# --- Step 5: PCA Scores colored by HPLC numeric labels ---
metadata_aligned = metadata.copy()
metadata_aligned["variété"] = metadata_aligned["variété"].astype(int)
metadata_aligned["plant"] = metadata_aligned["plant"].astype(int) + 1
metadata_aligned["date"] = metadata_aligned["date"].astype(int)

labels_fixed = labels.copy()
labels_fixed["variété"] = labels_fixed["variété"].astype(int)
labels_fixed["plant"]   = labels_fixed["plant"].astype(int)
labels_fixed["date"]    = labels_fixed["date"].astype(int)

replicated_hplc = metadata_aligned.merge(
    labels_fixed,
    on=["variété","plant","date"],
    how="left"
)[["mad","asiat","madAc","asiatAc","TTP"]]

numeric_labels = ["mad","asiat","madAc","asiatAc","TTP"]

for factor in numeric_labels:
    for pcx, pcy in pc_pairs:
        fig = plt.figure(figsize=(7,6))
        scatter = plt.scatter(
            pca_scores[:, pcx],
            pca_scores[:, pcy],
            c=replicated_hplc[factor],
            cmap="viridis",
            alpha=0.8
        )
        plt.xlabel(f"PC{pcx+1} ({pca.explained_variance_ratio_[pcx]*100:.1f}%)")
        plt.ylabel(f"PC{pcy+1} ({pca.explained_variance_ratio_[pcy]*100:.1f}%)")
        title = f"PCA_Scores_PC{pcx+1}_vs_PC{pcy+1}_colored_by_{factor}"
        plt.title(title.replace("_", " "))
        plt.colorbar(scatter, label=factor)
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f"05_{title}.pdf"))
        plt.close(fig)


