import os
import sys
import scipy as sp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
from src.utils.dataset_loader import DatasetLoader

root = os.getcwd()
data_path= root+"/data/dataset/Mango/mango_dm_full_outlier_removed2.mat" 
data = sp.io.loadmat(data_path)

out_dir = "figures_spectra"
os.makedirs(out_dir, exist_ok=True)


Ycal = data["DM_cal"]
Ytest = data["DM_test"]
Xcal = data["SP_all_train"]
Xtest = data["SP_all_test"]
wv=data['wave'].astype(np.float32).reshape(-1,1)


n_samples, n_features_total = Xcal.shape
n_modalities = 6
features_per_modality = n_features_total // n_modalities

assert features_per_modality == len(wv), f"{features_per_modality} != {len(wv)}"

modalities = [
    Xcal[:, i*features_per_modality:(i+1)*features_per_modality]
    for i in range(n_modalities)
]

names = [
    "Raw", 
    "SNV", 
    "SavgGol1", 
    "Savgol2", 
    "SNV + SavGol1", 
    "SNV + SavGol2"
]

# Plot
n_to_plot = 50  # number of random spectra per modality
cmap = plt.cm.get_cmap("PuOr", n_to_plot)
for mod, name in zip(modalities, names):
    idx = np.random.choice(mod.shape[0], n_to_plot, replace=False)  
    mod_subset = mod[idx, :]

    plt.figure(figsize=(8, 5))
    for i, spectrum in enumerate(mod_subset):
        plt.plot(wv, spectrum, alpha=0.8, 
                 color=cmap(i)) 

    plt.title(f"{name} spectra (subset of {n_to_plot})")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance (a.u.)")
    plt.tight_layout()
    plt.show(block=False);
    save_path = os.path.join(out_dir, f"{name.replace(' ', '_')}_spectra.pdf")
    plt.savefig(save_path, format="pdf")
    # plt.close()
    
plt.figure(figsize=(7,5))

plt.hist(Ycal.ravel(), bins=100, alpha=0.6, color="tab:blue", label="Calibration")
plt.hist(Ytest.ravel(), bins=100, alpha=0.6, color="tab:orange", label="Test")

plt.xlabel("Dry Matter (%)")
plt.ylabel("Frequency")
plt.title("Distribution of Dry Matter (Calibration vs Test)")
plt.legend()
plt.tight_layout()
plt.show();
save_path = os.path.join(out_dir, "dry_matter_distribution.pdf")
plt.savefig(save_path, format="pdf")
# plt.close()