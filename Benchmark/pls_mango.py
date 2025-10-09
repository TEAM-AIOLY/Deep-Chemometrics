import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

from src.utils.dataset_loader import DatasetLoader

from src.net.chemtools import PLS
from src.utils.misc import snv
from scipy.signal import savgol_filter
import scipy as sp



# dataset= {"data_path": "data/dataset/Mango/mango_dm_full_outlier_removed2.mat",
#     "dataset_type": "mango_new"}

# data = DatasetLoader.load(dataset)



data_path ="C:/00_aioly/GitHub/Deep-chemometrics/data/dataset/Mango/mango_splits.mat"
data = sp.io.loadmat(data_path)
print(data.keys())

X_cal = data["X_cal"]
Y_cal = data["y_cal"]
X_val = data["X_val"]
Y_val = data["y_val"]
X_test = data["X_test"]
Y_test = data["y_test"]
wavelengths = np.arange(435, 1050 + 3, 3)  



window_length = 7
polyorder = 2
deriv = 1


x_cal= savgol_filter(X_cal, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
x_val = savgol_filter(X_val, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
x_test = savgol_filter(X_test, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)


# # Calibration set
# plt.figure(figsize=(10,6))
# idx_cal = np.random.choice(x_cal.shape[0], 25, replace=False)
# for i in idx_cal:
#     plt.plot(wavelengths, x_cal[i], alpha=0.7)
# plt.title("Calibration set (25 examples)")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Preprocessed intensity")
# plt.show()

# # Validation set
# plt.figure(figsize=(10,6))
# idx_val = np.random.choice(x_val.shape[0], 25, replace=False)
# for i in idx_val:
#     plt.plot(wavelengths, x_val[i], alpha=0.7)
# plt.title("Validation set (25 examples)")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Preprocessed intensity")
# plt.show()

# # Test set
# plt.figure(figsize=(10,6))
# idx_test = np.random.choice(x_test.shape[0], 25, replace=False)
# for i in idx_test:
#     plt.plot(wavelengths, x_test[i], alpha=0.7)
# plt.title("Test set (25 examples)")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Preprocessed intensity")
# plt.show()


save_path = "C:/00_aioly/GitHub/Deep-chemometrics/Benchmark/Mango/PLS/"
os.makedirs(save_path, exist_ok=True)

nlv=50

pls = PLS(ncomp=nlv)
pls.fit(x_cal, Y_cal)

Rmsec = []
Rmsecv = []
Rmsep = []
R2 = []
for lv in range(nlv):
    y_pred_cal=pls.predict(x_cal, nlv=lv).numpy()
    y_pred_val =pls.predict(x_val, nlv=lv).numpy()
    
    rmsec = np.sqrt(np.mean((Y_cal - y_pred_cal) ** 2))
    rmsecv = np.sqrt(np.mean((Y_val - y_pred_val) ** 2))
    Rmsec.append(rmsec)
    Rmsecv.append(rmsecv)
    
    r2 = np.corrcoef(Y_cal.flatten(), y_pred_cal.flatten())[0, 1] ** 2
    R2.append(r2)
    
    

# --- RMSEC, RMSECV and R² vs LV ---
fig, ax1 = plt.subplots(figsize=(8, 6))

ln1 = ax1.plot(range(1, nlv + 1), Rmsec, label='RMSEC', marker='o', color='#1f77b4')
ln2 = ax1.plot(range(1, nlv + 1), Rmsecv, label='RMSECV', marker='o', color='#ff7f0e')
ax1.set_xlabel('Latent Variables')
ax1.set_ylabel('RMSE')
ax1.set_title('Training and Cross validation performances RMSE and R²')
ax1.grid()

ax2 = ax1.twinx()
ln3 = ax2.plot(range(1, nlv + 1), R2, label='R²', marker='o', color='#2ca02c', linestyle='--')
ax2.set_ylabel('R²', color='#2ca02c')
ax2.tick_params(axis='y', labelcolor='#2ca02c')
ax2.spines['right'].set_color('#2ca02c')

lines = ln1 + ln2 + ln3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)

plt.tight_layout()
plt.savefig(os.path.join(save_path, "RMSE_R2_vs_LV.pdf"))
plt.savefig(os.path.join(save_path, "RMSE_R2_vs_LV.png"), dpi=600)
plt.close()

opt_lv =30
y_pred_test = pls.predict(x_test, nlv=opt_lv).numpy()
rmsep = np.sqrt(np.mean((Y_test - y_pred_test) ** 2))
r2_test = np.corrcoef(Y_test.flatten(), y_pred_test.flatten())[0, 1] ** 2


min_val = min(np.min(Y_test), np.min(y_pred_test))
max_val = max(np.max(Y_test), np.max(y_pred_test))
padding = 0.05 * (max_val - min_val)
lims = [min_val - padding, max_val + padding]
# Scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(Y_test, y_pred_test, edgecolors='k', alpha=0.5)
ax.plot(lims, lims, 'r')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Observed Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Predicted vs Observed Values for DMC')
ax.grid()
ax.text(
    0.02, 0.98,
    f"R²: {r2_test:.2f}\nRMSEP: {rmsep:.3f}",
    transform=ax.transAxes, ha='left', va='top', fontsize=12,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
    color='red', fontweight='bold', fontfamily='serif'
)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "Scatter_Test.pdf"))
plt.savefig(os.path.join(save_path, "Scatter_Test.png"), dpi=600)
plt.close()

# Hexbin plot
fig, ax = plt.subplots(figsize=(8, 6))
hexbin = ax.hexbin(Y_test, y_pred_test, gridsize=50, cmap='viridis', mincnt=1)
cb = fig.colorbar(hexbin, ax=ax, orientation='vertical')
cb.set_label('Density')
ax.plot(lims, lims, 'k-')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Observed Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Predicted vs Observed Values for DMC')
ax.grid()
ax.text(
    0.02, 0.98,
    f"R²: {r2_test:.2f}\nRMSEP: {rmsep:.3f}",
    transform=ax.transAxes, ha='left', va='top', fontsize=12,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
    color='red', fontweight='bold', fontfamily='serif'
)
plt.tight_layout()
plt.savefig(os.path.join(save_path, "Hexbin_Test.pdf"))
plt.savefig(os.path.join(save_path, "Hexbin_Test.png"), dpi=600)
plt.close()