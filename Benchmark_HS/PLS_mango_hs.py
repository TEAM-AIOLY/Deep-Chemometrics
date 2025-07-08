import os
import sys
os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from src.utils.dataset_loader import DatasetLoader

from src.net.chemtools import PLS
from src.utils.misc import snv
from scipy.signal import savgol_filter

data_root = [{
    "data_path": "C:/00_aioly/GitHub/Deep-Chemometrics/data/dataset/mango/mango_dm_full_outlier_removed2.mat",
    "dataset_type": "mango"},        
{"data_path": "C:/00_aioly/GitHub/Deep-Chemometrics/data/dataset/ossl/ossl_database.mat",
    "dataset_type": "ossl"},
{"data_path": "C:/00_aioly/GitHub/Deep-Chemometrics/data/dataset/wheat/",
    "dataset_type": "wheat"}]    



dataset=data_root[0]
data = DatasetLoader.load(dataset)

x_cal = data["x_cal"]
y_cal = data["y_cal"]   
x_val = data["x_val"]
y_val = data["y_val"]
x_test = data["x_test"]
y_test = data["y_test"]



window_length = 7
polyorder = 2
deriv = 1


x_cal= savgol_filter(x_cal, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
x_val = savgol_filter(x_val, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
x_test = savgol_filter(x_test, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)



nlv=50

pls = PLS(ncomp=nlv)
pls.fit(x_cal, y_cal)

Rmsec = []
Rmsecv = []
Rmsep = []
R2 = []
for lv in range(nlv):
    y_pred_cal=pls.predict(x_cal, nlv=lv).numpy()
    y_pred_val =pls.predict(x_val, nlv=lv).numpy()
    
    rmsec = np.sqrt(np.mean((y_cal - y_pred_cal) ** 2))
    rmsecv = np.sqrt(np.mean((y_val - y_pred_val) ** 2))
    Rmsec.append(rmsec)
    Rmsecv.append(rmsecv)
    
    r2 = np.corrcoef(y_cal.flatten(), y_pred_cal.flatten())[0, 1] ** 2
    R2.append(r2)
    
    

fig, ax1 = plt.subplots(figsize=(8, 6))

# RMSEC and RMSECV on left y-axis
ln1 = ax1.plot(range(1, nlv + 1), Rmsec, label='RMSEC', marker='o', color='#1f77b4')
ln2 = ax1.plot(range(1, nlv + 1), Rmsecv, label='RMSECV', marker='o', color='#ff7f0e')
ax1.set_xlabel('Latent Variables')
ax1.set_ylabel('RMSE')
ax1.set_title('Training and Cross validation performances RMSE and R²')
ax1.grid()

# R² on right y-axis (green)
ax2 = ax1.twinx()
ln3 = ax2.plot(range(1, nlv + 1), R2, label='R²', marker='o', color='#2ca02c', linestyle='--')
ax2.set_ylabel('R²', color='#2ca02c')
ax2.tick_params(axis='y', labelcolor='#2ca02c')
ax2.spines['right'].set_color('#2ca02c')

# Combine legends for left plot
lines = ln1 + ln2 + ln3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)

plt.tight_layout()
plt.show()

opt_lv =50
y_pred_test = pls.predict(x_test, nlv=opt_lv).numpy()
rmsep = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
r2_test = np.corrcoef(y_test.flatten(), y_pred_test.flatten())[0, 1] ** 2


min_val = min(np.min(y_test), np.min(y_pred_test))
max_val = max(np.max(y_test), np.max(y_pred_test))
padding = 0.05 * (max_val - min_val)
lims = [min_val - padding, max_val + padding]
# Scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred_test, edgecolors='k', alpha=0.5)
ax.plot(lims, lims, 'r')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Real Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Predicted vs Real Values')
ax.grid()
ax.text(
    0.02, 0.98,
    f"R²: {r2_test:.2f}\nRMSEP: {rmsep:.3f}",
    transform=ax.transAxes, ha='left', va='top', fontsize=12,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
    color='red', fontweight='bold', fontfamily='serif'
)
plt.tight_layout()
plt.show()

# Hexbin plot
fig, ax = plt.subplots(figsize=(8, 6))
hexbin = ax.hexbin(y_test, y_pred_test, gridsize=50, cmap='viridis', mincnt=1)
cb = fig.colorbar(hexbin, ax=ax, orientation='vertical')
cb.set_label('Density')
ax.plot(lims, lims, 'k-')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('Real Values')
ax.set_ylabel('Predicted Values')
ax.set_title('Predicted vs Real Values for dry matter')
ax.grid()
ax.text(
    0.02, 0.98,
    f"R²: {r2_test:.2f}\nRMSEP: {rmsep:.3f}",
    transform=ax.transAxes, ha='left', va='top', fontsize=12,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
    color='red', fontweight='bold', fontfamily='serif'
)
plt.tight_layout()
plt.show()