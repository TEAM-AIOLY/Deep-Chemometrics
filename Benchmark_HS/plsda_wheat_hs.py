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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score
import seaborn as sns

data_root = [{
    "data_path": "C:/00_aioly/GitHub/Deep-Chemometrics/data/dataset/mango/mango_dm_full_outlier_removed2.mat",
    "dataset_type": "mango"},        
{"data_path": "C:/00_aioly/GitHub/Deep-Chemometrics/data/dataset/ossl/ossl_database.mat",
    "dataset_type": "ossl"},
{"data_path": "C:/00_aioly/GitHub/Deep-Chemometrics/data/dataset/wheat/",
    "dataset_type": "wheat"}]    


dataset=data_root[2]
data = DatasetLoader.load(dataset)

x_cal = data["x_cal"]
y_cal = data["y_cal"]   
x_val = data["x_val"]
y_val = data["y_val"]
x_test = data["x_test"]
y_test = data["y_test"]

y_cal_labels = np.argmax(y_cal, axis=1)
y_val_labels = np.argmax(y_val, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
# window_length = 7
# polyorder = 2
# deriv = 1

# x_cal= savgol_filter(x_cal, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
# x_val = savgol_filter(x_val, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
# x_test = savgol_filter(x_test, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)



nlv=100

pls = PLS(ncomp=nlv)
pls.fit(x_cal, y_cal)
score_cal=pls.transform(x_cal).numpy()
score_val =pls.transform(x_val).numpy()

lda_accs = []
f1_cal = []
f1_val = []

for lv in range(1,nlv+1):
    lda = LinearDiscriminantAnalysis()
    lda.fit(score_cal[:, :lv], y_cal_labels)
    
    # Predict on calibration set
    y_cal_pred = lda.predict(score_cal[:, :lv])
    f1_cal.append(f1_score(y_cal_labels, y_cal_pred, average='macro'))
    
    # Predict on validation set
    y_val_pred = lda.predict(score_val[:, :lv])
    f1_val.append(f1_score(y_val_labels, y_val_pred, average='macro'))


# Plot F1 scores
plt.plot(range(1, nlv + 1), f1_cal, label='Calibration F1 ')
plt.plot(range(1, nlv + 1), f1_val, label='Validation F1 ')
plt.xlabel('Number of Latent Variables (LVs)')
plt.ylabel('F1 Score ')
plt.title('PLS-DA: F1 Score vs. Number of LVs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


optimal_nlv =50

# Train LDA with optimal nlv on calibration set
lda = LinearDiscriminantAnalysis()
lda.fit(score_cal[:, :optimal_nlv], y_cal_labels)

# Predict on test set
score_test = pls.transform(x_test).numpy()
y_test_pred = lda.predict(score_test[:, :optimal_nlv])

# Confusion matrix
cm = confusion_matrix(y_test_labels, y_test_pred)
print("Confusion Matrix:\n", cm)

# Mean F1 score (macro), mean accuracy, mean recall (macro)
mean_f1 = f1_score(y_test_labels, y_test_pred, average='macro')
mean_acc = accuracy_score(y_test_labels, y_test_pred)
mean_recall = recall_score(y_test_labels, y_test_pred, average='macro')

print(f"Mean F1 score: {mean_f1:.4f}")
print(f"Mean accuracy: {mean_acc:.4f}")
print(f"Mean recall: {mean_recall:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.show()