import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter,detrend
from net.chemtools.PLS import PLS,PLSDA
from net.chemtools.metrics import ccc, r2_score
import torch 
from sklearn.model_selection import train_test_split

data_path='./data/dataset/3rd_data_sampling_and_microbial_data/data 3rd sampling 450-950nm.csv'
ref_path ='./data/dataset/3rd_data_sampling_and_microbial_data/MIX MICROBE dis.csv'

data = pd.read_csv(data_path, sep=',', header=0)
wv =np.array(data.keys()[1:]).astype(float)

spectral_data = np.array(data.iloc[:, 1:])
T_ref = np.array(data.iloc[:, 0]).astype(str)

ref= pd.read_csv(ref_path, sep=',', header=0)
ref_data = np.array(ref.iloc[:, 1:])
mixrobe_data = ref_data[:,1:]
ferti =ref_data[:,0]


# plt.figure(figsize=(10, 6))
# for i in range(spectral_data.shape[0]):
#     plt.plot(wv, spectral_data[i, :], label=f"Sample {i+1}" if i < 10 else "", alpha=0.7)

# plt.title("Spectral Data")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Intensity")
# plt.grid(True)
# plt.show();


spectral_data_log = np.log1p(spectral_data)  # log(1 + x) to handle zero or negative values

# plt.figure(figsize=(10, 6))
# for i in range(spectral_data_log.shape[0]):
#     plt.plot(wv, spectral_data_log[i, :], label=f"Sample {i+1}" if i < 10 else "", alpha=0.7)

# plt.title("Log-Transformed and Smoothed Spectral Data")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Intensity (log-transformed)")
# plt.grid(True)
# plt.show();


## Settings for the smooth derivatives using a Savitsky-Golay filter
w = 21 ## Sav.Gol window size
p = 2  ## Sav.Gol polynomial degree
d=1 ## Sav.Gol derive  order

spectral_data_sg = savgol_filter(spectral_data_log, window_length=w, polyorder=p,deriv=d, axis=1)

# plt.figure(figsize=(10, 6))
# for i in range(spectral_data_sg.shape[0]):
#     plt.plot(wv, spectral_data_sg[i, :], label=f"Sample {i+1}" if i < 10 else "", alpha=0.7)

# plt.title("First Derivative of Log-Transformed and Smoothed Spectral Data")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Intensity (1st Derivative)")
# plt.grid(True)
# plt.show();


spectral_data_dt= detrend(spectral_data_sg, axis=1)
# plt.figure(figsize=(10, 6))
# for i in range(spectral_data_dt.shape[0]):
#     plt.plot(wv, spectral_data_dt[i, :], label=f"Sample {i+1}" if i < 10 else "", alpha=0.7)

# plt.title("Detrended Spectral Data")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Intensity (Detrended)")
# plt.grid(True)
# plt.show();

# Split data into training and testing sets
X_train, X_test, y_train, y_test, T_ref_train, T_ref_test = train_test_split(
    spectral_data_dt, mixrobe_data, T_ref, test_size=0.2, random_state=42, shuffle=True
)

n_components = 20 
plsda_model = PLSDA(ncomp=n_components)

plsda_model.fit(X_train, y_train, class_labels=y_train)
y_pred = plsda_model.predict(X_test)


lda_loadings = plsda_model.lda_loadings  # Discriminant vectors
plt.figure(figsize=(10, 6))
for i in range(lda_loadings.shape[0]):  # Iterate over the number of discriminant vectors
    plt.plot(lda_loadings[i, :], label=f"Discriminant Vector {i+1}")

plt.title("LDA Loadings (Discriminant Vectors)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Loadings")
plt.legend()
plt.grid(True)
plt.show()


lda_scores = plsda_model.lda_scores  # LDA scores
# in one-hot encoding to labels (e.g., "B1 Present", "B2 Absent", etc.)
num_classes = y_train.shape[1]  # Number of classes (columns in y_train)
for col_index in range(num_classes):
    # Determine labels for the current class
    labels = [f"B{col_index + 1}" if row[col_index] == 1 else "∅" for row in y_train]

    # Plot the LDA scores for components 1 and 2
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.scatter(lda_scores[i, 0], lda_scores[i, 1], alpha=0.7)
        plt.annotate(label, (lda_scores[i, 0], lda_scores[i, 1]), fontsize=8, alpha=0.7)

    plt.title(f"LDA Scores Plot for B{col_index + 1} (First Two Components)")
    plt.xlabel("LDA Component 1")
    plt.ylabel("LDA Component 2")
    plt.grid(True)
    plt.show();
    
    
    # Plot for components 2 and 3
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.scatter(lda_scores[i, 1], lda_scores[i, 2], alpha=0.7)
        plt.annotate(label, (lda_scores[i, 1], lda_scores[i, 2]), fontsize=8, alpha=0.7)
    plt.title(f"LDA Scores Plot for B{col_index + 1} (Components 2 and 3)")
    plt.xlabel("LDA Component 2")
    plt.ylabel("LDA Component 3")
    plt.grid(True)
    plt.show();
    
    

