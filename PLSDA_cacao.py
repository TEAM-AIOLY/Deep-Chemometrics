import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter,detrend
from net.chemtools.PLS import PLS

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


plt.figure(figsize=(10, 6))
for i in range(spectral_data.shape[0]):
    plt.plot(wv, spectral_data[i, :], label=f"Sample {i+1}" if i < 10 else "", alpha=0.7)

plt.title("Spectral Data")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.grid(True)
plt.show();


spectral_data_log = np.log1p(spectral_data)  # log(1 + x) to handle zero or negative values

plt.figure(figsize=(10, 6))
for i in range(spectral_data_log.shape[0]):
    plt.plot(wv, spectral_data_log[i, :], label=f"Sample {i+1}" if i < 10 else "", alpha=0.7)

plt.title("Log-Transformed and Smoothed Spectral Data")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (log-transformed)")
plt.grid(True)
plt.show();


## Settings for the smooth derivatives using a Savitsky-Golay filter
w = 21 ## Sav.Gol window size
p = 2  ## Sav.Gol polynomial degree
d=1 ## Sav.Gol derive  order

spectral_data_sg = savgol_filter(spectral_data_log, window_length=w, polyorder=p,deriv=d, axis=1)

plt.figure(figsize=(10, 6))
for i in range(spectral_data_sg.shape[0]):
    plt.plot(wv, spectral_data_sg[i, :], label=f"Sample {i+1}" if i < 10 else "", alpha=0.7)

plt.title("First Derivative of Log-Transformed and Smoothed Spectral Data")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (1st Derivative)")
plt.grid(True)
plt.show();


spectral_data_dt= detrend(spectral_data_sg, axis=1)
plt.figure(figsize=(10, 6))
for i in range(spectral_data_dt.shape[0]):
    plt.plot(wv, spectral_data_dt[i, :], label=f"Sample {i+1}" if i < 10 else "", alpha=0.7)

plt.title("Detrended Spectral Data")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (Detrended)")
plt.grid(True)
plt.show();

# Split data into training and testing sets
X_train, X_test, y_train, y_test, T_ref_train, T_ref_test = train_test_split(
    spectral_data_dt, mixrobe_data, T_ref, test_size=0.2, random_state=42, shuffle=True
)


# n_components = 50 
# # plsda =PLSDA(ncomp=n_components)
# # plsda.fit(X_train, y_train)

# # y_pred=plsda.predict(X_test)

# pls =PLS(ncomp=n_components)
# pls.fit(X_train, y_train)
# T_new = pls.transform(X_test)
# T= pls.transform(X_train)

# lda=LDA(prior='unif')
# lda.fit(pls.T,y_train)
# y_train_pred = lda.predict(T)
# y_test_pred = lda.predict(T_new)

# print(y_test_pred)
# print(y_test)
