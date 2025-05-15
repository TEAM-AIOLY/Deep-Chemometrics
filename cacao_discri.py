import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,detrend
from net.chemtools.PLS import PLS,MLDA,MLDA_v2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from utils.make_discri_graphs import create_2d_plot,create_3d_discriminant_plots, create_3d_discriminant_plots_mpl



data_path='./data/dataset/3rd_data_sampling_and_microbial_data/spec_data-rep.csv'                                                                    
y_class_path='./data/dataset/3rd_data_sampling_and_microbial_data/y_grow_classes.csv'
y_microbe_path ='./data/dataset/3rd_data_sampling_and_microbial_data/y_mircob_rep.csv'

save_path = os.path.dirname(data_path)
base_save_path = os.path.join(save_path, "figures")

############################################################################################
############################################################################################
data = pd.read_csv(data_path, sep=';', header=0, lineterminator='\n', skip_blank_lines=True)
data = data.dropna(how='all')
first_col = data.iloc[1:1984, 0]
wv =np.array(first_col.str.replace(',', '.')).astype(float)

# Generate a single column array with repeated column names
column_labels = [f"T{i}" for i in range(1, 16)]  # Generate T1 to T15
repeated_labels = np.array([label for label in column_labels for _ in range(4)])  # Repeat each label 4 times
T_colors = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
    "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3"
]

sepctral_data = data.iloc[1:1984, 1:]
sepctral_data = sepctral_data.map(lambda x: str(x).replace(',', '.'))  # Replace commas with dots
X = np.array(sepctral_data).astype(float)
X=X.T
y_class = pd.read_csv(y_class_path,sep=';',header=0)
y_microbe= pd.read_csv(y_microbe_path,sep=';',header=0)


row_labels = y_microbe.iloc[2:, 0]
microbe_labels = np.array(row_labels).astype(str)
y_microbe=y_microbe.iloc[2:, 1:]
y_microbe_array=np.array(y_microbe).astype(float)
y_microbe_array = y_microbe_array.T

class_array =(y_class.values[:,1:]).astype(float)
class_array = np.repeat(class_array, 4, axis=0)

mat_data = {
    'wv': wv,
    'X': X,
    'y_microbe': y_microbe_array,
    'class_array': class_array
}

############################################################################################
############################################################################################
# plt.figure(figsize=(10, 6))
# for i in range(X.shape[0]):  # Iterate over columns (assuming each column is a spectrum)
#     plt.plot(wv,X[i,:], label=f"Column {i+1}")

# plt.title(" Raw Spectral Data")
# plt.xlabel("wavelength (nm)")
# plt.ylabel("Intensity")
# plt.grid(True)
# raw_spec_path =os.path.join(base_save_path,"raw_spectral.pdf")
# plt.savefig(raw_spec_path, bbox_inches='tight')

############################################################################################
############################################################################################

X_pp = np.log1p(X)  # log(1 + x) to handle zero or negative values


############################################################################################
############################################################################################
# plt.figure(figsize=(10, 6))
# for i in range(X.shape[0]):  # Iterate over columns (assuming each column is a spectrum)
#     plt.plot(wv,X_pp[i,:], label=f"Column {i+1}")

# plt.title(" Log(Spectral) Data")
# plt.xlabel("wavelength (nm)")
# plt.ylabel("Intensity")
# plt.grid(True)
# log_spec_path =os.path.join(base_save_path,"log_spec.pdf")
# plt.savefig(log_spec_path, bbox_inches='tight')

############################################################################################
############################################################################################

# Settings for the smooth derivatives using a Savitsky-Golay filter
w = 21 ## Sav.Gol window size
p = 2  ## Sav.Gol polynomial degree
d=1 ## Sav.Gol derive  order

X_pp = savgol_filter(X_pp, window_length=w, polyorder=p,deriv=d, axis=1)


############################################################################################
############################################################################################
# plt.figure(figsize=(10, 6))
# for i in range(X.shape[0]):  # Iterate over columns (assuming each column is a spectrum)
#     plt.plot(wv,X_pp[i,:], label=f"Column {i+1}")

# plt.title("  SG(Log(Spectral)  Data")
# plt.xlabel("wavelength (nm)")
# plt.ylabel("Intensity")
# plt.grid(True)
# log_SG_spec_path =os.path.join(base_save_path,"log_SG_spec.pdf")
# plt.savefig(log_SG_spec_path, bbox_inches='tight')

############################################################################################
############################################################################################

X_pp= detrend(X_pp, axis=1)


############################################################################################
############################################################################################
# plt.figure(figsize=(10, 6))
# for i in range(X.shape[0]):  # Iterate over columns (assuming each column is a spectrum)
#     plt.plot(wv,X_pp[i,:], label=f"Column {i+1}")

# plt.title(" DT(SG(Log(Spectral)) Data")
# plt.xlabel("wavelength (nm)")
# plt.ylabel("Intensity")
# plt.grid(True)
# log_SG_spec_path =os.path.join(base_save_path,"log_SG__DT_spec.pdf")
# plt.savefig(log_SG_spec_path, bbox_inches='tight')

############################################################################################
############################################################################################



X_train, X_test, y_train, y_test,  = train_test_split(
    X_pp, y_microbe_array, test_size=0.2, random_state=42, shuffle=True
)

n_components = 20      

plsr = PLS(ncomp=n_components)

plsr.fit(X_train, y_train)  # Fit PLSDA for the current class
T_train = plsr.transform(X_train)
scores_test =plsr.transform(X_test)


label_names = ['B1', 'B2', 'B3', 'B4']
axis_pairs = [(0, 1), (1, 2), (2, 3)]


mlda =MLDA()
mlda.fit(T_train,y_train)
preds =mlda.transform(scores_test)
# print(y_test)
classes =mlda.predict(scores_test)
# print(preds)
proj =mlda.projections


# Create 2D plots
create_2d_plot(preds, y_test, label_names, axis_pairs, save_path, show=False)
# Create 3D plots
create_3d_discriminant_plots(preds, y_test, proj, label_names, save_path, show=False)
create_3d_discriminant_plots_mpl(preds, y_test, proj, label_names, save_path, show=False)



############################################################################################
############################################################################################
W =(plsr.W)
DV = (W@proj.T).T 
DV = np.array(DV)
print(DV.shape)

plt.figure()
for i in range(DV.shape[0]):
    plt.plot(wv,DV[i,:])
plt.title('Discriminant Vectors (DV) in spectral space')
dv_path =os.path.join(base_save_path,"discrimiannt_vector_da_5lv.pdf")
plt.savefig(dv_path, bbox_inches='tight')
plt.xlabel("wavelength (nm)")
plt.grid(True)
# plt.show()


# plt.figure()
# for i in range(proj.shape[0]):
#     plt.plot(proj[i,:])
# plt.title('Discriminant Vectors (DV) in latent space')
# dv_path =os.path.join(base_save_path,"discrimiannt vector pls.pdf")
# plt.xlabel("Latent variables")
# plt.xticks(np.arange(0,n_components,1))
# plt.grid(True)
# # plt.savefig(dv_path, bbox_inches='tight')
# plt.show()
############################################################################################
############################################################################################



