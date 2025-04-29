import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,detrend
from net.chemtools.PLS import PLS,LDA, QDA


from sklearn.model_selection import train_test_split
from utils.make_discri_graphs import create_2d_plot,create_3d_discriminant_plots, create_3d_discriminant_plots_mpl



data_path='./data/dataset/3rd_data_sampling_and_microbial_data/spec_data-rep.csv'                                                                    
y_class_path='./data/dataset/3rd_data_sampling_and_microbial_data/y_grow_classes.csv'

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
class_array =(y_class.values[:,1:]).astype(float)
class_array = np.repeat(class_array, 4, axis=0)


X_pp = np.log1p(X)  # log(1 + x) to handle zero or negative values

# Settings for the smooth derivatives using a Savitsky-Golay filter
w = 21 ## Sav.Gol window size
p = 2  ## Sav.Gol polynomial degree
d=1 ## Sav.Gol derive  order

X_pp = savgol_filter(X_pp, window_length=w, polyorder=p,deriv=d, axis=1)


X_pp= detrend(X_pp, axis=1)


X_train, X_test, y_train, y_test,  = train_test_split(
    X_pp, class_array, test_size=0.2, random_state=42, shuffle=True
)

n_components=20
pls =PLS(ncomp=n_components)
pls.fit(X_train,y_train)

T_train = pls.transform(X_train)
T_test =pls.transform(X_test)

qda =QDA()
qda.fit(T_train,y_train)

classes=qda.predict(T_test)

# lda =LDA()
# lda.fit(T_train,y_train)
# preds=lda.transform(T_test)
# classes=lda.predict(T_test)

