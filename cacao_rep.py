import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import savgol_filter,detrend
from net.chemtools.PLS import PLS,PLSDA, LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
############################################################################################
############################################################################################
def threshold_predictions(probabilities):
    """
    Threshold multi-class probabilities to determine which classes are true.

    Parameters:
    - probabilities: np.array of shape (n_samples, n_classes), predicted probabilities for each class.
    - n_classes: int, total number of classes.

    Returns:
    - binary_predictions: np.array of shape (n_samples, n_classes), binary array indicating true classes.
    """
     # Determine the number of classes dynamically
    n_classes = probabilities.shape[1]
    # Define the threshold dynamically based on the number of classes
    threshold = 1 / n_classes

    # Apply the threshold to the probabilities
    binary_predictions = (probabilities >= threshold).int()

    return binary_predictions
############################################################################################
############################################################################################


data_path='./data/dataset/3rd_data_sampling_and_microbial_data/spec_data-rep.csv'                                                                    
y_class_path='./data/dataset/3rd_data_sampling_and_microbial_data/y_grow_classes.csv'
y_microbe_path ='./data/dataset/3rd_data_sampling_and_microbial_data/y_mircob_rep.csv'


############################################################################################
############################################################################################
data = pd.read_csv(data_path, sep=';', header=0, lineterminator='\n', skip_blank_lines=True)
data = data.dropna(how='all')
first_col = data.iloc[1:1984, 0]
wv =np.array(first_col.str.replace(',', '.')).astype(float)

# Generate a single column array with repeated column names
column_labels = [f"T{i}" for i in range(1, 16)]  # Generate T1 to T15
repeated_labels = np.array([label for label in column_labels for _ in range(4)])  # Repeat each label 4 times
unique_labels = np.unique(repeated_labels)  # Get unique labels
colors = [
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

############################################################################################
############################################################################################



############################################################################################
############################################################################################
# plt.figure(figsize=(10, 6))
# for i in range(X.shape[1]):  # Iterate over columns (assuming each column is a spectrum)
#     plt.plot(wv,X[:, i], label=f"Column {i+1}")

# plt.title("Spectral Data")
# plt.xlabel("wavelength (nm)")
# plt.ylabel("Intensity")
# plt.grid(True)
# plt.show()
############################################################################################
############################################################################################

# Perform PCA
# pca = PCA(n_components=8)  # Reduce to 2 principal components for visualization
# scores = pca.fit_transform(X) 

# plt.figure(figsize=(16, 12))

# # Define pairs of principal components to plot
# pc_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

# for i, (pc_x, pc_y) in enumerate(pc_pairs, start=1):
#     plt.subplot(2, 2, i)  # Create a 2x2 grid of subplots
#     for j, label in enumerate(unique_labels):
#         indices = np.where(repeated_labels == label)  # Find indices for the current label
#         plt.scatter(scores[indices, pc_x], scores[indices, pc_y], label=label, color=colors[j], alpha=0.7)
#         for idx in indices[0][::4]:  # Annotate every 4th point
#             plt.text(scores[idx, pc_x], scores[idx, pc_y], label, fontsize=8)
#     plt.title(f"PCA Scores Plot (PC{pc_x + 1} vs. PC{pc_y + 1})")
#     plt.xlabel(f"PC{pc_x + 1}")
#     plt.ylabel(f"PC{pc_y + 1}")
#     plt.grid(True)
# plt.legend(title="Labels", loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=8, fontsize="small")
# plt.tight_layout()
# plt.show()

############################################################################################
############################################################################################


X_train, X_test, y_train, y_test,  = train_test_split(
    X, y_microbe_array, test_size=0.2, random_state=42, shuffle=True
)

n_components = 20 

binary_tables = []
for i in range(y_train.shape[1]):
    binary_table = np.zeros_like(y_train).astype(int)  # Initialize a table of zeros
    binary_table[:, i] = y_train[:, i]  # Set the column for the current class to its values
    binary_tables.append(binary_table)

plsda =PLSDA(ncomp=n_components)
plsda.fit(X_train, binary_tables[0])
y_pred=plsda.predict(X_test)
print(y_pred)
print(y_test)
