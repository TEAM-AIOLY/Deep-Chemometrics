import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import savgol_filter,detrend
from net.chemtools.PLS import PLS,LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import os



data_path='./data/dataset/3rd_data_sampling_and_microbial_data/spec_data-rep.csv'                                                                    
y_class_path='./data/dataset/3rd_data_sampling_and_microbial_data/y_grow_classes.csv'
y_microbe_path ='./data/dataset/3rd_data_sampling_and_microbial_data/y_mircob_rep.csv'

save_path = os.path.dirname(data_path)


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
# for i in range(X.shape[0]):  # Iterate over columns (assuming each column is a spectrum)
#     plt.plot(wv,X[i,:], label=f"Column {i+1}")

# plt.title(" Raw Spectral Data")
# plt.xlabel("wavelength (nm)")
# plt.ylabel("Intensity")
# plt.grid(True)
# plt.show()
############################################################################################
############################################################################################
X_pp = np.log1p(X)  # log(1 + x) to handle zero or negative values

# Settings for the smooth derivatives using a Savitsky-Golay filter
w = 13 ## Sav.Gol window size
p = 2  ## Sav.Gol polynomial degree
d=1 ## Sav.Gol derive  order

X_pp = savgol_filter(X_pp, window_length=w, polyorder=p,deriv=d, axis=1)
X_pp= detrend(X_pp, axis=1)
############################################################################################
############################################################################################
# plt.figure(figsize=(10, 6))
# for i in range(X.shape[0]):  # Iterate over columns (assuming each column is a spectrum)
#     plt.plot(wv,X[i,:], label=f"Column {i+1}")

# plt.title(" PP Spectral Data")
# plt.xlabel("wavelength (nm)")
# plt.ylabel("Intensity")
# plt.grid(True)
# plt.show()


# # # Perform PCA
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

# ############################################################################################
# ############################################################################################



X_train, X_test, y_train, y_test, T_ref_train, T_ref_test  = train_test_split(
    X_pp, y_microbe_array, repeated_labels, test_size=0.2, random_state=42, shuffle=True
)

n_components = 20       

plsr = PLS(ncomp=n_components)

plsr.fit(X_train, y_train)  # Fit PLSDA for the current class
T_train = plsr.transform(X_train)
lda =LDA()
lda.fit(T_train,y_train)
proj =lda.projections
W =plsr.W
DV = (W@proj.T).T 
DV = np.array(DV)

ld = lda.projections

# for i in range(DV.shape[1]):
#     plt.plot(wv,DV[i,:])
# plt.title('Discriminant Vectors (DV) from LDA on PLSR Transformed Data')
# plt.show();


scores_test =plsr.transform(X_test)
preds =lda.transform(scores_test)


# Color map: blue for 1 (True), red for 0 (False)
color_map = {1: 'blue', 0: 'red'}
label_names = ['B1', 'B2', 'B3', 'B4']
axis_pairs = [(0, 1), (1, 2), (2, 3)]

base_save_path = os.path.join(save_path, "figures", "2D_discri_plots")

for i, label_name in enumerate(label_names):
    true_labels = y_test[:, i]
    label_save_path = os.path.join(base_save_path, label_name)
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)
    
    for j, (x_axis, y_axis) in enumerate(axis_pairs):
        x = preds[:, x_axis]
        y = preds[:, y_axis]

        # Define present and absent labels
        present = true_labels == 1
        absent = true_labels == 0

        # Create figure for scatter plot
        plt.figure(figsize=(5.5, 5.5))
        
        # Scatter plot for present and absent labels
        plt.scatter(x[present], y[present], color='blue', label='Present', s=80, edgecolor='k')
        plt.scatter(x[absent], y[absent], facecolors='none', edgecolors='red', label='Absent', s=80)

        # # Plot the linear discriminant line
        # # Calculate the slope and intercept of the line separating the classes
        # # The line is determined by the linear discriminant vector
        # # We'll assume the projection `proj` is a 4 x n_components matrix as discussed earlier
        # w = proj[i]  # Get the discriminant vector for class i (4 x 20 matrix)
        # line_x = np.linspace(min(x), max(x), 100)
        # line_y = -(w[x_axis] * line_x) / w[y_axis]  # The line equation: w1*x + w2*y = 0
        
        # # Plot the discriminant line
        # plt.plot(line_x, line_y, color='black', linestyle='--', label='Discriminant Line')

        # Set labels and title
        plt.xlabel(f'Score DV {x_axis + 1}')
        plt.ylabel(f'Score DV {y_axis + 1}')
        plt.title(f'{label_name} ')

        # Add legend
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
        
        # Adjust layout and show plot
        plt.tight_layout()
        plt.show(block=False);
        path_2d_fig = os.path.join(label_save_path, f"2d_plot_{label_name}_axes_{x_axis+1}_{y_axis+1}.png")
        plt.savefig(path_2d_fig, dpi=900, bbox_inches='tight')  # Save the figure
        plt.close()



for i, label_name in enumerate(label_names):
    true_labels = y_test[:, i]  # Get the true labels for this class (B1, B2, B3, or B4)

    # Set present (1) and absent (0) for each class
    present = true_labels == 1
    absent = true_labels == 0
    

    # Create the 3D scatter plot
    fig = go.Figure()

    # Plot 'Present' points (blue)
    fig.add_trace(go.Scatter3d(
        x=preds[present, 0].numpy(),  # Discriminant axis 1 (DV1)
        y=preds[present, 1].numpy(),  # Discriminant axis 2 (DV2)
        z=preds[present, 2].numpy(),  # Discriminant axis 3 (DV3)
        mode='markers',
        marker=dict(
            size=6,
            color='blue',
            symbol='circle',
            line=dict(width=1, color='black')
        ),
        name='Present'
    ))

    # Plot 'Absent' points (red)
    fig.add_trace(go.Scatter3d(
        x=preds[absent, 0].numpy(),  # Discriminant axis 1 (DV1)
        y=preds[absent, 1].numpy(),  # Discriminant axis 2 (DV2)
        z=preds[absent, 2].numpy(),  # Discriminant axis 3 (DV3)
        mode='markers',
        marker=dict(
            size=6,
            color='red',
            symbol='circle',
            line=dict(width=1, color='black')
        ),
        name='Absent'
    ))

    # Define the decision boundary plane based on the projection (proj)
    coeffs = proj[i, :3]  # Take the first three coefficients for the current class (use up to 3 for the decision plane)

    # The equation for a plane is: Ax + By + Cz + D = 0
    # Coefficients A, B, C correspond to the normal vector of the plane
    A, B, C = coeffs  # These will define the plane's normal

    # Get the min and max values for preds for grid generation
    x_min, x_max = torch.min(preds[:, 0]), torch.max(preds[:, 0])
    y_min, y_max = torch.min(preds[:, 1]), torch.max(preds[:, 1])

    # Define the range for the plane grid
    grid_size = 20
    x_range = np.linspace(x_min.item(), x_max.item(), grid_size)
    y_range = np.linspace(y_min.item(), y_max.item(), grid_size)
    
    # Create meshgrid
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = -(A * x_grid + B * y_grid) / C  # Solve for z in terms of x and y

    # Add the decision plane to the plot
    fig.add_trace(go.Surface(
        x=x_grid, y=y_grid, z=z_grid,
        colorscale='Viridis',
        opacity=0.5,
        name=f'Decision Plane {label_name}',
        showscale=False
    ))

    # Update layout for the 3D plot
    fig.update_layout(
        scene=dict(
            xaxis_title='Discriminant Axis 1',  # DV1
            yaxis_title='Discriminant Axis 2',  # DV2
            zaxis_title='Discriminant Axis 3'   # DV3
        ),
        title=f'{label_name}',
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            x=0.8, y=0.9,
            traceorder='normal',
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='Black',
            borderwidth=1
        )
    )

    # Save the 3D plot as an interactive HTML file
    path_3d_fig = os.path.join(save_path, f"3d_plot_{label_name}.html")
    fig.write_html(path_3d_fig);
    fig.show();

combinations_of_axes = list(combinations([0, 1, 2, 3], 3))  # Indices of axes, 0, 1, 2, 3 are for the 4 discriminant axes

# Loop through each class and generate 3D plots for the combinations of 3 axes
for i, label_name in enumerate(label_names):
    true_labels = y_test[:, i]  # Get the true labels for this class (B1, B2, B3, or B4)

    # Set present (1) and absent (0) for each class
    present = true_labels == 1
    absent = true_labels == 0
    label_save_dir = os.path.join(save_path, 'figures', '3D_discri_plots', label_name)
    os.makedirs(label_save_dir, exist_ok=True)  # Create the label-specific directory if it doesn't exist

    # Loop through each combination of 3 axes for the current class
    for combo in combinations_of_axes:
        fig = go.Figure()

        # Plot 'Present' points (blue)
        fig.add_trace(go.Scatter3d(
            x=preds[present, combo[0]].numpy(),  # Use the selected axes combination
            y=preds[present, combo[1]].numpy(),
            z=preds[present, combo[2]].numpy(),
            mode='markers',
            marker=dict(
                size=6,
                color='blue',
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            name='Present'
        ))

        # Plot 'Absent' points (red)
        fig.add_trace(go.Scatter3d(
            x=preds[absent, combo[0]].numpy(),
            y=preds[absent, combo[1]].numpy(),
            z=preds[absent, combo[2]].numpy(),
            mode='markers',
            marker=dict(
                size=6,
                color='red',
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            name='Absent'
        ))

        # Define the decision boundary plane based on the projection (proj)
        # Extract the coefficients for the selected axes combination
        coeffs = proj[i, list(combo)]  # Use the exact indices of the selected axes
        A, B, C = coeffs[0], coeffs[1], coeffs[2]  # Ensure we have the correct 3 coefficients

        # Get the min and max values for preds for grid generation
        x_min, x_max = torch.min(preds[:, combo[0]]), torch.max(preds[:, combo[0]])
        y_min, y_max = torch.min(preds[:, combo[1]]), torch.max(preds[:, combo[1]])

        # Define the range for the plane grid
        grid_size = 20
        x_range = np.linspace(x_min.item(), x_max.item(), grid_size)
        y_range = np.linspace(y_min.item(), y_max.item(), grid_size)
        
        # Create meshgrid
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        z_grid = -(A * x_grid + B * y_grid) / C  # Solve for z in terms of x and y

        # Add the decision plane to the plot
        fig.add_trace(go.Surface(
            x=x_grid, y=y_grid, z=z_grid,
            surfacecolor=np.ones_like(z_grid) * 0.5,  # Set the color to a uniform value (0.5 for light grey)
            colorscale=[[0, 'grey'], [1, 'grey']],  # Set the color scale to a single color (grey)
            opacity=0.5,  # Adjust the opacity if needed
            name=f'Decision Plane {label_name}',
            showscale=False  # Don't show the color scale bar
        ))

        # Update layout for the 3D plot
        fig.update_layout(
            scene=dict(
                xaxis_title=f'Axis {combo[0]+1}',  # Display the axes based on the combination
                yaxis_title=f'Axis {combo[1]+1}',
                zaxis_title=f'Axis {combo[2]+1}'
            ),
            title=f'{label_name} Projections (Axes {combo[0]+1}, {combo[1]+1}, {combo[2]+1}) with Decision Plane',
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(
                x=0.8, y=0.9,
                traceorder='normal',
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='Black',
                borderwidth=1
            )
        )

        # Save the 3D plot as an interactive HTML file
        path_3d_fig = os.path.join(label_save_dir, f"3d_discri_plot_axes__{combo[0]+1}_{combo[1]+1}_{combo[2]+1}.html")
        fig.write_html(path_3d_fig);





   
# binary_tables_train = []
# binary_tables_test = []

# for i in range(y_train.shape[1]):
#     binary_table_train = np.zeros_like(y_train).astype(int)  # Initialize a table of zeros
#     binary_table_train[:, i] = y_train[:, i]  # Set the column for the current class to its values
#     binary_tables_train.append(binary_table_train)
    
#     binary_table_test = np.zeros_like(y_test).astype(int)  # Initialize a table of zeros
#     binary_table_test[:, i] = y_test[:, i]  # Set the column for the current class to its values
#     binary_tables_test.append(binary_table_test)
    
#     plsda.fit(X_train, binary_table_train)  # Fit PLSDA for the current class
#     y_pred_prob = plsda.predict(X_test)  # Predict probabilities for the test set
    
#     print(y_pred_prob)
#     print(binary_table_test)



















############################################################################################
############################################################################################
# def threshold_predictions(probabilities):
#     """
#     Threshold multi-class probabilities to determine which classes are true.

#     Parameters:
#     - probabilities: np.array of shape (n_samples, n_classes), predicted probabilities for each class.
#     - n_classes: int, total number of classes.

#     Returns:
#     - binary_predictions: np.array of shape (n_samples, n_classes), binary array indicating true classes.
#     """
#      # Determine the number of classes dynamically
#     n_classes = probabilities.shape[1]
#     # Define the threshold dynamically based on the number of classes
#     threshold = 1 / n_classes

#     # Apply the threshold to the probabilities
#     binary_predictions = (probabilities >= threshold).int()

#     return binary_predictions
############################################################################################
############################################################################################







##############################################################################################################
##############################################################################################################

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Check if we have at least 3 dimensions in the preds array
# if preds.shape[1] >= 3:
#     # Plot the 3D scatter plot with discriminant scores along axes 1, 2, and 3
#     sc = ax.scatter(preds[:, 0], preds[:, 1], preds[:, 2], c=labels, cmap='viridis', edgecolor='k', s=100)

#     # Add labels and title
#     ax.set_title('Discriminant Scores: Axis 1 vs Axis 2 vs Axis 3')
#     ax.set_xlabel('Discriminant Axis 1')
#     ax.set_ylabel('Discriminant Axis 2')
#     ax.set_zlabel('Discriminant Axis 3')

#     # Add color bar to show true labels
#     cbar = plt.colorbar(sc)
#     cbar.set_label('True Labels')

#     # Display the plot
#     plt.tight_layout()
#     plt.show()
# else:
#     print("Not enough dimensions in the prediction results to plot a 3D scatter plot.")
##############################################################################################################
##############################################################################################################