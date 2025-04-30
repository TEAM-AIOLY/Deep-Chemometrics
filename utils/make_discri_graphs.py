import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from itertools import combinations
import torch
from mpl_toolkits.mplot3d import Axes3D
import pickle

def create_2d_plot(preds, y_test, label_names, axis_pairs, save_path, show=False):
    color_map = {1: 'blue', 0: 'red'}
    
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

            # Set labels and title
            plt.xlabel(f'Score DV {x_axis + 1}')
            plt.ylabel(f'Score DV {y_axis + 1}')
            plt.title(f'{label_name} ')

            # Add legend
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)

            # Adjust layout and show plot
            plt.tight_layout()
            if show == True:
             plt.show(block=False)
            path_2d_fig = os.path.join(label_save_path, f"2d_plot_{label_name}_axes_{x_axis+1}_{y_axis+1}.pdf")
            plt.savefig(path_2d_fig, bbox_inches='tight')  # Save the figure
            plt.close()

def create_3d_plot(preds, y_test, proj, label_names, save_path, show=False):
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
        fig.write_html(path_3d_fig)
        if show == True:
             fig.show()

def create_3d_discriminant_plots(preds, y_test, proj, label_names, save_path, show=False):
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
            if show == True:
             fig.show()
             
             
def create_3d_discriminant_plots_mpl(preds, y_test, proj, label_names, save_path, show=False):
    combinations_of_axes = list(combinations([0, 1, 2, 3], 3))  # Indices of axes, 0, 1, 2, 3 are for the 4 discriminant axes

    # Loop through each class and generate 3D plots for the combinations of 3 axes
    for i, label_name in enumerate(label_names):
        true_labels = y_test[:, i]  # Get the true labels for this class (B1, B2, B3, or B4)

        # Set present (1) and absent (0) for each class
        present = true_labels == 1
        absent = true_labels == 0
        label_save_dir = os.path.join(save_path, 'figures', '3D_discri_plots_mpl', label_name)
        os.makedirs(label_save_dir, exist_ok=True)  # Create the label-specific directory if it doesn't exist

        # Loop through each combination of 3 axes for the current class
        for combo in combinations_of_axes:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot 'Present' points (blue)
            ax.scatter(
                preds[present, combo[0]].cpu(),
                preds[present, combo[1]].cpu(),
                preds[present, combo[2]].cpu(),
                c='blue',
                label='Present',
                s=50,
                edgecolor='k'
            )

            # Plot 'Absent' points (red)
            ax.scatter(
                preds[absent, combo[0]].cpu(),
                preds[absent, combo[1]].cpu(),
                preds[absent, combo[2]].cpu(),
                c='red',
                label='Absent',
                s=50,
                edgecolor='k'
            )

            # Define the decision boundary plane based on the projection (proj)
            coeffs = proj[i, list(combo)]  # Use the exact indices of the selected axes
            A, B, C = coeffs[0], coeffs[1], coeffs[2]  # Ensure we have the correct 3 coefficients

            # Get the min and max values for preds for grid generation
            x_min, x_max = torch.min(preds[:, combo[0]]), torch.max(preds[:, combo[0]])
            y_min, y_max = torch.min(preds[:, combo[1]]), torch.max(preds[:, combo[1]])

            # Define the range for the plane grid
            grid_size = 20
            x_range = torch.linspace(x_min, x_max, grid_size)
            y_range = torch.linspace(y_min, y_max, grid_size)

            # Create meshgrid
            x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing='ij')
            z_grid = -(A * x_grid + B * y_grid) / C  # Solve for z in terms of x and y

            # Add the decision plane to the plot
            ax.plot_surface(
                x_grid.cpu().numpy(),
                y_grid.cpu().numpy(),
                z_grid.cpu().numpy(),
                color='grey',
                alpha=0.5,
                rstride=1,
                cstride=1,
                edgecolor='none'
            )

            # Set labels and title
            ax.set_xlabel(f'Axis {combo[0] + 1}')
            ax.set_ylabel(f'Axis {combo[1] + 1}')
            ax.set_zlabel(f'Axis {combo[2] + 1}')
            ax.set_title(f'{label_name} Projections (Axes {combo[0] + 1}, {combo[1] + 1}, {combo[2] + 1})')

            # Add legend
            ax.legend(loc='upper right')

            # Save the 3D plot
            path_3d_fig = os.path.join(label_save_dir, f"3d_discri_plot_axes_{combo[0] + 1}_{combo[1] + 1}_{combo[2] + 1}.pdf")
            plt.savefig(path_3d_fig, bbox_inches='tight')

            if show:
                plt.show()

            plt.close()