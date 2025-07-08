import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import itertools
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score

import torch
from torch import nn, optim
import torch.utils.data as data_utils

from src import utils
from src.net import ViT_1D
from src.training.training import Trainer
from src.utils.misc import TrainerConfig
from src.utils.dataset_loader import DatasetLoader

root = os.getcwd()

data_root = [{
    "data_path": "C:/00_aioly/GitHub/Deep-Chemometrics/data/dataset/mango/mango_dm_full_outlier_removed2.mat",
    "dataset_type": "mango"},        
{"data_path": "C:/00_aioly/GitHub/Deep-Chemometrics/data/dataset/ossl/ossl_database.mat",
    "dataset_type": "ossl"},
{"data_path": "C:/00_aioly/GitHub/Deep-Chemometrics/data/dataset/wheat/",
    "dataset_type": "wheat"}]            
             
base_params = {
    "spec_dims": None,
    "mean": None,
    "std": None,
    "DP" : 0.2,
    "LR": 0.00001,
    "EPOCH": 10,
     "WD": 0.003/2,
     "batch_size" : 512
}

PS = [40]
DE = [64]
TL = [6]
HDS=[ 6]
MLP = [32]

param_variations = [
    {"PS": ps, "DE": de, "TL": tl, "HDS": hds, "MLP": mlp}
    for ps, de, tl, hds, mlp in itertools.product(PS, DE, TL, HDS, MLP)
]


paramsets = [{**base_params, **variation} for variation in param_variations]
model_type = "Vit_1D"

dataset=data_root[2]

data = DatasetLoader.load(dataset)
mean = np.mean(data["x_cal"], axis=0)
std = np.std(data["x_cal"], axis=0)


cal_loader = data_utils.DataLoader(
    data_utils.TensorDataset(
        torch.tensor(data["x_cal"], dtype=torch.float32),
        torch.tensor(data["y_cal"], dtype=torch.float32)
    ),
    batch_size=base_params["batch_size"], shuffle=True
)
val_loader = data_utils.DataLoader(
    data_utils.TensorDataset(
        torch.tensor(data["x_val"], dtype=torch.float32),
        torch.tensor(data["y_val"], dtype=torch.float32)
    ),
    batch_size=base_params["batch_size"], shuffle=False
)
test_loader = data_utils.DataLoader(
    data_utils.TensorDataset(
        torch.tensor(data["x_test"], dtype=torch.float32),
        torch.tensor(data["y_test"], dtype=torch.float32)
    ),
    batch_size=base_params["batch_size"], shuffle=False
)

for i,param in enumerate(paramsets):
    param_id = f"Run_{i:02d}"

    typ =dataset["dataset_type"]
    spec_dims = data["x_cal"].shape[1]
    y_dim =data["y_cal"].shape[1]
    y_dim =data["y_cal"].shape[1]
    if typ == 'wheat': 
        cls=True
        crit = nn.BCEWithLogitsLoss(reduction='mean')
    else: 
        cls=False
        crit = nn.MSELoss(reduction='none')

    # Set up config for training
    config = TrainerConfig(model_name=typ)
    config.update_config(
        batch_size=param["batch_size"],
        learning_rate=param["LR"],
        num_epochs=param["EPOCH"],
        classification=cls
    )

    model = ViT_1D(
    mean=mean,
    std=std,
    seq_len=spec_dims,
    patch_size=param['PS'],
    dim_embed=param['DE'],
    trans_layers=param['TL'],
    heads=param['HDS'],
    mlp_dim=param['MLP'],
    out_dims=y_dim
)


    nb_train_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=param["WD"])

    local_run=f"Benchmark_HS/{typ}/{model_type}/{param_id}"
    base_path =os.path.join(root,local_run)
    os.makedirs(base_path, exist_ok=True)
    config.update_config(save_path=base_path)

    trainer = Trainer(model=model, optimizer=optimizer, criterion=crit, train_loader=cal_loader, val_loader=val_loader, config=config)
    train_losses, val_losses,  val_metrics, final_path,best_epoch = trainer.train()
   
    perf,y_pred =utils.test(model, final_path, test_loader,config)    

    if cls:
        # Classification metrics
        conf_mat = perf.get("confusion_matrix")
        f1 = perf.get("f1_score")
        accuracy = np.trace(conf_mat) / np.sum(conf_mat)
        precision = np.diag(conf_mat) / (np.sum(conf_mat, axis=0) + 1e-12)
        recall = np.diag(conf_mat) / (np.sum(conf_mat, axis=1) + 1e-12)
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)  
    else:
        # Regression metrics
        y_true = data["y_test"]
        y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.cpu().numpy()
        mse = np.mean((y_true - y_pred_np) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred_np) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
       

    train_losses_np = [loss.numpy() for loss in train_losses]
    val_losses_np = [loss.numpy() for loss in val_losses]


    maxplot_loss =20
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Colors
    train_color = '#1f77b4'  # blue
    val_color = '#ff7f0e'  # orange
    metric_color = '#2ca02c'  # green

    # Plot losses
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=train_color)
    ax1.plot(train_losses_np, label='Training Loss', color=train_color, linewidth=2)
    ax1.plot(val_losses_np, label='Validation Loss', color=val_color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=train_color)
    ax1.set_ylim(0, min(maxplot_loss, max(max(train_losses_np), max(val_losses_np)) * 1.1))  # Cap loss axis to 10

    # Legends for loss
    ax1.legend(loc='upper left')

    # Metrics on second axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Metric', color=metric_color)
    ax2.tick_params(axis='y', labelcolor=metric_color)

    if not cls:
        metric_label = "R² Score"
    else:
        metric_label = "F1 Score"

    # Handle both single and multi-output/class cases
    if isinstance(val_metrics[0], (list, np.ndarray)):
        for i in range(len(val_metrics[0])):
            metric_scores = [scores[i] for scores in val_metrics]
            ax2.plot(metric_scores, label=f'{metric_label} y{i}', linestyle='--', color=metric_color, linewidth=2)
    else:
        ax2.plot(val_metrics, label=metric_label, linestyle='--', color=metric_color, linewidth=2)

    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')

    plt.title('Training & Validation Loss and Metrics')
    fig.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    pdf_path =base_path+ f"/Training_{typ}.pdf"
    plt.savefig(pdf_path, format='pdf')
    plt.close('all')

    if  cls:
 
        conf_mat = perf.get("confusion_matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        confmat_pdf_path = base_path + f"/confusion_matrix_{typ}.pdf"
        plt.savefig(confmat_pdf_path, format='pdf')
        plt.close('all')
    else:
        y_true = data["y_test"].ravel()
        y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.cpu().numpy().ravel()

        min_val = min(np.min(y_true), np.min(y_pred_np))
        max_val = max(np.max(y_true), np.max(y_pred_np))
        padding = 0.05 * (max_val - min_val)
        lims = [min_val - padding, max_val + padding]

        # Scatter plot
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred_np, edgecolors='k', alpha=0.5)
        ax.plot(lims, lims, 'r')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Real Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Real Values')
        plt.tight_layout()
        plt.grid()
        fig.text(0.5, -0.05, f"CCC: {perf['ccc'][0]:.2f}   R²: {perf['r2'][0]:.2f}   RMSEP: {perf['rmsep'][0]:.3f}",
                ha='center', va='top', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                color='red', fontweight='bold', fontfamily='serif')
        pdf_path = base_path + f"/predicted_vs_observed_{typ}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close('all')

        # Hexbin plot
        fig, ax = plt.subplots()
        hexbin = ax.hexbin(data["y_test"], y_pred, gridsize=50, cmap='viridis', mincnt=1)
        cb = fig.colorbar(hexbin, ax=ax, orientation='vertical')
        cb.set_label('Density')
        ax.plot(lims, lims, 'k-')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Real Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Real Values for dry matter')
        plt.tight_layout()
        plt.grid()
        # Place annotation below the plot, centered
        fig.text(0.5, -0.05, f"CCC: {perf['ccc'][0]:.2f}   R²: {perf['r2'][0]:.2f}   RMSEP: {perf['rmsep'][0]:.3f}",
                ha='center', va='top', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                color='red', fontweight='bold', fontfamily='serif')
        hexbin_pdf_path = base_path + f"/fig_hexbin.pdf"
        plt.savefig(hexbin_pdf_path, format='pdf', bbox_inches='tight')
        plt.close('all')

    if cls:
            metrics_dict = {
                'dataset_type': typ,
                'num_epochs': param['EPOCH'],
                "batch_size": param['batch_size'],
                "LR": param['LR'],
                "WD": param['WD'],
                "F1": f1,
                "accuracy": accuracy,
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "precision": precision.tolist() if isinstance(precision, np.ndarray) else precision,
                "recall": recall.tolist() if isinstance(recall, np.ndarray) else recall,
                "N_parameters": nb_train_params,
                "model_name": model_type,
                "best_epoch": best_epoch,
                "Run_ID": f"run_{param_id}"
            }
    else:
        metrics_dict = {
            'dataset_type': typ,
            'num_epochs': param['EPOCH'],
            "batch_size": param['batch_size'],
            "LR": param['LR'],
            "WD": param['WD'],
            "RMSE": rmse,
            "R2": r2,
            "MSE": mse,
            "N_parameters": nb_train_params,
            "model_name": model_type,
            "best_epoch": best_epoch,
            "Run_ID": f"run_{param_id}"
        }

    with open(os.path.join(base_path, 'metrics.txt'), 'w') as f:
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")