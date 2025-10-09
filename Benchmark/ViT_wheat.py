import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import json

from pathlib import Path
import seaborn as sns
import random
import itertools
from sklearn.metrics import confusion_matrix, f1_score

import torch
from torch import nn, optim
import torch.utils.data as data_utils

from src import utils
from src.net import ViT_1D
from src.training.training_LR_Sched_stop import Trainer
from src.utils.misc import TrainerConfig
from src.utils.dataset_loader import DatasetLoader
from src.utils import test_benchmark

def set_seed(seed=42, strict_reproducibility=False):
    import random, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if strict_reproducibility:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster, allows some nondeterminism but robust overall
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = os.getcwd()

dataset= {"data_path": "data/dataset/Wheat_dt/",
    "dataset_type": "wheat"}


base_params = {
    "spec_dims": None,
    "mean": None,
    "std": None,
     "DP" : 0.2,
    "LR": 0.0001,
    "EPOCH": 1000,
     "WD": 0.003/2,
     "batch_size" : 512
}

PS = [40,60,80,200]
DE = [64,128]
TL = [16]
HDS=[ 10]
MLP = [256,512]

param_variations = [
    {"PS": ps, "DE": de, "TL": tl, "HDS": hds, "MLP": mlp}
    for ps, de, tl, hds, mlp in itertools.product(PS, DE, TL, HDS, MLP)
]
paramsets = [{**base_params, **variation} for variation in param_variations]
print(f"Total parameter sets to evaluate: {len(paramsets)}")

model_type = "ViT_1D"

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

all_params = []
all_metrics = []


for i,param in enumerate(paramsets):
    param_id = f"Run_{i:02d}"
    print(f"running {param_id} with parameters: {param}")
    
    param_with_id = param.copy()
    param_with_id["Run_ID"] = param_id
    all_params.append(param_with_id)

    typ =dataset["dataset_type"]
    spec_dims = data["x_cal"].shape[1]
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
    
    local_run= os.path.join("Benchmark",f"{typ}",f"{model_type}",f"{param_id}")   
    base_path =os.path.join(root,local_run)
    os.makedirs(base_path, exist_ok=True)
    config.update_config(save_path=Path(base_path)) 
    
    trainer = Trainer(model=model, optimizer=optimizer, criterion=crit, train_loader=cal_loader, val_loader=val_loader, config=config,verbose = False)
    train_losses, val_losses,  val_metrics, final_path = trainer.train()
    
    if cls:
        val_means = [np.mean(m) if isinstance(m, (list, np.ndarray)) else m for m in val_metrics]
        best_epoch = int(np.argmax(val_means))
    else:
   
        val_means = [np.mean(m) if isinstance(m, (list, np.ndarray)) else m for m in val_metrics]
        best_epoch = int(np.argmax(val_means))

    
    
    
    y_pred,Y =utils.test_benchmark(model, final_path, test_loader,config)    
    
    
    if cls:
        perf = {
        "confusion_matrix": confusion_matrix(
            np.argmax(Y, axis=1) if Y.ndim > 1 else Y,
            np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
        ),
        "f1_score": f1_score(
            np.argmax(Y, axis=1) if Y.ndim > 1 else Y,
            np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred,
            average="macro"
        )
        }
        conf_mat = perf.get("confusion_matrix")
        f1 = perf.get("f1_score")
        accuracy = np.trace(conf_mat) / np.sum(conf_mat)
        precision = np.diag(conf_mat) / (np.sum(conf_mat, axis=0) + 1e-12)
        recall = np.diag(conf_mat) / (np.sum(conf_mat, axis=1) + 1e-12)
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)  
        
    else:
        # Regression metrics
        y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.cpu().numpy()
        mse = np.mean((Y  - y_pred_np) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((Y  - y_pred_np) ** 2) / (np.sum((Y  - np.mean(Y )) ** 2) + 1e-12)
       

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
        sns.heatmap(conf_mat, annot=False, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        confmat_pdf_path = base_path + f"/confusion_matrix_{typ}.pdf"
        plt.savefig(confmat_pdf_path, format='pdf')
        plt.close('all')
    else:
        y_true = data["y_test"]
        y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.cpu().numpy()
        lims = [min(np.min(y_true), np.min(y_pred_np)), max(np.max(y_true), np.max(y_pred_np))]

        # Scatter plot
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred_np, edgecolors='k', alpha=0.5)
        ax.plot(lims, lims, 'r')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Real Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Real Values')
        ax.text(1.02, 1, f"CCC: {perf['ccc'][0]:.2f}\nR²: {perf['r2'][0]:.2f}\nRMSEP: {perf['rmsep'][0]:.3f}",
                transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                color='red', fontweight='bold', fontfamily='serif')
        plt.tight_layout()
        plt.grid()
        pdf_path = base_path + f"/predicted_vs_observed_{typ}.pdf"
        plt.savefig(pdf_path, format='pdf')
        plt.close('all')

        # Hexbin plot
        fig, ax = plt.subplots()
        hexbin = ax.hexbin(y_true, y_pred_np, gridsize=50, cmap='viridis', mincnt=1)
        cb = fig.colorbar(hexbin, ax=ax, orientation='vertical')
        cb.set_label('Density')
        ax.plot(lims, lims, 'k-', label=typ)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Real Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Real Values for dry matter')
        ax.text(1.02, 1, f"CCC: {perf['ccc'][0]:.2f}\nR²: {perf['r2'][0]:.2f}\nRMSEP: {perf['rmsep'][0]:.3f}",
                transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                color='red', fontweight='bold', fontfamily='serif')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=True, ncol=5, fontsize=12)
        plt.tight_layout()
        plt.grid()
        hexbin_pdf_path = base_path + f"/fig_hexbin.pdf"
        plt.savefig(hexbin_pdf_path, format='pdf')
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
            "Run_ID": f"run_{param_id}"
        }
    
    metrics_dict["best_epoch"] = best_epoch
    metrics_dict["Run_ID"] = param_id
    all_metrics.append(metrics_dict)
    
        
    with open(os.path.join(base_path, 'metrics.txt'), 'w') as f:
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")
    
summary_dir = os.path.dirname(base_path)
with open(os.path.join(summary_dir, "all_params.json"), "w") as f:
    json.dump(all_params, f, indent=2)

with open(os.path.join(summary_dir, "all_metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)


best_model_path = Path(base_path).with_name(f"{Path(base_path).stem}_best.pth")
if best_model_path.exists():
    try:
        os.remove(best_model_path)
    except Exception as e:
        print(f"Could not delete best model file {best_model_path}: {e}")





