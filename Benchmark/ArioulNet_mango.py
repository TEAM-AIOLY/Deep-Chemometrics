import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import json

from pathlib import Path
import random
import itertools
from sklearn.metrics import confusion_matrix, f1_score

import torch
from torch import nn, optim
import torch.utils.data as data_utils

from src import utils
from src.net import Arioul_net
from src.training.training import Trainer
from src.utils.misc import TrainerConfig
from src.utils.dataset_loader import DatasetLoader
from src.utils import test_benchmark
from src.utils.testing import RMSEP, ccc

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def build_conv_config(depth, nf, ks):
    convs = []
    for i in range(depth):
        stride = 2 if i > 0 else 1       # downsample after first
        n_filters = nf * (2**i)          # double filters each layer
        convs.append((n_filters, ks, stride))
    return convs

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = os.getcwd()
data_path ="data/dataset/Mango/mango_splits.mat"

dataset= {"data_path": os.path.join(root,data_path),
    "dataset_type": "mango_new"}
wavelengths = np.arange(435, 1050 + 3, 3)  

base_params = {
    "spec_dims": None,
    "mean": None,
    "std": None,
     "DP" : 0.1,
    "LR": 0.001,
    "EPOCH": 1000,
     "WD": 0.003/2,
     "batch_size" : 512
}


DEPTH = [1, 2, 3, 4]          # number of conv layers
KS    = [3, 5, 7, 11]         # kernel sizes
NF    = [1, 3]       # base filters
FC    = [32,64, 128, 256]   # fully connected
LR    = [0.0001, 0.001, 0.005]
DP    = [0.1, 0.2, 0.5]

param_variations = [
    {"DEPTH": d, "KS": ks, "NF": nf, "FC": fc, "LR": lr, "DP": dp}
    for d, ks, nf, fc, lr, dp in itertools.product(DEPTH, KS, NF, FC, LR, DP)
]

paramsets = [{**base_params, **variation} for variation in param_variations]
model_type = "ArioulNet_mango"


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
   
    cls=False
    crit = nn.MSELoss(reduction='mean')

    # Set up config for training
    config = TrainerConfig(model_name=typ)
    config.update_config(
        batch_size=param["batch_size"],
        learning_rate=param["LR"],
        num_epochs=param["EPOCH"],
        classification=cls

    )
    conv_config = build_conv_config(param["DEPTH"], param["NF"], param["KS"])
    
    model = Arioul_net(
    input_dims=spec_dims,
    conv_config=conv_config,
    fc1_dims=param["FC"],
    dropout=param["DP"],
    out_dims=y_dim,
    mean=mean,
    std=std
).to(device)

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

    
    
    Y,y_pred =utils.test_benchmark(model, final_path, test_loader,config)    
    
    # Regression metrics
    perf = {
    "ccc": ccc(Y,y_pred),
    "r2":  1 - np.sum((Y - y_pred) ** 2) / (np.sum((Y - np.mean(Y)) ** 2) + 1e-12),
    "rmsep": RMSEP(Y,y_pred)
    }
    
    perf = {k: float(np.ravel(v)[0]) for k, v in perf.items()}
        
        
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

    
    metric_label = "R² Score"
   
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
    
    y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.cpu().numpy()
    lims = [min(np.min(Y), np.min(y_pred_np)), max(np.max(Y), np.max(y_pred_np))]


    # Scatter plot
    fig, ax = plt.subplots()
    ax.scatter(Y, y_pred_np, edgecolors='k', alpha=0.5)
    ax.plot(lims, lims, 'r')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Expected Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('')
    ax.text(
            0.98, 0.02,   
            f"CCC: {perf['ccc']:.2f}\nR²: {perf['r2']:.2f}\nRMSEP: {perf['rmsep']:.3f}",
            transform=ax.transAxes,   
            fontsize=12,
            va='bottom', ha='right',  
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
            color='red',
            fontweight='bold',
            fontfamily='serif'
        )
    plt.tight_layout()
    plt.grid()
    pdf_path = base_path + f"/predicted_vs_observed_{typ}.pdf"
    plt.savefig(pdf_path, format='pdf')
    plt.close('all')

    # Hexbin plot
    fig, ax = plt.subplots()
    hexbin = ax.hexbin(Y, y_pred_np, gridsize=50, cmap='viridis', mincnt=1)
    cb = fig.colorbar(hexbin, ax=ax, orientation='vertical')
    cb.set_label('Density')
    ax.plot(lims, lims, 'k-', label=typ)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Expected Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('')
    ax.text(
            0.98, 0.02,   
            f"CCC: {perf['ccc']:.2f}\nR²: {perf['r2']:.2f}\nRMSEP: {perf['rmsep']:.3f}",
            transform=ax.transAxes,   
            fontsize=12,
            va='bottom', ha='right',  
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
            color='red',
            fontweight='bold',
            fontfamily='serif'
        )
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=5, fontsize=12)
    plt.tight_layout()
    plt.grid()
    hexbin_pdf_path = base_path + f"/fig_hexbin.pdf"
    plt.savefig(hexbin_pdf_path, format='pdf')
    plt.close('all')
    
    metrics_dict = {
            'dataset_type': typ,
            'num_epochs': param['EPOCH'],
            "batch_size": param['batch_size'],
            "LR": param['LR'],
            "WD": param['WD'],
            "RMSE": perf['rmsep'],
            "R2": perf['r2'],
            "MSE": perf['ccc'],
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