import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import json
import torch
from torch import nn, optim
import torch.utils.data as data_utils

from src import utils
from src.net import ViT_1D
from src.training.training import Trainer
from src.utils.misc import TrainerConfig
from src.utils.Loaddatasets import DatasetLoader

configs = [
    "data/dataset/ossl/config/_ViT1D_ossl_jz(oc_usda).json",
    "data/dataset/mango/config/_ViT1D_Mango_jz(dm_mango).json",
    "data/dataset/wheat/config/_ViT1D_Wheat_jz(Wheat_dt).json"
]

config_root = configs[0]
root = os.getcwd()
config_path = os.path.join(root, config_root)



try:
    params_dicts = DatasetLoader.parse_args(config_path)
    print(f"{len(params_dicts)} parameter sets")
except Exception as e:
    print(f"Error during execution: {e}")
    sys.exit(1)
    
data = DatasetLoader.get_data(config_path)


#########################################################################################################
# Main loop over configs
#########################################################################################################
for idx, params in enumerate(params_dicts):
    print(f"\n=== Running config {idx+1}/{len(params_dicts)}: {params.get('model_name', 'unnamed')} ===")

    batch_size = params.get('batch_size', 128)

    # Example: create DataLoaders from the already loaded data
    cal_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(data["x_cal"], dtype=torch.float32),
            torch.tensor(data["y_cal"], dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=True
    )
    val_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(data["x_val"], dtype=torch.float32),
            torch.tensor(data["y_val"], dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=False
    )
    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(
            torch.tensor(data["x_test"], dtype=torch.float32),
            torch.tensor(data["y_test"], dtype=torch.float32)
        ),
        batch_size=batch_size, shuffle=False
    )

    # Set up config for training
    config = TrainerConfig(model_name=params['model_name'])
    config.update_config(
        batch_size=params['batch_size'],
        learning_rate=params['LR'],
        num_epochs=params['num_epochs'],
        classification=False
    )

    params['spec_dims'] = mean.shape[0] if hasattr(mean, 'shape') else None
    params['mean'] = mean
    params['std'] = std

    # Model
    model = ViT_1D(
        mean=params['mean'],
        std=params['std'],
        seq_len=params['spec_dims'],
        patch_size=params['PS'],
        dim_embed=params['DE'],
        trans_layers=params['TL'],
        heads=params['HDS'],
        mlp_dim=params['MLP'],
        out_dims=len(params['y_labels'])
    )

    nb_train_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {nb_train_params}")
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=params.get('WD', 0.003/2))
    criterion = nn.MSELoss(reduction='none')

    # Training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=cal_loader,
        val_loader=val_loader,
        config=config
    )
