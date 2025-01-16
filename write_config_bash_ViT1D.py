import os
import json


base_params = {
    "dataset_type": "nir",
    "data_path": "./data/dataset/ossl/ossl_all_L1_v1.2.csv",  
    "y_labels": [],
    "batch_size": 1024,
    "num_epochs": 500,
    "model_name": "_ViT1D_",
    "seed": 42,
    "spec_dims": None,
    "mean": None,
    "std": None
}

# List of different possible y_labels
y_labels_options = [
    ["oc_usda.c729_w.pct"],
    ["na.ext_usda.a726_cmolc.kg"],
    ["clay.tot_usda.a334_w.pct"],
    ["k.ext_usda.a725_cmolc.kg"],
    ["ph.h2o_usda.a268_index"],
]

params_list = []
y_labels =y_labels_options[0]
base_params["y_labels"] = y_labels
y_labels_short_str = y_labels[0][:5]

param_file = os.path.join(
    os.path.dirname(base_params['data_path']),'config',
    f"{base_params['model_name']}({y_labels_short_str}).json"
).replace("\\", "/")




param_variations = [
    {"LR": 0.001, "WD": 0.0055, "PS": 32, "DE": 16, "TL": 4, "HDS": 4, "MLP": 16},
    {"LR": 0.0009, "WD": 0.0052, "PS": 48, "DE": 16, "TL": 4, "HDS": 4, "MLP": 16},
    {"LR": 0.0008, "WD": 0.005, "PS": 64, "DE": 24, "TL": 6, "HDS": 6, "MLP": 24},
    {"LR": 0.0004, "WD": 0.0038, "PS": 64, "DE": 16, "TL": 6, "HDS": 6, "MLP": 24},
    {"LR": 0.0007, "WD": 0.0048, "PS": 80, "DE": 24, "TL": 8, "HDS": 8, "MLP": 24},
    {"LR": 0.00035, "WD": 0.0035, "PS": 96, "DE": 32, "TL": 10, "HDS": 10, "MLP": 32},
    {"LR": 0.0006, "WD": 0.0045, "PS": 112, "DE": 36, "TL": 12, "HDS": 12, "MLP": 36},
    {"LR": 0.0005, "WD": 0.004, "PS": 128, "DE": 48, "TL": 14, "HDS": 14, "MLP": 48},
    {"LR": 0.0003, "WD": 0.003, "PS": 96, "DE": 32, "TL": 10, "HDS": 10, "MLP": 40},
    {"LR": 0.0002, "WD": 0.0025, "PS": 112, "DE": 36, "TL": 12, "HDS": 12, "MLP": 48}
]

# Add the variations to the param list
params_list = []
for idx, params in enumerate(param_variations):
    # Add 'id' to each parameter set dynamically
    params["id"] = f"Id_{idx+1:03d}"
    
    # Update the base params with the new values
    param_dict = base_params.copy()
    param_dict.update(params)
    
    # Append the updated param dictionary to the list
    params_list.append(param_dict)

with open(param_file, 'w') as f:
    json.dump(params_list, f, indent=2)

print(f"Parameters saved to {param_file}")



