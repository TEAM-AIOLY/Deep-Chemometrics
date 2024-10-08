import json
import os

# Define the base parameters
base_params = {
    "dataset_type": "mir",
    "data_path": "./data/dataset/ossl/ossl_all_L1_v1.2.csv",  
    "y_labels": [],
    "batch_size": 1024,
    "num_epochs": 500,
    "model_name": "_ResNet18_OSSL_",
    "seed": 42,
    "spec_dims": None,
    "mean": None,
    "std": None
}

# List of different y_labels
y_labels_options = [
    ["oc_usda.c729_w.pct"],
    ["na.ext_usda.a726_cmolc.kg"],
    ["clay.tot_usda.a334_w.pct"],
    ["k.ext_usda.a725_cmolc.kg"],
    ["ph.h2o_usda.a268_index"],
]


params_list = []

# Generate parameters for each y_labels option
for y_labels in y_labels_options:
    params = base_params.copy()  # Create a copy of the base params
    params["y_labels"] = y_labels
    params_list.append(params)
    
params_all = base_params.copy()
params_all["y_labels"] = [label for option in y_labels_options for label in option]  # Combine all y_labels
params_list.append(params_all)

data_path ="./data/dataset/ossl/ossl_all_L1_v1.2.csv"
param_file = os.path.dirname(data_path)+ '/optuna_ResNet18_params_mir.json'

# Write the list to a JSON file
with open(param_file, 'w') as f:
    json.dump(params_list, f, indent=2)