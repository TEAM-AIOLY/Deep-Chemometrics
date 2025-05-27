import os
import json


base_params = {
    "dataset_type": "Wheat",
    "data_path": "data/dataset/Wheat/", 
    "y_labels": ['wheat 30 classes'],
    "nb_classes":30,
    "batch_size": 512,
    "num_epochs": 1000,
    "model_name": "_ViT1D_Wheat_jz",
    "seed": 42,
    "spec_dims": None,
    "mean": None,
    "std": None,
     "DP" : 0.1,
     "LR": 0.00001,
     "WD": 0.0015
}

y_lab='Wheat_dt'

param_file = os.path.join(
    os.path.dirname(base_params['data_path']),'config',
    f"{base_params['model_name']}({y_lab}).json"
).replace("\\", "/")

os.makedirs(os.path.dirname(param_file), exist_ok=True)
    

param_variations = [
       {   "PS": 8, "DE": 64, "TL": 12, "HDS": 8, "MLP": 128},
       {   "PS": 12, "DE": 64, "TL": 12, "HDS": 8, "MLP": 128},
       {   "PS": 12, "DE": 64, "TL": 16, "HDS": 12, "MLP": 128},
       {   "PS": 12, "DE": 64, "TL": 16, "HDS": 8, "MLP": 256},
       {   "PS": 12, "DE": 128, "TL": 20, "HDS": 16, "MLP": 256},
       {   "PS": 16, "DE": 64, "TL": 12, "HDS": 8, "MLP": 128},
       {   "PS": 16, "DE": 64, "TL": 16, "HDS": 12, "MLP": 128},
       {   "PS": 16, "DE": 64, "TL": 16, "HDS": 8, "MLP": 256},
       {   "PS": 16, "DE": 128, "TL": 20, "HDS": 16, "MLP": 128},
       {   "PS": 24, "DE": 64, "TL": 12, "HDS": 8, "MLP": 128},
       {   "PS": 24, "DE": 128, "TL": 16, "HDS": 12, "MLP": 128},
       {   "PS": 24, "DE": 64, "TL": 16, "HDS": 8, "MLP": 256},
       {   "PS": 24, "DE": 128, "TL": 20, "HDS": 16, "MLP": 256},
]

# Add the variations to the param list
params_list = []
for idx, params in enumerate(param_variations):
    # Add 'id' to each parameter set dynamically
    params["ID"] = f"_ViT1D_Wheat_dt_jz{idx+1:03d}"
    # Update the base params with the new values
    param_dict = base_params.copy()
    param_dict.update(params)
    
    # Append the updated param dictionary to the list
    params_list.append(param_dict)

with open(param_file, 'w') as f:
    json.dump(params_list, f, indent=2)

print(f"Parameters saved to {param_file}")