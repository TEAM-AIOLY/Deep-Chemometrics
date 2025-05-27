import os
import json


base_params = {
    "dataset_type": "ossl",
    "ossl_type": "visnir",  # Options: "mir", "nir", "visnir"
    "data_path": "data/dataset/ossl/ossl_all_L1_v1.2.csv",  
    "y_labels": ['oc_usda.c729_w.pct'],
    "batch_size": 1024,
    "num_epochs": 1000,
    "model_name": "_ViT1D_ossl_jz",
    "seed": 42,
    "spec_dims": None,
    "mean": None,
    "std": None,
    "slope": 0.2,
    "offset": 0.2,
    "noise": 0.002,
    "shift": 0.05,
     "DP" : 0.5,
     "LR": 0.01,
     "WD": 0.0015
}

y_lab='oc_usda'

param_file = os.path.join(
    os.path.dirname(base_params['data_path']),'config',
    f"{base_params['model_name']}({y_lab}).json"
).replace("\\", "/")

os.makedirs(os.path.dirname(param_file), exist_ok=True)
    

param_variations = [
       {   "PS": 32, "DE": 16, "TL": 4, "HDS": 4, "MLP": 16},
       {   "PS": 32, "DE": 16, "TL": 8, "HDS": 8, "MLP": 16},
       {   "PS": 32, "DE": 16, "TL": 8, "HDS": 8, "MLP": 32},
       {   "PS": 32, "DE": 32, "TL": 12, "HDS": 12,"MLP": 32},
       {   "PS": 32, "DE": 32, "TL": 12, "HDS": 12, "MLP": 64},
       {   "PS": 32, "DE": 64, "TL": 12, "HDS": 12, "MLP": 64},
       {   "PS": 32, "DE": 32, "TL": 16, "HDS": 16, "MLP": 32},
       {   "PS": 64, "DE": 64, "TL": 8, "HDS": 8, "MLP": 32},
       {   "PS": 64, "DE": 64, "TL": 8, "HDS": 8, "MLP": 64},
       {   "PS": 64, "DE": 64, "TL": 16, "HDS": 16, "MLP": 32},
       {   "PS": 64, "DE": 64, "TL": 16, "HDS": 16, "MLP": 64},
       {   "PS": 96, "DE": 32, "TL": 12, "HDS": 12, "MLP": 32},
       {   "PS": 96, "DE": 64, "TL": 12, "HDS": 12, "MLP": 32},
       {   "PS": 96, "DE": 64, "TL": 12, "HDS": 12, "MLP": 64},
       {   "PS": 96, "DE": 32, "TL": 12, "HDS": 12, "MLP": 32},
       {   "PS": 96, "DE": 32, "TL": 16, "HDS": 16, "MLP": 32},
       {   "PS": 96, "DE": 32, "TL": 8, "HDS": 8, "MLP": 32},
       {   "PS": 128, "DE": 32, "TL": 8, "HDS": 8, "MLP": 32},
       {   "PS": 128, "DE": 32, "TL": 16, "HDS": 16, "MLP": 32},
       {   "PS": 128, "DE": 64, "TL": 16, "HDS": 16, "MLP": 64},
       {   "PS": 128, "DE": 32, "TL": 24, "HDS": 16, "MLP": 32},
       
           
]

# Add the variations to the param list
params_list = []
for idx, params in enumerate(param_variations):
    # Add 'id' to each parameter set dynamically
    params["ID"] = f"_ViT1D_ossl_jz{idx+1:03d}"
    
    # Update the base params with the new values
    param_dict = base_params.copy()
    param_dict.update(params)
    
    # Append the updated param dictionary to the list
    params_list.append(param_dict)

with open(param_file, 'w') as f:
    json.dump(params_list, f, indent=2)

print(f"Parameters saved to {param_file}")