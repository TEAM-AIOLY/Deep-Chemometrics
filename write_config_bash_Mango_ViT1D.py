import os
import json


base_params = {
    "dataset_type": "VisNIR",
    "data_path": "./data/dataset/Mango/mango_dm_full_outlier_removed2.mat",  
    "y_labels": ['dm_mango'],
    "batch_size": 512,
    "num_epochs": 2000,
    "model_name": "_ViT1D_Mango_optim",
    "seed": 42,
    "spec_dims": None,
    "mean": None,
    "std": None,
    "slope": 0.,
    "offset": 0.,
    "noise": 0.00005,
    "shift": 0.,
     "DP" : 0.5,
}

y_lab='dm_mango'

param_file = os.path.join(
    os.path.dirname(base_params['data_path']),'config',
    f"{base_params['model_name']}({y_lab}).json"
).replace("\\", "/")

os.makedirs(os.path.dirname(param_file), exist_ok=True)
    

param_variations = [
       { "LR": 0.01, "WD": 0.0015, "PS": 10, "DE": 64, "TL": 8, "HDS": 8, "MLP": 64},
       { "LR": 0.001, "WD": 0.0015, "PS": 10, "DE": 64, "TL": 8, "HDS": 8, "MLP": 64},
       { "LR": 0.02, "WD": 0.0015, "PS": 10, "DE": 64, "TL": 8, "HDS": 8, "MLP": 64},
       { "LR": 0.01, "WD": 0.0015, "PS": 10, "DE": 64, "TL": 8, "HDS": 8, "MLP": 32},
       { "LR": 0.01, "WD": 0.0015, "PS": 10, "DE": 32, "TL": 8, "HDS": 8, "MLP": 32},
       { "LR": 0.01, "WD": 0.0015, "PS": 8, "DE": 64, "TL": 16, "HDS": 16, "MLP": 32},
       { "LR": 0.01, "WD": 0.0015, "PS": 8, "DE": 64, "TL": 16, "HDS": 16, "MLP": 64},
       { "LR": 0.01, "WD": 0.0015, "PS": 4, "DE": 64, "TL": 16, "HDS": 16, "MLP": 64},
       { "LR": 0.01, "WD": 0.0015, "PS": 4, "DE": 32, "TL": 16, "HDS": 16, "MLP": 64}
]

# Add the variations to the param list
params_list = []
for idx, params in enumerate(param_variations):
    # Add 'id' to each parameter set dynamically
    params["ID"] = f"Id_mango_optim{idx+1:03d}"
    
    # Update the base params with the new values
    param_dict = base_params.copy()
    param_dict.update(params)
    
    # Append the updated param dictionary to the list
    params_list.append(param_dict)

with open(param_file, 'w') as f:
    json.dump(params_list, f, indent=2)

print(f"Parameters saved to {param_file}")



