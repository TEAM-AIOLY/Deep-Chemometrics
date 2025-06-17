import os
import json

# Fixed base parameters
base_params = {
    "dataset_type": "Mango",
    "data_path": "data/dataset/Mango/mango_dm_full_outlier_removed2.mat",  
    "y_labels": ['dm_mango'],
    "batch_size": 512,
    "num_epochs": 1000,
    "model_name": "_ViT1D_Mango_jz",
    "seed": 42,
    "spec_dims": None,
    "mean": None,
    "std": None,
    "slope": 0.,
    "offset": 0.,
    "noise": 0.00005,
    "shift": 0.,
    "DP": 0.1,
    "WD": 0.0015,
    "DE": 64,
    "TL": 12,
    "HDS": 8,
    "MLP": 64,
    "ED" : 0.02
}

# Target label name
y_lab = 'dm_mango'

# Generate the path to the JSON configuration file
param_file = os.path.join(
    os.path.dirname(base_params['data_path']), 'config',
    f"{base_params['model_name']}({y_lab}).json"
).replace("\\", "/")


os.makedirs(os.path.dirname(param_file), exist_ok=True)

# Values of Patch Size (PS) and Learning Rate (LR) to test
patch_sizes = [16, 32, 64, 128]
learning_rates = [5e-5, 1e-4, 5e-4, 1e-3, 1e-2]

# Generate all combinations of PS × LR
params_list = []
idx = 1
for ps in patch_sizes:
    for lr in learning_rates:
        param_dict = base_params.copy()
        param_dict["PS"] = ps
        param_dict["LR"] = lr
        param_dict["ID"] = f"_ViT1D_Mango_PS{ps}_LR{lr:.4f}".replace('.', 'p')
        params_list.append(param_dict)
        idx += 1

# Save the configuration list to the JSON file
with open(param_file, 'w') as f:
    json.dump(params_list, f, indent=2)

print(f"Parameters saved to {param_file}")
