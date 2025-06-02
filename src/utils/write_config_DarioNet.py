import os
import json
import itertools

# Define dataset configurations
datasets = {
    "Mango": {
        "data_path": "./data/dataset/Mango/mango_dm_full_outlier_removed2.mat",
        "y_lab": "dm_mango",
        "model_name": "_DarioNet_Mango_jz"
    },
    "ossl": {
        "data_path": "data/dataset/ossl/ossl_all_L1_v1.2.csv",
        "y_lab": "oc_usda",
        "model_name": "_DarioNet_ossl_jz"
    },
    "Wheat": {
        "data_path": "data/dataset/Wheat/",
        "y_lab": "Wheat_dt",
        "model_name": "_DarioNet_Wheat_jz"
    }
}

base_params = {
    "batch_size": 512,
    "num_epochs": 1000,
    "seed": 42,
    "spec_dims": None,
    "mean": None,
    "std": None,
    "WD": 0.0015
}

reg_beta = [0.015, 0.025, 0.05]
filter_size = [3, 5, 7]
lr_values = [0.002, 0.005, 0.01, 0.05]
dp_values = [0.005, 0.01, 0.02, 0.05, 0.1]




for dataset_type, config in datasets.items():
    params_list = []
    for idx, params in enumerate(param_variations):
        param_dict = base_params.copy()
        param_dict.update({
            "dataset_type": dataset_type,
            "data_path": config["data_path"],
            "y_labels": [config["y_lab"]],
            "model_name": config["model_name"]
        })
        param_dict.update(params)
        # Unique ID per dataset and variation
        param_dict["ID"] = f"{config['model_name']}{idx+1:03d}"
        params_list.append(param_dict)

    # Output config file per dataset
    param_file = os.path.join(
        os.path.dirname(config['data_path']), 'config',
        f"{config['model_name']}({config['y_lab']}).json"
    ).replace("\\", "/")
    os.makedirs(os.path.dirname(param_file), exist_ok=True)
    with open(param_file, 'w') as f:
        json.dump(params_list, f, indent=2)
    print(f"Parameters saved to {param_file}")