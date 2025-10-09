import itertools
import json

# ---- Base parameters ----
base_params = {
    "spec_dims": None,
    "mean": None,
    "std": None,
    "DP": 0.2,
    "LR": 0.001,
    "EPOCH": 40,
    "WD": 0.003 / 2,
    "batch_size": 512,
}

# ---- Search grid ----
PS = [40, 80, 100, 120]
DE = [64, 128]
TL = [12, 24]
HDS = [8, 12]
MLP = [128, 512]

# ---- Cartesian product of all variations ----
param_variations = [
    {"PS": ps, "DE": de, "TL": tl, "HDS": hds, "MLP": mlp}
    for ps, de, tl, hds, mlp in itertools.product(PS, DE, TL, HDS, MLP)
]

# ---- Merge base params with variations ----
paramsets = [{**base_params, **variation} for variation in param_variations]


config_path ="C:/00_aioly/GitHub/Deep-chemometrics/data/dataset/Mango/config/config_mango_vit.json"
# ---- Save to JSON ----
with open(config_path, "w") as f:
    json.dump(paramsets, f, indent=2)

print(f"Generated {len(paramsets)} parameter sets and saved to paramsets.json")