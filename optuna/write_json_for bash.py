import json
import random

def decide_randomly():
    """Randomly decide to return True (provide bounds) or None."""
    return random.choice([True, None])

def increase_bounds(min_val, max_val, step):
    """Increase the bounds from the minimum value by a fixed step."""
    new_min = min_val + step
    if new_min < max_val:
        return [new_min, max_val]
    return [min_val, max_val] 
def generate_configurations(num_configs):
    configurations = []

    # Define default ranges
    default_ranges = {
        "lr": (1e-5, 1e-1, 1e-1),  # (min, max, step)
        "weight_decay": (1e-4, 1e-2, 1e-2),
        "slope": (0.0, 0.3, 0.1),
        "offset": (0.0, 0.3, 0.05),
        "noise": (0.0, 0.3, 0.05),
        "shift": (0.0, 0.3, 0.05),
        "patch_size": (8, 48, 4),  # (min, max, step)
        "dim_embed": (16, 64, 4),  # (min, max, step)
        "trans_layers": (2, 16, 2),  # (min, max, step)
        "heads": (2, 16, 2),  # (min, max, step)
        "mlp_dim": (4, 64, 4)  # (min, max, step)
    }

    for _ in range(num_configs):
        config = {
            "data_path": "./data/soil_data.csv",
            "dataset_type": "mir",
            "y_labels":["oc_usda.c729_w.pct","na.ext_usda.a726_cmolc.kg","clay.tot_usda.a334_w.pct", "k.ext_usda.a725_cmolc.kg", "ph.h2o_usda.a268_index" ],
            "num_epochs": 1000,
            "batch_size": 1024
        }
            
            
        for param in ["lr", "weight_decay", "slope", "offset", "noise", "shift", "patch_size", "dim_embed", "trans_layers", "heads", "mlp_dim"]:
            decision = decide_randomly()
            if decision is True:
                min_val, max_val, step = default_ranges[param]
                config[param] = increase_bounds(min_val, max_val, step)
            else:
                config[param] = None

        configurations.append(config)

    return configurations

num_configs = 50
configurations = generate_configurations(num_configs)

# Save configurations to JSON
with open('configurations.json', 'w') as f:
    json.dump(configurations, f, indent=4)