#!/bin/bash

# Path to the JSON file containing configurations
CONFIG_FILE="configurations.json"

# Get the number of configurations
num_configs=$(python3 -c "import json; print(len(json.load(open('$CONFIG_FILE'))))")

# Loop through each configuration
for ((i=0; i<num_configs; i++)); do
    # Get the configuration as a JSON string
    config=$(python3 -c "import json; print(json.dumps(json.load(open('$CONFIG_FILE'))[$i]))")

    # Run the Python script with the current configuration
    python3 ./optim_ViT1D_batch.py --config "$config"
done