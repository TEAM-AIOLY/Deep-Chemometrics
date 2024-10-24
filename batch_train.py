import argparse
import json
import os

def load_config(file_path, index):
    with open(file_path, 'r') as file:
        config_list = json.load(file)
    if index < 0 or index >= len(config_list):
        raise IndexError(f"Index {index} is out of range for configuration list with {len(config_list)} entries.")
    return config_list[index]

def get_args():
 
    parser = argparse.ArgumentParser(description="Read one configuration from the JSON file.")
    
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the JSON configuration file containing a list of parameter sets.')
    parser.add_argument('--config_index', type=int, required=True,
                        help='Index of the configuration to read from the JSON file.')
    args = parser.parse_args()
    
    # Load and return the configuration dictionary based on the arguments
    config = load_config(args.config_path, args.config_index)
    
    return config

if __name__ == "__main__":
    config = get_args()
    print(config)