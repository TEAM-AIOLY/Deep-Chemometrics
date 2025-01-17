
PYTHON_SCRIPT="./train_ViT1D_batch_argparse.py"

# Specify the path to your JSON configuration file
CONFIG_FILE="./data/dataset/ossl/config/_ViT1D_(oc_us).json"

# Run the Python script with the JSON config file as an argument
python3 "$PYTHON_SCRIPT" --config_file "$CONFIG_FILE"*
