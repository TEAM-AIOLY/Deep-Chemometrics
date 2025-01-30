
PYTHON_SCRIPT="./train_ViT1D_mango_batch_argparse.py"

# Specify the path to your JSON configuration file
CONFIG_FILE="./data/dataset/Mango/config/_ViT1D_Mango(dm_mango).json"

# Run the Python script with the JSON config file as an argument
python3 "$PYTHON_SCRIPT" --config_file "$CONFIG_FILE"*
