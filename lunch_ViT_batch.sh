
#_______________ FOR WSL ONLY ON LOCAL ______________
# ______________activate venv ! _____________________
#/mnt/c/Users/fabdelghafo/im_regis/Scripts
#dos2unix ./activate
#source ./ activate
#____________________________________________________


VENV_PATH="/mnt/c/Users/fabdelghafo/im_regis/Scripts"
dos2unix "$VENV_PATH/activate"
source "$VENV_PATH/activate"
which python3

PYTHON_SCRIPT="./read_json_argparse_dict.py"

# Specify the path to your JSON configuration file
CONFIG_FILE="./data/dataset/ossl/config/_ViT1D_(oc_us).json"

# Run the Python script with the JSON config file as an argument
python3 "$PYTHON_SCRIPT" --config_file "$CONFIG_FILE"*
