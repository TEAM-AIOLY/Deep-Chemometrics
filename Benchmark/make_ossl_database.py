import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scipy.io as sio
from src.utils.Loaddatasets import DatasetLoader



config_root = "data/dataset/ossl/config/_ViT1D_ossl_jz(oc_usda).json"
root = os.getcwd()
config_path = os.path.join(root, config_root)



try:
    params_dicts = DatasetLoader.parse_args(config_path)
    print(f"{len(params_dicts)} parameter sets")
except Exception as e:
    print(f"Error during execution: {e}")
    sys.exit(1)

data = DatasetLoader.get_data(config_path)
 
data["x_cal"]
data["y_cal"]
data["x_val"]
data["y_val"]
data["x_test"]
data["y_test"]

wv =np.linspace(350, 2500,data["x_cal"].shape[1]) 


mat_dict = {
    "x_cal": data["x_cal"],
    "y_cal": data["y_cal"],
    "x_val": data["x_val"],
    "y_val": data["y_val"],
    "x_test": data["x_test"],
    "y_test": data["y_test"],
    "wv": wv
}

save_path = os.path.join(root, "data", "dataset", "ossl", "ossl_database.mat")
sio.savemat("ossl_database.mat", mat_dict)