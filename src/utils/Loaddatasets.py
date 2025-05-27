import os
import json
import pandas as pd
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from src.utils.misc import snv

class DatasetLoader:
    @staticmethod
    def parse_args(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, 'r') as f:
            params_list = json.load(f)
        return params_list

    @staticmethod
    def get_data(config_path):
        params = DatasetLoader.parse_args(config_path)[0]  # Only use the first config
        rel_data_path = params.get("data_path")
        dataset_type = params.get("dataset_type", "").lower()
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(src_dir)
        data_path = os.path.abspath(os.path.join(project_root, rel_data_path))

        if dataset_type == "mango":
            mat_file = os.path.join(data_path, "mango_dm_full_outlier_removed2.mat")
            data = sio.loadmat(mat_file)
            Xcal = data["Sp_cal"]
            Ycal = data["DM_cal"]
            Xtest = data["Sp_test"]
            Ytest = data["DM_test"]
            x_cal, x_val, y_cal, y_val = train_test_split(Xcal, Ycal, test_size=0.20, shuffle=True, random_state=42)
            x_test, y_test = Xtest, Ytest
            return {
                "x_cal": x_cal,
                "y_cal": y_cal,
                "x_val": x_val,
                "y_val": y_val,
                "x_test": x_test,
                "y_test": y_test
            }

        elif dataset_type == "wheat":
            cal_files = [os.path.join(data_path, f"DT_train-{i}.csv") for i in range(1, 6)]
            val_files = [os.path.join(data_path, f"DT_val-{i}.csv") for i in range(1, 3)]
            test_files = [os.path.join(data_path, f"DT_test-{i}.csv") for i in range(1, 4)]
            cal_dfs = [pd.read_csv(f, header=None) for f in cal_files]
            val_dfs = [pd.read_csv(f, header=None) for f in val_files]
            test_dfs = [pd.read_csv(f, header=None) for f in test_files]
            x_cal_raw = np.concatenate([df.iloc[:, 0:-1] for df in cal_dfs], axis=0)
            y_cal = np.concatenate([df.iloc[:, -1] for df in cal_dfs], axis=0)
            x_val_raw = np.concatenate([df.iloc[:, 0:-1] for df in val_dfs], axis=0)
            y_val = np.concatenate([df.iloc[:, -1] for df in val_dfs], axis=0)
            x_test_raw = np.concatenate([df.iloc[:, 0:-1] for df in test_dfs], axis=0)
            y_test = np.concatenate([df.iloc[:, -1] for df in test_dfs], axis=0)
            w = 13
            p = 2
            def augment(x):
                return np.concatenate((
                    x,
                    snv(x),
                    savgol_filter(x, w, polyorder=p, deriv=1),
                    savgol_filter(x, w, polyorder=p, deriv=2),
                    savgol_filter(snv(x), w, polyorder=p, deriv=1),
                    savgol_filter(snv(x), w, polyorder=p, deriv=2)
                ), axis=1)
            x_cal = augment(x_cal_raw)
            x_val = augment(x_val_raw)
            x_test = augment(x_test_raw)
            return {
                "x_cal": x_cal,
                "y_cal": y_cal,
                "x_val": x_val,
                "y_val": y_val,
                "x_test": x_test,
                "y_test": y_test
            }

        elif dataset_type == "ossl":
            y_labels = params.get("y_labels", "oc.usda.c729")
            if isinstance(y_labels, str):
                y_labels = [y_labels]
            reduce_lbd = params.get("reduce_lbd", False)
            dataset_type_ossl = params.get("ossl_type", "visnir")
            seed = params.get("seed", 42)
            data_raw = pd.read_csv(data_path, low_memory=False)
            Y = np.array(data_raw.filter(regex="|".join(y_labels)))
            mask = ~np.isnan(Y).any(axis=1)
            if dataset_type_ossl == "mir":
                spectral_data = np.array(data_raw.filter(regex="mir").filter(regex="abs"))
            elif dataset_type_ossl in ["nir", "visnir"]:
                spectral_data = np.array(data_raw.filter(regex="visnir").filter(regex="ref"))
            else:
                raise ValueError("dataset_type must be either 'mir' or 'nir'")
            mask_ = ~(np.isnan(spectral_data[:, 0])[mask])
            X = spectral_data[mask][mask_]
            Y = Y[mask][mask_]
            if reduce_lbd:
                X = X[:, :-1]
            # Split: 20% test, 80% train; then 75% cal, 25% val from train
            idx = np.arange(len(X))
            np.random.seed(seed)
            np.random.shuffle(idx)
            test_size = int(0.2 * len(X))
            test_idx = idx[:test_size]
            train_idx = idx[test_size:]
            cal_size = int(0.75 * len(train_idx))
            cal_idx = train_idx[:cal_size]
            val_idx = train_idx[cal_size:]
            # Apply log to Y
            Y = np.log(Y + 1)
            return {
                "x_cal": X[cal_idx],
                "y_cal": Y[cal_idx],
                "x_val": X[val_idx],
                "y_val": Y[val_idx],
                "x_test": X[test_idx],
                "y_test": Y[test_idx]
            }
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")