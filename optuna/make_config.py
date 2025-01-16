import json
import os
import numpy as np

N = 50
network = "ViT1D"


# Standard part of the configuration that remains constant
standard = {
    "data_path": "./data/dataset/ossl/ossl_all_L1_v1.2.csv",
    "dataset_type": "mir",
    "y_labels": [ "oc_usda.c729_w.pct","na.ext_usda.a726_cmolc.kg",  "clay.tot_usda.a334_w.pct",  "k.ext_usda.a725_cmolc.kg",  "ph.h2o_usda.a268_index"],
    "batch_size": 1024,
    "network" : network
}

config = {
    "LR": ("log", 1e-5, 1e-3), 
    "WD": ("log", 1e-4, 0.01),   
    "slope": ("uniform", 0.1, 0.3),        
    "offset": ("uniform", 0.1, 0.3),      
    "noise": ("uniform", 0.001, 0.01),     
    "shift": ("uniform", 0.01, 0.1),       
    "EPOCHS": ("int", 200, 1000,50)      
}

if "ViT1D" in network:
    config.update({
        "PS": ("int", 32, 64, 2),     
        "DE": ("int", 8, 32, 2),    
        "TL": ("int", 4, 16, 2),     
        "HDS": ("int", 4, 16, 2),  
        "MLP": ("int", 8, 32, 2)          
    })
    
elif "ResNet" in network or "DeepSpectra" in network:
    config.update({
        "DP": ("uniform", 0.1, 0.5)       
    })





output_path = os.path.dirname(standard["data_path"])+ "/config/"+network
if not os.path.exists(output_path):
            os.makedirs(output_path)
output_file = output_path+"/config_batch.json"


param_list = []


for i in range(N):
    params = {}
    
    params.update(standard)
    params['ID'] =f"ID_{i:03d}"
    
    for param, config_value  in config.items():
        draw_type = config_value[0]
        low = config_value[1]
        high = config_value[2]
        
        precision_low = abs(int(np.floor(np.log10(abs(low))))) if low != 0 else 0
        precision_high = abs(int(np.floor(np.log10(abs(high))))) if high != 0 else 0
        precision = max(precision_low, precision_high)
        
        if draw_type == "log":
            # Log-uniform draw
            raw_value = np.exp(np.random.uniform(np.log(low), np.log(high)))
            params[param] = round(raw_value, precision)
        elif draw_type == "int":
            step = config_value[3]
            raw_value = np.random.randint(low, high + 1)  
            params[param] = (raw_value // step) * step
        else:
            raw_value = np.random.uniform(low, high)
            params[param] = round(raw_value, precision)
    param_list.append(params)

with open(output_file, 'w') as json_file:
    json.dump(param_list, json_file, indent=4)

print(f"Generated {output_file} with {N} parameter sets.")