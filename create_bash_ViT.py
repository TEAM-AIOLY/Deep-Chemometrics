import random


num_trials = 50
trials = []


SCRIPT_PATH="path/to/vit.py"

# Base parameters that are mandatory
DATA_PATH="./data/dataset/ossl/ossl_all_L1_v1.2.csv"
DATASET_TYPE="mir"
Y_LABELS= ["oc_usda.c729_w.pct","na.ext_usda.a726_cmolc.kg","clay.tot_usda.a334_w.pct", "k.ext_usda.a725_cmolc.kg", "ph.h2o_usda.a268_index" ]
NUM_EPOCHS=1000
BATCH_SIZE=1024
SEED=42