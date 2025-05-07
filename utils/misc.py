import os
import sys
from utils.constants import *
import numpy as np
import torch
import matplotlib.pyplot as plt


def snv(input_data):
    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    return output_data

## data augmentation of NIR data, add slope, offset, noise and shift
class data_augmentation:
    def __init__(self, slope = 0.1, offset = 0.1, noise = 0.1, shift = 0.1):
        self.slope = slope
        self.offset = offset
        self.noise = noise
        self.shift = shift

    def __call__(self, X):
        X_aug = np.zeros_like(X)
        X_aug = X * (1 + np.random.uniform(-self.slope, self.slope)) + np.random.uniform(-self.offset, self.offset) + np.random.normal(0, self.noise, X.shape[1])
        return torch.tensor(X_aug)
    
def load_config(conf):
    model_name = conf[MODEL]["name"]
    model = model_name
    if "PS" in conf[MODEL] :
        model+=f"PS{conf[MODEL]['PS']}"
    if "DE" in conf[MODEL] :
        model+=f"DE{conf[MODEL]['DE']}"
    if "TL" in conf[MODEL] :
        model+=f"TL{conf[MODEL]['TL']}"
    if "HDS" in conf[MODEL] :
        model+=f"HDS{conf[MODEL]['HDS']}"
    if "MLP" in conf[MODEL] :
        model+=f"MLP{conf[MODEL]['MLP']}"
    dataset = conf[DATASET]
    preprocessing = None
    augmentation = None
    lr = None
    dt_folder_name = dataset['name']
    if 'fraction' in dataset:
        dt_folder_name += str(dataset["fraction"])
    save_folder = os.path.join(f"runs/{model}/{dt_folder_name}/")

    if LR in conf:
        lr = conf[LR]

    if 'target' in dataset:
        save_folder += f"{dataset['target']}/"

    
    if PREPROCESSING in conf:
        preprocessing = conf[PREPROCESSING]
        save_folder += f"{preprocessing['name']}"
        if 'order' in preprocessing:
            save_folder += f"_{preprocessing['order']}"
        if 'window_size' in preprocessing : 
            save_folder += f"{preprocessing['window_size']}"
        if 'poly_order' in preprocessing : 
            save_folder += f"{preprocessing['poly_order']}"
        if 'derivative' in preprocessing : 
            save_folder += f"{preprocessing['derivative']}"
        save_folder += "/"

    if AUGMENTATION in conf:
        if PREPROCESSING in conf:
            sys.exit("for now you cannot use preprocessing and data augmentation, choose one or another")
        augmentation = conf[AUGMENTATION]
        if 'name' in augmentation :
            if augmentation['name'] == 'EMSA':
                save_folder += f"EMSA{augmentation['order']}"
        else : 
            if (augmentation[FACTORIAL_METHOD]["name"] == MVN or augmentation[GENERATOR] == MVN) and (augmentation[GENERATOR] != augmentation[FACTORIAL_METHOD]["name"]):
                sys.exit("if you use MVN, you have to use it both for the factorial method and for the generator")
            aug_name = f"{augmentation[PREPROCESSING]['name']}"
            if 'order' in augmentation[PREPROCESSING]:
                aug_name += f"{augmentation[PREPROCESSING]['order']}"
            if 'window_size' in augmentation[PREPROCESSING]:
                aug_name += f"{augmentation[PREPROCESSING]['window_size']}"
            if 'poly_order' in augmentation[PREPROCESSING]: 
                aug_name += f"{augmentation[PREPROCESSING]['poly_order']}"
            if 'derivative' in augmentation[PREPROCESSING]: 
                aug_name += f"{augmentation[PREPROCESSING]['derivative']}"
            aug_name += f"_{augmentation[FACTORIAL_METHOD]['name']}"
            if 'var_explained' in augmentation[FACTORIAL_METHOD]:
                aug_name += f"{str(augmentation[FACTORIAL_METHOD]['var_explained']).replace('.','')}v"
            elif 'nb_comp' in augmentation[FACTORIAL_METHOD]:
                aug_name += f"{augmentation[FACTORIAL_METHOD]['nb_comp']}c"
            aug_name += f"_{augmentation[GENERATOR]['name']}"
            if 'alpha' in augmentation[GENERATOR]:
                aug_name += f"{str(augmentation[GENERATOR]['alpha']).replace('.','')}"
            save_folder += f"{aug_name}"

    return conf[MODEL], lr, dataset, preprocessing, augmentation, save_folder

def expected_calibration_error(samples, true_labels, save_path, M=10):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)
    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels
    ece = np.zeros(1)
    bins_conf = []
    bins_acc = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower &amp; upper), boolean list
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            bins_acc.append((accuracy_in_bin).item())
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            bins_conf.append((avg_confidence_in_bin).item())
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
        else:
            bins_acc.append(0)
            bins_conf.append(0)
    confidence_bins = np.linspace(0.05, 0.95, 10)  # Confidence bins (x-axis)
    accuracy = np.array(bins_acc)  # Accuracy values
    confidence = np.array(bins_conf)
    gap = confidence - accuracy  # Gap between confidence and accuracy
    bar_width = 0.1  # Adjust width of bars
    plt.figure(figsize=(8,6))
    plt.bar(confidence_bins, accuracy, width=bar_width, color='blue', label='Outputs', alpha=0.9, edgecolor='black')
    plt.bar(confidence_bins, gap + accuracy, width=bar_width, color='red', alpha=0.3, hatch='/', edgecolor='black', label='Gap')
    plt.plot([0, 1], [0, 1], '--', color='gray', label="perfect calibration")
    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Model calibration", fontsize=10)
    plt.legend(loc="upper left", fontsize=10)
    plt.text(0.5, 0.1, f"ECE={ece.item():.2f}", fontsize=12, bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(save_path)