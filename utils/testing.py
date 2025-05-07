
"""
Created on Thu Aug  1 14:58:26 2024

@author: metz
"""
import numpy as np
import torch
from torcheval.metrics import MulticlassF1Score
 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from utils.misc import expected_calibration_error

def RMSEP(y_true, y_pred):
    loss = np.sqrt(np.mean(np.square((y_true - y_pred)), axis=0))
    return loss

def ccc(y_true,y_pred):
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    cor = np.corrcoef(y_true, y_pred,rowvar = False)[0][1]

    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    ccc = numerator / denominator
    return(ccc)


def test(model, model_path, test_loader,lr,classification=False,device = "cuda",save_path=None) :
    
    Y = []
    probas = [] # for classification calibration
    y_pred = []
    model.load_state_dict(torch.load(model_path))
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            if not classification:
                Y.append(targets.to("cpu").numpy())
                inputs = inputs.to(device,non_blocking=True).float()
                y_pred.append(model(inputs[:,None]).to("cpu").numpy())
            else :
                Y += torch.argmax(targets,dim=1).to("cpu")
                inputs = inputs.to(device,non_blocking=True).float()
                outputs = model(inputs[:,None])
                probas += torch.softmax(outputs,dim=1).to("cpu")
                y_pred += torch.argmax(outputs,dim=1).to("cpu")
    if not classification:
        Y = np.concatenate(Y)
        y_pred = np.concatenate(y_pred)


    if not classification:
        ccc_score = ccc(y_pred,Y)
        r2_score_ = r2_score( Y,y_pred)
        rmsep_score = RMSEP(y_pred, Y)

        print(f"CCC: {ccc_score}, R2: {r2_score_}, RMSEP: {rmsep_score}")


        plt.figure(figsize=(8,6))

        # Scatter plot of X vs Y
        plt.scatter(Y,y_pred,edgecolors='k',alpha=0.5)

        # Plot of the 45 degree line
        plt.plot([Y.min()-1,Y.max()+1],[Y.min()-1,Y.max()+1],'r')
        # add text with cc_score and r2_score
        plt.text(0.95, 0.05, f'CCC: {ccc_score:.2f}\nR²: {r2_score_:.2f}',
            transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
            color='red', fontweight='bold', fontfamily='serif')


        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('Observed',fontsize=16)
        plt.ylabel('Predicted',fontsize=16)
        plt.title(f'Predicted vs Observed',fontsize=16)
        # plt.show(block=False)
        plt.savefig(f"{save_path}/lr{str(lr).replace('.','')}_test_plot.png")
    else:
        F1 = MulticlassF1Score()
        print(torch.tensor(Y))
        print(torch.tensor(y_pred))  
        F1.update(torch.tensor(Y), torch.tensor(y_pred))
        f1_scores = F1.compute()
        expected_calibration_error(torch.stack(probas,dim=0).numpy(),torch.tensor(Y).numpy(), f"{save_path}/lr{str(lr).replace('.','')}_calibration_plot.png")
        print("f1 on testing data", f1_scores.item())
