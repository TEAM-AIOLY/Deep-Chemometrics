
"""
Created on Thu Aug  1 14:58:26 2024

@author: metz
"""
import numpy as np
import torch
 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

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


def test(model, model_path, test_loader) : 
    
    Y = []
    y_pred = []
    
    model.load_state_dict(torch.load(model_path))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            Y += targets.to("cpu")
            inputs = inputs.to(device,non_blocking=True).float()
            outputs = model(inputs[:,None])
            y_pred += outputs.to("cpu")


    Y = np.array(Y)
    y_pred = np.array(y_pred)
    print("CCC: %5.5f, R2: %5.5f, RMSEP: %5.5f"%(ccc(y_pred,Y), r2_score(y_pred, Y), RMSEP(y_pred, Y)))
  
    plt.figure(figsize=(8,6))
    
    # Scatter plot of X vs Y
    plt.scatter(Y,y_pred,edgecolors='k',alpha=0.5)
    
    # Plot of the 45 degree line
    plt.plot([Y.min()-1,Y.max()+1],[Y.min()-1,Y.max()+1],'r')
      
    plt.text(0, 0.75*Y.max(), "CCC: %5.5f"%(ccc(Y,y_pred))+"\nR2: %5.5f"%(r2_score(Y,y_pred))+"\nRMSEP: %5.5f"%(RMSEP(Y,y_pred)),
             fontsize=16, bbox=dict(facecolor='white', alpha=0.5))

    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Observed',fontsize=16)
    plt.ylabel('Predicted',fontsize=16)
    
    plt.show(block=False)
