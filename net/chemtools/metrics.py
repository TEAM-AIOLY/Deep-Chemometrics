import torch


def ccc(y_true, y_pred):
    # Calculate correlation
    
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    cor = torch.mean((y_true - mean_true) * (y_pred - mean_pred)) / (torch.std(y_true) * torch.std(y_pred))

    # Population variances
    var_true = torch.var(y_true, unbiased=False)
    var_pred = torch.var(y_pred, unbiased=False)

    # Population standard deviations
    sd_true = torch.std(y_true, unbiased=False)
    sd_pred = torch.std(y_pred, unbiased=False)

    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    ccc = numerator / denominator

    return ccc.item()

def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()


def RMSEP(y_true, y_pred):
    loss = torch.sqrt(torch.mean((y_true - y_pred) ** 2, dim=0))
    return loss.item()