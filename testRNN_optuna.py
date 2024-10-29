import torch
import os
import numpy as np
import scipy as sp
from scipy.signal import savgol_filter
import pandas as pd
from utils.misc import snv
import utils
from torch import nn
from torch.nn import functional as F
from torch import optim
from net.chemtools.PLS import PLS
from tensorflow.keras.utils import to_categorical
from net.base_net import RNN, CuiNet , DeepSpectraCNN, ResNet18_1D , ViT_1D, FullyConvNet
from utils.testing import ccc,r2_score,RMSEP
import matplotlib.pyplot as plt
from utils.training import train, train_optuna
from utils.testing import test
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler

seed = 42
torch.manual_seed(seed)

cal1 = pd.read_csv('data/dataset/Wheat_dt/DT_train-1.csv',header=None)
cal2 = pd.read_csv('data/dataset/Wheat_dt/DT_train-2.csv',header=None)
cal3 = pd.read_csv('data/dataset/Wheat_dt/DT_train-3.csv',header=None)
cal4 = pd.read_csv('data/dataset/Wheat_dt/DT_train-4.csv',header=None)
cal5 = pd.read_csv('data/dataset/Wheat_dt/DT_train-5.csv',header=None)
## validation
val1 = pd.read_csv('data/dataset/Wheat_dt/DT_val-1.csv',header=None)
val2 = pd.read_csv('data/dataset/Wheat_dt/DT_val-2.csv',header=None)
## test
pre1 = pd.read_csv('data/dataset/Wheat_dt/DT_test-1.csv',header=None)
pre2 = pd.read_csv('data/dataset/Wheat_dt/DT_test-2.csv',header=None)
pre3 = pd.read_csv('data/dataset/Wheat_dt/DT_test-3.csv',header=None)

## Concatenate input variables, X
cal_features = np.concatenate((cal1.iloc[:, 0:-1],cal2.iloc[:, 0:-1],cal3.iloc[:, 0:-1],cal4.iloc[:, 0:-1],cal5.iloc[:, 0:-1]),axis=0)
val_features = np.concatenate((val1.iloc[:, 0:-1],val2.iloc[:, 0:-1]),axis = 0)
pre_features = np.concatenate((pre1.iloc[:, 0:-1],pre2.iloc[:, 0:-1],pre3.iloc[:, 0:-1]),axis = 0)

## Concatenate the target variable or lables, Y
cal_labels = np.concatenate((cal1.iloc[:, -1],cal2.iloc[:, -1],cal3.iloc[:, -1],cal4.iloc[:, -1],cal5.iloc[:, -1]),axis = 0)
val_labels = np.concatenate((val1.iloc[:, -1],val2.iloc[:, -1]),axis=0)
pre_labels = np.concatenate((pre1.iloc[:, -1],pre2.iloc[:, -1],pre3.iloc[:, -1]),axis = 0)

## Settings for the smooth derivatives using a Savitsky-Golay filter
w = 13 ## Sav.Gol window size
p = 2  ## Sav.Gol polynomial degree

## Perform data augmentation in the feature space by combining different types of typical chemometric spectral pre-processings
## [spectra, SNV, 1st Deriv., 2nd Deriv., 1st Deriv. SNV, 2nd Deriv. SNV]
x_cal = np.concatenate((cal_features, snv(cal_features),savgol_filter(cal_features, w, polyorder = p, deriv=1),\
                                savgol_filter(cal_features, w, polyorder = p, deriv=2),savgol_filter(snv(cal_features), w, polyorder = p, deriv=1),\
                                savgol_filter(snv(cal_features), w, polyorder = p, deriv=2)),axis = 1)
x_val = np.concatenate((val_features, snv(val_features),savgol_filter(val_features, w, polyorder = p, deriv=1),\
                                savgol_filter(val_features, w, polyorder = p, deriv=2),savgol_filter(snv(val_features), w, polyorder = p, deriv=1),\
                                savgol_filter(snv(val_features), w, polyorder = p, deriv=2)),axis =1)
x_test= np.concatenate((pre_features, snv(pre_features),savgol_filter(pre_features, w, polyorder = p, deriv=1),\
                                savgol_filter(pre_features, w, polyorder = p, deriv=2),savgol_filter(snv(pre_features), w, polyorder = p, deriv=1),\
                                savgol_filter(snv(pre_features), w, polyorder = p, deriv=2)),axis =1)

## Create wavelength x-scale by interpolating the range mentioned in the original wheat paper
delta_co = (1645-975)/200 # wavelenght step
# print(delta_co)
co=975+np.arange(200)*delta_co

# plt.plot(co, cal_features[:100,:].T)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Signal intensity')
# plt.show()

# create a temp directory


y_cal = np.eye(30)[cal_labels]
y_val = np.eye(30)[val_labels]
y_test = np.eye(30)[pre_labels]

## Concatenate input variables, X
# calculate mean and std per column

mean = np.mean(x_cal, axis=0)
std = np.std(x_cal, axis=0)


# Convert np.array to Dataloader
bsz = 1024
print(x_cal.shape[0],bsz)
# bsz = x_cal.shape[0]
cal = data_utils.TensorDataset(torch.Tensor(x_cal), torch.Tensor(y_cal))
cal_loader = data_utils.DataLoader(cal, batch_size=bsz, shuffle=True)

# bsz = x_val.shape[0]
val = data_utils.TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
val_loader = data_utils.DataLoader(val, batch_size=bsz, shuffle=True)

# bsz = x_test.shape[0]
test_dt = data_utils.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
test_loader = data_utils.DataLoader(test_dt, batch_size=bsz, shuffle=True)

from net.base_net import RNN
model_name ="_RNN_Wheat_optuna_2"
spec_dims = x_cal.shape[1]
data_path ="data/dataset/Wheat_dt/DT_test-1.csv"
save_path = f'models/{model_name}/' + model_name

def objective(trial, train_loader, val_loader, num_epochs, save_path=None, save_interval=10, classification=False):
    LR = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    DO = trial.suggest_float("do", 0.0, 1.0, log=False)
    HD = trial.suggest_int("hd",16,1024, log=True)
    NL = trial.suggest_int("nl",1,16, log=True)
    BD = trial.suggest_categorical("bd",[True,False])
    WE = trial.suggest_int("we",2,10, log=True)
    NUM_FC_LAYERS = trial.suggest_int("nfcl",1,5, step=1)
    NUM_FC_UNITS = [int(trial.suggest_int("nfcu_"+str(i), 1, 512, step=2)) for i in range(NUM_FC_LAYERS)]
    DO_FC = [float(trial.suggest_float("dofc_"+str(i), 0.0, 1.0, log=False)) for i in range(NUM_FC_LAYERS)]
    model = RNN(spec_dims, mean=mean, std=std, out_dims=30, dropout_lstm=DO, hidden_dim_lstm = HD, num_layers_lstm = NL, bidirectional=BD, num_fc = NUM_FC_LAYERS, out_dims_per_fc = NUM_FC_UNITS, dropout_fc = DO_FC)

    # Generate the optimizers.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    linear_lr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                                  end_factor=1.0, total_iters=WE)
    scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                  end_factor=0.1, total_iters=WE)

    # Combine warmup et Cosine Annealing
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_lr,scheduler2], milestones=[WE])
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    max_metric = train_optuna(model, optimizer, criterion, train_loader, val_loader, scheduler=scheduler, num_epochs=num_epochs, save_path=save_path, save_interval=save_interval,
                 classification=classification, trial=trial)
    return max_metric

optimize = True
if optimize:
    num_epochs = 250
    save_interval = 1000000
    sampler = TPESampler(seed=seed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(study_name='Aioly_Wheat_test2',direction="maximize",storage="sqlite:///db.sqlite3",sampler=sampler, load_if_exists=True)  # Create a new study with database.
    # study.optimize(objective, n_trials=100, timeout=600)
    study.optimize(lambda trial: objective(trial, cal_loader, val_loader, num_epochs=num_epochs, save_path=save_path, save_interval=save_interval, classification=True), n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

else:
    LR = 0.0010158042883135754
    num_epochs = 2500
    save_interval = 1000000
    warmup_epochs = 6
    # input_dims, mean, std, hidden_dim = 2048, num_layers = 1, dropout=0.2, out_dims=1, bidirectional=True
    # lr: 0.0010158042883135754, do: 0.7406051037812547, hd: 1444, nl: 2, bd: False, we: 6
    model = RNN(input_dims = spec_dims, mean = mean,std = std, hidden_dim_lstm = 1444, num_layers_lstm = 2, dropout_lstm = 0.7406051037812547, out_dims=30, bidirectional = False)

    # Generate the optimizers.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    linear_lr = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                                  end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                                  end_factor=0.1, total_iters=warmup_epochs)

    # Combine warmup et Cosine Annealing
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_lr,scheduler2], milestones=[warmup_epochs])
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)

    train_losses, val_losses,val_r2_scores, final_path = train_optuna(model, optimizer, criterion, cal_loader, val_loader, scheduler=scheduler,
                                                                      num_epochs = num_epochs, save_path = save_path, save_interval = save_interval,
                                                                      classification = True)

    #
    #
    # import torcheval.metrics
    # # device = "cpu"
    # model.load_state_dict(torch.load("models/" + model_name + "/" + model_name + "_best.pth", weights_only=True))
    # model.to(device)
    # model.eval()
    #
    #
    # out = []
    # tar = []
    # with torch.no_grad():
    #     for inputs, targets in test_loader:
    #         inputs = inputs.to(device,non_blocking=True).float()
    #         targets = targets.to(device,non_blocking=True).float()
    #         outputs = model(inputs[:,None])
    #         out.append(outputs.detach().cpu())
    #         tar.append(targets.detach().cpu())
    # all_outputs = torch.cat(out, dim=0)
    # all_targets = torch.cat(tar, dim=0)
    # F1 = torcheval.metrics.MulticlassF1Score()
    # F1.update(torch.argmax(all_targets,dim=1), torch.argmax(all_outputs,dim=1))
    # f1_scores = F1.compute()
    # print('F1 Score: ',f1_scores)
    # print(f'F1 Score: {f1_scores}')
    #
    # import torch.nn.utils.prune as prune
    # # Pruning
    # parameters_to_prune = (
    #     (model.lstm, 'weight_ih_l0'),
    #     (model.lstm, 'weight_hh_l0'),
    #     (model.fc, 'weight')
    # )
    # prune.global_unstructured(
    #     parameters_to_prune,
    #     pruning_method=prune.L1Unstructured,
    #     amount=0.85,
    # )
    # # print(list(model.lstm.named_parameters()))
    # prune.remove(model.lstm, 'weight_ih_l0')
    # prune.remove(model.lstm, 'weight_hh_l0')
    # prune.remove(model.fc, 'weight')
    #
    # for param in model.fc.parameters():
    #     param.requires_grad = False
    #
    #
    # # print(list(model.lstm.named_parameters()))
    #
    # out = []
    # tar = []
    # with torch.no_grad():
    #     for inputs, targets in test_loader:
    #         inputs = inputs.to(device,non_blocking=True).float()
    #         targets = targets.to(device,non_blocking=True).float()
    #         outputs = model(inputs[:,None])
    #         out.append(outputs.detach().cpu())
    #         tar.append(targets.detach().cpu())
    # all_outputs = torch.cat(out, dim=0)
    # all_targets = torch.cat(tar, dim=0)
    # F1 = torcheval.metrics.MulticlassF1Score()
    # F1.update(torch.argmax(all_targets,dim=1), torch.argmax(all_outputs,dim=1))
    # f1_scores = F1.compute()
    # print('F1 Score: ',f1_scores)
    # print(f'F1 Score: {f1_scores}')
    #
    # model_name += "_pruned"
    # save_path = f'models/{model_name}/' + model_name
    # LR = 0.0001
    # optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.00001)
    # criterion = nn.BCEWithLogitsLoss(reduction='mean')
    # criterion_test = nn.BCEWithLogitsLoss(reduction='mean')
    # train_losses, val_losses,val_r2_scores, final_path = train(model, optimizer, criterion, cal_loader, val_loader, num_epochs, save_path=save_path, save_interval=save_interval, classification = True, doWarmup=True)
