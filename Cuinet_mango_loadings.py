import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
import torch.utils.data as data_utils
from data.load_dataset_atonce import MangoDataset

from net.base_net import Darionet
from utils.misc import data_augmentation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         
data_path = "./data/dataset/Mango/mango_dm_full_outlier_removed2.mat"
data = sp.io.loadmat(data_path)

Ycal = data["DM_cal"]
Ytest = data["DM_test"]
Xcal = data["Sp_cal"]
Xtest = data["Sp_test"]
x_cal, x_val, y_cal, y_val = train_test_split(Xcal, Ycal, test_size=0.20, shuffle=True, random_state=42) 

wv = data['wave'].astype(np.float32).reshape(-1, 1)

mean = np.mean(x_cal, axis=0)
std = np.std(x_cal, axis=0)

cal = MangoDataset(x_cal, y_cal, transform=data_augmentation(slope=0., offset=0., noise=0.00005, shift=0.))
val = data_utils.TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
test_dt = data_utils.TensorDataset(torch.Tensor(Xtest), torch.Tensor(Ytest))

model_name = "_DarioNet_Mango"
spec_dims = x_cal.shape[1]

num_epochs = 500
BATCH = 128
FILTER_SIZE = 45
L2_BETA = 0.015
LR = 0.002
p = 0.02

save_path = os.path.dirname(os.path.abspath(data_path)) + f'/models/{model_name}/' + model_name

cal_loader = data_utils.DataLoader(cal, batch_size=BATCH, shuffle=True)
val_loader = data_utils.DataLoader(val, batch_size=BATCH, shuffle=True)
test_loader = data_utils.DataLoader(test_dt, batch_size=BATCH, shuffle=True)

model = Darionet(mean=mean, std=std, out_dims=1, filter_size=FILTER_SIZE, reg_beta=L2_BETA, input_dims=spec_dims, p=p)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_BETA)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=25, factor=0.5, min_lr=1e-6, verbose=True)

criterion = nn.MSELoss(reduction='none')

activation_save_step = 25  # every N epochs
activations_per_epoch = {}

def conv1_hook(module, input, output):
    # Store only the first batch and the first 4 samples
    if current_epoch % activation_save_step == 0 and current_batch == 0:
        activations_per_epoch[current_epoch] = output.detach().cpu()[:4]  # Capture first 4 samples

# Register hook
hook_handle = model.conv1.register_forward_hook(conv1_hook)

# Training loop
for epoch in range(num_epochs):
    current_epoch = epoch  # set for hook
    model.train()
    running_loss = torch.zeros(1, device=device)

    for current_batch, (inputs, targets) in enumerate(cal_loader):
        inputs = inputs.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).float()

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs[:, None])  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        torch.mean(loss).backward()  # Backward pass
        optimizer.step()  # Update weights

    # Only remove hook after the training loop is done
    if epoch == num_epochs - 1:
        hook_handle.remove()

# Plot saved activations after training
for epoch, act_list in activations_per_epoch.items():
    if epoch % activation_save_step == 0:
        # Create figure for output spectra
        plt.figure(figsize=(10, 6))
        for i, act in enumerate(act_list):  # One sample per batch
            plt.plot(wv.flatten(), act.numpy().flatten(), label=f"Output Sample {i+1}")
        plt.title(f"Epoch {epoch+1}, Output Spectra (4 Samples)")
        plt.xlabel("Wavelength")
        plt.ylabel("Output Value")
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)  # Avoid blocking the execution

        # Create figure for input spectra (original input data)
        plt.figure(figsize=(10, 6))
        for i, input_sample in enumerate(inputs[:4]):  # First 4 samples in batch
            input_sample_cpu = input_sample.cpu().numpy()
            plt.plot(wv.flatten(), input_sample_cpu.flatten(), label=f"Input Sample {i+1}")
        plt.title(f"Epoch {epoch+1}, Input Spectra (4 Samples)")
        plt.xlabel("Wavelength")
        plt.ylabel("Input Value")
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)  # Avoid blocking the execution

plt.show()  # Make sure the final plots display correctly
