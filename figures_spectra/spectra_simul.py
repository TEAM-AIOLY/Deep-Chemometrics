import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
# -------------------------------
# Create folder for figures
# -------------------------------
fig_dir = os.path.join(os.getcwd(), "figures_simu")
os.makedirs(fig_dir, exist_ok=True)

# -------------------------------
# Simulation parameters
# -------------------------------
L = 200
x = np.linspace(400, 1000, L)
n_components = 3
n_train = 1000
n_test = 200
n_peaks_per_component = 2
rng = np.random.default_rng(42)

# -------------------------------
# Generate pure components (multiple peaks)
# -------------------------------
def simulate_pure_component(x, rng, n_peaks=2):
    spectrum = np.zeros_like(x)
    for _ in range(n_peaks):
        mu = rng.uniform(450, 950)
        sigma = rng.uniform(10, 50)
        amplitude = rng.uniform(0.5, 1.0)
        spectrum += amplitude * np.exp(-(x - mu)**2 / (2*sigma**2))
    return spectrum

P = np.zeros((L, n_components))
for i in range(n_components):
    P[:, i] = simulate_pure_component(x, rng, n_peaks=n_peaks_per_component)

# plt.figure(figsize=(8,5))
# for i in range(n_components):
#     plt.plot(x, P[:,i], label=f'Component {i+1}')
# plt.title("Pure Component Spectra (multiple peaks)")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Absorbance (a.u.)")
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, "pure_components.png"), dpi=600)
# plt.close()

# -------------------------------
# Dataset generator
# -------------------------------
def generate_mixture_dataset(n_samples, P, x, rng, 
                             slope_range=(0.0,0.3), poly_range=(-0.5,0.0),
                             noise_std=0.005, mixture_shift=None):
    L, n_components = P.shape
    X = np.zeros((n_samples, L))
    Y = np.zeros(n_samples)
    scores = np.zeros((n_samples, n_components))
    
    for i in range(n_samples):
        props = rng.dirichlet(alpha=np.ones(n_components))
        if mixture_shift is not None:
            props = props * (1 + mixture_shift)
            props /= props.sum()
        scores[i,:] = props
        
        spectrum = P @ props
        slope_coef = rng.uniform(*slope_range)
        spectrum += slope_coef * (x - x.min()) / (x.max() - x.min())
        poly_coef = rng.uniform(*poly_range)
        spectrum += poly_coef * ((x - x.mean()) / (x.ptp()/2))**2
        spectrum += rng.normal(0, noise_std, L)
        X[i,:] = spectrum
        Y[i] = 50*props[0] + 30*props[1] + 20*props[2] + rng.normal(0,1.0)
    
    return X, Y, scores

train_X, train_y, train_scores = generate_mixture_dataset(
    n_train, P, x, rng, slope_range=(0.0,0.3), poly_range=(-0.5,0.0), noise_std=0.005
)
mixture_shift = np.array([0.1, -0.05, -0.05])
test_X, test_y, test_scores = generate_mixture_dataset(
    n_test, P, x, rng, slope_range=(0.0,0.3), poly_range=(-0.5,0.0),
    noise_std=0.005, mixture_shift=mixture_shift
)

# -------------------------------
# Compute mean/std from train set
# -------------------------------
train_mean = train_X.mean(axis=0, keepdims=True)
train_std = train_X.std(axis=0, keepdims=True)

# -------------------------------
# Plot sample spectra
# -------------------------------
# fig, axes = plt.subplots(1,2, figsize=(14,6), sharey=True)
# for i in range(25):
#     axes[0].plot(x, train_X[i], alpha=0.7)
# axes[0].set_title("25 Training Spectra")
# axes[0].set_xlabel("Wavelength (nm)")
# axes[0].set_ylabel("Absorbance (a.u.)")
# for i in range(25):
#     axes[1].plot(x, test_X[i], alpha=0.7)
# axes[1].set_title("25 Test Spectra")
# axes[1].set_xlabel("Wavelength (nm)")
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, "sample_train_test_spectra.png"), dpi=600)
# plt.close()

# plt.figure(figsize=(10,5))
# plt.hist(train_y, bins=30, alpha=0.7, label="Train")
# plt.hist(test_y, bins=30, alpha=0.7, label="Test")
# plt.xlabel("Label Y")
# plt.ylabel("Count")
# plt.title("Distribution of Y (Regression Targets)")
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, "Y_distribution.png"), dpi=600)
# plt.close()

# -------------------------------
# Mixture proportions
# -------------------------------
# fig = plt.figure(figsize=(12,5))
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(train_scores[:,0], train_scores[:,1], train_scores[:,2], c='blue', alpha=0.5)
# ax1.set_title("Training Mixture Proportions (C1,C2,C3)")
# ax1.set_xlabel("C1"); ax1.set_ylabel("C2"); ax1.set_zlabel("C3")
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.scatter(test_scores[:,0], test_scores[:,1], test_scores[:,2], c='red', alpha=0.5)
# ax2.set_title("Test Mixture Proportions (shifted)")
# ax2.set_xlabel("C1"); ax2.set_ylabel("C2"); ax2.set_zlabel("C3")
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, "mixture_proportions.png"), dpi=600)
# plt.close()

# -------------------------------
# CNN on GPU with standardization
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_X_t = torch.tensor(train_X, dtype=torch.float32).unsqueeze(1).to(device)
train_y_t = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1).to(device)
train_ds = TensorDataset(train_X_t, train_y_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# class SimpleCNN1Filter(nn.Module):
#     def __init__(self, input_length, mean, std):
#         super().__init__()
#         self.mean = torch.tensor(mean, dtype=torch.float32).to(device)
#         self.std = torch.tensor(std, dtype=torch.float32).to(device)
#         self.conv1 = nn.Conv1d(1,1,7,padding=3)
#         self.elu = nn.ELU()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(input_length,16)
#         self.fc2 = nn.Linear(16,8)
#         self.fc3 = nn.Linear(8,1)
#     def forward(self,x):
#         x = (x - self.mean) / self.std
#         x = self.conv1(x)
#         conv_out = self.elu(x)
#         flat = self.flatten(conv_out)
#         fc1_out = self.elu(self.fc1(flat))
#         fc2_out = self.elu(self.fc2(fc1_out))
#         y_pred = self.fc3(fc2_out)
#         return y_pred, conv_out, fc1_out

# model = SimpleCNN1Filter(L, train_mean, train_std).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# n_epochs = 500
# visualize_epochs = [0,100,200,300,400,500]

# for epoch in range(1,n_epochs+1):
#     model.train()
#     running_loss = 0
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         y_pred, _, _ = model(X_batch)
#         loss = criterion(y_pred,y_batch)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     if epoch % 50 == 0:
#         print(f"Epoch {epoch}/{n_epochs}, Loss: {running_loss/len(train_loader):.3f}")
    
#     if epoch in visualize_epochs:
#         model.eval()
#         with torch.no_grad():
#             sample_idx = 0
#             y_pred, conv_out, fc1_out = model(train_X_t[sample_idx:sample_idx+1])
            
#             # 1. Conv filter
#             filt = model.conv1.weight.cpu().numpy().squeeze()
#             # plt.figure(figsize=(6,3))
#             # plt.plot(filt)
#             # plt.title(f"Conv1D Filter Epoch {epoch}")
#             # plt.xlabel("Kernel index"); plt.ylabel("Weight")
#             # plt.tight_layout()
#             # plt.savefig(os.path.join(fig_dir, f"conv_filter_epoch{epoch}.png"), dpi=600)
#             # plt.close()
            
#             # # 2. Convolved spectrum
#             # plt.figure(figsize=(8,4))
#             # plt.plot(x, conv_out.cpu().numpy().squeeze())
#             # plt.plot(x, train_X[sample_idx], alpha=0.5, linestyle='--', label="Original spectrum")
#             # plt.title(f"Convolved Spectrum Epoch {epoch}")
#             # plt.xlabel("Wavelength (nm)"); plt.ylabel("Activation")
#             # plt.legend()
#             # plt.tight_layout()
#             # plt.savefig(os.path.join(fig_dir, f"conv_output_epoch{epoch}.png"), dpi=600)
#             # plt.close()
            
#             # 3. FC1 weights applied to spectra
#             fc1_weights = model.fc1.weight.cpu().numpy()
#             # plt.figure(figsize=(8,4))
#             # for i in range(fc1_weights.shape[0]):
#             #     plt.plot(fc1_weights[i], alpha=0.7, label=f"Neuron {i+1}")
#             # plt.title(f"FC1 Layer Weights Epoch {epoch}")
#             # plt.xlabel("Flattened Conv Feature Index")
#             # plt.ylabel("Weight")
#             # plt.legend()
#             # plt.tight_layout()
#             # plt.savefig(os.path.join(fig_dir, f"fc1_weights_epoch{epoch}.png"), dpi=600)
#             # plt.close()
            
#             # # 4. FC1 features for sample spectrum
#             # fc1_features = fc1_out.cpu().numpy().squeeze()
#             # plt.figure(figsize=(8,4))
#             # plt.plot(fc1_features)
#             # plt.title(f"FC1 Features Epoch {epoch}")
#             # plt.xlabel("Neuron index"); plt.ylabel("Activation")
#             # plt.tight_layout()
#             # plt.savefig(os.path.join(fig_dir, f"fc1_features_epoch{epoch}.png"), dpi=600)
#             # plt.close()


#####################################################################################################################


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # if dimensions change, add a projection shortcut
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) \
            if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.elu(out)
        return out


class BottleneckBlock1D(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(bottleneck_channels)
        
        self.conv2 = nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)
        
        self.conv3 = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.elu = nn.ELU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride) \
            if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.elu(self.bn1(self.conv1(x)))
        out = self.elu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.elu(out)
        return out
    
    
class ResidualDemoNet(nn.Module):
    def __init__(self, input_length, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32).to(device)
        self.std = torch.tensor(std, dtype=torch.float32).to(device)
        self.block = ResidualBlock1D(1, 8, kernel_size=7)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_length * 8, 1)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = self.block(x)
        flat = self.flatten(out)
        y_pred = self.fc(flat)
        return y_pred, out
    
    
class BottleneckDemoNet(nn.Module):
    def __init__(self, input_length, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32).to(device)
        self.std = torch.tensor(std, dtype=torch.float32).to(device)
        self.block = BottleneckBlock1D(1, 4, 8)  # in=1, bottleneck=4, out=8
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_length * 8, 1)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = self.block(x)
        flat = self.flatten(out)
        y_pred = self.fc(flat)
        return y_pred, out


# model = ResidualDemoNet(L, train_mean, train_std).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# n_epochs = 100
# for epoch in range(1, n_epochs+1):
#     model.train()
#     running_loss = 0.0
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         y_pred, _ = model(X_batch)
#         loss = criterion(y_pred, y_batch)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}/{n_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# # -------------------------------
# # Average block output after training
# # -------------------------------
# model.eval()
# with torch.no_grad():
#     n_samples_avg = 500
#     block_outs = []
#     for i in range(n_samples_avg):
#         y_pred, block_out = model(train_X_t[i:i+1])
#         block_outs.append(block_out.cpu().numpy())
#     block_out = np.mean(np.array(block_outs), axis=0)   # shape (1, C, L)

# plt.figure(figsize=(10,5))
# for c in range(block_out.shape[1]):
#     plt.plot(x, block_out[0,c], label=f"Channel {c+1}")
# plt.title(f"Residual Block Average Output after {n_epochs} epochs")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Activation")
# plt.legend()
# plt.tight_layout()
# plt.show()



# model = BottleneckDemoNet(L, train_mean, train_std).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# n_epochs = 400
# for epoch in range(1, n_epochs+1):
#     model.train()
#     running_loss = 0.0
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         y_pred, _ = model(X_batch)
#         loss = criterion(y_pred, y_batch)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     if epoch % 10 == 0:
#         print(f"[Bottleneck] Epoch {epoch}/{n_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# # -------------------------------
# # Average block output after training
# # -------------------------------
# model.eval()
# with torch.no_grad():
#     n_samples_avg = 500
#     block_outs = []
#     for i in range(n_samples_avg):
#         y_pred, block_out = model(train_X_t[i:i+1])
#         block_outs.append(block_out.cpu().numpy())
#     block_out = np.mean(np.array(block_outs), axis=0)   # shape (1, C, L)

# plt.figure(figsize=(10,5))
# for c in range(block_out.shape[1]):
#     plt.plot(x, block_out[0,c], label=f"Channel {c+1}")
# plt.title(f"Bottleneck Block Average Output after {n_epochs} epochs")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Activation")
# plt.legend()
# plt.tight_layout()
# plt.show()


# -------------------------------
# Building blocks
# -------------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvBlock1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.batchnorm1d = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.batchnorm1d(x)
        x = self.relu(x)
        return x
    

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        
        self.branch1 = ConvBlock1D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock1D(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock1D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock1D(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock1D(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
            ConvBlock1D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)  # concat along channel axis
        return out


# -------------------------------
# Inception demo net
# -------------------------------
class InceptionDemoNet(nn.Module):
    def __init__(self, L, mean, std, n_classes=1):
        super(InceptionDemoNet, self).__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)
        
        # Inception block: each branch produces `out_channels`, total = 4* out_channels
        out_channels = 4
        self.inception = InceptionModule(in_channels=1, out_channels=out_channels)
        
        n_channels = 4 * out_channels
        self.fc1 = nn.Linear(n_channels * L, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = (x - self.mean) / self.std
        block_out = self.inception(x)              # shape (B, n_channels, L)
        flat = block_out.view(block_out.size(0), -1)
        fc1_out = F.relu(self.fc1(flat))
        y_pred = self.fc2(fc1_out)
        return y_pred, block_out


# -------------------------------
# Train InceptionDemoNet
# -------------------------------
model = InceptionDemoNet(L, train_mean, train_std).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 200
for epoch in range(1, n_epochs+1):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred, _ = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch % 10 == 0:
        print(f"[Inception] Epoch {epoch}/{n_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# -------------------------------
# Average inception output after training
# -------------------------------
model.eval()
with torch.no_grad():
    n_samples_avg = 500
    block_outs = []
    for i in range(n_samples_avg):
        y_pred, block_out = model(train_X_t[i:i+1])
        block_outs.append(block_out.cpu().numpy())
    block_out = np.mean(np.array(block_outs), axis=0)   # (1, 16, L)

plt.figure(figsize=(10,5))
for c in range(block_out.shape[1]):
    plt.plot(x, block_out[0,c], label=f"Channel {c+1}")
plt.title(f"Inception Block Average Output after {n_epochs} epochs")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Activation")
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()
plt.show()
