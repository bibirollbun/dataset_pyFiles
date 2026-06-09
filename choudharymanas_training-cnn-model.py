import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split



data_train = np.load('/kaggle/input/complete-lightweight-data/data_train.npy', mmap_mode='r')
data_train_FGS = np.load('/kaggle/input/complete-lightweight-data/data_train_FGS.npy', mmap_mode='r')
train_df = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/train.csv")



data_train.shape


data_train_FGS.shape


train_df


import numpy as np
from tqdm import tqdm

def get_stats_in_batches(data_array, indices, batch_size=16):
    """
    Calculates mean and std deviation of a large, memory-mapped numpy array 
    iteratively in batches without loading the whole array into RAM.
    """
    num_samples = len(indices)
    # Get the shape of a single data point to initialize our accumulators
    sample_shape = data_array[0].shape 
    
    # --- PASS 1: Calculate Mean ---
    print("Pass 1/2: Calculating mean...")
    # Use float64 for accumulators to maintain precision
    mean_sum = np.zeros(sample_shape, dtype=np.float64)
    
    for i in tqdm(range(0, num_samples, batch_size)):
        # Correctly slice the indices for the current batch
        batch_indices = indices[i:i+batch_size]
        # Load only the data for the current batch using the correct indices
        batch_data = data_array[batch_indices]
        # Sum along the batch dimension (axis=0)
        mean_sum += batch_data.sum(axis=0)
        
    global_mean = mean_sum / num_samples

    # --- PASS 2: Calculate Standard Deviation ---
    print("\nPass 2/2: Calculating standard deviation...")
    std_sum_sq = np.zeros(sample_shape, dtype=np.float64)
    
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_indices = indices[i:i+batch_size]
        batch_data = data_array[batch_indices]
        # Sum the squared differences along the batch dimension (axis=0)
        std_sum_sq += ((batch_data - global_mean)**2).sum(axis=0)
        
    global_std = np.sqrt(std_sum_sq / num_samples)

    # Add a small epsilon to std to avoid division by zero
    global_std[global_std == 0] = 1e-9

    print("\nCalculation complete.")
    # Return as float32, which is what PyTorch expects
    return global_mean.astype(np.float32), global_std.astype(np.float32)


num_examples = data_train.shape[0]
train_target = train_df.head(num_examples)
train_target = train_target.drop(columns=['planet_id'])

# Combine targets for the unified model
targets_combined = train_target.values.astype(np.float32)






import torch
import torch.nn as nn
import torch.nn.functional as F

class FGS_1D_CNN(nn.Module):
    """ A 1D CNN for the FGS data. """
    def __init__(self):
        super(FGS_1D_CNN, self).__init__()
        # Input will be reshaped to (batch, 187, 32*32=1024)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(187, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 256, 512), # 1024 -> 512 -> 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2) # Output mean and sigma for 1 wavelength
            ,nn.ReLU()
        )

    def forward(self, x):
        # Reshape for 1D convolution: (batch, channels, height, width) -> (batch, channels, sequence_length)
        batch_size = x.shape[0]
        x = x.view(batch_size, 187, -1)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        output = self.fc(x)
        
        y_pred = output[:, :1]
        sigma_pred = F.softplus(output[:, 1:]) + 1e-6
        return y_pred, sigma_pred

class AIRS_1D_CNN(nn.Module):
    """ A 1D CNN for the AIRS data. """
    def __init__(self):
        super(AIRS_1D_CNN, self).__init__()
        # Input will be reshaped to (batch, 187, 356*32=11392)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(187, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 712, 512), # 11392 -> 2848 -> 712
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,512),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(512, 282 * 2) # Output mean and sigma for 282 wavelengths
            ,nn.ReLU()
        )

    def forward(self, x):
        # Reshape for 1D convolution
        batch_size = x.shape[0]
        x = x.view(batch_size, 187, -1)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        output = self.fc(x)
        
        y_pred = output[:, :282]
        sigma_pred = F.softplus(output[:, 282:]) + 1e-6
        return y_pred, sigma_pred

class CombinedArielModel(nn.Module):
    """ A wrapper model to combine the 1D CNNs for joint training. """
    def __init__(self):
        super(CombinedArielModel, self).__init__()
        self.model_fgs = FGS_1D_CNN()
        self.model_airs = AIRS_1D_CNN()

    def forward(self, x_fgs, x_airs):
        y_pred_fgs, sigma_pred_fgs = self.model_fgs(x_fgs)
        y_pred_airs, sigma_pred_airs = self.model_airs(x_airs)

        y_pred_combined = torch.cat([y_pred_fgs, y_pred_airs], dim=1)
        sigma_pred_combined = torch.cat([sigma_pred_fgs, sigma_pred_airs], dim=1)

        return y_pred_combined, sigma_pred_combined


class GLLLoss(nn.Module):
    """
    This class converts the scoring metric into a PyTorch loss function.
    It re-implements the Gaussian Log-Likelihood calculation using PyTorch tensors,
    making it differentiable for model training.
    """
    def __init__(self, naive_mean, naive_sigma, fsg_sigma_true=1e-6, airs_sigma_true=1e-5, fgs_weight=1.0, n_wavelengths=283, device='cpu'):
        super(GLLLoss, self).__init__()
        self.naive_mean = torch.tensor(naive_mean, dtype=torch.float32).to(device)
        self.naive_sigma = torch.tensor(naive_sigma, dtype=torch.float32).to(device)

        # Pre-calculate constants and move them to the correct device as tensors.
        sigma_true_np = np.append(np.array([fsg_sigma_true]), np.ones(n_wavelengths - 1) * airs_sigma_true)
        self.sigma_true = torch.tensor(sigma_true_np, dtype=torch.float32).to(device)

        weights_np = np.append(np.array([fgs_weight]), np.ones(n_wavelengths - 1))
        self.weights = torch.tensor(weights_np, dtype=torch.float32).to(device)

    def log_pdf(self, x, loc, scale):
        """
        PyTorch implementation of the Gaussian log probability density function.
        """
        eps = 1e-9
        return -torch.log(scale + eps) - 0.5 * np.log(2 * np.pi) - 0.5 * torch.pow((x - loc) / (scale + eps), 2)

    def forward(self, y_pred, sigma_pred, y_true):
        """
        Calculates the competition metric as a loss. Since the goal is to MAXIMIZE
        the score, the loss is the NEGATIVE of the score, which we MINIMIZE.
        """
        GLL_pred = self.log_pdf(y_true, loc=y_pred, scale=sigma_pred)
        GLL_true = self.log_pdf(y_true, loc=y_true, scale=self.sigma_true)
        GLL_mean = self.log_pdf(y_true, loc=self.naive_mean, scale=self.naive_sigma)

        ind_scores = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean + 1e-9)
        weighted_scores = ind_scores * self.weights.unsqueeze(0)
        submit_score = torch.sum(weighted_scores) / (torch.sum(self.weights) * y_true.shape[0])

        # The loss is the negative of the score.
        loss = -submit_score
        return loss


from torch.utils.data import Dataset
import torch
import numpy as np

class ArielDataset(Dataset):
    """
    Custom PyTorch Dataset for the Ariel data.
    This class handles loading data from memory-mapped files and applies
    on-the-fly normalization using pre-computed statistics.
    """
    def __init__(self, data_fgs, data_airs, targets, indices, mean_fgs, std_fgs, mean_airs, std_airs):
        """
        Args:
            data_fgs (np.memmap): Memory-mapped array for FGS data.
            data_airs (np.memmap): Memory-mapped array for AIRS data.
            targets (np.array): Numpy array of target values.
            indices (np.array): The indices of the data to use (e.g., train_idx or val_idx).
            mean_fgs (np.array): Pre-computed mean for the FGS training data.
            std_fgs (np.array): Pre-computed std deviation for the FGS training data.
            mean_airs (np.array): Pre-computed mean for the AIRS training data.
            std_airs (np.array): Pre-computed std deviation for the AIRS training data.
        """
        self.data_fgs = data_fgs
        self.data_airs = data_airs
        self.targets = targets
        self.indices = indices
        
        # Store the normalization statistics as attributes of the dataset
        self.mean_fgs = mean_fgs
        self.std_fgs = std_fgs
        self.mean_airs = mean_airs
        self.std_airs = std_airs

    def __len__(self):
        """ Returns the total number of samples in the dataset. """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Fetches one sample from the dataset at the given index, applies
        normalization, and returns it as a tuple of PyTorch tensors.
        """
        # Get the actual index from the provided list of indices (train or val)
        i = self.indices[idx]
        
        # Load the raw data for one sample
        x_fgs_raw = self.data_fgs[i]
        x_airs_raw = self.data_airs[i]
        y = self.targets[i]

        # Apply normalization on-the-fly using the stored statistics.
        # A small epsilon is added to the standard deviation to prevent division by zero.
        x_fgs = (x_fgs_raw - self.mean_fgs) / (self.std_fgs + 1e-9)
        x_airs = (x_airs_raw - self.mean_airs) / (self.std_airs + 1e-9)

        # Convert the numpy arrays to PyTorch tensors and return
        return (torch.from_numpy(x_fgs).float(),
                torch.from_numpy(x_airs).float(),
                torch.from_numpy(y).float())




indices = np.arange(num_examples)
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=69)




mean_FGS, std_FGS = get_stats_in_batches(data_train_FGS,train_idx,batch_size=16)
mean_AIRS, std_AIRS = get_stats_in_batches(data_train,train_idx,batch_size=16)


data_train_FGS.shape


train_dataset = ArielDataset(data_train_FGS, data_train, targets_combined, train_idx,mean_FGS,std_FGS,mean_AIRS,std_AIRS)
val_dataset = ArielDataset(data_train_FGS, data_train, targets_combined, val_idx,mean_FGS,std_FGS,mean_AIRS,std_AIRS)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, pin_memory=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=16, pin_memory=True, num_workers=2)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CombinedArielModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5,weight_decay = 1e-4)
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, verbose=True)

# Calculate naive mean and sigma from the training set for the loss function
# Note: Using the whole dataset here for simplicity. In a strict setting,
# you'd only use the training split.
NAIVE_MEAN = np.mean(targets_combined)
NAIVE_SIGMA = np.std(targets_combined)

fn_loss = GLLLoss(
    naive_mean=NAIVE_MEAN,
    naive_sigma=NAIVE_SIGMA,
    device=device
)

best_val_loss = float('inf')
patience = 3
wait = 0
num_epochs = 50

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    train_losses = []

    for data_fgs, data_airs, targets in tqdm(train_dataloader):
        data_fgs = data_fgs.to(device, non_blocking=True)
        data_airs = data_airs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        y_pred, sigma_pred = model(data_fgs, data_airs)
        loss = fn_loss(y_pred, sigma_pred, targets)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    avg_train_loss = sum(train_losses) / len(train_losses)

    # --- Validation ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for data_fgs, data_airs, targets in val_dataloader:
            data_fgs = data_fgs.to(device, non_blocking=True)
            data_airs = data_airs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            y_pred, sigma_pred = model(data_fgs, data_airs)
            loss = fn_loss(y_pred, sigma_pred, targets)
            val_losses.append(loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    scheduler.step(avg_val_loss)
    # Since loss is -score, a lower loss value is better.
    print(f"Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f}")

    # --- Early stopping ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait = 0
        torch.save(model.state_dict(), "Combined_Ariel_Model.pt")
        print(f"Validation loss improved. Best val loss: {best_val_loss:.6f}. Model saved.")
    else:
        wait += 1
        if wait >= patience:
            print("EARLY STOPPING TRIGGERED")
            break
        print(f"No improvement. Early stop patience: {wait}/{patience}")



























