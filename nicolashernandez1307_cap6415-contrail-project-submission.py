import sys
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from pathlib import Path



# Weights dataset 
WEIGHTS_PATH = "/kaggle/input/cap6415-contrail-weights/best_model.pth" 
# Competition test dataset
DATA_DIR = Path('/kaggle/input/google-research-identify-contrails-reduce-global-warming')
TEST_DIR = DATA_DIR / 'test'
BATCH_SIZE = 4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
T11_BOUNDS = (243, 303)
CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
TDIFF_BOUNDS = (-4, 2)


# Bounds for Ash Color Scheme
_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)

# Normalization function for Ash Color Scheme
def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


class ContrailDataset(Dataset):
    """
    Loads the sequences, calculates Ash Color Scheme, and returns 3D tensors.
    Input Shape: (H, W, T) from numpy files.
    Output Shape: (C, T, H, W) for PyTorch 3D Conv.
    """
    def __init__(self, data_dir, record_ids):
        self.root = Path(data_dir) 
        self.record_ids = list(record_ids)
        

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, idx):
        rid = self.record_ids[idx]
        rid_path = self.root / rid

        # Load Bands (Shape: 256, 256, 8)
        band11 = np.load(rid_path / "band_11.npy").astype(np.float32)
        band14 = np.load(rid_path / "band_14.npy").astype(np.float32)
        band15 = np.load(rid_path / "band_15.npy").astype(np.float32)

        # Calculate Ash Color Scheme
        # R = Band 15 - Band 14
        r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
        # G = Band 14 - Band 11
        g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        # B = Band 14
        b = normalize_range(band14, _T11_BOUNDS)

        # Stack to (3, 256, 256, 8)
        rgb = np.stack([r, g, b], axis=0)
        rgb = np.clip(rgb, 0, 1)

        # Transpose from (C, H, W, T) to (C, T, H, W)
        rgb = np.transpose(rgb, (0, 3, 1, 2)) 

        return torch.from_numpy(rgb), rid


# Sigle Convolution Long Short-Term Memory
class ConvLSTMCell(nn.Module):
    """
    A single step of ConvLSTM
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        """
        Initialize ConvLSTM cell

        input_channels (int): Number of channels of input tensor.  
        hidden_channels (int): Number of channels of hidden state.   
        kernel_size (int): Size of the convolutional kernel.
        """
        super().__init__()
        # Initialize
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # Compute input, forget, cell, and output gates
        self.conv = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, x, h, c):
        """
        x: (B, C_in, H, W)
        h, c: (B, C_hidden, H, W)
        returns: h_next, c_next
        """
        # Concatenate input and previous hidden state along channel axis
        combined = torch.cat([x, h], dim=1)
        # Convolution 
        gates = self.conv(combined)
        # Split into input gate, forget gate, candidate, output gate
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        # Nonlinearities
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        # Update
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    """
    Full ConvLSTM using ConvLSTMCell
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

    def forward(self, x):
        """
        x: (Batch, Channel, Time, Height, Width)
        """
        B, C, T, H, W = x.shape
        # Initial h and c 
        h = torch.zeros(B, self.cell.hidden_channels, H, W, device=x.device)
        c = torch.zeros(B, self.cell.hidden_channels, H, W, device=x.device)
        
        outputs = []
        # Loop through each time step
        for t in range(T):
            # time step
            x_t = x[:, :, t, :, :] 
            # One step of ConvLSTMCell
            h, c = self.cell(x_t, h, c)
            # add time dimension back (B, C, 1, H, W)
            outputs.append(h.unsqueeze(2)) 
            
        # Concatenate along time axis
        return torch.cat(outputs, dim=2), (h, c)

# Code from Wen, Q. (2020). ConvLSTM PyTorch implementation [Code repository]. GitHub.
# https://github.com/ndrplz/ConvLSTM_pytorch

class Conv3DBlock(nn.Module):
    """
    Standard 3D Convolution Block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            # First 3×3×3 conv
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # Second 3×3×3 conv
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

# Main model
class UNet3D_ConvLSTM(nn.Module):
    """
    3D U-Net + ConvLSTM bottleneck.
    Input:  x  (B, C=3, T=8, H=256, W=266)
    Output: (B, 1, H, W) binary mask
    """
    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        
        # Encoder
        self.enc1 = Conv3DBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2)) 
        
        self.enc2 = Conv3DBlock(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.enc3 = Conv3DBlock(base_channels*2, base_channels*4)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        # ConvLSTM Bottle neck
        # Input: (B, 64, 8, 32, 32)
        self.proj = nn.Conv3d(base_channels*4, base_channels*4, kernel_size=1)
        self.lstm = ConvLSTM(base_channels*4, base_channels*8, kernel_size=3)
        
        # Decoder (Expanding Path)
        self.up3 = nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec3 = Conv3DBlock(base_channels*8, base_channels*4)
        
        self.up2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec2 = Conv3DBlock(base_channels*4, base_channels*2)
        
        self.up1 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=(1,2,2), stride=(1,2,2))
        self.dec1 = Conv3DBlock(base_channels*2, base_channels)
        
        # Head
        self.final = nn.Conv3d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, 8, 256, 256)
        
        # Encoder
        e1 = self.enc1(x) # (16, 8, 256, 256)
        p1 = self.pool1(e1) # (16, 8, 128, 128)
        
        e2 = self.enc2(p1) # (32, 8, 128, 128)
        p2 = self.pool2(e2) # (32, 8, 64, 64)
        
        e3 = self.enc3(p2) # (64, 8, 64, 64)
        p3 = self.pool3(e3) # (64, 8, 32, 32)
        
        # Bottleneck
        p3 = self.proj(p3)
        lstm_out, _ = self.lstm(p3) # (128, 8, 32, 32)
        
        # Decoder
        u3 = self.up3(lstm_out) # (64, 8, 64, 64)
        cat3 = torch.cat([u3, e3], dim=1) # Skip connection
        d3 = self.dec3(cat3)
        
        u2 = self.up2(d3) # (32, 8, 128, 128)
        cat2 = torch.cat([u2, e2], dim=1) # Skip connection
        d2 = self.dec2(cat2)
        
        u1 = self.up1(d2) # (16, 8, 256, 256)
        cat1 = torch.cat([u1, e1], dim=1) # Skip connection
        d1 = self.dec1(cat1)
        
        # Final Projection
        out_3d = self.final(d1) # (B, 1, 8, 256, 256)
        
        # Select the labeld image (5th image)
        out_2d = out_3d[:, :, 4, :, :] 
        
        return out_2d


def rle_encode(x, fg_val=1):
    """
    Encoding for submission.
    x (numpy array): mask (1=contrail, 0=bg)
    Returns: list of run lengths
    """
    # 1d array with that finds the indices of pixels where there are contrails
    dots = np.where(x.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2 # Because indices start at 0
    for b in dots:
        # Check if the current pixel is not the neighbor of the previous pixel
        if b > prev + 1:
            # Add start position
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b # update
    return run_lengths

def list_to_string(x):
    """
    Converts RLE list to string for CSV
    """
    if x:
        s = str(x).replace("[", "").replace("]", "").replace(",", "")
    else:
        s = '-'
    return s



def run_inference():
    print("Loading Model...")
    # Load model
    model = UNet3D_ConvLSTM(in_channels=3, base_channels=16).to(DEVICE)
    
    # Load Weights
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Make sure you are pointing to the correct .pth file in your input dataset.")
        return

    model.eval()
    
    print("Preparing Data...")
    if not os.path.exists(TEST_DIR):
        print("Test directory not found, skipping.")
        return

    test_ids = sorted(os.listdir(TEST_DIR))
    test_ds = ContrailDataset(TEST_DIR, test_ids)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    submission_data = []
    
    print("Starting Prediction...")
    with torch.no_grad():
        for x, rids in tqdm(test_loader):
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits)
            
            # Thresholding
            preds = (probs > 0.5).float().cpu().numpy()[:, 0, :, :]
            
            for i, rid in enumerate(rids):
                mask = preds[i]
                rle = rle_encode(mask)
                rle_str = list_to_string(rle)
                submission_data.append({"record_id": rid, "encoded_pixels": rle_str})
                
    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv("submission.csv", index=False)
    print("submission.csv generated successfully!")
    print(df_sub.head())




run_inference()




