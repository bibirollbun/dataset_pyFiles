import numpy as np
import pandas as pd
import os
import glob
import gc
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# --- Configuration ---
COMPETITION_NAME = "waveform-inversion"
BASE_PATH = f"/kaggle/input/{COMPETITION_NAME}"
TRAIN_SAMPLES_PATH = os.path.join(BASE_PATH, "train_samples")
TEST_PATH = os.path.join(BASE_PATH, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

# Model & Training Params (EXAMPLES - need tuning!)
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16 # Adjust based on GPU memory
IMG_HEIGHT = 70 # Assuming velocity map height from typical data/desc
IMG_WIDTH = 70  # Assuming velocity map width (consistent with submission x_69)
N_INPUT_CHANNELS = 1 # Placeholder - how to represent 4D seismic data as input channels?
N_OUTPUT_CHANNELS = 1 # Predicting a single velocity map
EPOCHS = 20 # Start small, increase for real training
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.1 # Use 10% of training data for validation

# Submission Params
SUBMISSION_ODD_COLS_ONLY = True
N_SUBMISSION_COLS = 70 # Number of columns in the full velocity map (0 to 69)

print(f"Using device: {DEVICE}")
print(f"Base path: {BASE_PATH}")
print(f"Image dimensions (H, W): ({IMG_HEIGHT}, {IMG_WIDTH})") # Verify these dimensions!

# Set seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_train_files_df(train_path):
    all_files = []

    # --- Vel and Style families ---
    # data/*.npy paired with model/*.npy
    data_files_vel_style = sorted(glob.glob(os.path.join(train_path, "data", "*.npy")))
    model_files_vel_style = sorted(glob.glob(os.path.join(train_path, "model", "*.npy")))

    if len(data_files_vel_style) == len(model_files_vel_style) and len(data_files_vel_style) > 0:
        print(f"Found {len(data_files_vel_style)} Vel/Style data/model pairs.")
        # Basic check: Assume files correspond based on sorting by name parts
        # Example: data1.npy -> model1.npy
        file_map = {}
        for f in data_files_vel_style:
            basename = os.path.basename(f)
            key = basename.replace('data','').replace('.npy','')
            file_map[key] = {'data': f}
        for f in model_files_vel_style:
            basename = os.path.basename(f)
            key = basename.replace('model','').replace('.npy','')
            if key in file_map:
                 file_map[key]['model'] = f

        for key, paths in file_map.items():
             if 'data' in paths and 'model' in paths:
                 all_files.append({'data_path': paths['data'], 'model_path': paths['model'], 'family': 'Vel/Style'})
             else:
                 print(f"Warning: Missing pair for key {key} in Vel/Style")

    else:
        print("Mismatch or no Vel/Style files found.")
        if len(data_files_vel_style) != len(model_files_vel_style):
             print(f"Warning: Mismatch in Vel/Style file counts: {len(data_files_vel_style)} data vs {len(model_files_vel_style)} model files.")


    # --- Fault family ---
    # seis_{n}_1_{i}.npy paired with vel_{n}_1_{i}.npy
    data_files_fault = sorted(glob.glob(os.path.join(train_path, "seis_*.npy")))
    model_files_fault = sorted(glob.glob(os.path.join(train_path, "vel_*.npy")))

    if len(data_files_fault) > 0:
        print(f"Found {len(data_files_fault)} Fault seismic files and {len(model_files_fault)} Fault velocity files.")
        # Create a map based on the common part of the filename
        file_map_fault = {}
        for f in data_files_fault:
            basename = os.path.basename(f)
            key = basename.replace('seis_','').replace('.npy','') # e.g., "10_1_0"
            file_map_fault[key] = {'data': f}
        for f in model_files_fault:
             basename = os.path.basename(f)
             key = basename.replace('vel_','').replace('.npy','') # e.g., "10_1_0"
             if key in file_map_fault:
                 file_map_fault[key]['model'] = f
             else:
                  # This case might happen if vel files exist without corresponding seis files
                  pass # Or add logging

        for key, paths in file_map_fault.items():
             if 'data' in paths and 'model' in paths:
                 all_files.append({'data_path': paths['data'], 'model_path': paths['model'], 'family': 'Fault'})
             else:
                 print(f"Warning: Missing pair for key {key} in Fault family")

    else:
         print("No Fault files found.")


    if not all_files:
        print("ERROR: No training file pairs were found. Check paths and file structures.")
        return pd.DataFrame()

    df = pd.DataFrame(all_files)
    print(f"Total training file pairs found: {len(df)}")
    return df

train_df = get_train_files_df(TRAIN_SAMPLES_PATH)
display(train_df.head())

# --- Get Test Files ---
test_files = sorted(glob.glob(os.path.join(TEST_PATH, "*.npy")))
test_oids = [os.path.basename(f).replace('.npy', '') for f in test_files]
print(f"\nFound {len(test_files)} test files.")
# print(test_oids[:5]) # Example test oids


# Placeholder: Store min/max velocity values for potential normalization
# These might be estimated from the training data or known physical bounds
VELOCITY_MIN = 1500 # Example value (m/s) - **MUST BE ADJUSTED BASED ON DATA**
VELOCITY_MAX = 4500 # Example value (m/s) - **MUST BE ADJUSTED BASED ON DATA**

def preprocess_input(seismic_data_4d):
    """
    Placeholder function to process 4D seismic data into a format suitable for the U-Net.
    Input: numpy array (batch_size, num_sources, time_steps, num_receivers)
    Output: torch tensor (batch_size, N_INPUT_CHANNELS, height, width) - requires careful design!

    Current simple strategy:
    1. Select the first source. -> (batch, time, receivers)
    2. Treat time_steps as channels? Or maybe use Conv1D first?
    3. Normalize.
    4. Reshape/Interpolate if needed to match expected H, W for U-Net?

    THIS IS A MAJOR SIMPLIFICATION AND LIKELY NEEDS SIGNIFICANT IMPROVEMENT.
    """
    # Example: Select first source, keep time and receivers
    # Shape: (batch_size, time_steps, num_receivers)
    processed_data = seismic_data_4d[:, 0, :, :]

    # Normalize (example: standardize per sample)
    mean = np.mean(processed_data, axis=(1, 2), keepdims=True)
    std = np.std(processed_data, axis=(1, 2), keepdims=True)
    processed_data = (processed_data - mean) / (std + 1e-6) # Add epsilon for stability

    # Reshape/Adapt to U-Net input (e.g., treat time as channels or use specific layers)
    # Assuming IMG_HEIGHT=time_steps, IMG_WIDTH=num_receivers FOR THIS PLACEHOLDER
    # This assumption is likely INCORRECT and depends heavily on the actual data dimensions
    # and the chosen model architecture.
    if processed_data.shape[1] != IMG_HEIGHT or processed_data.shape[2] != IMG_WIDTH:
         # This part needs real implementation - maybe interpolation, padding, or different architecture
         # For now, we'll just add a channel dimension assuming shapes match (WHICH THEY PROBABLY DON'T)
         print(f"Warning: Input shape mismatch {processed_data.shape} vs expected ({IMG_HEIGHT}, {IMG_WIDTH}). Requires proper handling.")
         # As a fallback placeholder, let's just take a slice or resize crudely if possible.
         # This is highly problematic and just for code structure demonstration.
         # Let's assume we can reshape/select to get (batch, N_INPUT_CHANNELS, H, W)
         # Example: adding a channel dim - this assumes T=H, R=W
         if N_INPUT_CHANNELS == 1:
             processed_data = processed_data[:, np.newaxis, :, :] # Add channel dim
             # If shapes *still* don't match H, W, resizing/cropping/padding needed here
             # This step is complex and data-dependent.
         else:
             # Handle multiple input channels (e.g. if time is treated as channels)
             # processed_data = processed_data.transpose(0, 1, 2) # Example, needs correct logic
             raise NotImplementedError("Input channel handling > 1 needs specific implementation")


    # Ensure final shape matches (batch, N_INPUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    # Add crude resizing/padding if shapes don't match (VERY basic placeholder)
    current_h, current_w = processed_data.shape[2], processed_data.shape[3]
    if current_h != IMG_HEIGHT or current_w != IMG_WIDTH:
        # Use torch functional interpolate (requires tensor)
        temp_tensor = torch.tensor(processed_data, dtype=torch.float32)
        temp_tensor = nn.functional.interpolate(temp_tensor, size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=False)
        processed_data = temp_tensor.numpy()
        print(f"Resized input from {(current_h, current_w)} to {(IMG_HEIGHT, IMG_WIDTH)}")


    return torch.tensor(processed_data, dtype=torch.float32)


def preprocess_output(velocity_map_3d):
    """
    Process 3D velocity map (batch, H, W) for model target.
    Normalize and add channel dimension.
    """
    # Add channel dim: (batch, H, W) -> (batch, 1, H, W)
    velocity_map_4d = velocity_map_3d[:, np.newaxis, :, :]

    # Normalize (Example: Min-Max scaling to [0, 1])
    normalized_map = (velocity_map_4d - VELOCITY_MIN) / (VELOCITY_MAX - VELOCITY_MIN)
    normalized_map = np.clip(normalized_map, 0, 1) # Ensure values are within [0, 1]

    return torch.tensor(normalized_map, dtype=torch.float32)

def postprocess_output(prediction_tensor):
    """
    Inverse transform the model's normalized prediction back to original velocity scale.
    Input: torch tensor (batch, 1, H, W)
    Output: numpy array (batch, H, W)
    """
    prediction = prediction_tensor.detach().cpu().numpy()
    # Denormalize (from [0, 1] back to original scale)
    velocity_map = prediction * (VELOCITY_MAX - VELOCITY_MIN) + VELOCITY_MIN
    # Remove channel dim: (batch, 1, H, W) -> (batch, H, W)
    velocity_map = velocity_map.squeeze(1)
    return velocity_map


class WaveformDataset(Dataset):
    def __init__(self, df, file_indices, transform_input=None, transform_output=None):
        self.df = df
        self.file_indices = file_indices # Indices of files in df to use for this dataset split
        self.transform_input = transform_input
        self.transform_output = transform_output

        self.samples = []
        print(f"Loading data pointers for {len(self.file_indices)} files...")
        for idx in tqdm(self.file_indices):
            row = self.df.iloc[idx]
            try:
                # Load the entire file content once
                seismic_data_batch = np.load(row['data_path'])
                velocity_map_batch = np.load(row['model_path'])

                # Verify shapes (example: check first sample)
                # print(f"Loaded seismic shape: {seismic_data_batch.shape}, velocity shape: {velocity_map_batch.shape}")
                # Expected: (batch_size, num_sources, time_steps, num_receivers) and (batch_size, height, width)

                num_samples_in_file = seismic_data_batch.shape[0]
                if num_samples_in_file != velocity_map_batch.shape[0]:
                     print(f"Warning: Mismatch in batch size for file {idx}. Skipping.")
                     continue

                for i in range(num_samples_in_file):
                    # Store pointers or indices to individual samples within the loaded batches
                    # Option 1: Store indices (memory efficient if files loaded lazily)
                    self.samples.append({'file_idx': idx, 'sample_idx': i})
                    # Option 2: Store actual data (simpler but uses more RAM if all loaded upfront)
                    # self.samples.append({'seismic': seismic_data_batch[i], 'velocity': velocity_map_batch[i]})

            except Exception as e:
                print(f"Error loading file index {idx} ({row['data_path']}/{row['model_path']}): {e}")

        print(f"Total individual samples found: {len(self.samples)}")
        # If using Option 2 above, clear the large loaded batches now if needed
        # del seismic_data_batch, velocity_map_batch
        # gc.collect()


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        file_idx = sample_info['file_idx']
        sample_idx_in_file = sample_info['sample_idx']

        # Load data for the specific sample
        # This might involve reloading the file or accessing pre-loaded data
        try:
            # --- Reload file (if not pre-loaded) ---
            row = self.df.iloc[file_idx]
            # Using mmap_mode='r' can help with large files if memory is an issue,
            # but might be slower for random access depending on usage pattern.
            seismic_data_full_batch = np.load(row['data_path']) # Potentially use mmap_mode='r'
            velocity_map_full_batch = np.load(row['model_path']) # Potentially use mmap_mode='r'

            seismic_sample = seismic_data_full_batch[sample_idx_in_file]
            velocity_sample = velocity_map_full_batch[sample_idx_in_file]
            # --- ---

            # --- Access pre-loaded data (if using Option 2 in __init__) ---
            # seismic_sample = sample_info['seismic']
            # velocity_sample = sample_info['velocity']
            # --- ---

            # Apply transformations (preprocessing)
            # The preprocessing functions expect batches, so add a batch dim temporarily
            if self.transform_input:
                seismic_tensor = self.transform_input(seismic_sample[np.newaxis, ...]) # Add batch dim
                seismic_tensor = seismic_tensor.squeeze(0) # Remove batch dim
            else:
                seismic_tensor = torch.tensor(seismic_sample, dtype=torch.float32) # Basic tensor conversion

            if self.transform_output:
                velocity_tensor = self.transform_output(velocity_sample[np.newaxis, ...]) # Add batch dim
                velocity_tensor = velocity_tensor.squeeze(0) # Remove batch dim
            else:
                velocity_tensor = torch.tensor(velocity_sample, dtype=torch.float32) # Basic tensor conversion


            # Verify output tensor shape (should be ~ [C_out, H, W]) after processing
            # print(f"Sample {idx}: Input shape {seismic_tensor.shape}, Output shape {velocity_tensor.shape}")
            if velocity_tensor.shape[0] != N_OUTPUT_CHANNELS or velocity_tensor.shape[1] != IMG_HEIGHT or velocity_tensor.shape[2] != IMG_WIDTH:
                 print(f"Warning: Unexpected output tensor shape after processing: {velocity_tensor.shape}")


            return seismic_tensor, velocity_tensor

        except Exception as e:
             print(f"Error getting item {idx} (file {file_idx}, sample {sample_idx_in_file}): {e}")
             # Return dummy data or raise error? For now, return None and handle in DataLoader.
             # This needs robust error handling.
             return None, None


# --- Data Splitting ---
if not train_df.empty:
    train_indices, val_indices = train_test_split(
        range(len(train_df)), # Split based on file indices
        test_size=VALIDATION_SPLIT,
        random_state=SEED
    )

    print(f"\nSplitting {len(train_df)} files into:")
    print(f"Training files: {len(train_indices)}")
    print(f"Validation files: {len(val_indices)}")

    # Create Datasets
    train_dataset = WaveformDataset(train_df, train_indices, transform_input=preprocess_input, transform_output=preprocess_output)
    val_dataset = WaveformDataset(train_df, val_indices, transform_input=preprocess_input, transform_output=preprocess_output)

    # Create DataLoaders
    # Handle potential None values returned by dataset __getitem__ due to errors
    def collate_fn(batch):
        batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
        if not batch: return torch.Tensor(), torch.Tensor() # Return empty tensors if batch is empty
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), pin_memory=True, collate_fn=collate_fn)

    print(f"\nDataLoaders created.")
    # Optional: Check a batch shape
    # try:
    #     inputs, targets = next(iter(train_loader))
    #     print(f"Sample batch input shape: {inputs.shape}") # Should be [B, C_in, H_in, W_in]
    #     print(f"Sample batch target shape: {targets.shape}") # Should be [B, C_out, H, W]
    # except Exception as e:
    #     print(f"Could not load a sample batch: {e}")

else:
    print("\nSkipping Dataset/DataLoader creation due to missing training files.")
    train_loader, val_loader = None, None


# Basic U-Net implementation (taken from PyTorch Hub or common implementations)
# (Ensure you have the necessary U-Net code defined here or imported)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # Add a final activation like Sigmoid if output is normalized to [0, 1]
        self.final_activation = nn.Sigmoid() # Or nn.Identity() if not normalizing to [0,1]

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # Apply final activation
        outputs = self.final_activation(logits)
        return outputs

# Instantiate the model
model = UNet(n_channels=N_INPUT_CHANNELS, n_classes=N_OUTPUT_CHANNELS).to(DEVICE)
print(f"U-Net model created with {N_INPUT_CHANNELS} input channels and {N_OUTPUT_CHANNELS} output classes.")
# Optional: Print model summary (requires torchinfo)
# try:
#     from torchinfo import summary
#     # Input size needs to match preprocessed input tensor shape B,C,H,W
#     # Use a dummy batch size, e.g., 1
#     # WARNING: This assumes your preprocess_input correctly shapes data to (B, N_INPUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
#     # Adjust input_size if your preprocessing yields different H, W for the input tensor
#     dummy_input_size = (BATCH_SIZE, N_INPUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
#     summary(model, input_size=dummy_input_size)
# except ImportError:
#     print("torchinfo not installed, skipping model summary.")
# except Exception as e:
#      print(f"Could not generate model summary. Check input size/preprocessing. Error: {e}")


# --- Loss Function ---
# MAE Loss
criterion = nn.L1Loss()

# --- Optimizer ---
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Learning Rate Scheduler (Optional) ---
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

# --- Training Loop ---
best_val_mae = float('inf')
best_model_path = "best_model.pth"

if train_loader and val_loader: # Only train if data loaders are available
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for inputs, targets in pbar_train:
            if inputs.numel() == 0 or targets.numel() == 0: continue # Skip empty batches
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Ensure shapes match for loss calculation
            if outputs.shape != targets.shape:
                 print(f"Shape mismatch! Output: {outputs.shape}, Target: {targets.shape}. Skipping batch.")
                 # This often indicates an issue in preprocessing or model output layer
                 continue

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            pbar_train.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset) # Average loss over all samples

        # --- Validation ---
        model.eval()
        val_mae = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for inputs, targets in pbar_val:
                if inputs.numel() == 0 or targets.numel() == 0: continue # Skip empty batches
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)

                if outputs.shape != targets.shape:
                     print(f"Shape mismatch! Output: {outputs.shape}, Target: {targets.shape}. Skipping batch.")
                     continue

                mae_batch = criterion(outputs, targets) # MAE loss
                val_mae += mae_batch.item() * inputs.size(0)
                pbar_val.set_postfix(mae=mae_batch.item())

        val_mae /= len(val_loader.dataset) # Average MAE over all validation samples

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f} - Val MAE: {val_mae:.6f}")

        # Optional: Update learning rate scheduler
        # scheduler.step(val_mae)

        # Save the best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_model_path)
            print(f"✨ New best model saved with Val MAE: {best_val_mae:.6f} to {best_model_path}")

        # Clean up GPU memory
        del inputs, targets, outputs
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nTraining finished. Best Validation MAE: {best_val_mae:.6f}")

else:
     print("Skipping training as DataLoaders could not be created.")


class WaveformTestDataset(Dataset):
    """Dataset for loading test data (only seismic input)."""
    def __init__(self, test_files, oids, transform_input=None):
        self.test_files = test_files
        self.oids = oids
        self.transform_input = transform_input
        # Assume test files also contain batches like training files
        self.samples = []
        print(f"Mapping test samples...")
        for i, file_path in enumerate(tqdm(self.test_files)):
            try:
                # Get the number of samples in the file without loading everything (if possible)
                # This is tricky with .npy without loading. Let's assume we load it to find out.
                # Use mmap_mode potentially if files are huge.
                test_data_batch = np.load(file_path) # Potentially use mmap_mode='r'
                num_samples_in_file = test_data_batch.shape[0]
                oid = self.oids[i]
                for j in range(num_samples_in_file):
                     # Store file path and index within file for each sample
                     self.samples.append({'file_path': file_path, 'sample_idx': j, 'oid': oid})
                del test_data_batch # Free memory
                gc.collect()
            except Exception as e:
                 print(f"Error processing test file {file_path}: {e}")
        print(f"Total individual test samples found: {len(self.samples)}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        file_path = sample_info['file_path']
        sample_idx_in_file = sample_info['sample_idx']
        oid = sample_info['oid']

        try:
            # Load the specific sample
            # Again, potentially more efficient ways for huge files (mmap)
            test_data_full_batch = np.load(file_path) # Potentially use mmap_mode='r'
            seismic_sample = test_data_full_batch[sample_idx_in_file]
            del test_data_full_batch # Free memory
            gc.collect()

            # Apply input transformation
            if self.transform_input:
                seismic_tensor = self.transform_input(seismic_sample[np.newaxis, ...]) # Add batch dim
                seismic_tensor = seismic_tensor.squeeze(0) # Remove batch dim
            else:
                seismic_tensor = torch.tensor(seismic_sample, dtype=torch.float32)

            # Return the processed input tensor and the original oid/index info
            # OID might need careful handling if test files don't map 1:1 with submission OIDs
            # Assuming each test file {oid}.npy corresponds to one oid in submission.
            # If test files contain *batches* that need individual predictions, the OID logic needs adjustment.
            # Let's assume each sample within a test file needs prediction, but they all share the root OID.
            # The submission format implies one output map per OID.
            # This part is ambiguous in the problem description if test files are batched.
            # For now, assume one prediction needed per file {oid}.npy --> average predictions if batched?
            # Let's refine: Assume one prediction per file {oid}.npy. We'll process the first sample only.
            # *** THIS NEEDS CLARIFICATION FROM COMPETITION HOSTS or FORUM ***
            # If each test file needs *one* map prediction, we might average predictions across samples inside, or just use one sample.
            # Let's proceed assuming we predict only for the *first sample* in each test file for simplicity.
            if sample_idx_in_file == 0:
                 return seismic_tensor, oid
            else:
                 # Skip other samples within the same file for now
                 return None, None # Need collate_fn to handle this

        except Exception as e:
             print(f"Error getting test item {idx} (file {file_path}, sample {sample_idx_in_file}): {e}")
             return None, None


# --- Create Test Dataset & DataLoader ---
# Adjusting the test dataset assumption: Predict one map per file {oid}.npy
# Modify the dataset to only yield the first sample from each file.

class WaveformTestDatasetPerFile(Dataset):
    """Dataset for loading test data (one sample per file)."""
    def __init__(self, test_files, oids, transform_input=None):
        self.test_files = test_files
        self.oids = oids
        self.transform_input = transform_input
        print(f"Found {len(test_files)} test files to process.")

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, idx):
        file_path = self.test_files[idx]
        oid = self.oids[idx]

        try:
            test_data_batch = np.load(file_path) # Potentially use mmap_mode='r'
            # Use only the first sample from the batch/file
            seismic_sample = test_data_batch[0]
            del test_data_batch
            gc.collect()

            # Apply input transformation
            if self.transform_input:
                seismic_tensor = self.transform_input(seismic_sample[np.newaxis, ...]) # Add batch dim
                seismic_tensor = seismic_tensor.squeeze(0) # Remove batch dim
            else:
                seismic_tensor = torch.tensor(seismic_sample, dtype=torch.float32)

            return seismic_tensor, oid

        except Exception as e:
             print(f"Error getting test item {idx} (file {file_path}): {e}")
             return None, None # Handle potential errors


# Helper collate function for test loader
def collate_fn_test(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if not batch: return torch.Tensor(), []
    inputs = [item[0] for item in batch]
    oids = [item[1] for item in batch]
    inputs_collated = torch.utils.data.dataloader.default_collate(inputs)
    return inputs_collated, oids


if os.path.exists(best_model_path):
    print(f"Loading best model from {best_model_path}")
    # Ensure model architecture is defined before loading state_dict
    model = UNet(n_channels=N_INPUT_CHANNELS, n_classes=N_OUTPUT_CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    test_dataset = WaveformTestDatasetPerFile(test_files, test_oids, transform_input=preprocess_input)
    # Use batch size 1 for prediction if memory is tight or processing one file at a time
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count(), collate_fn=collate_fn_test)

    predictions = []
    print("Starting prediction on test set...")
    with torch.no_grad():
        for inputs, oids_batch in tqdm(test_loader):
             if inputs.numel() == 0: continue # Skip empty batches

             inputs = inputs.to(DEVICE)
             outputs_norm = model(inputs) # Normalized output [B, 1, H, W]

             # Postprocess (denormalize) and move to CPU
             outputs_denorm = postprocess_output(outputs_norm) # Numpy array [B, H, W]

             # Format for submission
             for i, oid in enumerate(oids_batch):
                 pred_map = outputs_denorm[i] # Shape (H, W), e.g., (70, 70)
                 # Iterate through rows (ypos)
                 for y_pos in range(pred_map.shape[0]): # Iterate through height (rows)
                     row_data = {'oid_ypos': f"{oid}_y_{y_pos}"}
                     # Extract odd columns (x_1, x_3, ..., x_69)
                     # Assuming width is N_SUBMISSION_COLS (e.g., 70), so columns are 0 to 69
                     # Odd indices are 1, 3, 5, ..., 69
                     for x_pos in range(1, N_SUBMISSION_COLS, 2): # Step by 2
                          if x_pos < pred_map.shape[1]: # Check bounds
                              row_data[f'x_{x_pos}'] = pred_map[y_pos, x_pos]
                          else:
                              # Handle cases where prediction width might be smaller? Pad? Error?
                              # For now, let's assume width matches or is larger.
                              # If smaller, maybe fill with a default value (e.g., 0 or nan) or error out.
                              # Setting to 0 for now if out of bounds.
                              print(f"Warning: x_pos {x_pos} out of bounds for prediction width {pred_map.shape[1]} for oid {oid}. Setting to 0.")
                              row_data[f'x_{x_pos}'] = 0.0
                     predictions.append(row_data)

    print(f"Generated {len(predictions)} prediction rows.")

    # Create submission DataFrame
    if predictions:
        submission_df = pd.DataFrame(predictions)
        # Ensure correct column order (oid_ypos, x_1, x_3, ...)
        cols = ['oid_ypos'] + [f'x_{i}' for i in range(1, N_SUBMISSION_COLS, 2)]
        submission_df = submission_df[cols]

        submission_df.to_csv("submission.csv", index=False)
        print("submission.csv created successfully.")
        display(submission_df.head())
    else:
        print("No predictions were generated. Cannot create submission file.")

elif not train_loader or not val_loader:
     print("Skipping prediction as model was not trained due to data loading issues.")
     # Create dummy submission based on sample if needed for platform checks
     if os.path.exists(SAMPLE_SUB_PATH):
          print("Creating dummy submission from sample file.")
          sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
          # Potentially populate with a constant value if required
          # sample_sub.iloc[:, 1:] = 3000.0 # Example: Fill with constant velocity
          sample_sub.to_csv("submission.csv", index=False)
     else:
          print("Sample submission file not found, cannot create dummy submission.")

else:
     print("Best model file not found. Cannot run prediction.")
     # Create dummy submission if possible (as above)
     if os.path.exists(SAMPLE_SUB_PATH):
          print("Creating dummy submission from sample file.")
          sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
          sample_sub.to_csv("submission.csv", index=False)
     else:
          print("Sample submission file not found, cannot create dummy submission.")

