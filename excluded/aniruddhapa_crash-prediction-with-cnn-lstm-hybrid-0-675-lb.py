!pip install decord


# Cell: Imports and Configuration
import os
# import cv2 # Keep if needed for other potential ops, but not core sampling/resize now
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader # Moved DataLoader import here
from torchvision import transforms
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import gc # Import garbage collector

# --- Decord Import ---
import decord
from decord import VideoReader, cpu
# Initialize Decord context (do this once globally)
decord.bridge.set_bridge('torch') # Let Decord return PyTorch tensors directly

# --- AMP Imports ---
from torch.cuda.amp import GradScaler, autocast

# --- Collate Function Import ---
from torch.utils.data.dataloader import default_collate


print("--- Initial Setup ---")
#--- Basic Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BASE_DATA_PATH = '/kaggle/input/nexar-collision-prediction'
TRAIN_VIDEO_DIR = os.path.join(BASE_DATA_PATH, 'train')
TEST_VIDEO_DIR = os.path.join(BASE_DATA_PATH, 'test')
ORIGINAL_TRAIN_CSV = os.path.join(BASE_DATA_PATH, 'train.csv')
TEST_CSV_PATH = os.path.join(BASE_DATA_PATH, 'test.csv') # Corrected variable name
SAMPLE_SUB_CSV = os.path.join(BASE_DATA_PATH, 'sample_submission.csv')


# --- Working Directory Paths ---
WORKING_DIR = '/kaggle/working/'
TRAIN_CSV_SPLIT = os.path.join(WORKING_DIR, 'train_split.csv')
VAL_CSV_SPLIT = os.path.join(WORKING_DIR, 'val_split.csv')
SUBMISSION_CSV = os.path.join(WORKING_DIR, 'submission.csv')
CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model_cnn_lstm.pth")


# --- Model/Training Hyperparameters ---

IMG_SIZE = 224
SEQ_LEN = 16 # Number of frames per sample
N_CHANNELS = 3
NUM_CLASSES = 1    # Binary classification (crash/no-crash)
LSTM_HIDDEN_SIZE = 512
LSTM_LAYERS = 2
PRETRAINED = True # Use pretrained ResNet
LEARNING_RATE = 5e-5
MAX_EPOCHS = 100     # Set high, let early stopping decide
EARLY_STOPPING_PATIENCE = 5 # Example: Stop after 10 epochs with no val loss improvement

# !! CRITICAL TUNING PARAMETERS for GPU !!
BATCH_SIZE = 32     # << START HERE for GPU (e.g., 16, 32, 64) - TUNE THIS
NUM_WORKERS = 2     # << START HERE (e.g., 2, 4) - TUNE THIS
PREFETCH_FACTOR = 2 # For DataLoader prefetching


# --- Set final paths for train/val CSVs (will use splits if they exist) ---
TRAIN_CSV_PATH_FINAL = TRAIN_CSV_SPLIT
VAL_CSV_PATH_FINAL = VAL_CSV_SPLIT

print("Initial configuration and paths defined.")


# Initialize Decord context (do this once globally if preferred)
decord.bridge.set_bridge('torch') # Let Decord return PyTorch tensors directly

FRAME_STRIDE = 3 # Sample every 3rd frame (effective 10 FPS) - TUNE THIS (2, 3, 4)

def sample_frames_robust_stride(video_path, seq_len, stride):
    """ Samples seq_len frames using stride. Falls back to OpenCV. """
    frames_tensor = None
    # --- Try Decord ---
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0: return None

        # Calculate indices using stride
        end_frame = total_frames - 1
        # Start sampling potentially earlier to ensure seq_len frames are available
        # Max possible start index: end_frame - (seq_len - 1) * stride
        effective_len = (seq_len -1) * stride + 1
        if effective_len > total_frames:
            # Not enough frames even with stride=1, sample all available
            indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)
        else:
            # Sample a starting point randomly
            start_max = end_frame - effective_len + 1
            start_index = np.random.randint(0, start_max + 1)
            indices = np.arange(start_index, start_index + effective_len, stride)
            # Ensure we don't exceed total_frames due to rounding/edge cases
            indices = indices[indices <= end_frame][:seq_len] # Take only up to seq_len

        indices = np.clip(indices, 0, total_frames - 1).astype(int)

        # Ensure we have exactly seq_len indices, pad if necessary due to sampling near end
        if len(indices) < seq_len:
            padding_needed = seq_len - len(indices)
            last_idx = indices[-1] if len(indices) > 0 else total_frames - 1
            padding_indices = np.full(padding_needed, last_idx)
            indices = np.concatenate((indices, padding_indices)).astype(int)

        frames_tensor = vr.get_batch(indices) # (T, H, W, C), torch.uint8

        # Final check for shape (should match seq_len now)
        if frames_tensor.shape[0] != seq_len:
             print(f"WARN: Frame count mismatch after Decord+Stride ({frames_tensor.shape[0]} vs {seq_len}) for {video_path}. Trying OpenCV.")
             frames_tensor = None # Force OpenCV fallback

    except Exception as e_decord:
        print(f"Decord failed for {video_path}: {e_decord}. Trying OpenCV...")
        frames_tensor = None

    # --- Fallback to OpenCV (with stride) ---
    if frames_tensor is None:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return None
            total_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames_cv == 0:
                 cap.release(); return None

            frames_list = []
            effective_len_cv = (seq_len -1) * stride + 1
            if effective_len_cv > total_frames_cv:
                # Fallback: Sample uniformly if stride doesn't fit
                indices_cv = np.linspace(0, total_frames_cv - 1, seq_len, dtype=int)
            else:
                start_max_cv = total_frames_cv - effective_len_cv
                start_index_cv = np.random.randint(0, start_max_cv + 1)
                indices_cv = np.arange(start_index_cv, start_index_cv + effective_len_cv, stride)
                indices_cv = indices_cv[indices_cv < total_frames_cv][:seq_len]

            indices_cv = np.clip(indices_cv, 0, total_frames_cv - 1).astype(int)

            # Read specific frames (more efficient than reading all)
            for idx_to_read in indices_cv:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx_to_read)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_list.append(torch.from_numpy(frame_rgb))
                else:
                    # If read fails, append last good frame (or handle differently)
                    if frames_list: frames_list.append(frames_list[-1].clone())
                    else: break # Cannot read even first frame

            cap.release()

            if len(frames_list) < seq_len: # Pad if necessary
                if not frames_list: return None
                padding = frames_list[-1].unsqueeze(0).repeat(seq_len - len(frames_list), 1, 1, 1)
                frames_tensor = torch.cat((torch.stack(frames_list, dim=0), padding), dim=0)
            else:
                frames_tensor = torch.stack(frames_list[:seq_len], dim=0)

        except Exception as e_cv:
            print(f"OpenCV also failed for {video_path}: {e_cv}")
            return None

    # Final check: ensure tensor is not None and has correct shape
    if frames_tensor is None or frames_tensor.shape[0] != seq_len:
         print(f"ERROR: Final frame tensor shape incorrect for {video_path}. Got {frames_tensor.shape if frames_tensor is not None else 'None'}")
         return None

    return frames_tensor # Shape (T, H, W, C) uint8


import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset 
from torchvision import transforms


# --- PyTorch Dataset ---


class DashcamDataset(Dataset):
    def __init__(self, csv_file, video_dir, seq_len, transform=None, mode='train', apply_hflip=False, flip_p=0.5):
        """
        Args:
            csv_file (string): Path to the csv file (train, val, or test).
            video_dir (string): Directory with all the video files.
            seq_len (int): Number of frames per sequence.
            transform (callable, optional): Transform applied AFTER potential HFlip.
            mode (string): 'train', 'val', or 'test'.
            apply_hflip (bool): Whether to apply consistent RandomHorizontalFlip (train mode only).
            flip_p (float): Probability of applying the horizontal flip.
        """
        self.metadata = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.seq_len = seq_len
        self.transform = transform
        self.mode = mode
        self.apply_hflip = apply_hflip and self.mode == 'train'
        self.flip_p = flip_p

        # --- Corrected Target Column Handling ---
        # The target is *already* in train.csv (0 or 1). For val split, it's copied.
        # For test.csv, there is no target.
        if self.mode in ['train', 'val']:
            if 'target' not in self.metadata.columns:
                 raise ValueError(f"'target' column not found in {csv_file}. Ensure the CSV is correct.")
            # Ensure target is numeric (it should be 0/1 already)
            self.metadata['target'] = pd.to_numeric(self.metadata['target'], errors='coerce')
            if self.metadata['target'].isnull().any():
                 print(f"Warning: Found NaN values in target column of {csv_file}. Check data integrity.")
                 # Decide on handling: dropna, fillna(0)? For now, keep going.

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]
        video_id_raw = row['id'] # Get the ID as it is in the CSV

        # --- Determine Correct Filename ---
        # Try converting to int first for padding, fallback to float/string if needed
        try:
            video_id_int = int(video_id_raw)
            video_id_padded = str(video_id_int).zfill(5)
        except ValueError:
            # Handle cases where ID might be float (like 667.0) or already a padded string
            video_id_str = str(video_id_raw)
            if '.' in video_id_str: # Handle float IDs like '667.0'
                video_id_padded = str(int(float(video_id_str))).zfill(5)
            else: # Assume it might already be padded or just needs zfill
                 video_id_padded = video_id_str.zfill(5)
                
        video_path = os.path.join(self.video_dir, f"{video_id_padded}.mp4")

        # Use Decord to sample frames -> (T, H, W, C), uint8, CPU
        frames_tensor_hwc = sample_frames_robust_stride(video_path, self.seq_len,FRAME_STRIDE)

        if frames_tensor_hwc is None:
            print(f"Error processing video {video_id_padded}, returning None.")
            return None # collate_fn handles this

        # Permute HWC -> CHW and convert to float [0.0, 1.0]
        frames_tensor_chw = frames_tensor_hwc.permute(0, 3, 1, 2).float() / 255.0

        # Apply Consistent Augmentations (like HFlip) BEFORE Compose
        if self.apply_hflip and torch.rand(1) < self.flip_p:
             frames_tensor_chw = transforms.functional.hflip(frames_tensor_chw)

        # Apply Compose Transforms (Resize, Normalize, etc.)
        if self.transform:
            frames_processed = self.transform(frames_tensor_chw)
        else:
            frames_processed = frames_tensor_chw

        # Get Label/ID
        if self.mode in ['train', 'val']:
            label = row['target'] # Get target directly from the loaded CSV
            # Handle potential NaNs read from CSV if any issues occurred
            if pd.isna(label):
                print(f"Warning: NaN label encountered for video {video_id_padded} at index {idx}. Returning None.")
                return None # Or return a default label like 0.0? None is safer for collate_fn.
            label = torch.tensor(label, dtype=torch.float32)
            return frames_processed, label
        else: # 'test' mode
            # Return the original, non-padded ID as string
            return frames_processed, str(video_id_raw)

print("DashcamDataset class defined.")


# --- Transforms ---
# Define normalization (using ImageNet stats as a standard starting point)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Define Resize transform - works directly on (T, C, H, W)
# Use antialias=True for better quality when downsampling
resize = transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True)

# --- Validation/Test Transforms ---
# Input: Tensor (T, C, H, W), float [0.0, 1.0]
# Output: Tensor (T, C, IMG_SIZE, IMG_SIZE), normalized
val_test_transform = transforms.Compose([
    resize,      # Resize each frame in the sequence
    normalize    # Normalize the sequence tensor
])

# --- Training Transforms ---
# Input: Tensor (T, C, H, W), float [0.0, 1.0]
# Output: Tensor (T, C, IMG_SIZE, IMG_SIZE), augmented, normalized
# Note: RandomHorizontalFlip is applied *before* this Compose in Dataset.__getitem__ for consistency

# Optional: Define RandomCrop if desired (applied independently per frame)
# If using RandomCrop, Resize should output slightly larger first
# RESIZE_FOR_CROP_SIZE = IMG_SIZE + 32 # Example padding
# resize_for_crop = transforms.Resize((RESIZE_FOR_CROP_SIZE, RESIZE_FOR_CROP_SIZE), antialias=True)
# random_crop = transforms.RandomCrop((IMG_SIZE, IMG_SIZE))

train_transform = transforms.Compose([resize,transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.1),transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),normalize])

# Assign to variables used later (if needed, e.g., when creating datasets)
train_transform_final = train_transform
val_transform_final = val_test_transform
test_transform_final = val_test_transform

print("Transform pipelines defined (Decord compatible).")
print("Note: RandomHorizontalFlip handled in Dataset.__getitem__ for sequence consistency.")

# --- Collate Function ---
def collate_fn(batch):
    """
    Custom collate function to handle batches from DashcamDataset.
    """
    # 1. Filter out None values
    batch = [item for item in batch if item is not None]

    # 2. Handle empty batch case
    if not batch:
        print("Warning: Collate received an empty batch after filtering.")
        # Return structure suitable for either mode if possible, or handle in loop
        # Returning structure potentially suitable for test mode prediction loop
        return torch.empty((0, SEQ_LEN, N_CHANNELS, IMG_SIZE, IMG_SIZE)), []

    # 3. Determine mode
    is_test_mode = isinstance(batch[0][1], str)

    # 4. Unpack and stack
    if is_test_mode: # Test mode
        frames, ids = zip(*batch)
        frames_batch = default_collate(frames) # Use default_collate for tensors
        return frames_batch, list(ids)
    else: # Train/Val mode
        # Default collate works for (tensor, tensor) tuples
        return default_collate(batch)

print("Collate function defined.")


import torch.optim as optim
from sklearn.model_selection import train_test_split
import os # Make sure os is imported
import pandas as pd # Make sure pandas is imported

# --- (Previous code: DEVICE, Model Hyperparams, Training Hyperparams) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
NUM_CLASSES = 1
LSTM_HIDDEN_SIZE = 512
LSTM_LAYERS = 2
PRETRAINED = True
LEARNING_RATE = 1e-4

# --- Paths ---
BASE_DATA_PATH = '/kaggle/input/nexar-collision-prediction'
ORIGINAL_TRAIN_CSV = os.path.join(BASE_DATA_PATH, 'train.csv') # Keep original path separate

# Define paths for the split files in the working directory
TRAIN_CSV_SPLIT = '/kaggle/working/train_split.csv'
VAL_CSV_SPLIT = '/kaggle/working/val_split.csv'

# --- Perform Split ONLY if files don't exist ---
# --- Perform Split ONLY if files don't exist ---
if not os.path.exists(TRAIN_CSV_SPLIT) or not os.path.exists(VAL_CSV_SPLIT):
    print(f"Split files not found in /kaggle/working/. Performing train_test_split on {ORIGINAL_TRAIN_CSV}...")
    try:
        full_train_df = pd.read_csv(ORIGINAL_TRAIN_CSV)

        # --- Use the EXISTING 'target' column for stratification ---
        if 'target' not in full_train_df.columns:
             print("ERROR: Cannot perform stratified split because 'target' column is missing from train.csv.")
             # Handle error: maybe raise exception or proceed without stratification
             stratify_col = None
        else:
            # Ensure target is suitable for stratification (handle potential NaNs if any)
            full_train_df['target'] = pd.to_numeric(full_train_df['target'], errors='coerce').fillna(-1).astype(int) # Temp fillna for stratify
            if (full_train_df['target'] == -1).any():
                print("Warning: Found missing/non-numeric targets in train.csv. Stratification might be affected.")
            stratify_col = full_train_df['target']
            print(f"Stratifying split using 'target' column. Distribution:\n{stratify_col.value_counts(normalize=True)}")

        train_df, val_df = train_test_split(full_train_df,
                                            test_size=0.15,          # Validation set size
                                            random_state=42,       # For reproducibility
                                            stratify=stratify_col) # Stratify based on target

        # Before saving, potentially revert the fillna if it was temporary
        # train_df['target'] = train_df['target'].replace(-1, np.nan)
        # val_df['target'] = val_df['target'].replace(-1, np.nan)

        train_df.to_csv(TRAIN_CSV_SPLIT, index=False)
        val_df.to_csv(VAL_CSV_SPLIT, index=False)
        print(f"Created {TRAIN_CSV_SPLIT} and {VAL_CSV_SPLIT}")
        del full_train_df, train_df, val_df # Clean up memory
        gc.collect()

    except FileNotFoundError:
        print(f"ERROR: Original train CSV not found at {ORIGINAL_TRAIN_CSV}. Cannot perform split.")
        TRAIN_CSV_PATH_FINAL = None # Indicate failure
        VAL_CSV_PATH_FINAL = None
    except Exception as e:
         print(f"An error occurred during train/test split: {e}")
         TRAIN_CSV_PATH_FINAL = None
         VAL_CSV_PATH_FINAL = None
else:
    print(f"Using existing split files: {TRAIN_CSV_PATH_FINAL} and {VAL_CSV_PATH_FINAL}")

# Check if paths are valid before proceeding
if TRAIN_CSV_PATH_FINAL is None or VAL_CSV_PATH_FINAL is None:
    raise RuntimeError("Train/Validation split failed. Cannot proceed.")


# --- Create Datasets ---
print("\nCreating Datasets using split files...")

train_dataset = DashcamDataset(csv_file=TRAIN_CSV_PATH_FINAL,
                               video_dir=TRAIN_VIDEO_DIR,
                               seq_len=SEQ_LEN,
                               transform=train_transform_final,
                               mode='train',
                               apply_hflip=True,
                               flip_p=0.5)
print(f"Training dataset size: {len(train_dataset)}")

val_dataset = DashcamDataset(csv_file=VAL_CSV_PATH_FINAL,
                             video_dir=TRAIN_VIDEO_DIR,
                             seq_len=SEQ_LEN,
                             transform=val_transform_final,
                             mode='val',
                             apply_hflip=False)
print(f"Validation dataset size: {len(val_dataset)}")

# --- Create DataLoaders ---
print("\nCreating DataLoaders...")
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS,
                          pin_memory=True,
                          prefetch_factor=PREFETCH_FACTOR,
                          collate_fn=collate_fn)

val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE * 2, # Can use larger BS for validation
                        shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=True,
                        prefetch_factor=PREFETCH_FACTOR,
                        collate_fn=collate_fn)

print("Train and Validation DataLoaders created.")

# --- Optional: Sanity Check Iteration (before training) ---
print("\nIterating through one batch of Train Loader for sanity check...")
try:
    if len(train_loader) == 0:
         print("Train loader is empty. Cannot iterate.")
    else:
        first_batch_data = next(iter(train_loader))
        if first_batch_data is None or not first_batch_data[0].numel():
            print(f"First batch is empty or None.")
        else:
            frames_batch, labels_batch = first_batch_data
            print(f"Example Batch 1:")
            print("  Frames batch shape:", frames_batch.shape) # Should be (B, T, C, H, W)
            print("  Labels batch shape:", labels_batch.shape) # Should be (B,)
            print("  Labels:", labels_batch)
            print("\nIteration Example Complete.")
            del first_batch_data, frames_batch, labels_batch # Clean up memory
            gc.collect()
except StopIteration:
     print("Train loader is empty or could not fetch the first batch.")
except Exception as e:
    print(f"\nError during DataLoader iteration check: {e}")
    import traceback
    traceback.print_exc()


import torch
import torch.nn as nn
import torchvision.models as models

class VideoClassifierCNN_LSTM(nn.Module):
    def __init__(self, num_classes=1, lstm_hidden_size=512, lstm_layers=2, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes (1 for binary classification with sigmoid).
            lstm_hidden_size (int): Number of features in the LSTM hidden state.
            lstm_layers (int): Number of recurrent layers in LSTM.
            pretrained (bool): Whether to use a pretrained CNN backbone.
        """
        super().__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        # --- CNN Backbone ---
        # Load a pretrained ResNet (or another model like EfficientNet)
        # We'll remove the final fully connected layer (the original classifier)
        base_model = models.resnet18(pretrained=pretrained) # Example: ResNet18
        # Get the number of features output by the ResNet's pooling layer
        num_cnn_features = base_model.fc.in_features
        # Remove the final layer
        modules = list(base_model.children())[:-1]
        self.cnn_backbone = nn.Sequential(*modules)
        # Freeze backbone layers if desired (transfer learning)
        # for param in self.cnn_backbone.parameters():
        #     param.requires_grad = False


        # --- LSTM Layer ---
        # Input features to LSTM will be the output features from the CNN backbone
        self.lstm = nn.LSTM(input_size=num_cnn_features,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True, # Input shape: (batch, seq_len, features)
                            dropout=0.6 if lstm_layers > 1 else 0) # Add dropout if multiple layers

        # --- Classifier Head ---
        self.fc1 = nn.Linear(lstm_hidden_size, lstm_hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.65)
        self.fc2 = nn.Linear(lstm_hidden_size // 2, num_classes)
        # Sigmoid activation will be applied later (usually with BCEWithLogitsLoss for stability)
        # or explicitly here if using BCELoss

    def forward(self, x):
        # x shape: (batch_size, seq_len, C, H, W)

        batch_size, seq_len, C, H, W = x.shape

        # --- Pass through CNN ---
        # Reshape input for CNN: Treat sequence dimension as part of the batch
        cnn_input = x.view(batch_size * seq_len, C, H, W)
        # Get features from CNN
        cnn_output = self.cnn_backbone(cnn_input) # Shape: (batch*seq_len, num_cnn_features, 1, 1)
        # Remove spatial dimensions (squeeze)
        cnn_features = cnn_output.view(batch_size * seq_len, -1) # Shape: (batch*seq_len, num_cnn_features)
        # Reshape back into sequence: (batch_size, seq_len, num_cnn_features)
        lstm_input = cnn_features.view(batch_size, seq_len, -1)

        # --- Pass through LSTM ---
        # Initialize hidden and cell states (optional, defaults to zeros)
        # h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size).to(x.device)
        # c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size).to(x.device)
        # Get LSTM output (output features for each time step + final hidden/cell states)
        # lstm_output shape: (batch_size, seq_len, lstm_hidden_size)
        # hn shape: (num_layers, batch_size, lstm_hidden_size)
        # cn shape: (num_layers, batch_size, lstm_hidden_size)
        lstm_output, (hn, cn) = self.lstm(lstm_input) #, (h0, c0))

        # --- Classifier ---
        # We usually use the output of the *last* time step from the LSTM
        last_lstm_output = lstm_output[:, -1, :] # Shape: (batch_size, lstm_hidden_size)

        # Pass through classifier head
        out = self.fc1(last_lstm_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out) # Shape: (batch_size, num_classes)

        # Remove the last dimension if num_classes is 1
        if out.shape[1] == 1:
            out = out.squeeze(1) # Shape: (batch_size,)

        return out

# --- Example Usage (how to create the model) ---
# model = VideoClassifierCNN_LSTM(num_classes=1, lstm_hidden_size=512, lstm_layers=2)
# print(model)

# --- Move to GPU if available ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


# -------------------------------------------------------------
# --- Step 2: Instantiate Model, Loss, Optimizer ---
# -------------------------------------------------------------
print("\nInstantiating Model, Loss, and Optimizer...")

model = VideoClassifierCNN_LSTM(
    num_classes=NUM_CLASSES,
    lstm_hidden_size=LSTM_HIDDEN_SIZE,
    lstm_layers=LSTM_LAYERS,
    pretrained=PRETRAINED
).to(DEVICE)

# Loss Function - BCEWithLogitsLoss is recommended for binary classification
# as it combines Sigmoid and BCELoss for numerical stability.
criterion = nn.BCEWithLogitsLoss()

# Optimizer - Adam is a popular choice
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

# Learning Rate Scheduler (e.g., reduce LR if validation loss plateaus)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

print("Model, Loss, Optimizer instantiated.")
# You can print the model summary if needed:
# print(model)


import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm # Or regular tqdm

# --- AMP Imports ---
from torch.cuda.amp import GradScaler, autocast

# --- Dummy model for example ---
# Assume your VideoClassifierCNN_LSTM model is defined elsewhere
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16*3*224*224, 1) # Example structure
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1)).squeeze(1) # Output (B,)

# -------------------------------------------------------------
# --- Step 4 & 5: Implement Training and Evaluation Loops ---
# -------------------------------------------------------------

# --- Initialize GradScaler ONCE, typically before the main training loop ---
# It should only be enabled when using CUDA
# DEVICE should be defined globally (e.g., DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# We pass the scaler into the training function
# scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch_num, max_epochs):
    """Trains the model for one epoch using AMP."""
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Use tqdm for progress bar
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{max_epochs} [Train]", leave=False, unit="batch")

    for batch_idx, batch_data in enumerate(progress_bar):
        # Check for empty batch from collate_fn
        if batch_data is None or not batch_data[0].numel():
            print(f"Warning: Skipping empty training batch {batch_idx}")
            continue

        videos, labels = batch_data
        videos = videos.to(device) # (B, T, C, H, W)
        labels = labels.to(device) # (B,) - ensure criterion expects this shape

        # Zero gradients BEFORE the forward pass in this batch
        optimizer.zero_grad()

        # --- Automatic Mixed Precision Context ---
        # Runs the forward pass under autocast
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            outputs = model(videos) # Should be (B,) for BCEWithLogitsLoss
            loss = criterion(outputs, labels)

        # --- Scaled Backward Pass ---
        # Scales loss. Calls backward() on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()

        # Optional: Gradient Clipping (apply *before* scaler.step)
        # If you clip, unscale the gradients first
        # scaler.unscale_(optimizer) # Unscale gradients back to fp32
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # --- Scaler Step and Update ---
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called.
        # Otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        # --- Update Statistics ---
        running_loss += loss.item() * videos.size(0) # Loss per batch * batch size
        total_samples += labels.size(0)

        # Calculate accuracy (outside autocast context)
        with torch.no_grad():
            preds = torch.sigmoid(outputs) > 0.5 # Shape: (B, 1) if output was (B,1), or (B,) if output was (B,)
            # Ensure labels are boolean and match pred shape for comparison
            correct_predictions += (preds == labels.bool().view_as(preds)).sum().item() # Use view_as for safety
        # Update progress bar postfix
        current_loss = loss.item()
        current_acc = correct_predictions / total_samples if total_samples > 0 else 0
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    # End of Epoch Calculations
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    progress_bar.close() # Close the tqdm bar for this epoch

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, epoch_num, max_epochs):
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_outputs = []
    all_labels = []

    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{max_epochs} [Val.] ", leave=False, unit="batch")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None or not batch_data[0].numel():
                print(f"Warning: Skipping empty validation batch {batch_idx}")
                continue

            videos, labels = batch_data
            videos = videos.to(device)
            labels = labels.to(device) # Shape (B,)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                outputs = model(videos) # Shape (B,)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * videos.size(0)
            total_samples += labels.size(0)

            # --- CORRECTED Accuracy Calculation ---
            preds = torch.sigmoid(outputs) > 0.5 # Get probabilities and threshold -> Shape (B,) boolean
            correct_predictions += (preds == labels.bool()).sum().item() # Direct comparison if labels are (B,)

            all_outputs.append(torch.sigmoid(outputs).cpu())
            all_labels.append(labels.cpu())

            # Display *batch* accuracy (optional, but useful for debugging)
            batch_acc = (preds == labels.bool()).float().mean().item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", batch_acc=f"{batch_acc:.4f}") # Show batch acc

    # --- CORRECTED Epoch Accuracy ---
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0
    progress_bar.close()

    # --- Optional: Calculate AUC here ---
    # try:
    #     all_outputs_cat = torch.cat(all_outputs).numpy()
    #     all_labels_cat = torch.cat(all_labels).numpy()
    #     val_auc = roc_auc_score(all_labels_cat, all_outputs_cat)
    #     print(f"  Val AUC: {val_auc:.4f}")
    # except ValueError:
    #     print("  Val AUC: Could not calculate (likely only one class present in batch)")
    #     val_auc = -1.0 # Or some indicator

    return epoch_loss, epoch_acc #, val_auc


# Cell [13]: Main Training Loop (Corrected Checkpoint Paths)

# -------------------------------------------------------------
# --- Step 6: Run the Training Process ---
# -------------------------------------------------------------

# --- Initialize variables BEFORE checking for checkpoint ---
start_epoch = 0
best_val_loss = float('inf')
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# Initialize scaler (using updated torch.amp syntax)
scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

# --- Early Stopping Parameters ---

patience_counter = 0

# --- Check for and load existing checkpoint ---
if os.path.exists(CHECKPOINT_PATH): # <-- Use CHECKPOINT_PATH variable
    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint and 'epoch' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if 'scheduler_state_dict' in checkpoint and 'scheduler' in locals() and scheduler is not None: # Check if scheduler exists
                try: # Add extra try-except for scheduler loading
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Loaded scheduler state.")
                except Exception as e_sched:
                    print(f"Warning: Could not load scheduler state: {e_sched}")
            # Load patience counter if saved, otherwise reset
            patience_counter = checkpoint.get('patience_counter', 0)
            print(f"Resuming training from Epoch {start_epoch}")
            print(f"Previous best validation loss: {best_val_loss:.4f}")
            print(f"Resuming patience counter: {patience_counter}")
        else:
            print("Checkpoint file seems incomplete or only model state_dict. Loading weights only.")
            model.load_state_dict(checkpoint)
            print("Starting training from epoch 0, optimizer/scheduler state not resumed.")
            start_epoch = 0
            best_val_loss = float('inf')
            patience_counter = 0 # Reset patience

    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting training from scratch.")
        start_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0
else:
    print("No checkpoint found. Starting training from scratch.")


print("\n--- Starting Training ---")

# MAX_EPOCHS should be defined in your config cell
for epoch in range(start_epoch, MAX_EPOCHS):
    # --- Training Phase ---
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, epoch, MAX_EPOCHS) # Pass scaler

    # --- Validation Phase ---
    if 'val_loader' in locals() and val_loader is not None: # Check if val_loader was created
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE, epoch, MAX_EPOCHS)
    else:
        print("Warning: No validation loader available. Skipping validation and early stopping.")
        val_loss, val_acc = -1.0, -1.0

    # --- Log Epoch Results ---
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    if 'val_loader' in locals() and val_loader is not None:
        print(f"  Val. Loss:  {val_loss:.4f} | Val. Acc:  {val_acc:.4f}") # Ensure acc calc is correct

    # --- Store history ---
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    if 'val_loader' in locals() and val_loader is not None:
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    else: # Append placeholders if no validation
        history['val_loss'].append(None)
        history['val_acc'].append(None)


    # --- Learning Rate Scheduling ---
    if 'val_loader' in locals() and val_loader is not None and 'scheduler' in locals() and scheduler is not None:
       scheduler.step(val_loss) # Step scheduler based on validation loss

    # --- Early Stopping Logic & Checkpoint Saving ---
    if 'val_loader' in locals() and val_loader is not None:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # *** CORRECTED PATH ***
            model_save_path = CHECKPOINT_PATH # Use the full path variable
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() and scheduler is not None else None,
                'patience_counter': patience_counter # Optional: Save patience counter state
            }
            print(f"  * Validation loss improved to {best_val_loss:.4f}. Saving checkpoint to {model_save_path}. Patience reset.")
            # Add a disk space check before saving (optional but helpful)
            # !df -h /kaggle/working/
            print(f"DEBUG: Attempting to save checkpoint to {model_save_path} for epoch {epoch+1}") # Add debug print
            try:
                torch.save(checkpoint, model_save_path)
                print(f"DEBUG: Save successful for epoch {epoch+1}") # Confirm success
            except Exception as e:
                print(f"DEBUG: Error saving checkpoint: {e}") # Catch specific save errors
        else:
            # Validation loss did not improve
            patience_counter += 1
            print(f"  * Validation loss ({val_loss:.4f}) did not improve from best ({best_val_loss:.4f}). Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"--- Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement. ---")
                break # Exit the training loop
    else:
        # --- Optional Periodic Saving (if no validation) ---
        if (epoch + 1) % 5 == 0: # Example: Save every 5 epochs
             # *** CORRECTED PATH ***
             periodic_save_path = os.path.join(WORKING_DIR, f"model_epoch_{epoch+1}.pth")
             checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'scheduler_state_dict': scheduler.state_dict() if 'scheduler' in locals() and scheduler is not None else None,
                'patience_counter': patience_counter # Still save current patience state
             }
             print(f"Saving periodic checkpoint to {periodic_save_path}")
             try:
                 torch.save(checkpoint, periodic_save_path)
             except Exception as e:
                 print(f"ERROR saving periodic checkpoint: {e}")


print("\n--- Training Finished ---")
if 'val_loader' in locals() and val_loader is not None:
    print(f"Best Validation Loss achieved: {best_val_loss:.4f}") # This holds the best loss found
    # Determine the actual epoch number where training stopped
    final_epoch_completed = epoch + 1 # Add 1 because epoch is 0-indexed
    print(f"Training stopped after completing epoch: {final_epoch_completed}")
else:
    print("Training finished (fixed epochs or periodic saves).")

# --- IMPORTANT: Load BEST model before inference ---
print("\nLoading best model weights for evaluation/submission...")
try:
    # Ensure CHECKPOINT_PATH points to the file saved for the best validation loss
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode
    print("Successfully loaded best model weights.")
except FileNotFoundError:
    print(f"ERROR: Could not find best checkpoint at {CHECKPOINT_PATH} after training.")
    # Handle error - maybe use the model state from the very last epoch if needed, but it's not the best
except Exception as e:
     print(f"Error loading best model weights after training: {e}")


# --- Step 2 & 3: Instantiate Model and Load Weights ---
print("Loading trained model...")
MODEL_PATH='best_model_cnn_lstm.pth' # Path to your saved CHECKPOINT file
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Instantiate the model architecture EXACTLY as used during training
model = VideoClassifierCNN_LSTM(
    num_classes=1,             # Ensure this matches training
    lstm_hidden_size=512,      # Ensure this matches training
    lstm_layers=2,             # Ensure this matches training
    pretrained=False           # Usually False when loading fine-tuned weights
                               # Set based on how you initialized the model
                               # *before* loading the checkpoint during training/saving.
                               # If you loaded pretrained weights *then* trained and saved,
                               # you still initialize with pretrained=False here because
                               # the state_dict contains the fine-tuned weights.
).to(DEVICE)

try:
    # 2. Load the entire checkpoint dictionary
    # Use weights_only=False (default) because the checkpoint contains more than just weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # 3. Extract the model's state dictionary from the checkpoint
    # Check if the expected key exists for robustness
    if 'model_state_dict' in checkpoint:
        model_weights = checkpoint['model_state_dict']
    else:
        # Handle case where the file might *only* contain the state dict
        # (e.g., from an older saving method or different script)
        print("Warning: Checkpoint dictionary does not contain 'model_state_dict'. Assuming the file IS the state dict.")
        model_weights = checkpoint # Assume the loaded object IS the state dict

    # 4. Load the extracted weights into the model instance
    model.load_state_dict(model_weights)

    print("Model weights loaded successfully.")

except FileNotFoundError:
    print(f"ERROR: Model checkpoint not found at {MODEL_PATH}. Cannot proceed.")
    # Decide how to handle - exit(), raise error, return None, etc.
    exit() # Or raise FileNotFoundError("Checkpoint not found")
except Exception as e:
    print(f"Error loading model weights: {e}")
    # Log the full traceback for debugging complex errors
    import traceback
    traceback.print_exc()
    # Decide how to handle - exit(), raise error, return None, etc.
    exit() # Or raise e

# 5. Set the model to evaluation mode (important!)
model.eval()
print("Model set to evaluation mode.")

# --- Now you can proceed with inference using the loaded model ---


# --- Step 7: Inference on Test Set ---
# -------------------------------------------------------------
print("\n--- Starting Inference on Test Set ---")

# --- Load Best Model Weights (Use your corrected loading logic) ---
print("Loading best trained model weights...")

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE) # Load the whole checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Best model weights loaded successfully from checkpoint.")
    else:
        # Fallback if only state dict was saved (less likely with your saving code)
        model.load_state_dict(checkpoint)
        print("Checkpoint contained only state_dict, loaded successfully.")
    # Load best val loss achieved (optional, for reference)
    loaded_best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"(Best validation loss during training was: {loaded_best_val_loss:.4f})")

except FileNotFoundError:
    print(f"ERROR: Best model checkpoint not found at {CHECKPOINT_PATH}. Using model from last epoch (if available) or initial state.")
    # Handle appropriately - maybe exit or raise error if inference requires the best model
except Exception as e:
    print(f"Error loading best model weights: {e}")
    # Handle appropriately

model.eval() # Set model to evaluation mode

# --- Create Test DataLoader HERE (after training) ---
print("\nCreating Test Dataset and DataLoader...")
try:
    test_dataset = DashcamDataset(csv_file=TEST_CSV_PATH,
                                  video_dir=TEST_VIDEO_DIR,
                                  seq_len=SEQ_LEN,
                                  transform=test_transform_final, # Use test transforms
                                  mode='test',
                                  apply_hflip=False) # No flip for test

    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE * 2, # Inference batch size
                             shuffle=False,
                             num_workers=NUM_WORKERS,
                             pin_memory=True,
                             prefetch_factor=PREFETCH_FACTOR,
                             collate_fn=collate_fn)
    print(f"Test dataset size: {len(test_dataset)}")
except FileNotFoundError:
    print(f"ERROR: Test CSV not found at {TEST_CSV_PATH}. Cannot run inference.")
    test_loader = None
except Exception as e:
    print(f"Error creating test dataset/loader: {e}")
    test_loader = None

# --- Run Inference (only if test_loader exists) ---
predictions = {}
if test_loader:
    with torch.no_grad():
        test_progress_bar = tqdm(test_loader, desc="Inference", leave=False)
        for inputs_test, ids_test in test_progress_bar:
            # Check for empty batch from collate_fn
            if inputs_test is None or not inputs_test.numel():
                print(f"Warning: Skipping empty test batch.")
                continue

            inputs_test = inputs_test.to(DEVICE)

            with autocast(enabled=(DEVICE.type == 'cuda')):
                outputs_test = model(inputs_test)

            probs = torch.sigmoid(outputs_test).cpu().numpy().flatten()

            for video_id, prob in zip(ids_test, probs):
                predictions[str(video_id)] = prob
    print(f"Inference complete. Generated predictions for {len(predictions)} videos.")
else:
    print("Skipping inference as test loader was not created.")


# --- Step 8: Create Submission File ---
# -------------------------------------------------------------
print("\nCreating submission file...")
try:
    # Load sample submission to get all required test IDs in the correct order
    sample_sub = pd.read_csv(os.path.join(BASE_DATA_PATH, 'sample_submission.csv'))
    submission_df = pd.DataFrame({'id': sample_sub['id']}) # Use IDs from sample submission

    # Map predictions - handle cases where a test video might have failed processing
    submission_df['score'] = submission_df['id'].astype(str).map(predictions).fillna(0.5) # Fill missing with 0.5? Or maybe 0? Check competition baseline.

    # Ensure scores are within [0, 1]
    submission_df['score'] = submission_df['score'].clip(0.0, 1.0)

    # Save submission file
    submission_df.to_csv(SUBMISSION_CSV, index=False)
    print(f"Submission file saved to {SUBMISSION_CSV}")
    print(submission_df.head())

except FileNotFoundError:
     print("ERROR: sample_submission.csv not found. Cannot create submission file in correct format.")
except Exception as e:
     print(f"Error creating submission file: {e}")


#torch.save({'test': 1}, '/kaggle/working/test_save.pth')




