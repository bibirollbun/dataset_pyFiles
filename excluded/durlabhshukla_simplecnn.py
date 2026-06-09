import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import librosa
import numpy as np
import pandas as pd

import gc
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Tuple, List
import traceback  # Import traceback module


# Define file paths 
test_data = "/kaggle/input/birdclef-2025/test_soundscapes"
submission = "/kaggle/input/birdclef-2025/sample_submission.csv"
train_csv = "/kaggle/input/birdclef-2025/train.csv"
taxonomy_csv = "/kaggle/input/birdclef-2025/taxonomy.csv"

transform: Optional[Callable] = None  # Type hint for transform
audio_transform: Optional[Callable] = None # Type hint for audio_transform

@dataclasses.dataclass
class AudioParam:
    SR: int = 32_000  # Sample rate
    NFFT: int = 2048  # Number of FFT points
    NMEL: int = 128   # Number of Mel bands
    FMAX: int = 16_000 # Maximum frequency
    FMIN: int = 20   # Minimum frequency
    HOP_LENGTH: int = NFFT // 4  # Hop length

audio_param = AudioParam()

# Load submission CSV to get class names
try:
    sub_csv = pd.read_csv(submission)
    idx2cls = sub_csv.columns.drop("row_id").tolist()  # List of bird species (class names)
    cls2idx = {c: i for i, c in enumerate(idx2cls)} # Class name to index mapping
except FileNotFoundError as e:
    print(f"Error: sample_submission.csv not found! {e}")
    idx2cls = [] # Provide a default for testing, but the code will likely fail
    cls2idx = {}


DEBUG = True # Enable Debugging
file_names = [os.path.join(test_data, fp) for fp in os.listdir(test_data) if fp.endswith(".ogg")]

# Use a single file for debugging.  This makes the matrix dimension calculations easier.
if len(file_names) == 0:
    file_names = [
        "/kaggle/input/birdclef-2025/train_soundscapes/H02_20230420_074000.ogg",
    ]
    DEBUG = True


#  a simpler, randomly initialized CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Calculate the input size to the linear layer dynamically
        self._to_linear = None  # Placeholder, will be calculated during the first forward pass
        self.fc1 = nn.Linear(1, num_classes)  # Placeholder Linear layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)

            # Dynamically determine the input size of the linear layer
            if self._to_linear is None:
                self._to_linear = x.shape[1]
                if self._to_linear == 0:
                   print("Error: self._to_linear is zero!")
                   #Handle this better - e.g., skip or set a min size
                   return torch.zeros((1, len(idx2cls)))  # or a zero tensor of the right size
                self.fc1 = nn.Linear(self._to_linear, len(idx2cls))  # Update the linear layer
            x = self.fc1(x)
            return x
        except Exception as e:
            print(f"Error in SimpleCNN.forward: {e}")
            return torch.zeros((1, len(idx2cls)))


# Instantiate the SimpleCNN model.
model = SimpleCNN(num_classes=len(idx2cls))
model.eval() # Set the model to evaluation mode.

def pipeline(x: np.ndarray) -> np.ndarray:
    """
    Converts audio data to MFCCs (Mel-frequency cepstral coefficients).
    """
    try:
        mfccs = librosa.feature.mfcc(
            y=x,
            sr=audio_param.SR,
            n_mfcc=audio_param.NMEL,  # Using the same dimension as previously defined for mels
            n_fft=audio_param.NFFT,
            hop_length=audio_param.HOP_LENGTH,
            fmin=audio_param.FMIN,
            fmax=audio_param.FMAX
        )
        # Normalize the MFCCs
        mfccs_normalized = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-6)
        if np.isnan(mfccs_normalized).any():
            print("Warning: NaN values detected in mfccs!")
            mfccs_normalized = np.nan_to_num(mfccs_normalized)  # Replace with 0
        return mfccs_normalized[None, :, :]  # Add a channel dimension (1, height, width)
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return np.zeros((1, audio_param.NMEL, 1))  # return a zero array


def generate_submission(file_names):
    """
    Generate predictions for the test files and create a submission CSV
    formatted according to Kaggle's requirements.
    
    Args:
        file_names (list): List of file paths to predict on
    """
    row_id = []
    matrix = []

    # Function to process a single file for thread pooling
    def process_file(fp):
        try:
            out, rid = predict(fp)
            if len(rid) > 0:  # Only return if there are valid results
                return out, rid
            else:
                print(f"Warning: No predictions generated for file: {fp}")
                return None, None
        except Exception as e:
            print(f"Failed to run predict for file {fp}: {e}")
            return None, None

    # Using a ThreadPoolExecutor to parallelize the predictions
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, fp) for fp in file_names]
        
        # Process results as they complete
        for fp_idx, future in enumerate(as_completed(futures)):
            try:
                out, rid = future.result()
                if rid is not None and len(rid) > 0:
                    row_id.extend(rid)  # Extend to add all row IDs from this file
                    matrix.extend(out)  # Extend to add all predictions from this file
                gc.collect()  # Collect garbage after processing each file's results
                print(f"Finished {fp_idx+1}/{len(file_names)}")  # Track progress
            except Exception as e:
                print(f"Error processing result: {e}")

    try:
        if len(row_id) > 0 and len(matrix) > 0:
            # CRITICAL FIX: Correct the row_id format to match Kaggle's requirements
            # Extract the test file basename without extension and use the timestamp
            corrected_row_ids = []
            for r in row_id:
                # Original format: "filename_start_end" 
                # Required format: "filename_end" (just the end timestamp)
                parts = r.split('_')
                filename = parts[0]
                end_time = parts[2]  # Get the end timestamp
                corrected_row_id = f"{filename}_{end_time}"
                corrected_row_ids.append(corrected_row_id)
            
            # Convert predictions to numpy array
            matrix_np = np.array(matrix)
            
            # Ensure we have exactly the expected classes from the competition
            # First, create a dataframe with just row_ids
            submission_df = pd.DataFrame({"row_id": corrected_row_ids})
            
            # Add each class column correctly
            for i, cls_name in enumerate(idx2cls):
                submission_df[cls_name] = matrix_np[:, i]
            
            # Verify columns match expected format
            expected_columns = ["row_id"] + idx2cls
            missing_cols = set(expected_columns) - set(submission_df.columns)
            if missing_cols:
                print(f"Warning: Missing columns in submission: {missing_cols}")
                # Add missing columns with zeros
                for col in missing_cols:
                    if col != "row_id":  # Skip row_id as it should already be there
                        submission_df[col] = 0.0
            
            # Ensure columns are in the correct order
            submission_df = submission_df[expected_columns]
            
            # Save to CSV
            submission_df.to_csv('submission.csv', index=False)
            print("Submission file created successfully!")
            print(f"Shape: {submission_df.shape}")
            print(submission_df.head())
            
            # Validate submission format
            print("\nValidating submission format...")
            print(f"Number of rows: {len(submission_df)}")
            print(f"Number of columns: {len(submission_df.columns)}")
            print(f"Missing values: {submission_df.isnull().sum().sum()}")
            
        else:
            print("Error: No valid predictions were generated.")
    except Exception as e:
        print(f"Error creating submission file: {e}")
        traceback.print_exc()  # Print full traceback for debugging

    print("Finished!")
    gc.collect()

# Updated predict function to fix row_ID format
@torch.no_grad()
def predict(fp: str) -> tuple[np.ndarray, list[str]]:
    """
    Predicts bird calls in a given audio file using MFCC features.
    Args:
        fp (str): File path of the audio file.
    Returns:
        Tuple[np.ndarray, List[str]]: Tuple containing the model output and the list of row IDs.
    """
    try:
        # Fix the syntax error in loading audio
        x, sr = librosa.load(fp, sr=audio_param.SR)  # Load the audio file.
        if x.size == 0:
            print(f"Warning: Audio file {fp} is empty!")
            return np.array([]), [] # return empty arrays
    except Exception as e:
        print(f"Error loading file {fp}: {e}")
        return np.array([]), []
        
    # Number of 5-second segments
    num_segments = int(np.floor(len(x) / audio_param.SR / 5))
    all_outs = []
    all_row_ids = []
    
    for i in range(num_segments):
        start = i * audio_param.SR * 5
        end = (i + 1) * audio_param.SR * 5
        segment = x[start:end]
        
        # Apply audio transforms if defined
        if 'audio_transform' in globals() and audio_transform is not None:
            try:
                segment = audio_transform(sample=segment, sample_rate=audio_param.SR) # Apply audio transform
            except Exception as e:
                print(f"Audio Transform Failed {e}")
                
        try:
            # This will now use the updated pipeline that extracts MFCCs instead of mel spectrograms
            segment_features = pipeline(segment)  # Convert waveform to MFCCs
        except Exception as e:
            print(f"Pipeline failed {e}")
            continue
            
        # Apply image transforms if defined
        if 'transform' in globals() and transform is not None:
            try:
                segment_features = transform(image=segment_features)["image"] # Apply image transform
            except Exception as e:
                print(f"Transform failed {e}")
                continue
                
        try:
            # Convert to tensor and add batch dimension
            segment_tensor = torch.from_numpy(segment_features).float().unsqueeze(0)
            
            # Get the model output
            out = model(segment_tensor).sigmoid().detach().cpu().numpy()
            all_outs.append(out[0])
            
            # Extract the base filename and create row ID
            # KAGGLE FORMAT: Use just the filename and end time
            fp_name = os.path.basename(fp).split(".")[0]
            end_time = (i + 1) * 5  # End time in seconds
            row_id = f"{fp_name}_{end_time}"  # Format matches Kaggle requirements: "filename_endtime"
            all_row_ids.append(row_id)
        except Exception as e:
            print(f"Error during processing of segment {i} in {fp}: {e}\n{traceback.format_exc()}")  # Print trace
            
    return np.array(all_outs), all_row_ids  # return all values


# Import the necessary modules
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

row_id = []
matrix = []

# Function to process a single file for thread pooling
def process_file(fp):
    try:
        out, rid = predict(fp)
        if len(rid) > 0:  # Only return if there are valid results
            return out, rid
        else:
            print(f"Warning: No predictions generated for file: {fp}")
            return None, None
    except Exception as e:
        print(f"Failed to run predict for file {fp}: {e}")
        return None, None

# Using a ThreadPoolExecutor to parallelize the predictions
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_file, fp) for fp in file_names]
    
    # Process results as they complete
    for fp_idx, future in enumerate(as_completed(futures)):
        try:
            out, rid = future.result()
            if rid is not None and len(rid) > 0:
                row_id.extend(rid)  # Extend to add all row IDs from this file
                matrix.extend(out)  # Extend to add all predictions from this file
            gc.collect()  # Collect garbage after processing each file's results
            print(f"Finished {fp_idx+1}/{len(file_names)}")  # Track progress
        except Exception as e:
            print(f"Error processing result: {e}")

try:
    if len(row_id) > 0 and len(matrix) > 0:
        # Convert row_id to string type to ensure it's handled correctly
        row_id = [str(r) for r in row_id]
        
        # Create DataFrame directly from dictionary
        sub = pd.DataFrame({
            "row_id": row_id,
            **{cls_name: [pred[i] for pred in matrix] for i, cls_name in enumerate(idx2cls)}
        })
        
        # Save to CSV
        sub.to_csv('submission.csv', index=False)
        print("Submission file created successfully!")
        print(sub.head())
    else:
        print("Error: No valid predictions were generated.")
except Exception as e:
    print(f"Error creating submission file: {e}")
    traceback.print_exc()  # Print full traceback for debugging

print("Finished!")  # If you see this, then great!
gc.collect()







