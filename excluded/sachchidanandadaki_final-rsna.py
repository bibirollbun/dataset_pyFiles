# --- 1. Imports and Setup ---
import os
import gc
import warnings
import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
from pathlib import Path
from scipy import ndimage
from typing import List, Tuple

warnings.filterwarnings('ignore')

# --- 2. Configuration ---
class CFG:
    # !!! IMPORTANT: EDIT THIS PATH !!!
    # Point this to the .pth file from your training notebook's output
    MODEL_PATH = "/kaggle/input/final-train/checkpoints/best_model_fold_0.pth"
    
    # Device and model parameters must match your training setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"
    TARGET_SHAPE = (32, 384, 384) # (Depth/Channels, Height, Width)
    
    # Target columns must be in the same order as in training
    TARGET_COLUMNS = [
        'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
        'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
        'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
        'Anterior Communicating Artery', 'Left Anterior Cerebral Artery',
        'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
        'Right Posterior Communicating Artery', 'Basilar Tip',
        'Other Posterior Circulation', 'Aneurysm Present'
    ]

print(f"Using device: {CFG.DEVICE}")

# --- 3. Model Definition ---
# This function must be identical to the one in your training script
def build_model():
    model = timm.create_model(
        CFG.MODEL_NAME,
        pretrained=False, # We are loading our own trained weights
        num_classes=len(CFG.TARGET_COLUMNS),
        in_chans=CFG.TARGET_SHAPE[0]
    )
    return model

# Load the trained model weights
model = build_model().to(CFG.DEVICE)
model.load_state_dict(torch.load(CFG.MODEL_PATH, map_location=CFG.DEVICE))
model.eval()
print("Model loaded successfully.")

# --- 4. Preprocessing Pipeline ---
# This class MUST be an exact copy of the one from your successful preprocessing notebook
class DICOMPreprocessor:
    def __init__(self, target_shape: Tuple[int, int, int] = CFG.TARGET_SHAPE):
        self.target_depth, self.target_height, self.target_width = target_shape

    def load_dicom_series(self, series_path: str) -> List[pydicom.Dataset]:
        dicom_files = list(Path(series_path).rglob('*.dcm'))
        if not dicom_files: raise ValueError(f"No DICOM files in {series_path}")
        datasets = [pydicom.dcmread(fp, force=True) for fp in dicom_files]
        return datasets

    def sort_slices(self, datasets: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
        slice_info = [{'dataset': ds, 'z_pos': float(ds.ImagePositionPatient[2]) if hasattr(ds, 'ImagePositionPatient') else int(ds.InstanceNumber)} for i, ds in enumerate(datasets)]
        return [info['dataset'] for info in sorted(slice_info, key=lambda x: x['z_pos'])]

    def apply_windowing(self, img: np.ndarray) -> np.ndarray:
        window_min, window_max = 0, 500
        img = np.clip(img, window_min, window_max)
        return (img - window_min) / (window_max - window_min)

    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        img = ds.pixel_array.astype(np.float32)
        slope = getattr(ds, 'RescaleSlope', 1.0)
        intercept = getattr(ds, 'RescaleIntercept', 0.0)
        img = img * float(slope) + float(intercept)
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img if img.size > 0 else None

    def resize_depth(self, volume: np.ndarray) -> np.ndarray:
        if volume.shape[0] == self.target_depth: return volume
        zoom_factor = self.target_depth / volume.shape[0]
        return ndimage.zoom(volume, (zoom_factor, 1, 1), order=1, mode='nearest')

    def process_series(self, series_path: str) -> np.ndarray:
        datasets = self.load_dicom_series(series_path)
        if not datasets: raise ValueError("Could not read DICOMs.")
        sorted_datasets = self.sort_slices(datasets)
        processed_slices = []
        for ds in sorted_datasets:
            img = self.extract_pixel_array(ds)
            if img is None: continue
            windowed_img = self.apply_windowing(img)
            resized_img = cv2.resize(windowed_img, (self.target_width, self.target_height))
            processed_slices.append(resized_img)
        if not processed_slices: raise ValueError("No valid slices processed.")
        volume = np.stack(processed_slices, axis=0)
        final_volume = self.resize_depth(volume)
        return (final_volume * 255).astype(np.uint8)

def preprocess_for_inference(series_path: str) -> torch.Tensor:
    """Orchestrates the full preprocessing pipeline for a single series."""
    preprocessor = DICOMPreprocessor()
    
    # 1. DICOM to NumPy array (uint8)
    volume_np = preprocessor.process_series(series_path)
    
    # 2. NumPy to PyTorch Tensor, scale to [0, 1]
    volume_tensor = torch.from_numpy(volume_np).float() / 255.0
    
    # 3. Apply the same normalization as in training (mean=0.5, std=0.5)
    mean = torch.tensor([0.5] * CFG.TARGET_SHAPE[0]).view(CFG.TARGET_SHAPE[0], 1, 1)
    std = torch.tensor([0.5] * CFG.TARGET_SHAPE[0]).view(CFG.TARGET_SHAPE[0], 1, 1)
    volume_tensor = (volume_tensor - mean) / std
    
    # 4. Add batch dimension for the model
    return volume_tensor.unsqueeze(0)

# --- 5. Prediction Function (Kaggle API) ---
def predict(test_series_path: str) -> pd.DataFrame:
    """Processes a DICOM series, runs inference with TTA, and returns results."""
    print(f"--> Processing series: {os.path.basename(test_series_path)}")
    try:
        # Preprocess the original test scan
        scan_tensor_original = preprocess_for_inference(test_series_path).to(CFG.DEVICE)
        
        # TTA: Create a horizontally flipped version (W-axis is the last one)
        scan_tensor_flipped = torch.flip(scan_tensor_original, dims=[-1])

        with torch.no_grad():
            # Get predictions for both versions
            logits_original = model(scan_tensor_original)
            probs_original = torch.sigmoid(logits_original)
            
            logits_flipped = model(scan_tensor_flipped)
            probs_flipped = torch.sigmoid(logits_flipped)

        # TTA: "Un-flip" the predictions from the flipped scan
        probs_flipped_corrected = probs_flipped.clone()
        swap_map = {0: 1, 2: 3, 4: 5, 7: 8, 9: 10} # Indices of Left/Right pairs
        for left_idx, right_idx in swap_map.items():
            probs_flipped_corrected[0, left_idx] = probs_flipped[0, right_idx]
            probs_flipped_corrected[0, right_idx] = probs_flipped[0, left_idx]
            
        # TTA: Average the two sets of probabilities for a more robust prediction
        final_probabilities = (probs_original + probs_flipped_corrected) / 2.0
        
        # Format the final results
        results = {'SeriesInstanceUID': os.path.basename(test_series_path)}
        for i, col in enumerate(CFG.TARGET_COLUMNS):
            results[col] = float(final_probabilities[0, i].item())
            
        return pd.DataFrame([results])

    except Exception as e:
        # Fallback: If anything fails, return a DataFrame of zeros
        print(f"!!! FAILED processing {test_series_path}: {e}")
        results = {'SeriesInstanceUID': os.path.basename(test_series_path)}
        for col in CFG.TARGET_COLUMNS:
            results[col] = 0.0
        return pd.DataFrame([results])
    finally:
        gc.collect()

print("Inference pipeline with TTA is ready.")

# --- 6. Kaggle Submission Execution ---
# This part of the code is provided by Kaggle to run your `predict` function
from kaggle_evaluation import rsna_inference_server

inference_server = rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # This runs in the interactive notebook, using a local test set
    inference_server.run_local_gateway()
    try:
        display(pd.read_parquet('/kaggle/working/submission.parquet'))
    except FileNotFoundError:
        print("Local run complete. No submission.parquet generated.")

print("Submission script finished.")

