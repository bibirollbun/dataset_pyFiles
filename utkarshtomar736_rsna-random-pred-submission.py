import os, glob
import pandas as pd
import numpy as np
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt


DATA_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection"
print("Data directory contents:", os.listdir(DATA_DIR))


train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
print(f"Train shape: {train_df.shape}")
train_df.head(2)


locs_df = pd.read_csv(f"{DATA_DIR}/train_localizers.csv") 
print(f"Localizers shape: {locs_df.shape}")
locs_df.head(2)


location_cols = [
    "Left Infraclinoid Internal Carotid Artery", "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery", "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery", "Right Middle Cerebral Artery", "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery", "Right Anterior Cerebral Artery", "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery", "Basilar Tip", "Other Posterior Circulation"
]

print(f"Total location columns: {len(location_cols)}")


series_root = f"{DATA_DIR}/series"
available_series = set(os.listdir(series_root)) if os.path.exists(series_root) else set()
print(f"Available series folders: {len(available_series)}")


train_df["has_dicom"] = train_df["SeriesInstanceUID"].isin(available_series)
coverage = train_df["has_dicom"].mean()
print(f"Coverage: {coverage:.1%} of train series have DICOM files")


from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


gkf = GroupKFold(n_splits=5)
groups = train_df["SeriesInstanceUID"]


splits = list(gkf.split(train_df, train_df["Aneurysm Present"], groups))
print(f"Created {len(splits)} folds")


for i, (train_idx, val_idx) in enumerate(splits[:2]):  # Check first 2 folds
    val_pos_rate = train_df.iloc[val_idx]["Aneurysm Present"].mean()
    print(f"Fold {i}: val positive rate = {val_pos_rate:.3f}")


train_idx, val_idx = splits[0]  # Pick first fold
val_df = train_df.iloc[val_idx].copy()
n_val = len(val_df)
print(f"Using fold 0: {n_val} validation samples")


np.random.seed(42)
random_preds = np.random.uniform(0, 1, size=(n_val, 14))  # 14 = 1 presence + 13 locations
print(f"Random predictions shape: {random_preds.shape}")


presence_col = "Aneurysm Present"
true_labels = val_df[location_cols + [presence_col]].values
print(f"True labels shape: {true_labels.shape}")


aucs = []
for i in range(14):
    if len(np.unique(true_labels[:, i])) > 1:  # Need both classes
        auc = roc_auc_score(true_labels[:, i], random_preds[:, i])
        aucs.append(auc)
print(f"Random baseline AUCs: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")


test_series = val_df[val_df["has_dicom"]]["SeriesInstanceUID"].iloc[0]
print(f"Testing with series: {test_series}")


series_path = f"{series_root}/{test_series}"
print(f"Series path: {series_path}")
print(f"Path exists: {os.path.exists(series_path)}")


if os.path.exists(series_path):
    contents = os.listdir(series_path)
    print(f"Contents: {contents[:5]}")  # First 5 items
else:
    print("Series directory not found")


dcm_pattern = f"{series_root}/{test_series}/*.dcm"
dcm_files = sorted(glob.glob(dcm_pattern))
print(f"Found {len(dcm_files)} DICOM files")


if dcm_files:
    sample_dcm = pydicom.dcmread(dcm_files[0])  # Use first file: dcm_files[0]
    print(f"Modality: {sample_dcm.Modality}")
    print(f"Image shape: {sample_dcm.pixel_array.shape}")


slices = []
for dcm_path in dcm_files[:5]:  # Just first 5 for testing
    ds = pydicom.dcmread(dcm_path)
    slices.append((ds.InstanceNumber, ds.pixel_array, ds))


slices.sort(key=lambda x: x[0])  # Sort by InstanceNumber (index 0)
arrays = [s[1] for s in slices]  # Extract pixel arrays (index 1)
volume_sample = np.stack(arrays, axis=0)  # Shape: (slices, H, W)
print(f"Sample volume shape: {volume_sample.shape}")


if len(slices) > 0:
    ds = slices[0][2]  # Get DICOM metadata from first slice
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        volume_sample = volume_sample * ds.RescaleSlope + ds.RescaleIntercept
        print("Applied rescale transformation")
else:
    print("No slices available for rescale")


plt.figure(figsize=(6, 4))
mid_slice = volume_sample[len(volume_sample)//2]
plt.imshow(mid_slice, cmap='gray')
plt.title(f"Middle slice from {test_series}")
plt.axis('off')
plt.show()


# Define exact column order for submission (critical for API)
SUBMISSION_COLS = [
    'SeriesInstanceUID', 'Aneurysm Present',
    'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
    'Anterior Communicating Artery', 'Left Anterior Cerebral Artery', 'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery', 'Right Posterior Communicating Artery',
    'Basilar Tip', 'Other Posterior Circulation'
]
print(f"Submission has {len(SUBMISSION_COLS)} columns")


import numpy as np

def generate_random_predictions(series_id, seed=42):
    """Generate random predictions for one series"""
    np.random.seed(seed + hash(series_id) % 1000)  # Series-specific seed
    preds = np.random.uniform(0.1, 0.9, size=14)  # 14 targets
    return [series_id] + list(preds)


# Test with dummy series ID
test_series = "dummy_series_123"
test_preds = generate_random_predictions(test_series)
print(f"Generated {len(test_preds)} values for series {test_series}")
print(f"Sample predictions: {test_preds[1:4]}")  # First 3 target probabilities


import os
import numpy as np
import pandas as pd

# Import the RSNA inference server
import kaggle_evaluation.rsna_inference_server

# Define the 14 target columns in exact order
TARGET_COLS = [
    'Aneurysm Present',
    'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
    'Anterior Communicating Artery', 'Left Anterior Cerebral Artery', 'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery', 'Right Posterior Communicating Artery',
    'Basilar Tip', 'Other Posterior Circulation'
]

def predict(series_id):
    """
    Random baseline prediction function
    
    Args:
        series_id: SeriesInstanceUID for the test series
        
    Returns:
        list: List of 14 predictions (not dict!)
    """
    # Series-specific random seed for reproducibility
    np.random.seed(abs(hash(series_id)) % (2**32))
    
    # Generate 14 random probabilities (0.1 to 0.9)
    predictions = np.random.uniform(0.1, 0.9, size=14)
    
    # Return as LIST (not dictionary)
    return predictions.tolist()

# Create the inference server with the predict function
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

# Run the inference server
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # Production submission mode
    inference_server.serve()
else:
    # Local testing mode - this runs your predictions on test data
    inference_server.run_local_gateway()
    
    # Try to display results (use pandas since polars may not be available)
    try:
        if os.path.exists('/kaggle/working/submission.parquet'):
            submission_df = pd.read_parquet('/kaggle/working/submission.parquet')
        else:
            # Fallback to CSV if parquet not available
            submission_df = pd.read_csv('/kaggle/working/submission.csv')
            
        print(f"Generated predictions for {len(submission_df)} test series")
        print(f"\nSubmission shape: {submission_df.shape}")
        print(f"Columns: {list(submission_df.columns)}")
        print("\nFirst few random predictions:")
        print(submission_df.head())
        
        # Show statistics of random predictions
        pred_cols = [col for col in submission_df.columns if col != 'SeriesInstanceUID']
        print(f"\nRandom baseline statistics:")
        for col in pred_cols[:5]:  # Show first 5 columns
            mean_pred = submission_df[col].mean()
            print(f"{col}: mean = {mean_pred:.3f}")
            
    except Exception as e:
        print(f"Could not display submission results: {e}")

print("Random baseline submission process complete!")




