# ==========================================
# RSNA Intracranial Aneurysm Detection - Baseline Inference Code
# ==========================================

# Install necessary libraries
!pip install polars -q      # High-performance DataFrame library
!pip install pydicom -q     # For handling DICOM medical images

# Import required libraries
import kaggle_evaluation.rsna_inference_server   # Kaggle inference server for competition submissions
from collections import defaultdict              # Useful for dictionary-like structures
import pydicom                                   # For DICOM medical image processing
import shutil                                    # For file and directory operations
import os                                        # For path operations
import pandas as pd                              # For handling tabular data
import polars as pl                              # Alternative fast DataFrame library
import numpy as np                               # Numerical operations
from sklearn import *                            # For machine learning utilities (if needed)


# ==========================================
# Step 1: Define paths and load dataset
# ==========================================

# Path to the dataset
data_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/'

# Read training metadata
train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
train_localizers = pd.read_csv(os.path.join(data_path, 'train_localizers.csv'))


# ==========================================
# Step 2: Define column names
# ==========================================

# Unique identifier for each series
ID_COL = 'SeriesInstanceUID'

# Labels for different aneurysm locations
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]


# ==========================================
# Step 3: Define allowed DICOM tags
# ==========================================

# These metadata fields from DICOM files are considered safe to use
DICOM_TAG_ALLOWLIST = [
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit',
    'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber', 'Modality',
    'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
    'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType',
    'Rows', 'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness',
    'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID',
]


# ==========================================
# Step 4: Simple Baseline Model - Mean Strategy
# ==========================================

# Compute mean values of labels across the training dataset
# This will be used as a naive prediction (always predicting class mean probability)
means = train_data[LABEL_COLS].mean().to_dict()


# ==========================================
# Step 5: Define Prediction Function
# ==========================================

def predict(series_path: str):
    """
    Prediction function for RSNA Aneurysm Detection.

    Args:
        series_path (str): Path to the DICOM series folder.

    Returns:
        pl.DataFrame: Predictions for the given series without the ID column.
    """
    
    # Extract series ID from folder name
    series_id = os.path.basename(series_path)

    # Create a prediction DataFrame with mean values
    predictions = pl.DataFrame(
        data=[[series_id] + [means[k] for k in LABEL_COLS]],
        schema=[ID_COL, *LABEL_COLS],
        orient='row'
    )

    # Validate that predictions have the correct format
    if isinstance(predictions, pl.DataFrame):
        assert predictions.columns == [ID_COL, *LABEL_COLS], "Invalid Polars DataFrame schema"
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == [ID_COL, *LABEL_COLS]).all(), "Invalid Pandas DataFrame schema"
    else:
        raise TypeError('The predict function must return a DataFrame (Polars or Pandas).')

    # Clean up temporary shared directory (used by Kaggle inference server)
    shutil.rmtree('/kaggle/shared', ignore_errors=True)

    # Return predictions (excluding ID column, as required)
    return predictions.drop(ID_COL)


# ==========================================
# Step 6: Initialize Inference Server
# ==========================================

# Initialize the RSNA inference server with our predict() function
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)


# ==========================================
# Step 7: Run Inference
# ==========================================

# If running in competition mode, start the server
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()

# Otherwise, run locally for debugging
else:
    inference_server.run_local_gateway()

    # Display generated submission file for verification
    display(pl.read_parquet('/kaggle/working/submission.parquet'))


