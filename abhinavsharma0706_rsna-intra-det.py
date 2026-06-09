# Import necessary libraries
!pip install polars -q
!pip install pydicom -q
import kaggle_evaluation.rsna_inference_server
from collections import defaultdict
import pydicom
import shutil
import os
import pandas as pd
import polars as pl
import numpy as np
from sklearn import *

# Define paths and read CSV files
data_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/'
train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
train_localizers = pd.read_csv(os.path.join(data_path, 'train_localizers.csv'))

# Define column names
ID_COL = 'SeriesInstanceUID'

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

# Define allowed DICOM tags
DICOM_TAG_ALLOWLIST = [
    'BitsAllocated',
    'BitsStored',
    'Columns',
    'FrameOfReferenceUID',
    'HighBit',
    'ImageOrientationPatient',
    'ImagePositionPatient',
    'InstanceNumber',
    'Modality',
    'PatientID',
    'PhotometricInterpretation',
    'PixelRepresentation',
    'PixelSpacing',
    'PlanarConfiguration',
    'RescaleIntercept',
    'RescaleSlope',
    'RescaleType',
    'Rows',
    'SOPClassUID',
    'SOPInstanceUID',
    'SamplesPerPixel',
    'SliceThickness',
    'SpacingBetweenSlices',
    'StudyInstanceUID',
    'TransferSyntaxUID',
]

# Calculate means of label columns
means = train_data[LABEL_COLS].mean().to_dict()

def predict(series_path: str):
    # Extract series ID from the path
    series_id = os.path.basename(series_path)

    # Create a DataFrame with predictions
    predictions = pl.DataFrame(
        data=[[series_id] + [means[k] for k in LABEL_COLS]],
        schema=[ID_COL, *LABEL_COLS],
        orient='row'
    )

    # Validate the predictions DataFrame
    if isinstance(predictions, pl.DataFrame):
        assert predictions.columns == [ID_COL, *LABEL_COLS]
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == [ID_COL, *LABEL_COLS]).all()
    else:
        raise TypeError('The predict function must return a DataFrame')

    # Clean up the shared directory
    shutil.rmtree('/kaggle/shared', ignore_errors=True)

    # Return predictions without the ID column
    return predictions.drop(ID_COL)

# Initialize the inference server
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

# Check if the environment variable is set to run the server or locally
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    # Display the submission parquet file
    display(pl.read_parquet('/kaggle/working/submission.parquet'))

