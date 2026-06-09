# RSNA Intracranial Aneurysm Detection - MINIMAL WORKING SUBMISSION
import os
import shutil
from collections import defaultdict
import pandas as pd
import polars as pl
import pydicom
import kaggle_evaluation.rsna_inference_server

# Competition constants (copy from demo)
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

DICOM_TAG_ALLOWLIST = [
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit',
    'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber', 'Modality',
    'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
    'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows',
    'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness',
    'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID',
]

# REPLACE THIS FUNCTION WITH YOUR INFERENCE CODE
def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction - MINIMAL VERSION"""
    series_id = os.path.basename(series_path)
    
    # Default predictions (0.5 for all classes)
    predictions = pl.DataFrame(
        data=[[series_id] + [0.5] * len(LABEL_COLS)],
        schema=[ID_COL] + LABEL_COLS,
        orient='row',
    )
    
    # IMPORTANT: Clean up to prevent disk space errors
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    return predictions.drop(ID_COL)

# Competition execution flow
print("ğŸš€ Starting RSNA Aneurysm Detection Submission")
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("ğŸ�† COMPETITION MODE: Serving inference server...")
    inference_server.serve()
else:
    print("ğŸ’» LOCAL MODE: Running local gateway...")
    inference_server.run_local_gateway()
    result = pl.read_parquet('/kaggle/working/submission.parquet')
    print("âœ… submission.parquet created successfully!")
    print("ğŸ“Š Submission preview:")
    display(result)

