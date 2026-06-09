# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#       print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import shutil
from collections import defaultdict
import polars as pl
import pydicom
import kaggle_evaluation.rsna_inference_server

ID_COL = 'SeriesInstanceUID'

LABEL_COLS = [
    'Other Posterior Circulation',
    'Right Middle Cerebral Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Right Anterior Cerebral Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Right Posterior Communicating Artery',
    'Anterior Communicating Artery',
    'Left Middle Cerebral Artery',
    'Left Supraclinoid Internal Carotid Artery',    
    'Left Infraclinoid Internal Carotid Artery',
    'Left Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Basilar Tip',
    'Aneurysm Present',
]

# All tags (other than PixelData and SeriesInstanceUID) that may be in a test set dcm file
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

# Replace this function with your inference code.
# You can return either a Pandas or Polars dataframe, though Polars is recommended.
# Each prediction (except the very first) must be returned within 30 minutes of the series being provided.
def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # --------- Replace this section with your own prediction code ---------
    series_id = os.path.basename(series_path)
    
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()
    
    # Collect tags from the dicoms
    tags = defaultdict(list)
    tags['SeriesInstanceUID'] = series_id
    global dcms
    for filepath in all_filepaths:
        ds = pydicom.dcmread(filepath, force=True)
        tags['filepath'].append(filepath)
        for tag in DICOM_TAG_ALLOWLIST:
            tags[tag].append(getattr(ds, tag, None))
        # The image is in ds.PixelData

    # ... do some machine learning magic ...
    predictions = pl.DataFrame(
        data=[[series_id] +  len(LABEL_COLS)],
        schema=[ID_COL, *LABEL_COLS],
        orient='row',
    )
    # ----------------------------------------------------------------------

    if isinstance(predictions, pl.DataFrame):
        assert predictions.columns == [ID_COL, *LABEL_COLS]
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == [ID_COL, *LABEL_COLS]).all()
    else:
        raise TypeError('The predict function must return a DataFrame')

    # ----------------------------- IMPORTANT ------------------------------
    # You MUST have the following code in your `predict` function
    # to prevent "out of disk space" errors. This is a temporary workaround
    # as we implement improvements to our evaluation system.
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    # ----------------------------------------------------------------------
    
    return predictions.drop(ID_COL)


inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))

