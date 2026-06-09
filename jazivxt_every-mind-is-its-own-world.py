import kaggle_evaluation.rsna_inference_server
from collections import defaultdict
import pydicom, shutil, os
import pandas as pd
import polars as pl
import numpy as np
from sklearn import *

p = '/kaggle/input/rsna-intracranial-aneurysm-detection/'
train = pd.read_csv(p+'train.csv')
trainl = pd.read_csv(p+'train_localizers.csv')


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


means = train[LABEL_COLS].mean().to_dict()


def predict(series_path: str):
    series_id = os.path.basename(series_path)
    #all_filepaths = []
    #for root, _, files in os.walk(series_path):
        #for file in files:
            #if file.endswith('.dcm'):
                #all_filepaths.append(os.path.join(root, file))
    #all_filepaths.sort()
    #tags = defaultdict(list)
    #tags['SeriesInstanceUID'] = series_id
    #global dcms
    #for filepath in all_filepaths:
        #ds = pydicom.dcmread(filepath, force=True)
        #tags['filepath'].append(filepath)
        #for tag in DICOM_TAG_ALLOWLIST:
            #tags[tag].append(getattr(ds, tag, None))
    predictions = pl.DataFrame(data=[[series_id] + [means[k] for k in LABEL_COLS]], schema=[ID_COL, *LABEL_COLS], orient='row',)
    if isinstance(predictions, pl.DataFrame):
        assert predictions.columns == [ID_COL, *LABEL_COLS]
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == [ID_COL, *LABEL_COLS]).all()
    else:
        raise TypeError('The predict function must return a DataFrame')
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    return predictions.drop(ID_COL)


inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))

