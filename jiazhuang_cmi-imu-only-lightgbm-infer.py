import os
import polars as pl
import pandas as pd
import numpy as np


import lightgbm as lgb
from pathlib import Path
import joblib


MODEL_DIR = '/kaggle/input/cmi-imu-only-lightgbm-train'
models = []
for model_file in Path(MODEL_DIR).glob('lgb_model_fold*.txt'):
    models.append(
        lgb.Booster(model_file=model_file)
    )


label_encoder = joblib.load(f'{MODEL_DIR}/label_encoder.joblib')


def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

def kurt(x):
    return x.kurt()

agg_funs = ['mean', 'std', 'min', 'max', q25, q75, 'skew', kurt]
seq_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']


feat_cols = pd.read_csv(f'{MODEL_DIR}/feat_cols.txt', names=['feat']).feat.tolist()


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    sequence = sequence.to_pandas()
    demographics = demographics.to_pandas()

    # FE
    sequence = sequence.groupby(['sequence_id', 'subject']).agg({c: agg_funs for c in seq_cols})
    sequence.columns = [x[0] + '_' + x[1] for x in sequence.columns]
    sequence.reset_index(inplace=True)
    
    sequence = pd.merge(
        sequence,
        demographics,
        on='subject',
        how='left'
    )
    
    feat_df = sequence[feat_cols]

    # predict
    preds = []
    for model in models:
        preds.append( model.predict(feat_df).argmax(axis=-1)[0] )
    
    voted_pred = pd.Series(preds).mode().iloc[0]
    
    return label_encoder.classes_[voted_pred]


import kaggle_evaluation.cmi_inference_server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )




