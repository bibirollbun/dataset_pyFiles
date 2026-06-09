import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import polars as pl
from sklearn.preprocessing import LabelEncoder

train_df = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
GESTURE_LABELS = train_df["gesture"].unique().sort().to_numpy()

le = LabelEncoder()
le.fit(GESTURE_LABELS)


import numpy as np
import pandas as pd
from tqdm import tqdm

def data_process(sequence_df, demo_df):
    exclude_cols = {"row_id", "sequence_type", "sequence_id", "sequence_counter", "subject", "orientation", "behavior", "phase", "gesture", "gesture_id"}
    value_cols = [col for col in sequence_df.columns if col not in exclude_cols]

    features = []
    for seq_id, group in tqdm(sequence_df.group_by("sequence_id")):
        for col in value_cols:
            if group.select(pl.col(col).is_null().all()).to_numpy()[0][0]:
                group = group.with_columns(pl.lit(0).alias(col))
            else:
                mean_val = group.select(pl.col(col).mean()).to_numpy()[0][0]
                group = group.with_columns(pl.col(col).fill_null(mean_val).alias(col))

        if group["subject"].n_unique() > 1:
            print("Warning: Duplicate `gesture` detected! The same `sequence_id` cannot have multiple `gesture`.")
        
        subject = group["subject"].first()
        row = {
            "sequence_id": seq_id[0],
            "subject": subject
        }
        for col in value_cols:
            values = group[col].to_numpy()
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_std"] = values.std()
            row[f"{col}_min"] = values.min()
            row[f"{col}_max"] = values.max()
            row[f"{col}_median"] = np.median(values)
            row[f"{col}_q25"] = np.percentile(values, 25)
            row[f"{col}_q75"] = np.percentile(values, 75)

        features.append(row)

    feature_df = pd.DataFrame(features)

    for col in ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"]:
        fill_val = 0
        if not demo_df.select(pl.col(col).is_null().all()).to_numpy()[0][0]:
            fill_val = demo_df.select(pl.col(col).median()).to_numpy()[0][0]
        demo_df = demo_df.with_columns(
            pl.col(col).fill_null(fill_val)
        )

    for col in ["adult_child", "sex", "handedness"]:
        fill_val = 0
        if not demo_df.select(pl.col(col).is_null().all()).to_numpy()[0][0]:
            modes = demo_df.select(pl.col(col).mode()).to_series()
            if modes.is_empty() or modes[0] is None:
                fill_val = 0
            else:
                fill_val = modes[0]
        demo_df = demo_df.with_columns(
            pl.col(col).fill_null(fill_val)
        )
    
    demo_df = demo_df.to_pandas()
    
    return feature_df.merge(demo_df, on="subject", how="left")


import pickle

with open('/kaggle/input/cmi25-randomforestclassifier-pretrain/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    process = data_process(sequence, demographics)
    X_test = process.drop(["sequence_id", "subject"], axis=1)
    preds = model.predict(X_test)
    return le.classes_[preds][0]


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


if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print(pd.read_parquet("submission.parquet"))

