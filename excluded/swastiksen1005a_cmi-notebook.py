# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import pandas as pd
import numpy as np
import polars as pl
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import kaggle_evaluation.cmi_inference_server

# === 1. Load and prepare training data ===
train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
merged = pd.merge(train, train_demo, on='subject', how='left')

# === 2. ToF Imputation ===
tof_cols = [col for col in merged.columns if col.startswith('tof')]
merged[tof_cols] = merged[tof_cols].replace(-1, np.nan)
medians = np.nanmedian(merged[tof_cols].values, axis=0)
merged[tof_cols] = np.where(np.isnan(merged[tof_cols]), medians, merged[tof_cols])

# === 3. Add ToF aggregates ===
tof_agg_df = pd.DataFrame({
    'tof_mean': merged[tof_cols].mean(axis=1),
    'tof_std': merged[tof_cols].std(axis=1),
    'tof_min': merged[tof_cols].min(axis=1),
    'tof_max': merged[tof_cols].max(axis=1),
    'tof_median': merged[tof_cols].median(axis=1),
}, index=merged.index)

merged = pd.concat([merged.reset_index(drop=True), tof_agg_df.reset_index(drop=True)], axis=1)

# === 4. Feature columns ===
features = [
    'acc_x', 'acc_y', 'acc_z',
    'rot_w', 'rot_x', 'rot_y', 'rot_z',
    'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5',
    'adult_child', 'age', 'sex', 'handedness',
    'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm',
    'tof_mean', 'tof_std', 'tof_min', 'tof_max', 'tof_median'
]

X = merged[features]
y = merged["gesture"]

# === 5. Encode target ===
gesture_le = LabelEncoder()
y_encoded = gesture_le.fit_transform(y)

# === 6. Train model ===
model = XGBClassifier(
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_jobs=-1,
    verbosity=0,
    random_state=42
)
model.fit(X, y_encoded)

# === 7. Inference function ===
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()

    # ToF Fix
    for col in tof_cols:
        if col in seq_df.columns:
            seq_df[col] = seq_df[col].replace(-1, np.nan)
    med_vals = pd.Series(medians, index=tof_cols)
    seq_df[tof_cols] = seq_df[tof_cols].fillna(med_vals)

    # Add ToF aggregates
    tof_agg = pd.DataFrame({
        'tof_mean': seq_df[tof_cols].mean(axis=1),
        'tof_std': seq_df[tof_cols].std(axis=1),
        'tof_min': seq_df[tof_cols].min(axis=1),
        'tof_max': seq_df[tof_cols].max(axis=1),
        'tof_median': seq_df[tof_cols].median(axis=1),
    })
    seq_df = pd.concat([seq_df.reset_index(drop=True), tof_agg.reset_index(drop=True)], axis=1)

    # Final features
    sensor_features = seq_df.iloc[-1][[
        'acc_x', 'acc_y', 'acc_z',
        'rot_w', 'rot_x', 'rot_y', 'rot_z',
        'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5',
        'tof_mean', 'tof_std', 'tof_min', 'tof_max', 'tof_median'
    ]]
    demo_features = demo_df.iloc[0][[
        'adult_child', 'age', 'sex', 'handedness',
        'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'
    ]]

    full_input = pd.concat([sensor_features, demo_features]).values.reshape(1, -1)
    pred = model.predict(full_input)[0]
    return gesture_le.inverse_transform([pred])[0]

# === 8. CMI Inference Server ===
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # === 9. Local Prediction & Submission File Generation ===
    test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
    test_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')

    results = []
    for seq_id, seq_df in test.groupby("sequence_id"):
        demo = test_demo[test_demo["subject"] == seq_df["subject"].iloc[0]]

        sequence_pl = pl.from_pandas(seq_df.reset_index(drop=True))
        demo_pl = pl.from_pandas(demo.reset_index(drop=True))

        pred = predict(sequence_pl, demo_pl)
        results.append({"row_id": seq_id, "gesture": pred})

    # Create and save submission
    submission_df = pd.DataFrame(results)
    submission_df.to_parquet("submission.parquet", index=False)
    print("submission.parquet saved successfully!")


