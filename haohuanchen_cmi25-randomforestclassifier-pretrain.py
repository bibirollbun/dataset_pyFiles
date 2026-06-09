import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import polars as pl

train_df = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demo_df = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")


from sklearn.preprocessing import LabelEncoder

GESTURE_LABELS = train_df["gesture"].unique().sort().to_numpy()

le = LabelEncoder()
le.fit(GESTURE_LABELS)

gesture_ids = le.transform(train_df["gesture"].to_numpy())
train_df = train_df.with_columns(pl.Series("gesture_id", gesture_ids).cast(pl.Int32))


# result = train_df.group_by("sequence_id").agg(
#     pl.col("subject").n_unique()
# )
# print(result)


import numpy as np
import pandas as pd
from tqdm import tqdm

def data_process(sequence_df, demo_df):
    exclude_cols = {"row_id", "sequence_type", "sequence_id", "sequence_counter", "subject", "orientation", "behavior", "phase", "gesture", "gesture_id"}
    value_cols = [col for col in sequence_df.columns if col not in exclude_cols]

    features = []
    for seq_id, group in tqdm(sequence_df.group_by("sequence_id")):
        group = group.with_columns([
            pl.when(pl.col(col).is_null().all())
            .then(pl.lit(0))
            .otherwise(pl.col(col).fill_null(pl.col(col).mean()))
            .alias(col)
            for col in value_cols
        ])

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

    demo_df = demo_df.to_pandas()
    return feature_df.merge(demo_df, on="subject", how="left")


def add_gesture_id(sequence_df, process_df):
    gestureIds = []
    for seq_id, group in tqdm(sequence_df.group_by("sequence_id")):
        if group["gesture_id"].n_unique() > 1:
            print("Warning: Duplicate `gesture_id` detected! The same `sequence_id` cannot have multiple `gesture_id`.")
        gesture_id = group["gesture_id"].first()
        gestureIds.append({
            "sequence_id": seq_id[0],
            "gesture_id": gesture_id
        })
        
    gestureId_df = pd.DataFrame(gestureIds)
    
    return process_df.merge(gestureId_df, on="sequence_id", how="left")


train_process_df = data_process(train_df, train_demo_df)


train_process_df = add_gesture_id(train_df, train_process_df)


train_process_df.head()


X = train_process_df.drop(["sequence_id", "subject", "gesture_id"], axis=1)
y = train_process_df["gesture_id"]


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)


train_score = model.score(X, y)
print(f"Training Accuracy: {train_score:.4f}")


from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")


import pickle

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

