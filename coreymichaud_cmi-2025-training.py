# Libraries
import os

import pandas as pd
import numpy as np
import polars as pl

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from metric import score, CompetitionMetric  # metric given by the competition

import xgboost as xgb


# CONFIG
CROSSVAL = 1
TRAINING = 0
SAVING = 0


# Importing training data
train = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_demographics = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")


# Combining train with train_demographics ... should have 348 columns if successful
train_merged = pd.merge(train, train_demographics, on = "subject", how = "left")
train_merged.shape[1]  # Num of columns


# Getting orientation, thermopile, and time-of-flight column names
acc_cols = [f'acc_{axis}' for axis in ['x', 'y', 'z']]
rot_cols = [f'rot_{axis}' for axis in ['w', 'x', 'y', 'z']]
thm_cols = [f'thm_{i}' for i in range(1, 6)]
tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]

# Combine all numeric columns into the final list
numerical_cols = ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"] + acc_cols + rot_cols + thm_cols + tof_cols

# Getting already labeled column names
already_labeled_categorical_cols = ["adult_child", "sex", "handedness"]


# Grouping by sequence_id and then merging remaining columns
cols = ['sequence_id'] + acc_cols + rot_cols + thm_cols + tof_cols

summary = (train_merged[cols].groupby('sequence_id').agg(['mean', 'std', 'min', 'max', 'median']))

summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

summary = summary.reset_index()

summary.head()


# Adding summary statistics back to whole grouped dataset
remaining = train_merged.drop_duplicates('sequence_id')

train_df = pd.merge(remaining, summary, on = "sequence_id", how = "left")
train_df.head()


# Getting new numeric columns
exclude_cols = set(already_labeled_categorical_cols) | {"sequence_id"}

numerical_cols = [
    col for col in train_df.select_dtypes(include=["number"]).columns
    if col not in exclude_cols
]


# Label encoding the target
le = LabelEncoder()
train_df["gesture"] = le.fit_transform(train_df["gesture"])


# Storing target gestures
target_classes = ["Above ear - pull hair", "Cheek - pinch skin", "Eyebrow - pull hair", "Eyelash - pull hair",
                  "Forehead - pull hairline", "Forehead - scratch", "Neck - pinch skin", "Neck - scratch"]
target_class_indices = [np.where(le.classes_ == cls)[0][0] for cls in target_classes]


# Column Transformer
num = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num, numerical_cols),
    ("cat_pass", "passthrough", already_labeled_categorical_cols)
], remainder = "drop")


# Model
xgboost = xgb.XGBClassifier(
    subsample = 0.8,
    reg_lambda = 0.1,
    reg_alpha = 0,
    n_estimators = 300,
    min_child_weight = 3,
    max_depth = 5,
    gamma = 0.2,
    eta = 0.1,
    colsample_bytree = 0.9,
    tree_method='hist',
    device='cuda',
    objective="multi:softprob",
    random_state = 42
)


# Pipeline
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("xgboost", xgboost)
])


# Getting X and y
X = train_df.drop("gesture", axis = 1)
y = train_df["gesture"]


%%time
# Cross validation
if CROSSVAL:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    hierarchical_scores = []

    metric = CompetitionMetric()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        y_test_labels = le.inverse_transform(y_test)
        y_pred_labels = le.inverse_transform(y_pred)

        solution_df = pd.DataFrame({
            'id': range(len(y_test_labels)),
            'gesture': y_test_labels
        })

        submission_df = pd.DataFrame({
            'id': range(len(y_pred_labels)),
            'gesture': y_pred_labels
        })

        score_fold = metric.calculate_hierarchical_f1(solution_df, submission_df)
        hierarchical_scores.append(score_fold)

    print("Mean Hierarchical F1 Score:", np.mean(hierarchical_scores))


%%time
# Fitting model on whole training set
if TRAINING:
    inference_pipeline = pipe.fit(X,y)


# Saving pipeline and label encoder
if SAVING:
    joblib.dump(inference_pipeline, 'pipeline.pkl')
    joblib.dump(le, 'label_encoder.pkl')

