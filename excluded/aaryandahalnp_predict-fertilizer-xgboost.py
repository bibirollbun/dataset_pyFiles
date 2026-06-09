!pip install pandas numpy opendatasets scikit-learn xgboost --quiet
import pandas as pd
import numpy as np
import os
import opendatasets as od
import random


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


def rename_columns(df_name):
    df_name.rename(columns = {'Temparature': 'Temperature'}, inplace=True)

rename_columns(df)
rename_columns(test_df)


# Split into training and validation
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare columns and dataframes
input_cols = list(train_df.columns)[1:-1]
target_col = 'Fertilizer Name'

train_inputs = train_df[input_cols].copy()
val_inputs = val_df[input_cols].copy()
test_inputs = test_df[input_cols].copy()
train_targets = train_df[target_col].copy()
val_targets = val_df[target_col].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


# Label Encoding Outputs
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col])
y_val = le.fit_transform(val_df[target_col])


X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

from xgboost import XGBClassifier

model = XGBClassifier(random_state=42, n_jobs=-1, n_estimators=400).fit(X_train, y_train)

def make_predictions(inputs, model):
    probs = model.predict_proba(inputs)
    top3_idx = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    predictions = le.inverse_transform(top3_idx.ravel()).reshape(top3_idx.shape)
    return predictions


def accuracy(actual, predicted, k=3):
    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: actual has {len(actual)}, predicted has {len(predicted)}")

    score = 0.0
    for a, p in zip(actual, predicted):
        if a in p[:k]:
            rank = p.index(a)
            score += 1.0 / (rank + 1)
        # If not in top-k, score += 0
    return score / len(actual)

def probability(actual, predictions):
    actual_labels = le.inverse_transform(actual)
    actual_labels = actual_labels.tolist()
    predictions = predictions.tolist()
    score = accuracy(actual_labels, predictions, k=3)
    print(f"Score: {score:.5f}")

train_predictions = make_predictions(X_train, model)
probability(y_train, train_predictions)

val_predictions = make_predictions(X_val, model)
probability(y_val, val_predictions)


test_predictions = make_predictions(X_test, model)
outputs = [' '.join(row) for row in test_predictions]
sample_submission['Fertilizer Name'] = outputs
sample_submission.to_csv('submission.csv', index=False)

