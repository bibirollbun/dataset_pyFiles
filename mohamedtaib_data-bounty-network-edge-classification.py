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


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Load Data ---
train_data_1 = pd.read_csv("/kaggle/input/data-bounty-3/Problem-3-train.csv")
train_data_2 = pd.read_csv("/kaggle/input/datat3/training.csv")
test_data = pd.read_csv("/kaggle/input/data-bounty-3/Problem-3-test.csv")

# Merge training data
train_data = pd.concat([train_data_1, train_data_2], ignore_index=True)

# --- Preprocessing ---

# Numerical features
numerical_features = ['rssi', 'snr', 'fcnt', 'frequency']

# Handle missing and infinite values
for df in [train_data, test_data]:
    df[numerical_features] = df[numerical_features].replace([np.inf, -np.inf], np.nan)
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].mean())

# Handle 'time' column
if 'time' in train_data.columns:
    train_data = train_data.drop(columns=['time'])
if 'time' in test_data.columns:
    test_data = test_data.drop(columns=['time'])

# Feature engineering
for df in [train_data, test_data]:
    df['rssi_snr'] = df['rssi'] * df['snr']
    df['rssi_freq_ratio'] = df['rssi'] / (df['frequency'] + 1e-6)
    df['snr_freq_ratio'] = df['snr'] / (df['frequency'] + 1e-6)

# Statistical features per deveui
stat_features = ['rssi', 'snr', 'fcnt', 'frequency']
train_grouped = train_data.groupby('deveui')[stat_features].agg(['mean', 'std', 'min', 'max']).reset_index()
train_grouped.columns = ['_'.join(col).strip() for col in train_grouped.columns.values]

test_grouped = test_data.groupby('deveui')[stat_features].agg(['mean', 'std', 'min', 'max']).reset_index()
test_grouped.columns = ['_'.join(col).strip() for col in test_grouped.columns.values]

train_data = train_data.merge(train_grouped, how='left', left_on='deveui', right_on='deveui_')
test_data = test_data.merge(test_grouped, how='left', left_on='deveui', right_on='deveui_')

train_data = train_data.drop(columns=['deveui_'])
test_data = test_data.drop(columns=['deveui_'])

# Label encode 'deveui' and 'gateway'
label_encoder_deveui = LabelEncoder()
all_deveui = pd.concat([train_data['deveui'], test_data['deveui']], axis=0).unique()
label_encoder_deveui.fit(all_deveui)
train_data['deveui'] = label_encoder_deveui.transform(train_data['deveui'])
test_data['deveui'] = label_encoder_deveui.transform(test_data['deveui'])

label_encoder_gateway = LabelEncoder()
train_data['gateway'] = label_encoder_gateway.fit_transform(train_data['gateway'])

# Normalize features
scaler = StandardScaler()
feature_cols = numerical_features + ['rssi_snr', 'rssi_freq_ratio', 'snr_freq_ratio'] + [
    f"{feature}_{stat}" for feature in stat_features for stat in ['mean', 'std', 'min', 'max']
]
train_data[feature_cols] = scaler.fit_transform(train_data[feature_cols])
test_data[feature_cols] = scaler.transform(test_data[feature_cols])

# --- Define Features and Target ---
drop_cols = []
if 'id' in train_data.columns:
    drop_cols.append('id')
drop_cols.append('gateway')

X = train_data.drop(columns=drop_cols)
y = train_data['gateway']

if 'id' in test_data.columns:
    X_test = test_data.drop(columns=['id'])
else:
    X_test = test_data

# --- Model Training ---

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBoost DMatrix
train_dmatrix = xgb.DMatrix(X_train, label=y_train)
val_dmatrix = xgb.DMatrix(X_val, label=y_val)
test_dmatrix = xgb.DMatrix(X_test)

# Parameters
params = {
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y)),
    'max_depth': 8,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'merror',
    'tree_method': 'hist',
    'seed': 42
}

# Training
model = xgb.train(
    params,
    train_dmatrix,
    num_boost_round=500,
    evals=[(train_dmatrix, 'train'), (val_dmatrix, 'validation')],
    verbose_eval=10
)

# --- Evaluation ---
val_preds = model.predict(val_dmatrix)
val_f1 = f1_score(y_val, val_preds, average='micro')
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation F1 Score: {val_f1:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# --- Predict on Test Set ---
test_preds = model.predict(test_dmatrix)
test_preds_labels = label_encoder_gateway.inverse_transform(test_preds.astype(int))

# Save submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'gateway': test_preds_labels
})
submission.to_csv("submission.csv", index=False)

print("\n✅ Submission file 'submission.csv' is ready!")

