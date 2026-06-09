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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.layers import LeakyReLU
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
import matplotlib.pyplot as plt

# -----------------------------
# Data Loading
# -----------------------------
train_path = "/kaggle/input/thapar-kaggle-hack-v02/train.csv"
test_path = "/kaggle/input/thapar-kaggle-hack-v02/test.csv"
submission_path = "/kaggle/input/thapar-kaggle-hack-v02/sample_submission.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_submission = pd.read_csv(submission_path)

# Store 'id' column for test data if available
test_ids = df_test.pop("id") if "id" in df_test.columns else None
df_train.drop(columns=["id"], errors="ignore", inplace=True)

# Separate features and target
X = df_train.drop(columns=["target"])
y = df_train["target"]
X_test = df_test.copy()

# -----------------------------
# Data Cleaning & Preprocessing
# -----------------------------

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

# Ensure categorical data is of type string
if cat_cols:
    X[cat_cols] = X[cat_cols].astype(str)
    X_test[cat_cols] = X_test[cat_cols].astype(str)

# --- Numerical Features ---
# Impute missing values using IterativeImputer (better for capturing relationships)
num_imputer = IterativeImputer(random_state=42)
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# Optional: Winsorize numerical features to cap extreme values (e.g., 1st and 99th percentiles)
def winsorize_series(s, lower_quantile=0.01, upper_quantile=0.99):
    lower = s.quantile(lower_quantile)
    upper = s.quantile(upper_quantile)
    return s.clip(lower, upper)

for col in num_cols:
    X[col] = winsorize_series(X[col])
    X_test[col] = winsorize_series(X_test[col])

# --- Categorical Features ---
if cat_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# --- Encoding Categorical Features ---
if cat_cols:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
    X_cat = encoder.fit_transform(X[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])
else:
    X_cat = np.empty((len(X), 0))
    X_test_cat = np.empty((len(X_test), 0))

# --- Scale Numerical Features ---
# Use RobustScaler to reduce the effect of outliers
scaler = RobustScaler()
X_num_scaled = scaler.fit_transform(X[num_cols])
X_test_num_scaled = scaler.transform(X_test[num_cols])

# Combine numerical and categorical features
X_processed = np.hstack((X_num_scaled, X_cat))
X_test_processed = np.hstack((X_test_num_scaled, X_test_cat))

# -----------------------------
# Feature Selection
# -----------------------------
# 1. Remove features with low variance
vt = VarianceThreshold(threshold=0.01)
X_var = vt.fit_transform(X_processed)
# (Apply same mask to test data)
X_test_var = vt.transform(X_test_processed)

# 2. Remove highly correlated features
# Create a DataFrame for convenience
df_features = pd.DataFrame(X_var)
corr_matrix = df_features.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
X_corr = df_features.drop(columns=to_drop).values
X_test_corr = pd.DataFrame(X_test_var).drop(columns=to_drop, errors='ignore').values

# 3. Use XGBoost-based feature importance via SelectFromModel
xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
xgb.fit(X_corr, y)
selector = SelectFromModel(xgb, threshold="median", prefit=True)
X_selected = selector.transform(X_corr)
X_test_selected = selector.transform(X_test_corr)

# -----------------------------
# Balance the Data using SMOTE
# -----------------------------
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_selected, y)

# -----------------------------
# Train-Test Split for NN training
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y))
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)

# -----------------------------
# Build a More Complex Neural Network Model
# -----------------------------
model_complex = keras.Sequential([
    layers.Dense(512, input_shape=(X_train.shape[1],)),
    LeakyReLU(alpha=0.2),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(256),
    LeakyReLU(alpha=0.2),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(128),
    LeakyReLU(alpha=0.2),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    layers.Dense(64),
    LeakyReLU(alpha=0.2),
    layers.BatchNormalization(),
    
    layers.Dense(num_classes, activation='softmax')
])

model_complex.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history_complex = model_complex.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate on Validation Data
val_loss, val_accuracy = model_complex.evaluate(X_val, y_val_cat)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# -----------------------------
# Train on Full Data and Predict Test Set
# -----------------------------
y_full_cat = keras.utils.to_categorical(y, num_classes)
model_complex.fit(X_selected, y_full_cat, epochs=50, batch_size=32, verbose=1)

y_test_pred = np.argmax(model_complex.predict(X_test_selected), axis=1)

# Create submission file
df_submission["target"] = y_test_pred
if test_ids is not None and "id" not in df_submission.columns:
    df_submission.insert(0, "id", test_ids)

df_submission.to_csv("submission.csv", index=False)
print(" Submission file created: submission.csv")



