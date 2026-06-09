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


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Load datasets
train_path = "/kaggle/input/thapar-kaggle-hack-v02/train.csv"
test_path = "/kaggle/input/thapar-kaggle-hack-v02/test.csv"
submission_path = "/kaggle/input/thapar-kaggle-hack-v02/sample_submission.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_submission = pd.read_csv(submission_path)

# Store 'id' column for test data
test_ids = df_test["id"] if "id" in df_test.columns else None
df_train.drop(columns=["id"], errors="ignore", inplace=True)
df_test.drop(columns=["id"], errors="ignore", inplace=True)

# Separate features and target
X = df_train.drop(columns=["target"])
y = df_train["target"]
X_test = df_test.copy()

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X[num_cols] = imputer.fit_transform(X[num_cols])
X_test[num_cols] = imputer.transform(X_test[num_cols])

# Create ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

# Transform data
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

# Convert labels to categorical
num_classes = len(np.unique(y))
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat = keras.utils.to_categorical(y_val, num_classes)

# Define optimized Neural Network Model
model = keras.Sequential([
    layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(num_classes, activation='softmax')  # Output layer
])

# Compile model with improved optimizer
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for training
early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(X_train, y_train_cat,
                    validation_data=(X_val, y_val_cat),
                    epochs=200,
                    batch_size=64,
                    verbose=1, 
                    callbacks=[early_stopping, reduce_lr])

# Evaluate model
val_loss, val_accuracy = model.evaluate(X_val, y_val_cat)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Train on full data
y_full_cat = keras.utils.to_categorical(y, num_classes)
model.fit(X_processed, y_full_cat, epochs=50, batch_size=64, verbose=1)

# Make predictions
y_test_pred = np.argmax(model.predict(X_test_processed), axis=1)

# Create submission file
df_submission["target"] = y_test_pred
if test_ids is not None and "id" not in df_submission.columns:
    df_submission.insert(0, "id", test_ids)

df_submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")


