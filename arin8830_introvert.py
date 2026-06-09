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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Load data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

X = train_df.drop(columns=["id", "Personality"])
y = train_df["Personality"]
X_test = test_df.drop(columns=["id"])
test_ids = test_df["id"]

# Binary features
binary_cols = ["Stage_fear", "Drained_after_socializing"]
for col in binary_cols:
    X[col] = X[col].map({"Yes": 1, "No": 0})
    X_test[col] = X_test[col].map({"Yes": 1, "No": 0})

# Impute missing
num_cols = X.select_dtypes(include=np.number).columns.tolist()
num_imputer = SimpleImputer(strategy="mean")
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# Feature engineering
X["Social_vs_Post"] = X["Social_event_attendance"] * X["Post_frequency"]
X["Alone_to_Friend_ratio"] = X["Time_spent_Alone"] / (X["Friends_circle_size"] + 1)
X_test["Social_vs_Post"] = X_test["Social_event_attendance"] * X_test["Post_frequency"]
X_test["Alone_to_Friend_ratio"] = X_test["Time_spent_Alone"] / (X_test["Friends_circle_size"] + 1)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(np.unique(y_encoded))

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert to categorical
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train_cat, epochs=100, batch_size=32,
                    validation_data=(X_val, y_val_cat), verbose=1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

# Evaluate
val_preds = model.predict(X_val)
val_preds_labels = np.argmax(val_preds, axis=1)
val_acc = accuracy_score(y_val, val_preds_labels)
print("MLP Validation Accuracy:", val_acc)

# Predict test
test_preds = model.predict(X_test)
test_preds_labels = np.argmax(test_preds, axis=1)
test_preds_decoded = le.inverse_transform(test_preds_labels)

# Save submission
submission = pd.DataFrame({
    "id": test_ids,
    "Personality": test_preds_decoded
})
submission.to_csv("mlp_submission.csv", index=False)
print("Saved: mlp_submission.csv")





