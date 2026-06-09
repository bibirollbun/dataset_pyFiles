import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


files_to_delete = [
    '/kaggle/working/submission.csv'
]

for file_path in files_to_delete:
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print(f"File not found: {file_path}")


df_train= pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
df_test= pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
df_subm= pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


df_test['winddirection'].fillna(df_test['winddirection'].median(), inplace=True)


# # --- Data Preparation ---

# # Assuming df_train and df_test are already loaded:
# # df_train contains columns including 'day' and 'rainfall'
# # df_test contains a 'day' column and feature columns

# # Separate features and target
# X_train = df_train.drop(columns=['day', 'rainfall'])
# y_train = df_train['rainfall']
# X_test = df_test.drop(columns=['day'])

# # Scale the features (using StandardScaler for deep learning)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # --- Model Definition ---

# def build_dnn_model(input_dim):
#     model = Sequential([
#         Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(input_dim,)),
#         Dropout(0.3),
#         Dense(64, activation='relu', kernel_initializer='he_normal'),
#         Dropout(0.3),
#         Dense(32, activation='relu', kernel_initializer='he_normal'),
#         Dropout(0.2),
#         Dense(16, activation='relu', kernel_initializer='he_normal'),
#         Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
#     ])
#     optimizer = Adam(learning_rate=0.001)
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # Build and summarize the model
# model = build_dnn_model(X_train_scaled.shape[1])
# model.summary()

# # --- Model Training ---

# # Early stopping callback to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# # Train the model with a validation split
# history = model.fit(
#     X_train_scaled, y_train,
#     epochs=200,
#     batch_size=32,
#     validation_split=0.2,
#     callbacks=[early_stopping],
#     verbose=1
# )

# # --- Evaluation and Prediction ---

# # Evaluate on the training set
# y_train_pred = (model.predict(X_train_scaled) > 0.5).astype(int)
# train_accuracy = accuracy_score(y_train, y_train_pred)
# print("Training Accuracy:", train_accuracy)

# # Generate predictions on the test set
# y_test_pred = model.predict(X_test_scaled)



# --- Data Preparation ---

X_train = df_train.drop(columns=['day', 'rainfall'])
y_train = df_train['rainfall']
X_test = df_test.drop(columns=['day'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Definition ---

def build_dnn_model(input_dim):
    model = Sequential([
        Dense(128, kernel_initializer='he_normal', input_shape=(input_dim,)),
        BatchNormalization(),
        # Use relu activation after BN:
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(64, kernel_initializer='he_normal'),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(32, kernel_initializer='he_normal'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu', kernel_initializer='he_normal'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.0005)  # Adjusted learning rate
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_dnn_model(X_train_scaled.shape[1])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    epochs=300,        # You might increase epochs since early stopping is used
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# --- Evaluation and Prediction ---

y_train_pred = (model.predict(X_train_scaled) > 0.5).astype(int)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

y_test_pred = model.predict(X_test_scaled)


df_subm['rainfall'] = y_test_pred
df_subm.to_csv('submission.csv', index=False)
df_subm.head()




