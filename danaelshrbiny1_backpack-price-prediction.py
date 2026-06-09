# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


df_train.head()


df_test.head()


# Display dataset overview
print("Train Dataset Info:")
print(df_train.info())
print(df_train.describe())


categorical_cols = ['Brand', 'Material', 'Style', 'Color', 'Size']
numerical_cols = ['Compartments', 'Weight Capacity (kg)']
target_col = 'Price'


cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='mean')


df_train[categorical_cols] = cat_imputer.fit_transform(df_train[categorical_cols])
df_train[numerical_cols] = num_imputer.fit_transform(df_train[numerical_cols])



df_test[categorical_cols] = cat_imputer.transform(df_test[categorical_cols])
df_test[numerical_cols] = num_imputer.transform(df_test[numerical_cols])


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le  # Store for later use


scaler = StandardScaler()
df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])


X_train = df_train[categorical_cols + numerical_cols].values
X_test = df_test[categorical_cols + numerical_cols].values
y_train = df_train[target_col].values


X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(df_train['Brand'].unique()) + 1, output_dim=4, input_length=1),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Output layer for regression
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])



history = model.fit(
    X_train_split, y_train_split,
    validation_data=(X_val, y_val),
    epochs=25,  # Increase epochs for better learning
    batch_size=32,
    verbose=1
)


y_pred = model.predict(X_test).flatten()


sample_predictions = pd.DataFrame({'id': df_test['id'].head(10), 'Predicted Price': y_pred[:10]})
print(sample_predictions)


submission = pd.DataFrame({'id': df_test['id'], 'Price': y_pred})
submission.to_csv("submission.csv", index=False)


import matplotlib.pyplot as plt
import seaborn as sns 


# Plot Loss
plt.figure(figsize=(18, 7)) 
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('MSE Loss', fontsize=14)
plt.title('Model Training Loss', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)



# Plot MAE
plt.figure(figsize=(18, 7)) 
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Mean Absolute Error', fontsize=14)
plt.title('Model Training MAE', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.show()


# Distribution of Actual vs Predicted Prices
plt.figure(figsize=(18, 7)) 
plt.subplot(1, 2, 1)
sns.histplot(y_train, label="Actual Prices", kde=True, color='blue', alpha=0.6, bins=30)
sns.histplot(y_pred, label="Predicted Prices", kde=True, color='red', alpha=0.6, bins=30)
plt.xlabel('Price', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title(' Distribution of Actual vs Predicted Prices', fontsize=16)
plt.legend(fontsize=12)

plt.grid(True)


# Scatter Plot: Actual vs Predicted
plt.figure(figsize=(18, 7)) 
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_train_split[:len(y_pred)], y=y_pred, alpha=0.6, edgecolor='k')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='dashed')  # Ideal prediction line
plt.xlabel("Actual Price", fontsize=14)
plt.ylabel("Predicted Price", fontsize=14)
plt.title("Actual vs Predicted Prices (Regression Fit)", fontsize=16)
plt.grid(True)

plt.show()


print("\n Model Training & Submission Complete!")
print("\nSample Predictions:")
print(submission.head(10))

