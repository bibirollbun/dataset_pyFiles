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


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import itertools

warnings.filterwarnings('ignore') # Disable warnings


train_data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
original_data = pd.read_csv('/kaggle/input/orginal-dataset/calories.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


def feature_engineering(df):
    df = df.copy()
    df.drop(columns=['id'], inplace=True, errors='ignore')
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    df['AgeSex'] = df['Age'].astype(str) + df['Sex'].astype(str)
    df['AgeSex'] = LabelEncoder().fit_transform(df['AgeSex'])
    features = ['Weight', 'Height', 'Body_Temp', 'Heart_Rate', 'Duration', 'Age', 'Sex', 'AgeSex']
    

    for comb in itertools.combinations(features, 2):  # Create pairwise combinations of features
        df[f"{comb[0]}_x_{comb[1]}"] = df[comb[0]] * df[comb[1]]
        df[f"{comb[0]}_div_{comb[1]}"] = df[comb[0]] / (df[comb[1]] + 1e-6)
    

    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)  # BMI-like feature
    
    return df


train_data = feature_engineering(train_data)
original_data = feature_engineering(original_data)
test_data = feature_engineering(test_data)

combined_data = pd.concat([train_data, original_data], axis=0).sample(frac=1, random_state=42)

# Split into features and target variable
X = combined_data.drop(columns=['Calories'])
y = np.log1p(combined_data['Calories'])  # Log transform of the target variable

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_data)

# model params
BATCH_SIZE = 1024
EPOCHS = 100
PATIENCE = 10
FOLDS = 5

# KFold cross-validator
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# array
test_predictions = np.zeros((test_data.shape[0],))


for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"\nFold {fold + 1}/{FOLDS}")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    

    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1)
    ])
    

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=[keras.metrics.RootMeanSquaredError()]
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    
    val_pred = model.predict(X_val, batch_size=BATCH_SIZE).flatten()
   
    from sklearn.metrics import mean_squared_error
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {val_rmse:.4f}")
    
    
    test_predictions += model.predict(test_scaled, batch_size=BATCH_SIZE).flatten() / FOLDS


test_predictions = np.expm1(test_predictions)

submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission['Calories'] = test_predictions
submission.to_csv('submission_tf.csv', index=False)

print("Submission file created!")


# import matplotlib.pyplot as plt

# plt.figure(figsize=(14, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.title(f'Fold {fold + 1} - Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss (MSE)')
# plt.legend()
# plt.grid(True)
# plt.subplot(1, 2, 2)
# plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')
# plt.plot(history.history['val_root_mean_squared_error'], label='Val RMSE')
# plt.title(f'Fold {fold + 1} - RMSE')
# plt.xlabel('Epochs')
# plt.ylabel('RMSE')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

