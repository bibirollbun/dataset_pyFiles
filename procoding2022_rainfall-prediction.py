import pandas as pd
import numpy as np
import seaborn as sns
import optuna
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
df_subm = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")

df_train.fillna(df_train.median(), inplace=True)
df_test.fillna(df_test.median(), inplace=True)

X = df_train.drop(columns=['day', 'rainfall'])
y = df_train['rainfall']
X_test = df_test.drop(columns=['day'])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

def objective(trial):
    units = trial.suggest_int('units', 64, 256, step=64)
    dropout_rate = trial.suggest_float('dropout', 0.2, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    
    model = Sequential([
        Dense(units, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(units // 2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(units // 4, activation='relu'),
        Dense(1, activation='linear') 
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping, lr_scheduler])
    
    return min(history.history['val_loss'])

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("Best Hyperparameters:", best_params)

model = Sequential([
    Dense(best_params['units'], activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(best_params['dropout']),
    Dense(best_params['units'] // 2, activation='relu'),
    BatchNormalization(),
    Dropout(best_params['dropout']),
    Dense(best_params['units'] // 4, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse', metrics=['mae'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping, lr_scheduler], verbose=1)

nn_pred = model.predict(X_test_scaled).flatten()

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_scores = cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_squared_error')
print("Random Forest Cross-Validation MSE:", -rf_scores.mean())

rf_pred = rf.predict(X_test)

final_pred = (nn_pred * 0.5) + (rf_pred * 0.5)

df_submission = pd.DataFrame({'id': df_test.index, 'rainfall': final_pred})
df_submission.to_csv('submission.csv', index=False)



