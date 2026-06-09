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


import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import glob
import os


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


TARGET = 'accident_risk'


def preprocess_features(df):
    df = df.copy()
    # Convert booleans to integers
    for col in ['road_signs_present', 'public_road', 'holiday', 'school_season']:
        df[col] = df[col].astype(int)
    # One-hot encode categorical variables
    categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


train_processed = preprocess_features(train)
test_processed = preprocess_features(test)


all_oof_data = []
all_test_data = []
oof_files = glob.glob('/kaggle/input/**/oof_*.csv', recursive=True)
print(f"Found {len(oof_files)} oof files.")


for oof_path in oof_files:
    test_path = oof_path.replace('oof_', 'test_')
    base_name = os.path.basename(oof_path)
    model_name = base_name.replace('oof_', '').replace('.csv', '')
    all_oof_data.append({
        'df': pd.read_csv(oof_path),
        'name': model_name
    })
    all_test_data.append({
        'df': pd.read_csv(test_path),
        'name': model_name
    })


def merge_dataframes_by_id(data_list, id_col='id', feature_col=TARGET):
    first_data = data_list[0]
    merged_df = first_data['df'].rename(columns={feature_col: f"{feature_col}_{first_data['name']}"})
    for data in data_list[1:]:
        renamed_df = data['df'].rename(columns={feature_col: f"{feature_col}_{data['name']}"})
        merged_df = pd.merge(merged_df, renamed_df, on=id_col, how='outer')
    return merged_df

oof_df = merge_dataframes_by_id(all_oof_data)
test_df = merge_dataframes_by_id(all_test_data)


feature_cols = [col for col in train_processed.columns if col not in ['id', TARGET]]
oof_df = pd.merge(oof_df, train_processed[['id'] + feature_cols], on='id', how='left')
test_df = pd.merge(test_df, test_processed[['id'] + feature_cols], on='id', how='left')
oof_df[TARGET] = train[TARGET].values


FEATURES = [col for col in oof_df.columns if col not in ['id', TARGET]]


X = oof_df[FEATURES]
y = oof_df[TARGET]
X_test = test_df[FEATURES]


N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_df))


SEEDS = [32, 12, 377, 1234, 9012, 3456]


for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f'---Fold {fold+1}/{N_SPLITS}---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    for seed in SEEDS:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        model = Sequential([
            Input(shape=(X_train_scaled.shape[1],)),
            Dense(128, activation='gelu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='gelu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='gelu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='gelu'),
            BatchNormalization(),
            Dense(1)
        ])
        
        # Compile with Huber loss and Adam optimizer
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                     loss=tf.keras.losses.Huber(delta=1.0))
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        
        # Train model
        model.fit(X_train_scaled, y_train,
                  validation_data=(X_val_scaled, y_val),
                  epochs=15,
                  batch_size=256,
                  callbacks=[early_stopping, lr_scheduler],
                  verbose=2)
        
        # Predict and accumulate
        val_preds = model.predict(X_val_scaled, verbose=0).flatten()
        oof_preds[val_idx] += val_preds / len(SEEDS)
        test_preds += model.predict(X_test_scaled, verbose=0).flatten() / len(SEEDS)
    
    # Compute fold RMSE
    fold_rmse = mean_squared_error(y_val, oof_preds[val_idx], squared=False)
    print(f"Fold {fold+1} RMSE: {fold_rmse:.6f}")


test_preds /= N_SPLITS


oof_preds = np.clip(oof_preds, 0, 1)
test_preds = np.clip(test_preds, 0, 1)


overall_oof_rmse = mean_squared_error(y, oof_preds, squared=False)
print(f"\nOverall OOF RMSE: {overall_oof_rmse:.6f}")


pd.DataFrame({'id': train['id'], TARGET: oof_preds}).to_csv('oof_nn_ensemble.csv', index=False)
pd.DataFrame({'id': test['id'], TARGET: test_preds}).to_csv('test_nn_ensemble.csv', index=False)

