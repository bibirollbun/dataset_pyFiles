import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# データ読み込み
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

# simple length feature
train['smiles_len'] = train['SMILES'].str.len()
test ['smiles_len'] = test ['SMILES'].str.len()
features = ['smiles_len']
target_cols = ['Tg','FFV','Tc','Density','Rg']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# prepare dataframes
test_preds = pd.DataFrame({'id': test['id']})
oof_preds  = pd.DataFrame(index=train.index, columns=target_cols)

# Neural Network model definition
def create_nn_model(input_dim):
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Callbacks
def get_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )
    ]

for target in target_cols:
    print(f'\n==> Training for target: {target}')
    # 1) only keep rows where this target exists
    mask    = train[target].notnull()
    X_full  = train.loc[mask, features]
    y_full  = train.loc[mask, target].astype(float)

    fold_rmses = []
    test_fold_preds = np.zeros(len(test))

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_full), 1):
        X_tr, y_tr = X_full.iloc[tr_idx], y_full.iloc[tr_idx]
        X_vl, y_vl = X_full.iloc[vl_idx], y_full.iloc[vl_idx]

        # データの標準化
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_vl_scaled = scaler.transform(X_vl)
        X_test_scaled = scaler.transform(test[features])

        # モデル作成
        model = create_nn_model(len(features))

        # 学習
        history = model.fit(
            X_tr_scaled, y_tr,
            validation_data=(X_vl_scaled, y_vl),
            epochs=200,
            batch_size=32,
            callbacks=get_callbacks(),
            verbose=0
        )

        # predict & score
        vl_pred = model.predict(X_vl_scaled, verbose=0).flatten()
        rmse    = np.sqrt(mean_squared_error(y_vl, vl_pred))
        fold_rmses.append(rmse)
        print(f'  Fold {fold} RMSE: {rmse:.4f}')

        # store OOF -- map back to original train index
        orig_idx = y_vl.index
        oof_preds.loc[orig_idx, target] = vl_pred

        # accumulate test predictions
        test_fold_preds += model.predict(X_test_scaled, verbose=0).flatten() / kf.n_splits

        # Clear session to free memory
        keras.backend.clear_session()

    test_preds[target] = test_fold_preds
    print(f'  >>> Avg RMSE for {target}: {np.mean(fold_rmses):.4f}')

# overall OOF
oof_rmse = np.sqrt(
    ((train[target_cols] - oof_preds[target_cols].astype(float))**2).mean().mean()
)
print(f'\nOverall OOF RMSE: {oof_rmse:.4f}')

# write submission
submission = test_preds[['id'] + target_cols]
submission.to_csv('submission.csv', index=False)
print(submission.head())

