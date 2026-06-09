import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import time


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})


TARGET = 'Calories'
FEATURES = [col for col in train.columns if col not in ['id', TARGET]]


X_test = test[FEATURES].copy()


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.maximum(0, y_true), np.maximum(0, y_pred)))


def make_callbacks():
    return [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]


def build_model(units, input_dim):
    x_in = Input(shape=(input_dim,))
    x = x_in
    for u in units:
        x = Dense(u)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)
    x = Dense(1, activation='linear')(x)
    return Model(inputs=x_in, outputs=x)


SEEDS = [42, 2020, 7]
FOLDS = 5
EPOCHS = 100
ARCH = [256, 128, 64]


kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))


for seed in SEEDS:
    print(f"\n SEED: {seed}")
    np.random.seed(seed)
    tf.random.set_seed(seed)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
        print(f"\n--- Fold {fold}/{FOLDS} ---")

        X_tr = train.loc[tr_idx, FEATURES].copy()
        y_tr = np.log1p(train.loc[tr_idx, TARGET])
        X_va = train.loc[va_idx, FEATURES].copy()
        y_va = np.log1p(train.loc[va_idx, TARGET])
        X_test_fold = X_test.copy()

        # Normalize
        scaler = StandardScaler()
        X_tr[FEATURES] = scaler.fit_transform(X_tr[FEATURES])
        X_va[FEATURES] = scaler.transform(X_va[FEATURES])
        X_test_fold[FEATURES] = scaler.transform(X_test_fold[FEATURES])

        # Compile + train
        K.clear_session()
        model = build_model(ARCH, input_dim=len(FEATURES))
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        start = time.time()
        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=EPOCHS,
            batch_size=256,
            callbacks=make_callbacks(),
            verbose=2
        )
        print(f"Time for fold: {time.time() - start:.1f} sec")

        # Predict
        va_pred = model.predict(X_va, batch_size=512, verbose=0).flatten()
        test_pred = model.predict(X_test_fold, batch_size=512, verbose=0).flatten()

        rmse = np.sqrt(mean_squared_error(y_va, va_pred))
        print(f"Fold {fold} RMSE (log-space): {rmse:.5f}")

        oof_preds[va_idx] += va_pred / len(SEEDS)
        test_preds += test_pred / (len(SEEDS) * FOLDS)



overall_rmse = np.sqrt(mean_squared_error(np.log1p(train[TARGET]), oof_preds))
print(f"\n=== Overall CV RMSLE (log-space RMSE): {overall_rmse:.5f} ===")


final_preds = np.expm1(test_preds).clip(0, None)
submission = pd.DataFrame({'id': test['id'], 'Calories': final_preds})
submission.to_csv('submission.csv', index=False)
print("submission.csv saved")

