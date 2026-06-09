# Install dependencies (GPU)
%pip -q install -U optuna "optuna-integration[tfkeras]" xgboost lightgbm catboost


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os, json, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

import optuna
from optuna.integration import (
    TFKerasPruningCallback,
    LightGBMPruningCallback,
    XGBoostPruningCallback,
    CatBoostPruningCallback
)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

SEED = 191
DATA_DIR = "/kaggle/input/playground-series-s5e9/"
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Assume GPU is available
try:
    tf.config.list_physical_devices('GPU')
except Exception:
    pass

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


class PreprocessData:
    def __init__(self, path):
        self.file_path = path
        self.train_path = os.path.join(path, "train.csv")
        self.test_path = os.path.join(path, "test.csv")
        self.eps = 1e-9

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        null_dict_train = {col: int(self.train_data[col].isnull().sum()) for col in self.train_data.columns}
        null_dict_test = {col: int(self.test_data[col].isnull().sum()) for col in self.test_data.columns}
        print("Data loaded successfully.")
        print("Null values in training data:", null_dict_train)
        print("Null values in test data:", null_dict_test)

    def feature_engineering(self, df):
        df_new = df.copy()

        # Numeric coercion for expected columns
        for col in [
            'TrackDurationMs','AudioLoudness','VocalContent','Energy','RhythmScore','MoodScore',
            'LivePerformanceLikelihood','AcousticQuality','InstrumentalScore'
        ]:
            if col in df_new.columns:
                df_new[col] = df_new[col].astype(float)

        # Duration
        df_new['log_Duration'] = np.log1p(df_new['TrackDurationMs'])

        # Audio loudness: convert dB to linear amplitude then log1p
        df_new['AudioLoudness_lin'] = 10 ** (df_new['AudioLoudness'] / 20.0)
        df_new['log_AudioLoudness_lin'] = np.log1p(df_new['AudioLoudness_lin'])

        # Vocal content
        df_new['log_VocalContent'] = np.log1p(df_new['VocalContent'])

        # Ratios and interactions
        df_new['Vocal_to_Instrumental'] = df_new['VocalContent'] / (df_new['InstrumentalScore'] + self.eps)
        df_new['Mood_x_Energy'] = df_new['MoodScore'] * df_new['Energy']
        df_new['Live_x_Acoustic'] = df_new['LivePerformanceLikelihood'] * df_new['AcousticQuality']
        df_new['Rhythm_sq'] = df_new['RhythmScore'] ** 2
        df_new['Rhythm_per_Ms'] = df_new['RhythmScore'] / (df_new['TrackDurationMs'] + self.eps)
        df_new['Vocal_Acoustic_Ratio'] = df_new['VocalContent'] / (df_new['AcousticQuality'] + self.eps)
        df_new['Energy_Mood_Product'] = df_new['Energy'] * df_new['MoodScore']
        df_new['Instrumental_Live_Interaction'] = df_new['InstrumentalScore'] * df_new['LivePerformanceLikelihood']

        # Polynomial features (interaction-only) for selected columns
        poly_in_cols = ['RhythmScore', 'AudioLoudness_lin', 'Energy']
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(df_new[poly_in_cols].fillna(0.0))
        feature_names = poly.get_feature_names_out(poly_in_cols)
        poly_df = pd.DataFrame(poly_features, columns=[f'poly_{name}' for name in feature_names], index=df_new.index)
        df_new = pd.concat([df_new, poly_df], axis=1)

        return df_new.fillna(0.0)

    def min_max_scale(self, series, min_val, max_val):
        if max_val - min_val == 0:
            return series - min_val
        return (series - min_val) / (max_val - min_val)

    def scale_data(self, df, min_max_dict=None):
        if min_max_dict is None:
            min_max_dict = {}
            df_scaled = df.copy()
            for col in df_scaled.columns:
                if col == 'id':
                    continue
                col_vals = df_scaled[col].astype(float)
                min_val = float(col_vals.min())
                max_val = float(col_vals.max())
                df_scaled[col] = self.min_max_scale(col_vals, min_val, max_val)
                min_max_dict[col] = {"min": min_val, "max": max_val}
            # path = os.path.join(self.file_path, 'min_max_values.json')
            # with open(path, 'w') as f:
            #     json.dump(min_max_dict, f)
            # print(f"Min and max values saved to {path}")
            return df_scaled, min_max_dict
        else:
            df_scaled = df.copy()
            for col in df_scaled.columns:
                if col == 'id':
                    continue
                if col in min_max_dict:
                    mm = min_max_dict[col]
                    df_scaled[col] = self.min_max_scale(df_scaled[col].astype(float), float(mm['min']), float(mm['max']))
                else:
                    col_vals = df_scaled[col].astype(float)
                    df_scaled[col] = self.min_max_scale(col_vals, float(col_vals.min()), float(col_vals.max()))
            return df_scaled, min_max_dict

# Keras objective (MLP)
def objective_keras(trial):
    hidden_1 = trial.suggest_int("hidden_1", 32, 256, step=32)
    hidden_2 = trial.suggest_int("hidden_2", 32, 256, step=32)
    dropout_1 = trial.suggest_float("dropout_1", 0.0, 0.4)
    dropout_2 = trial.suggest_float("dropout_2", 0.0, 0.4)
    momentum_ = trial.suggest_float("momentum", 0.01, 0.9)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    inputs = layers.Input(shape=(X_train.shape[1],))
    x = layers.Dense(hidden_1, activation=None)(inputs)
    x = layers.BatchNormalization(momentum=momentum_)(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_1)(x)
    x = layers.Dense(hidden_2, activation=None)(x)
    x = layers.BatchNormalization(momentum=momentum_)(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_2)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

    callbacks = [
        TFKerasPruningCallback(trial, "val_rmse"),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=3,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    preds = model.predict(X_val, verbose=0).ravel()
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# LightGBM objective (GPU)
def objective_lgb(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': 0,
        'boosting_type': 'gbdt',
        'seed': SEED,
        'feature_pre_filter': False,
        'device': 'gpu',
        'learning_rate': trial.suggest_float('lgb_learning_rate', 1e-4, 1e-1, log=True),
        'num_leaves': trial.suggest_int('lgb_num_leaves', 16, 256),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('lgb_min_data_in_leaf', 5, 200),
        'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lgb_lambda_l1', 0.0, 5.0),
        'lambda_l2': trial.suggest_float('lgb_lambda_l2', 0.0, 5.0),
    }

    dtrain_local = lgb.Dataset(X_train, label=y_train)
    dvalid_local = lgb.Dataset(X_val, label=y_val, reference=dtrain_local)

    callbacks = [
        lgb.early_stopping(stopping_rounds=100),
        LightGBMPruningCallback(trial, "rmse"),
        lgb.log_evaluation(period=0),
    ]

    gbm = lgb.train(param, dtrain_local, num_boost_round=2000, valid_sets=[dvalid_local], callbacks=callbacks)
    preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# XGBoost objective (GPU)
def objective_xgb(trial):
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': SEED,
        'verbosity': 0,
        'eta': trial.suggest_float('xgb_eta', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.4, 1.0),
        'lambda': trial.suggest_float('xgb_lambda', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('xgb_alpha', 1e-8, 10.0, log=True),
        'tree_method': 'hist',
        'device': 'cuda',
        'predictor': 'gpu_predictor',
    }

    dtrain_local = xgb.DMatrix(X_train, label=y_train)
    dvalid_local = xgb.DMatrix(X_val, label=y_val)

    bst = xgb.train(
        params=param,
        dtrain=dtrain_local,
        num_boost_round=2000,
        evals=[(dvalid_local, 'valid')],
        early_stopping_rounds=100,
        callbacks=[XGBoostPruningCallback(trial, "valid-rmse")],
    )

    preds = bst.predict(dvalid_local, iteration_range=(0, bst.best_iteration + 1))
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

# CatBoost objective (GPU)
def objective_cat(trial):
    param = {
        'loss_function': 'RMSE',
        'random_seed': SEED,
        'task_type': 'GPU',
        'devices': '0',
        'verbose': 0,
        'learning_rate': trial.suggest_float('cat_learning_rate', 1e-4, 1e-1, log=True),
        'depth': trial.suggest_int('cat_depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1e-2, 10.0),
        'random_strength': trial.suggest_float('cat_random_strength', 0.0, 20.0),
        'bagging_temperature': trial.suggest_float('cat_bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('cat_border_count', 32, 255),
        'iterations': 2000,
    }

    cat_train_local = cb.Pool(X_train, y_train)
    cat_val_local = cb.Pool(X_val, y_val)

    model = cb.CatBoostRegressor(**param)
    model.fit(
        cat_train_local,
        eval_set=cat_val_local,
        early_stopping_rounds=100,
        use_best_model=True,
        verbose=False,
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse


# Load, transform, and scale data
preprocess = PreprocessData(DATA_DIR)
preprocess.load_data()

train = preprocess.feature_engineering(preprocess.train_data)
test = preprocess.feature_engineering(preprocess.test_data)

# MinMax scaling saving train min/max
train_scaled, min_max = preprocess.scale_data(train)
train_mlp, val_mlp = train_test_split(train_scaled, test_size=0.1, random_state=SEED)

# Set for ensemble models (unscaled for tree-based)
train_ensemble, val_ensemble = train_test_split(train, test_size=0.1, random_state=SEED)

# Inputs/targets for MLP
X_train = train_mlp.drop(['id', 'BeatsPerMinute'], axis=1)
y_train = train_mlp['BeatsPerMinute']
X_val = val_mlp.drop(['id', 'BeatsPerMinute'], axis=1)
y_val = val_mlp['BeatsPerMinute']

# Scale test with the same min/max
test_scaled, _ = preprocess.scale_data(test, min_max_dict=min_max)


# Optuna studies
N_TRIALS = 20
TIMEOUT = None
TPESAMPLER = optuna.samplers.TPESampler(seed=SEED)
PRUNER = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

# 1) Keras MLP (uses scaled data)
study = optuna.create_study(direction="minimize", sampler=TPESAMPLER, pruner=PRUNER)
study.optimize(objective_keras, n_trials=N_TRIALS, timeout=TIMEOUT)
print("Best RMSE:", study.best_value)
print(study.best_params)

# 2) Reset splits for tree models (unscaled)
X_train = train_ensemble.drop(['id', 'BeatsPerMinute'], axis=1)
y_train = train_ensemble['BeatsPerMinute']
X_val = val_ensemble.drop(['id', 'BeatsPerMinute'], axis=1)
y_val = val_ensemble['BeatsPerMinute']

# LightGBM
study_lgb = optuna.create_study(direction="minimize", sampler=TPESAMPLER, pruner=PRUNER)
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS, timeout=TIMEOUT)
print("LightGBM best RMSE:", study_lgb.best_value)
print("LightGBM best params:", study_lgb.best_params)

# XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_val, label=y_val)
study_xgb = optuna.create_study(direction="minimize", sampler=TPESAMPLER, pruner=PRUNER)
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, timeout=TIMEOUT)
print("XGBoost best RMSE:", study_xgb.best_value)
print("XGBoost best params:", study_xgb.best_params)

# CatBoost
cat_train = cb.Pool(X_train, y_train)
cat_val = cb.Pool(X_val, y_val)
study_cat = optuna.create_study(direction="minimize", sampler=TPESAMPLER, pruner=PRUNER)
study_cat.optimize(objective_cat, n_trials=N_TRIALS, timeout=TIMEOUT)
print("CatBoost best RMSE:", study_cat.best_value)
print("CatBoost best params:", study_cat.best_params)


# Final training with best hyperparameters
best_params_mlp = study.best_params

hidden_1 = best_params_mlp['hidden_1']
hidden_2 = best_params_mlp['hidden_2']
dropout_1 = best_params_mlp['dropout_1']
dropout_2 = best_params_mlp['dropout_2']
momentum_ = best_params_mlp['momentum']
lr = best_params_mlp['lr']
batch_size = best_params_mlp['batch_size']

# Rebuild MLP for scaled data
X_train_mlp = train_mlp.drop(['id', 'BeatsPerMinute'], axis=1)
y_train_mlp = train_mlp['BeatsPerMinute']
X_val_mlp = val_mlp.drop(['id', 'BeatsPerMinute'], axis=1)
y_val_mlp = val_mlp['BeatsPerMinute']

inputs = layers.Input(shape=(X_train_mlp.shape[1],))
x = layers.Dense(hidden_1, activation=None)(inputs)
x = layers.BatchNormalization(momentum=momentum_)(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(dropout_1)(x)
x = layers.Dense(hidden_2, activation=None)(x)
x = layers.BatchNormalization(momentum=momentum_)(x)
x = layers.Activation("relu")(x)
x = layers.Dropout(dropout_2)(x)
outputs = layers.Dense(1)(x)
model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
model.fit(
    X_train_mlp, y_train_mlp,
    validation_data=(X_val_mlp, y_val_mlp),
    epochs=10,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=0,
)

# LightGBM with best params
lgb_param = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': 1,
    'boosting_type': 'gbdt',
    'seed': SEED,
    'feature_pre_filter': False,
    'device': 'gpu',
    'learning_rate': study_lgb.best_params['lgb_learning_rate'],
    'num_leaves': study_lgb.best_params['lgb_num_leaves'],
    'max_depth': study_lgb.best_params['lgb_max_depth'],
    'min_data_in_leaf': study_lgb.best_params['lgb_min_data_in_leaf'],
    'feature_fraction': study_lgb.best_params['lgb_feature_fraction'],
    'bagging_fraction': study_lgb.best_params['lgb_bagging_fraction'],
    'bagging_freq': study_lgb.best_params['lgb_bagging_freq'],
    'lambda_l1': study_lgb.best_params['lgb_lambda_l1'],
    'lambda_l2': study_lgb.best_params['lgb_lambda_l2'],
}

dtrain_local = lgb.Dataset(X_train, label=y_train)
dvalid_local = lgb.Dataset(X_val, label=y_val, reference=dtrain_local)

gbm = lgb.train(
    lgb_param,
    dtrain_local,
    num_boost_round=2000,
    valid_sets=[dvalid_local],
    callbacks=[lgb.early_stopping(100)],
)

# XGBoost with best params
xgb_param = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': SEED,
    'verbosity': 0,
    'eta': study_xgb.best_params['xgb_eta'],
    'max_depth': study_xgb.best_params['xgb_max_depth'],
    'min_child_weight': study_xgb.best_params['xgb_min_child_weight'],
    'subsample': study_xgb.best_params['xgb_subsample'],
    'colsample_bytree': study_xgb.best_params['xgb_colsample_bytree'],
    'lambda': study_xgb.best_params['xgb_lambda'],
    'alpha': study_xgb.best_params['xgb_alpha'],
    'tree_method': 'hist',
    'device': 'cuda',
    'predictor': 'gpu_predictor',
}

bst = xgb.train(
    params=xgb_param,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=[(dvalid, 'valid')],
    early_stopping_rounds=100,
)

# CatBoost with best params
cat_param = {
    'loss_function': 'RMSE',
    'random_seed': SEED,
    'task_type': 'GPU',
    'devices': '0',
    'verbose': 0,
    'learning_rate': study_cat.best_params['cat_learning_rate'],
    'depth': study_cat.best_params['cat_depth'],
    'l2_leaf_reg': study_cat.best_params['cat_l2_leaf_reg'],
    'random_strength': study_cat.best_params['cat_random_strength'],
    'bagging_temperature': study_cat.best_params['cat_bagging_temperature'],
    'border_count': study_cat.best_params['cat_border_count'],
    'iterations': 2000,
}

cat_model = cb.CatBoostRegressor(**cat_param)
cat_model.fit(cat_train, eval_set=cat_val, early_stopping_rounds=100, use_best_model=True, verbose=False)


# Test predictions and submission using the single best model by validation RMSE
id_series = test['id']

# Inverse scaling params for BPM
min_bpm = min_max['BeatsPerMinute']['min']
max_bpm = min_max['BeatsPerMinute']['max']

# 1) Compute validation RMSE for each model in BPM units
# MLP (trained on scaled features/target) -> inverse-transform to BPM
mlp_val_preds_scaled = model.predict(X_val_mlp, verbose=0).ravel()
mlp_val_preds = mlp_val_preds_scaled * (max_bpm - min_bpm) + min_bpm
y_val_mlp_true = y_val_mlp * (max_bpm - min_bpm) + min_bpm
rmse_mlp = np.sqrt(mean_squared_error(y_val_mlp_true, mlp_val_preds))

# LightGBM
lgb_val_preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
rmse_lgb = np.sqrt(mean_squared_error(y_val, lgb_val_preds))

# XGBoost
xgb_val_preds = bst.predict(xgb.DMatrix(X_val), iteration_range=(0, bst.best_iteration + 1))
rmse_xgb = np.sqrt(mean_squared_error(y_val, xgb_val_preds))

# CatBoost
cat_val_preds = cat_model.predict(X_val)
rmse_cat = np.sqrt(mean_squared_error(y_val, cat_val_preds))

rmse_by_model = {
    'mlp': rmse_mlp,
    'lgb': rmse_lgb,
    'xgb': rmse_xgb,
    'cat': rmse_cat,
}

best_model = min(rmse_by_model, key=rmse_by_model.get)
print("Validation RMSEs:", rmse_by_model)
print("Best model:", best_model, "with RMSE:", rmse_by_model[best_model])

# 2) Generate test predictions only from the best model
if best_model == 'mlp':
    preds_test = model.predict(test_scaled.drop(['id'], axis=1), verbose=0).ravel()
    preds_test = preds_test * (max_bpm - min_bpm) + min_bpm
elif best_model == 'lgb':
    preds_test = gbm.predict(test.drop(['id'], axis=1), num_iteration=gbm.best_iteration)
elif best_model == 'xgb':
    preds_test = bst.predict(xgb.DMatrix(test.drop(['id'], axis=1)), iteration_range=(0, bst.best_iteration + 1))
else:  # 'cat'
    preds_test = cat_model.predict(test.drop(['id'], axis=1))

# 3) Build submission
submission = pd.DataFrame({'id': id_series, 'BeatsPerMinute': preds_test})
submission_path = os.path.join('/kaggle/working/', 'submission.csv')
submission.to_csv(submission_path, index=False)
print(f"Submission saved to: {submission_path}")

