import pandas as pd 
import numpy as np 
import os 
import time
import logging 
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

from category_encoders import TargetEncoder

from tqdm.auto import tqdm
from itertools import combinations
import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")


df_external_1 = pd.read_csv("/kaggle/input/s5e9-26-37962/submission.csv")


def bin_column(df, column, bins, bin_names=None):
    if bin_names is None:
        bin_names = [f'{b:.1f}_to_{b_next:.1f}' for b, b_next in zip(bins[:-1], bins[1:])]
    df[column + '_binned'] = pd.cut(df[column], bins=bins, labels=bin_names, include_lowest=True)
    return df

bins = [0.025, 0.1, 0.15, 0.2]
train = bin_column(train, 'VocalContent', bins)
test = bin_column(test, 'VocalContent', bins)

bins = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
train = bin_column(train, 'AcousticQuality', bins)
test = bin_column(test, 'AcousticQuality', bins)

bins = [0.001, 0.2, 0.4, 0.6, 0.8, 1.0]
train = bin_column(train, 'InstrumentalScore', bins)
test = bin_column(test, 'InstrumentalScore', bins)

bins = [0.05, 0.2, 0.4]
train = bin_column(train, 'LivePerformanceLikelihood', bins)
test = bin_column(test, 'LivePerformanceLikelihood', bins)


bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
train = bin_column(train, 'MoodScore', bins)
test = bin_column(test, 'MoodScore', bins)



numerical_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
       'TrackDurationMs', 'Energy']

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    df['TrackDurationMin'] = df['TrackDurationMs'] / 60000 
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):  
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    
    df_new['acoustic_instrumental_ratio'] = df_new['AcousticQuality'] / (df_new['InstrumentalScore'] + 1e-6)
    df_new['RhythmEnergyRatio'] = df_new['RhythmScore'] / (df_new['Energy'] + 1e-8)
    df_new['VocalInstrumentalRatio'] = df_new['VocalContent'] / (df_new['InstrumentalScore'] + 1e-8)
    df['EnergyBin'] = pd.cut(df['Energy'], bins=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
    df['RhythmBin'] = pd.cut(df['RhythmScore'], bins=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
    
    return df_new

train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)



def add_feature_sq_terms(df, numerical_features):
    for feature in numerical_features:
        df[f'{feature}_squared'] = df[feature] ** 2
        df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))
    return df
    
train = add_feature_sq_terms(train, numerical_features)
test = add_feature_sq_terms(test, numerical_features)


num_features = train.select_dtypes(include='number')


BeatsPerMinute_global_avg = train['BeatsPerMinute'].mean()


X = train.drop(columns=["id", "BeatsPerMinute"])
y = train["BeatsPerMinute"]
X_test = test.drop(columns=["id"])


train.describe()


FOLDS = 10
FEATURES = X.columns.tolist()

# KFold setup
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# Arrays to store predictions
oof = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

# Start CV loop
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    
    x_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    x_valid = X.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx]
    x_test = X_test.copy()

    # No categorical target encoding in this dataset, but you can add if needed
    
    start = time.time()

    # Train model
    model = XGBRegressor(
        device="cuda" if XGBRegressor().get_params().get("device") == "cuda" else "cpu",
        max_depth=6,
        colsample_bytree=0.9,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.02,
        gamma=10.0, 
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric="rmse",
        enable_categorical=True
    )

    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=100
    )

    # Predict OOF and test
    oof[valid_idx] = model.predict(x_valid)
    pred_xgb += model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
    print(f"Feature engineering & training time: {time.time() - start:.1f} sec")

# Average test predictions
pred_xgb /= FOLDS

# Final RMSE
full_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"\nFinal CV RMSE: {full_rmse:.4f}")


# KFold setup
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=2025)

# Arrays to store predictions
oof = np.zeros(len(train))
pred_lgb = np.zeros(len(test))

# Start CV loop
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    
    x_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    x_valid = X.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx]
    x_test = X_test.copy()

    # No categorical target encoding in this dataset, but you can add if needed
    
    start = time.time()

    # Train model
    model = LGBMRegressor(
        device="gpu" if LGBMRegressor().get_params().get("device") == "gpu" else "cpu",
        max_depth=6,
        colsample_bytree=0.9,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.03,
        reg_alpha=0.8, 
        reg_lambda=4.0,
        early_stopping_rounds=100,
        metric="rmse",
        verbose=0,
    )

    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
    )

    # Predict OOF and test
    oof[valid_idx] = model.predict(x_valid)
    pred_lgb += model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
    print(f"Feature engineering & training time: {time.time() - start:.1f} sec")

# Average test predictions
pred_lgb /= FOLDS

# Final RMSE
full_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"\nFinal CV RMSE: {full_rmse:.4f}")


cat_features = [col for col in FEATURES if X[col].dtype == 'category']
for col in cat_features:
    X[col] = X[col].cat.add_categories(['missing']).fillna('missing')
    X_test[col] = X_test[col].cat.add_categories(['missing']).fillna('missing')


 
# KFold setup
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=2026)

# Arrays to store predictions
oof = np.zeros(len(train))
pred_cat = np.zeros(len(test))

# Start CV loop
for i, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\n{'#'*10} Fold {i+1} {'#'*10}")
    
    x_train = X.iloc[train_idx].copy()
    y_train = y.iloc[train_idx]
    x_valid = X.iloc[valid_idx].copy()
    y_valid = y.iloc[valid_idx]
    x_test = X_test.copy()

    # No categorical target encoding in this dataset, but you can add if needed
    
    start = time.time()

    # Train model
    model = CatBoostRegressor(
        task_type="GPU" if CatBoostRegressor().get_params().get("task_type") == "GPU" else "CPU",
        max_depth=6,
        colsample_bylevel=0.9,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.08,
        random_strength=0.1, 
        early_stopping_rounds=100,
        loss_function="RMSE",
        verbose=100
    )

    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        cat_features=cat_features,  
        use_best_model=True,
        verbose=100
    )

    # Predict OOF and test
    oof[valid_idx] = model.predict(x_valid)
    pred_cat += model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_valid, oof[valid_idx]))
    print(f"Fold {i+1} RMSE: {rmse:.4f}")
    print(f"Feature engineering & training time: {time.time() - start:.1f} sec")

# Average test predictions
pred_cat /= FOLDS

# Final RMSE
full_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"\nFinal CV RMSE: {full_rmse:.4f}")


pred = df_external_1["BeatsPerMinute"]*1.15 - pred_xgb*0.05 - pred_lgb * 0.05 - pred_cat * 0.05
print('predict mean :',pred.mean())
print('predict median :',np.median(pred))

y_pred_after = np.clip(pred, 46.718, 206.037)
print('predict mean after clip:',y_pred_after.mean())
print('predict median after clip:',np.median(y_pred_after))

submission["BeatsPerMinute"] = y_pred_after
submission.to_csv("submission.csv", index=False)
submission.head()

