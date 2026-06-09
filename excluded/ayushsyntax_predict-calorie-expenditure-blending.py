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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
import time

warnings.simplefilter('ignore')


import torch
print("CUDA available:", torch.cuda.is_available())


def load_data(train_path, test_path, submission_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(submission_path)

    # Ensure no leaks or inconsistencies
    assert 'id' in train.columns and 'Calories' in train.columns, "Missing required columns"
    
    numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    return train, test, submission, numerical_features


def add_feature_cross_terms(df, features):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df

def add_interaction_features(df, features):
    df_new = df.copy()
    for f1, f2 in itertools.combinations(features, 2):
        df_new[f"{f1}_plus_{f2}"] = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
        df_new[f"{f1}_div_{f2}"] = df_new[f1] / (df_new[f2] + 1e-5)
        df_new[f"{f2}_div_{f1}"] = df_new[f2] / (df_new[f1] + 1e-5)
    return df_new

def add_statistical_features(df, features):
    df["row_mean"] = df[features].mean(axis=1)
    df["row_std"] = df[features].std(axis=1)
    df["row_max"] = df[features].max(axis=1)
    df["row_min"] = df[features].min(axis=1)
    df["row_median"] = df[features].median(axis=1)
    return df


def apply_feature_engineering(df, numerical_features, le=None, poly=None, fit=False):
    df = df.copy()

    # Apply transformations
    df = add_feature_cross_terms(df, numerical_features)
    df = add_interaction_features(df, numerical_features)
    df = add_statistical_features(df, numerical_features)

    # Encode categorical variable
    if 'Sex' in df.columns:
        if fit and le is None:
            le = LabelEncoder()
            df['Sex'] = le.fit_transform(df['Sex'])
        else:
            df['Sex'] = le.transform(df['Sex'].astype(str))
        df['Sex'] = df['Sex'].astype('category')

    # Add polynomial features (only fit on train)
    if fit and poly is None:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_train = poly.fit_transform(df[numerical_features])
        poly_df = pd.DataFrame(poly_train, columns=poly.get_feature_names_out(numerical_features), index=df.index)
    else:
        poly_df = pd.DataFrame(poly.transform(df[numerical_features]), columns=poly.get_feature_names_out(numerical_features), index=df.index)

    # Concat engineered features safely
    df = pd.concat([df.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)

    return df, le, poly


train, test, submission, numerical_features = load_data(
    "/kaggle/input/playground-series-s5e5/train.csv",
    "/kaggle/input/playground-series-s5e5/test.csv",
    "/kaggle/input/playground-series-s5e5/sample_submission.csv"
)

# Apply feature engineering safely
X, le, poly = apply_feature_engineering(train, numerical_features, fit=True)
X_test, _, _ = apply_feature_engineering(test, numerical_features, le=le, poly=poly)

# Clean up final input
X = X.drop(columns=['id', 'Calories'], errors='ignore')
y = np.log1p(train['Calories'])
X_test = X_test.drop(columns=['id'], errors='ignore')


models = {
    'CatBoost': CatBoostRegressor(
        verbose=100,
        random_seed=42,
        cat_features=['Sex'],
        early_stopping_rounds=100,
        task_type="GPU" if torch.cuda.is_available() else "CPU"
    ),
    'XGBoost': XGBRegressor(
        max_depth=10,
        colsample_bytree=0.7,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.02,
        gamma=0.01,
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric='rmse',
        enable_categorical=True,
        tree_method='gpu_hist' if torch.cuda.is_available() else 'auto',
        predictor='gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor'
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=10,
        colsample_bytree=0.7,
        subsample=0.9,
        random_state=42,
        verbose=-1
    )
}


FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
results = {name: {'oof': np.zeros(len(X)), 'pred': np.zeros(len(X_test)), 'rmsle': []} for name in models}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {fold+1}")
        x_train, y_train = X.iloc[train_idx], y[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

        # Remove duplicate columns
        x_train = x_train.loc[:, ~x_train.columns.duplicated()]
        x_valid = x_valid.loc[:, ~x_valid.columns.duplicated()]
        x_test = X_test.loc[:, ~X_test.columns.duplicated()]

        start = time.time()

        if name == 'XGBoost':
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)
        elif name == 'CatBoost':
            model.fit(x_train, y_train, eval_set=(x_valid, y_valid))
        else:
            model.fit(x_train, y_train)

        oof_pred = model.predict(x_valid)
        test_pred = model.predict(x_test)

        results[name]['oof'][valid_idx] = oof_pred
        results[name]['pred'] += test_pred / FOLDS

        rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(oof_pred)))
        results[name]['rmsle'].append(rmsle)

        print(f"Fold {fold+1} RMSLE: {rmsle:.4f}")
        print(f"Training Time: {time.time() - start:.1f}s")

print("\n=== Model Comparison ===")
for name in models:
    mean_rmsle = np.mean(results[name]['rmsle'])
    std_rmsle = np.std(results[name]['rmsle'])
    print(f"{name} - Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")


oof_preds = {name: np.expm1(results[name]['oof']) for name in results}
test_preds = {name: np.expm1(results[name]['pred']) for name in results}
y_true = np.expm1(y)

def rmsle_loss(weights):
    blended = (
        weights[0] * oof_preds['CatBoost'] +
        weights[1] * oof_preds['XGBoost'] +
        weights[2] * oof_preds['LightGBM']
    )
    return np.sqrt(mean_squared_log_error(y_true, blended))

initial_weights = [1/3, 1/3, 1/3]
bounds = [(0, 1)] * 3
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

res = minimize(rmsle_loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
best_weights = res.x

print("\n✅ Optimized Weights:")
print(f"CatBoost = {best_weights[0]:.4f}")
print(f"XGBoost  = {best_weights[1]:.4f}")
print(f"LightGBM = {best_weights[2]:.4f}")

blended_preds = (
    best_weights[0] * test_preds['CatBoost'] +
    best_weights[1] * test_preds['XGBoost'] +
    best_weights[2] * test_preds['LightGBM']
)
blended_preds = np.clip(blended_preds, 1, 314)

submission['Calories'] = blended_preds
submission.to_csv('submission.csv', index=False)


submission


from IPython.display import FileLink
print("Click on the link below to download your submission file:")
display(FileLink('submission.csv'))




