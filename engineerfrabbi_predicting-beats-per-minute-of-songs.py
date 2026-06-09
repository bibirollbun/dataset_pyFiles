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
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import optuna

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)


# Drop id
train.drop('id', axis=1, inplace=True)
test_ids = test['id']
test.drop('id', axis=1, inplace=True)


# Inspect
print(train.head())
print(train.describe())
print(train.isnull().sum())  # Check missing values - seems none


# Target
target = 'BeatsPerMinute'
y = train[target]
X = train.drop(target, axis=1)


# Fill missing if any (median for numeric)
X.fillna(X.median(), inplace=True)
test.fillna(test.median(), inplace=True)


# Bin target for stratified CV
bins = pd.qcut(y, q=10, labels=False)


# Target distribution
plt.figure(figsize=(10, 5))
sns.histplot(y, kde=True)
plt.title('Distribution of BeatsPerMinute')
plt.show()

print("Mean BPM:", y.mean())  # ~120 expected
print("Median BPM:", y.median())


# Feature histograms
X.hist(figsize=(15, 10), bins=30)
plt.suptitle('Feature Distributions')
plt.show()


# Correlations
corr = train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Boxplots for outliers
plt.figure(figsize=(15, 10))
sns.boxplot(data=X)
plt.title('Boxplots of Features')
plt.xticks(rotation=45)
plt.show()


# Handle skewness (log1p for positive skewed features)
skewed_feats = ['AudioLoudness', 'TrackDurationMs', 'InstrumentalScore', 'LivePerformanceLikelihood']
for feat in skewed_feats:
    if (X[feat] > 0).all():
        X[feat] = np.log1p(X[feat])
        test[feat] = np.log1p(test[feat])


# Clip outliers based on IQR
for col in X.columns:
    Q1 = X[col].quantile(0.01)
    Q3 = X[col].quantile(0.99)
    X[col] = np.clip(X[col], Q1, Q3)
    test[col] = np.clip(test[col], Q1, Q3)


# Interaction features
X['Energy_Loudness'] = X['Energy'] * X['AudioLoudness']
test['Energy_Loudness'] = test['Energy'] * test['AudioLoudness']

X['Energy_Acoustic'] = X['Energy'] / (X['AcousticQuality'] + 1e-6)
test['Energy_Acoustic'] = test['Energy'] / (test['AcousticQuality'] + 1e-6)

X['Mood_Energy'] = X['MoodScore'] * X['Energy']
test['Mood_Energy'] = test['MoodScore'] * test['Energy']

X['Duration_Energy'] = X['TrackDurationMs'] * X['Energy']
test['Duration_Energy'] = test['TrackDurationMs'] * test['Energy']


# Polynomial features (example: square)
X['RhythmScore_sq'] = X['RhythmScore'] ** 2
test['RhythmScore_sq'] = test['RhythmScore'] ** 2


# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
test_pca = pca.transform(test)


X['PCA1'] = X_pca[:, 0]
X['PCA2'] = X_pca[:, 1]
test['PCA1'] = test_pca[:, 0]
test['PCA2'] = test_pca[:, 1]


# Clustering as feature
kmeans = KMeans(n_clusters=5, random_state=42)
X['Cluster'] = kmeans.fit_predict(X)
test['Cluster'] = kmeans.predict(test)


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

X = pd.DataFrame(X_scaled, columns=X.columns)
test = pd.DataFrame(test_scaled, columns=test.columns)


# CV setup
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Function for OOF predictions
def get_oof_predictions(model, X, y, test, skf, bins):
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((len(test), n_folds))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, bins)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)
        
        test_preds[:, fold] = model.predict(test)
    
    return oof_preds, np.mean(test_preds, axis=1)


# Optuna tuning function
def tune_model(trial, model_type):
    if model_type == 'lgbm':
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        }
        model = LGBMRegressor(**params, random_state=42)
    elif model_type == 'xgb':
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        model = XGBRegressor(**params, random_state=42, enable_categorical=True)
    elif model_type == 'cat':
        params = {
            'loss_function': 'RMSE',
            'verbose': False,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        }
        model = CatBoostRegressor(**params, random_state=42)
    
    oof, _ = get_oof_predictions(model, X, y, test, skf, bins)
    rmse = np.sqrt(mean_squared_error(y, oof))
    return rmse


# Tune LGBM
study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(lambda trial: tune_model(trial, 'lgbm'), n_trials=20)
lgbm_params = study_lgbm.best_params
print("Best LGBM params:", lgbm_params)


# Tune XGB
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(lambda trial: tune_model(trial, 'xgb'), n_trials=20)
xgb_params = study_xgb.best_params
print("Best XGB params:", xgb_params)


# Tune CatBoost
study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(lambda trial: tune_model(trial, 'cat'), n_trials=20)
cat_params = study_cat.best_params
print("Best Cat params:", cat_params)


# Train final models with tuned params and seed averaging (3 seeds)
def train_with_seeds(model_cls, params, seeds=[42, 43, 44]):
    oof_all = []
    test_all = []
    for seed in seeds:
        model = model_cls(**params, random_state=seed)
        if 'lgbm' in model_cls.__name__.lower():
            model.set_params(verbosity=-1)
        elif 'cat' in model_cls.__name__.lower():
            model.set_params(verbose=False)
        oof, test_pred = get_oof_predictions(model, X, y, test, skf, bins)
        oof_all.append(oof)
        test_all.append(test_pred)
    return np.mean(oof_all, axis=0), np.mean(test_all, axis=0)

lgbm_oof, lgbm_test = train_with_seeds(LGBMRegressor, {**lgbm_params, 'objective': 'regression', 'metric': 'rmse'})
xgb_oof, xgb_test = train_with_seeds(XGBRegressor, {**xgb_params, 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'enable_categorical': True})
cat_oof, cat_test = train_with_seeds(CatBoostRegressor, {**cat_params, 'loss_function': 'RMSE'})

print("LGBM OOF RMSE:", np.sqrt(mean_squared_error(y, lgbm_oof)))
print("XGB OOF RMSE:", np.sqrt(mean_squared_error(y, xgb_oof)))
print("Cat OOF RMSE:", np.sqrt(mean_squared_error(y, cat_oof)))


# Simple blend (equal weights)
blend_oof = (lgbm_oof + xgb_oof + cat_oof) / 3
blend_test = (lgbm_test + xgb_test + cat_test) / 3
print("Blend OOF RMSE:", np.sqrt(mean_squared_error(y, blend_oof)))


# Optimize weights
def optimize_weights(trial):
    w1 = trial.suggest_float('w1', 0, 1)
    w2 = trial.suggest_float('w2', 0, 1)
    w3 = 1 - w1 - w2
    if w3 < 0:
        return 1e6
    oof = w1 * lgbm_oof + w2 * xgb_oof + w3 * cat_oof
    return np.sqrt(mean_squared_error(y, oof))

study_weights = optuna.create_study(direction='minimize')
study_weights.optimize(optimize_weights, n_trials=50)
w1 = study_weights.best_params['w1']
w2 = study_weights.best_params['w2']
w3 = 1 - w1 - w2
print("Best weights:", w1, w2, w3)

weighted_oof = w1 * lgbm_oof + w2 * xgb_oof + w3 * cat_oof
weighted_test = w1 * lgbm_test + w2 * xgb_test + w3 * cat_test
print("Weighted Blend OOF RMSE:", np.sqrt(mean_squared_error(y, weighted_oof)))


# Stacking as alternative
estimators = [
    ('lgbm', LGBMRegressor(**lgbm_params, random_state=42, verbosity=-1)),
    ('xgb', XGBRegressor(**xgb_params, random_state=42, enable_categorical=True)),
    ('cat', CatBoostRegressor(**cat_params, random_state=42, verbose=False))
]
stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=5)
stack_oof, stack_test = get_oof_predictions(stack, X, y, test, skf, bins)
print("Stack OOF RMSE:", np.sqrt(mean_squared_error(y, stack_oof)))


# Final ensemble: average blend and stack
final_oof = (weighted_oof + stack_oof) / 2
final_test = (weighted_test + stack_test) / 2
print("Final OOF RMSE:", np.sqrt(mean_squared_error(y, final_oof)))


# Clip to train min-max
min_bpm = y.min()
max_bpm = y.max()
final_test = np.clip(final_test, min_bpm, max_bpm)


submission['BeatsPerMinute'] = final_test
submission.to_csv('submission.csv', index=False)
print(submission.head())


# Pseudo-labeling example (optional)
conf_threshold = 5  # RMSE-like threshold
conf_mask = np.abs(lgbm_test - cat_test) < conf_threshold  # Agreement between models
pseudo_y = (lgbm_test[conf_mask] + cat_test[conf_mask]) / 2
pseudo_X = test[conf_mask]

# Append to train
X_pseudo = pd.concat([X, pseudo_X])
y_pseudo = pd.concat([y, pd.Series(pseudo_y)])

# Retrain final model (e.g., stack) on augmented data
# stack.fit(X_pseudo, y_pseudo)
# final_test = stack.predict(test)
# But skip for now to avoid overfitting risk.




