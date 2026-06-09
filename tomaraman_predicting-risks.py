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
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200)


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
origin = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
# train = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')


train.info()


BINARY_COLS=['road_signs_present','public_road','holiday','school_season']
for df in [train,test,origin]:
    df['BINARY']=0
    for i in range(len(BINARY_COLS)):
        df['BINARY']+=df[BINARY_COLS[i]].astype(int)*(2**i)



BINARY_COLS=['road_signs_present','public_road','holiday','school_season']
for df in [train,test,origin]:
    df['base_risk'] = (
        0.3 * df["curvature"] + 
        0.2 * (df["lighting"] == "night").astype(int) + 
        0.1 * (df["weather"] != "clear").astype(int) + 
        0.2 * (df["speed_limit"] >= 60).astype(int) + 
        0.1 * (np.array(df["num_reported_accidents"]) > 2).astype(int)
    )



for c in ['curvature','speed_limit']:
    for i in range(-3,3):
        train[c+f"_{i}"]=(train[c]*(10**i)%10).astype(np.int8)
        test[c+f"_{i}"]=(test[c]*(10**i)%10).astype(np.int8)
        origin[c+f"_{i}"]=(origin[c]*(10**i)%10).astype(np.int8)
        if train[c+f"_{i}"].nunique()==1:
            train.drop([c+f"_{i}"],axis=1,inplace=True)
            test.drop([c+f"_{i}"],axis=1,inplace=True)
            origin.drop([c+f"_{i}"],axis=1,inplace=True)



FEATURES=[c for c in test.columns if c not in ['id']+BINARY_COLS]
print(FEATURES)
target_col = 'accident_risk'
aggs = ['mean','max','min','nunique','count']

for c in FEATURES:
    tmp = (origin.groupby(c)[target_col]           
                .agg(aggs)                         
                .rename(columns=lambda a: f'{c}_{target_col}_{a}')  
                .reset_index())
    train = train.merge(tmp, on=c, how='left')
    test  = test.merge(tmp, on=c, how='left')

train.head()



train.info()


# road_type_dt = {'urban' : 1, 'rural' : 0, 'highway' : 2}
# lighting_dt = {'daylight' : 0, 'dim' : 1, 'night' : 2}
# weather_dt = {'rainy' : 1, 'foggy' : 2, 'clear' : 0}
# time_of_day_dt = {'afternoon' : 0, 'morning' : 1, 'evening' : 2}


# train['road_type_v'] = train['road_type'].map(road_type_dt)
# train['lighting_v'] = train['lighting'].map(lighting_dt)
# train['weather_v'] = train['weather'].map(weather_dt)
# train['time_of_day_dt'] = train['time_of_day'].map(time_of_day_dt)


train.drop(['road_type', 'lighting', 'weather', 'time_of_day'], axis = 1, inplace = True)
test.drop(['road_type', 'lighting', 'weather', 'time_of_day'], axis = 1, inplace = True)


train.sample(5)


train.drop(['id'], axis = 1, inplace = True)


# Basic statistics
train.describe()

# Check for missing values
train.isnull().sum()

# Look at target distribution
plt.hist(train['accident_risk'].dropna(), bins=100)
plt.title('Target Distribution')
plt.show()

# Correlation with target
correlations = train.corr()['accident_risk'].sort_values(ascending=False)
print(correlations)


features = train.columns
features = list(features)
features.remove('accident_risk')


features


# features = ['num_lanes', 'curvature', 'speed_limit', 'road_signs_present',
#        'public_road', 'holiday', 'school_season', 'num_reported_accidents', 'road_type_v', 'lighting_v', 'weather_v',
#        'time_of_day_dt']


X = train[features]
y = train['accident_risk']

X_train = X.iloc[:500000]
y_train = y.iloc[:500000]
X_val = X.iloc[500000:]
y_val = y.iloc[500000:]


# from sklearn.model_selection import train_test_split
# import pandas as pd

# X = train[features]
# y = train['accident_risk']

# total_size = len(X) # e.g., 600,000
# train_size_count = 500000
# train_ratio = train_size_count / total_size

# # 2. Perform the random split
# X_train, X_val, y_train, y_val = train_test_split(
#     X,
#     y,
#     train_size=train_ratio, # or use test_size=1-train_ratio
#     random_state=42,        # Set a seed for reproducible results
#     shuffle=True            # This is True by default, ensures the random split
# )

# # 3. Verification (Optional)
# print(f"Total samples: {total_size}")
# print(f"X_train size: {len(X_train)}") # Should be close to 500,000
# print(f"X_val size: {len(X_val)}")     # Should be the rest


# 20844
# # LightGBM model (same as before)
# model = LGBMRegressor(
#     n_estimators=1000,
#     learning_rate=0.05,
#     max_depth=8,
#     num_leaves=512,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     random_state=42
# )

# 0.2084251466558015
# model = LGBMRegressor(
#     n_estimators=10000,
#     learning_rate=0.01,
#     max_depth=8,
#     num_leaves=512,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     random_state=42
# )

# #best score on this till now
# model = LGBMRegressor(
#     n_estimators=1000,
#     learning_rate=0.05,
#     max_depth=8,
#     num_leaves=512,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     random_state=42
# )


# from lightgbm import early_stopping, log_evaluation
# from lightgbm import LGBMRegressor
# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.metrics import mean_squared_error, make_scorer
# import numpy as np

# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = []

# for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
#     print(f'\n--- Fold {fold} ---')
    
#     X_train_fold = X.iloc[train_idx]
#     y_train_fold = y.iloc[train_idx]
#     X_val_fold = X.iloc[val_idx]
#     y_val_fold = y.iloc[val_idx]
    
#     # Create a fresh model for each fold
#     # model = LGBMRegressor(
#     #     n_estimators=50000,
#     #     learning_rate=0.05,
#     #     max_depth=8,
#     #     num_leaves=255,
#     #     subsample=0.9,
#     #     subsample_freq=1,
#     #     colsample_bytree=0.9,
#     #     reg_alpha=0.01,
#     #     reg_lambda=0.01,
#     #     min_child_samples=20,
#     #     min_split_gain=0.01,
#     #     n_jobs=-1,
#     #     random_state=42,
#     #     objective='rmse',
#     #     metric='rmse',
#     #     verbose=-1
#     # )
#     model = LGBMRegressor(
#         n_estimators=1000,
#         learning_rate=0.05,
#         max_depth=8,
#         num_leaves=512,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         random_state=42
#     )
    
#     # Train with early stopping
#     es_callback = early_stopping(stopping_rounds=50, verbose=False)
#     log_callback = log_evaluation(period=100)
    
#     model.fit(
#         X_train_fold, y_train_fold,
#         eval_set=[(X_val_fold, y_val_fold)],
#         callbacks=[es_callback, log_callback]
#     )
    
#     # Predict and evaluate
#     y_pred_fold = model.predict(X_val_fold)
#     rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
#     cv_scores.append(rmse_fold)
    
#     print(f'Fold {fold} RMSE: {rmse_fold:.6f}')

# cv_scores = np.array(cv_scores)
# print(f'\n=== Cross-Validation Results ===')
# print(f'All fold scores: {cv_scores}')
# print(f'Mean CV RMSE: {cv_scores.mean():.6f} (+/- {cv_scores.std():.6f})')


# print('\n=== Approach 1: Averaging predictions from CV folds ===')

# # Store models from each fold
# fold_models = []
# kfold = KFold(n_splits=3, shuffle=True, random_state=42)
# cv_scores = []

# for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
#     print(f'\n--- Fold {fold} ---')
    
#     X_train_fold = X.iloc[train_idx]
#     y_train_fold = y.iloc[train_idx]
#     X_val_fold = X.iloc[val_idx]
#     y_val_fold = y.iloc[val_idx]
    
#     # model = LGBMRegressor(
#     #     n_estimators=1000,
#     #     learning_rate=0.05,
#     #     max_depth=8,
#     #     num_leaves=255,
#     #     subsample=0.9,
#     #     subsample_freq=1,
#     #     colsample_bytree=0.9,
#     #     reg_alpha=0.01,
#     #     reg_lambda=0.01,
#     #     min_child_samples=20,
#     #     min_split_gain=0.01,
#     #     n_jobs=-1,
#     #     random_state=42,
#     #     objective='rmse',
#     #     metric='rmse',
#     #     verbose=-1
#     # )
#     model = LGBMRegressor(
#         n_estimators=1000,
#         learning_rate=0.05,
#         max_depth=8,
#         num_leaves=511,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         random_state=42,
#         objective='huber',
#     )
    
#     es_callback = early_stopping(stopping_rounds=50, verbose=False)
#     log_callback = log_evaluation(period=100)
    
#     model.fit(
#         X_train_fold, y_train_fold,
#         eval_set=[(X_val_fold, y_val_fold)],
#         callbacks=[es_callback, log_callback]
#     )
    
#     # Store the trained model
#     fold_models.append(model)
    
#     y_pred_fold = model.predict(X_val_fold)
#     rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
#     cv_scores.append(rmse_fold)
    
#     print(f'Fold {fold} RMSE: {rmse_fold:.6f}')

# cv_scores = np.array(cv_scores)
# print(f'\n=== Cross-Validation Results ===')
# print(f'Mean CV RMSE: {cv_scores.mean():.6f} (+/- {cv_scores.std():.6f})')


# # Now predict on test set using all fold models
# X_test = test
# test_predictions = np.zeros(len(X_test))

# for i, model in enumerate(fold_models, 1):
#     fold_pred = model.predict(X_test)
#     test_predictions += fold_pred
#     print(f'Fold {i} predictions added')

# # Average the predictions
# test_predictions = test_predictions / len(fold_models)
# print(f'\nFinal test predictions shape: {test_predictions.shape}')


# from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import pandas as pd

# # Define your features and target
# X = train[features]
# y = train['accident_risk']

# # ========================================================
# # K-FOLD CROSS-VALIDATION WITH 3 MODELS
# # ========================================================

# kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# # Storage for models and scores
# lgbm_models = []
# xgb_models = []
# catboost_models = []

# lgbm_scores = []
# xgb_scores = []
# catboost_scores = []

# for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
#     print(f'\n{"="*60}')
#     print(f'FOLD {fold}')
#     print(f'{"="*60}')
    
#     X_train_fold = X.iloc[train_idx]
#     y_train_fold = y.iloc[train_idx]
#     X_val_fold = X.iloc[val_idx]
#     y_val_fold = y.iloc[val_idx]
    
#     # -------------------- LightGBM --------------------
#     print('\n[LightGBM] Training...')
#     lgbm = LGBMRegressor(
#         n_estimators=1000,
#         learning_rate=0.05,
#         max_depth=8,
#         num_leaves=255,
#         subsample=0.9,
#         subsample_freq=1,
#         colsample_bytree=0.9,
#         reg_alpha=0.01,
#         reg_lambda=0.01,
#         min_child_samples=20,
#         min_split_gain=0.01,
#         n_jobs=-1,
#         random_state=42,
#         objective='rmse',
#         metric='rmse',
#         verbose=-1
#     )
    
#     es_callback = early_stopping(stopping_rounds=50, verbose=False)
#     log_callback = log_evaluation(period=500)
    
#     lgbm.fit(
#         X_train_fold, y_train_fold,
#         eval_set=[(X_val_fold, y_val_fold)],
#         callbacks=[es_callback, log_callback]
#     )
    
#     lgbm_pred = lgbm.predict(X_val_fold)
#     lgbm_rmse = np.sqrt(mean_squared_error(y_val_fold, lgbm_pred))
#     lgbm_scores.append(lgbm_rmse)
#     lgbm_models.append(lgbm)
#     print(f'[LightGBM] Fold {fold} RMSE: {lgbm_rmse:.6f}')
    
#     # -------------------- XGBoost --------------------
#     print('\n[XGBoost] Training...')
#     xgb = XGBRegressor(
#         n_estimators=1000,
#         learning_rate=0.05,
#         max_depth=8,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         reg_alpha=0.01,
#         reg_lambda=0.01,
#         min_child_weight=20,
#         gamma=0.01,
#         n_jobs=-1,
#         random_state=42,
#         objective='reg:squarederror',
#         eval_metric='rmse',
#         tree_method='hist',
#         early_stopping_rounds=50,
#         verbosity=0
#     )
    
#     xgb.fit(
#         X_train_fold, y_train_fold,
#         eval_set=[(X_val_fold, y_val_fold)],
#         verbose=500
#     )
    
#     xgb_pred = xgb.predict(X_val_fold)
#     xgb_rmse = np.sqrt(mean_squared_error(y_val_fold, xgb_pred))
#     xgb_scores.append(xgb_rmse)
#     xgb_models.append(xgb)
#     print(f'[XGBoost] Fold {fold} RMSE: {xgb_rmse:.6f}')
    
#     # -------------------- CatBoost --------------------
#     print('\n[CatBoost] Training...')
#     catboost = CatBoostRegressor(
#         iterations=1000,
#         learning_rate=0.05,
#         depth=8,
#         subsample=0.9,
#         colsample_bylevel=0.9,
#         reg_lambda=0.01,
#         min_child_samples=20,
#         random_state=42,
#         loss_function='RMSE',
#         eval_metric='RMSE',
#         early_stopping_rounds=50,
#         verbose=500,
#         task_type='CPU',
#         thread_count=-1
#     )
    
#     catboost.fit(
#         X_train_fold, y_train_fold,
#         eval_set=(X_val_fold, y_val_fold),
#         verbose=500
#     )
    
#     catboost_pred = catboost.predict(X_val_fold)
#     catboost_rmse = np.sqrt(mean_squared_error(y_val_fold, catboost_pred))
#     catboost_scores.append(catboost_rmse)
#     catboost_models.append(catboost)
#     print(f'[CatBoost] Fold {fold} RMSE: {catboost_rmse:.6f}')
    
#     # -------------------- Ensemble (40/30/30) --------------------
#     ensemble_pred = (0.4 * catboost_pred + 
#                      0.3 * xgb_pred + 
#                      0.3 * lgbm_pred)
#     ensemble_rmse = np.sqrt(mean_squared_error(y_val_fold, ensemble_pred))
#     print(f'\n[ENSEMBLE] Fold {fold} RMSE: {ensemble_rmse:.6f}')

# # ========================================================
# # CROSS-VALIDATION SUMMARY
# # ========================================================
# print(f'\n{"="*60}')
# print('CROSS-VALIDATION SUMMARY')
# print(f'{"="*60}')

# lgbm_scores = np.array(lgbm_scores)
# xgb_scores = np.array(xgb_scores)
# catboost_scores = np.array(catboost_scores)

# print(f'\n[LightGBM] CV RMSE: {lgbm_scores.mean():.6f} (+/- {lgbm_scores.std():.6f})')
# print(f'[XGBoost]  CV RMSE: {xgb_scores.mean():.6f} (+/- {xgb_scores.std():.6f})')
# print(f'[CatBoost] CV RMSE: {catboost_scores.mean():.6f} (+/- {catboost_scores.std():.6f})')

# # ========================================================
# # TEST PREDICTIONS - ENSEMBLE AVERAGING
# # ========================================================
# print(f'\n{"="*60}')
# print('GENERATING TEST PREDICTIONS')
# print(f'{"="*60}')

# X_test = test

# # Initialize prediction arrays
# lgbm_test_preds = np.zeros(len(X_test))
# xgb_test_preds = np.zeros(len(X_test))
# catboost_test_preds = np.zeros(len(X_test))

# # Get predictions from each fold
# for i, (lgbm_model, xgb_model, catboost_model) in enumerate(zip(lgbm_models, xgb_models, catboost_models), 1):
#     print(f'Predicting with Fold {i} models...')
    
#     lgbm_test_preds += lgbm_model.predict(X_test)
#     xgb_test_preds += xgb_model.predict(X_test)
#     catboost_test_preds += catboost_model.predict(X_test)

# # Average predictions across folds
# lgbm_test_preds /= len(lgbm_models)
# xgb_test_preds /= len(xgb_models)
# catboost_test_preds /= len(catboost_models)

# # Blend with 40/30/30 weights
# final_test_predictions = (0.4 * catboost_test_preds + 
#                           0.3 * xgb_test_preds + 
#                           0.3 * lgbm_test_preds)

# print(f'\nFinal blended predictions shape: {final_test_predictions.shape}')

# # ========================================================
# # CREATE SUBMISSION FILE
# # # ========================================================
# # submission = pd.DataFrame({
# #     'id': test['id'],
# #     'accident_risk': final_test_predictions
# # })

# # submission.to_csv('submission_ensemble_40_30_30.csv', index=False)
# # print(f'\n✓ Submission file created: submission_ensemble_40_30_30.csv')

# # ========================================================
# # OPTIONAL: Save individual model predictions
# # ========================================================
# # In case you want to experiment with different weights later
# # individual_preds = pd.DataFrame({
# #     'id': test['id'],
# #     'catboost': catboost_test_preds,
# #     'xgboost': xgb_test_preds,
# #     'lightgbm': lgbm_test_preds,
# #     'ensemble_40_30_30': final_test_predictions
# # })

# # individual_preds.to_csv('individual_predictions.csv', index=False)
# # print('✓ Individual predictions saved: individual_predictions.csv')

# # print(f'\n{"="*60}')
# # print('COMPLETED!')
# # print(f'{"="*60}')


# test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

# test['road_type_v'] = test['road_type'].map(road_type_dt)
# test['lighting_v'] = test['lighting'].map(lighting_dt)
# test['weather_v'] = test['weather'].map(weather_dt)
# test['time_of_day_dt'] = test['time_of_day'].map(time_of_day_dt)
# test.drop(['road_type', 'lighting', 'weather', 'time_of_day'], axis = 1, inplace = True)

subm = test.copy()
subm.drop(features, axis = 1, inplace = True)

test.drop(['id'], axis = 1, inplace = True)


from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import torch

# ========================================================
# GPU SETUP
# ========================================================

# Check GPU availability
gpu_available = torch.cuda.is_available()
print(f"GPU Available: {gpu_available}")
if gpu_available:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Define your features and target
X = train[features]
y = train['accident_risk']

# ========================================================
# OPTUNA OBJECTIVE FUNCTIONS (GPU-OPTIMIZED)
# ========================================================

def objective_lgbm(trial, X, y, n_splits=3):
    """Objective function for LightGBM hyperparameter tuning"""
    
    # GPU-optimized parameters
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'num_leaves': trial.suggest_int('num_leaves', 31, 511),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'subsample_freq': 1,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.001, 0.1, log=True),
        'random_state': 42,
        'objective': 'rmse',
        'metric': 'rmse',
        'verbose': -1,
        'device': 'gpu' if gpu_available else 'cpu',
        'gpu_use_dp': False,  # Disable double precision for speed
    }
    
    # Cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kfold.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        model = LGBMRegressor(**params)
        
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, pred))
        scores.append(rmse)
    
    return np.mean(scores)


def objective_xgb(trial, X, y, n_splits=3):
    """Objective function for XGBoost hyperparameter tuning"""
    
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
        'random_state': 42,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist' if gpu_available else 'hist',  # GPU-accelerated
        'predictor': 'gpu_predictor' if gpu_available else 'auto',  # GPU prediction
        'gpu_id': 0 if gpu_available else None,
        'early_stopping_rounds': 50,
        'verbosity': 0
    }
    
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kfold.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        model = XGBRegressor(**params)
        
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, pred))
        scores.append(rmse)
    
    return np.mean(scores)


def objective_catboost(trial, X, y, n_splits=3):
    """Objective function for CatBoost hyperparameter tuning"""
    
    params = {
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
        'random_state': 42,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 50,
        'verbose': False,
        'task_type': 'GPU' if gpu_available else 'CPU',  # GPU acceleration
        'devices': '0' if gpu_available else None,
        'bootstrap_type': 'Bernoulli',  # Required to use subsample parameter
    }
    
    # colsample_bylevel (rsm) only supported on CPU or pairwise modes on GPU
    if not gpu_available:
        params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.6, 1.0)
    
    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kfold.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            verbose=False
        )
        
        pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, pred))
        scores.append(rmse)
    
    return np.mean(scores)


def objective_ensemble_weights(trial, X, y, lgbm_params, xgb_params, catboost_params, n_splits=5):
    """Objective function for optimizing ensemble weights"""
    
    # Suggest weights (they will be normalized)
    w_lgbm = trial.suggest_float('w_lgbm', 0.0, 1.0)
    w_xgb = trial.suggest_float('w_xgb', 0.0, 1.0)
    w_catboost = trial.suggest_float('w_catboost', 0.0, 1.0)
    
    # Normalize weights
    total = w_lgbm + w_xgb + w_catboost
    w_lgbm /= total
    w_xgb /= total
    w_catboost /= total
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kfold.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train all three models
        lgbm = LGBMRegressor(**lgbm_params)
        lgbm.fit(X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
        
        xgb = XGBRegressor(**xgb_params)
        xgb.fit(X_train_fold, y_train_fold,
               eval_set=[(X_val_fold, y_val_fold)],
               verbose=False)
        
        catboost = CatBoostRegressor(**catboost_params)
        catboost.fit(X_train_fold, y_train_fold,
                    eval_set=(X_val_fold, y_val_fold),
                    verbose=False)
        
        # Weighted ensemble prediction
        ensemble_pred = (w_lgbm * lgbm.predict(X_val_fold) +
                        w_xgb * xgb.predict(X_val_fold) +
                        w_catboost * catboost.predict(X_val_fold))
        
        rmse = np.sqrt(mean_squared_error(y_val_fold, ensemble_pred))
        scores.append(rmse)
    
    return np.mean(scores)


# ========================================================
# STEP 1: OPTIMIZE EACH MODEL SEPARATELY (PARALLEL)
# ========================================================

print("="*60)
print("STEP 1: OPTIMIZING INDIVIDUAL MODELS (GPU ACCELERATED)")
print("="*60)

# Optimize LightGBM
print("\n[1/3] Optimizing LightGBM...")
study_lgbm = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    study_name='lgbm_optimization',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # Early stopping for trials
)
study_lgbm.optimize(
    lambda trial: objective_lgbm(trial, X, y, n_splits=3),
    n_trials=100,  # Increased for GPU speed
    show_progress_bar=True,
    n_jobs=1  # Set to 1 for GPU (parallel jobs cause issues with GPU)
)

print(f"\nBest LightGBM RMSE: {study_lgbm.best_value:.6f}")
print("Best LightGBM parameters:")
for key, value in study_lgbm.best_params.items():
    print(f"  {key}: {value}")

# Optimize XGBoost
print("\n[2/3] Optimizing XGBoost...")
study_xgb = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    study_name='xgb_optimization',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)
study_xgb.optimize(
    lambda trial: objective_xgb(trial, X, y, n_splits=3),
    n_trials=100,
    show_progress_bar=True,
    n_jobs=1
)

print(f"\nBest XGBoost RMSE: {study_xgb.best_value:.6f}")
print("Best XGBoost parameters:")
for key, value in study_xgb.best_params.items():
    print(f"  {key}: {value}")

# Optimize CatBoost
print("\n[3/3] Optimizing CatBoost...")
study_catboost = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    study_name='catboost_optimization',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)
study_catboost.optimize(
    lambda trial: objective_catboost(trial, X, y, n_splits=3),
    n_trials=100,
    show_progress_bar=True,
    n_jobs=1
)

print(f"\nBest CatBoost RMSE: {study_catboost.best_value:.6f}")
print("Best CatBoost parameters:")
for key, value in study_catboost.best_params.items():
    print(f"  {key}: {value}")

# ========================================================
# STEP 2: OPTIMIZE ENSEMBLE WEIGHTS
# ========================================================

print("\n" + "="*60)
print("STEP 2: OPTIMIZING ENSEMBLE WEIGHTS")
print("="*60)

# Prepare best parameters with fixed values
best_lgbm_params = {
    **study_lgbm.best_params,
    'n_estimators': 1000,
    'random_state': 42,
    'objective': 'rmse',
    'metric': 'rmse',
    'verbose': -1,
    'device': 'gpu' if gpu_available else 'cpu',
    'gpu_use_dp': False,
}

best_xgb_params = {
    **study_xgb.best_params,
    'n_estimators': 1000,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'gpu_hist' if gpu_available else 'hist',
    'predictor': 'gpu_predictor' if gpu_available else 'auto',
    'early_stopping_rounds': 50,
    'verbosity': 0
}

if gpu_available:
    best_xgb_params['gpu_id'] = 0

best_catboost_params = {
    **study_catboost.best_params,
    'iterations': 1000,
    'random_state': 42,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 50,
    'verbose': False,
    'task_type': 'GPU' if gpu_available else 'CPU',
    'bootstrap_type': 'Bernoulli',  # Required for subsample
}

# colsample_bylevel not supported on GPU
if 'colsample_bylevel' in best_catboost_params and gpu_available:
    del best_catboost_params['colsample_bylevel']

if gpu_available:
    best_catboost_params['devices'] = '0'

# Optimize ensemble weights
study_weights = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42),
    study_name='ensemble_weights'
)
study_weights.optimize(
    lambda trial: objective_ensemble_weights(
        trial, X, y, best_lgbm_params, best_xgb_params, best_catboost_params, n_splits=5
    ),
    n_trials=50,  # Increased for better weight optimization
    show_progress_bar=True,
    n_jobs=1
)

# Get normalized best weights
best_w = study_weights.best_params
total_w = best_w['w_lgbm'] + best_w['w_xgb'] + best_w['w_catboost']
final_weights = {
    'lgbm': best_w['w_lgbm'] / total_w,
    'xgb': best_w['w_xgb'] / total_w,
    'catboost': best_w['w_catboost'] / total_w
}

print(f"\nBest Ensemble RMSE: {study_weights.best_value:.6f}")
print("Best ensemble weights:")
for model, weight in final_weights.items():
    print(f"  {model}: {weight:.4f}")

# ========================================================
# STEP 3: FINAL TRAINING WITH OPTIMIZED PARAMETERS
# ========================================================

print("\n" + "="*60)
print("STEP 3: FINAL TRAINING WITH OPTIMIZED PARAMETERS")
print("="*60)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

lgbm_models = []
xgb_models = []
catboost_models = []
ensemble_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
    print(f'\nFold {fold}/5')
    
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]
    
    # Train LightGBM
    lgbm = LGBMRegressor(**best_lgbm_params)
    lgbm.fit(X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
    lgbm_pred = lgbm.predict(X_val_fold)
    lgbm_models.append(lgbm)
    
    # Train XGBoost
    xgb = XGBRegressor(**best_xgb_params)
    xgb.fit(X_train_fold, y_train_fold,
           eval_set=[(X_val_fold, y_val_fold)],
           verbose=False)
    xgb_pred = xgb.predict(X_val_fold)
    xgb_models.append(xgb)
    
    # Train CatBoost
    catboost = CatBoostRegressor(**best_catboost_params)
    catboost.fit(X_train_fold, y_train_fold,
                eval_set=(X_val_fold, y_val_fold),
                verbose=False)
    catboost_pred = catboost.predict(X_val_fold)
    catboost_models.append(catboost)
    
    # Ensemble prediction with optimized weights
    ensemble_pred = (final_weights['lgbm'] * lgbm_pred +
                    final_weights['xgb'] * xgb_pred +
                    final_weights['catboost'] * catboost_pred)
    
    ensemble_rmse = np.sqrt(mean_squared_error(y_val_fold, ensemble_pred))
    ensemble_scores.append(ensemble_rmse)
    print(f'  Ensemble RMSE: {ensemble_rmse:.6f}')

print(f'\nFinal CV RMSE: {np.mean(ensemble_scores):.6f} (+/- {np.std(ensemble_scores):.6f})')

# ========================================================
# STEP 4: GENERATE TEST PREDICTIONS
# ========================================================

print("\n" + "="*60)
print("GENERATING TEST PREDICTIONS")
print("="*60)

X_test = test

lgbm_test_preds = np.zeros(len(X_test))
xgb_test_preds = np.zeros(len(X_test))
catboost_test_preds = np.zeros(len(X_test))

for i, (lgbm_model, xgb_model, catboost_model) in enumerate(zip(lgbm_models, xgb_models, catboost_models), 1):
    lgbm_test_preds += lgbm_model.predict(X_test)
    xgb_test_preds += xgb_model.predict(X_test)
    catboost_test_preds += catboost_model.predict(X_test)

lgbm_test_preds /= len(lgbm_models)
xgb_test_preds /= len(xgb_models)
catboost_test_preds /= len(catboost_models)

final_test_predictions = (final_weights['lgbm'] * lgbm_test_preds +
                         final_weights['xgb'] * xgb_test_preds +
                         final_weights['catboost'] * catboost_test_preds)

print(f'\nFinal predictions shape: {final_test_predictions.shape}')
print("\n✓ Optimization and training completed!")

# ========================================================
# SAVE RESULTS
# ========================================================

# Save optimized parameters to file
import json

optimization_results = {
    'gpu_used': gpu_available,
    'lgbm_params': {k: (float(v) if isinstance(v, np.floating) else v) 
                    for k, v in best_lgbm_params.items()},
    'xgb_params': {k: (float(v) if isinstance(v, np.floating) else v) 
                   for k, v in best_xgb_params.items()},
    'catboost_params': {k: (float(v) if isinstance(v, np.floating) else v) 
                        for k, v in best_catboost_params.items()},
    'ensemble_weights': final_weights,
    'cv_rmse': float(np.mean(ensemble_scores)),
    'cv_std': float(np.std(ensemble_scores)),
    'optimization_trials': {
        'lgbm': len(study_lgbm.trials),
        'xgb': len(study_xgb.trials),
        'catboost': len(study_catboost.trials),
        'ensemble_weights': len(study_weights.trials)
    }
}

with open('optuna_optimization_results.json', 'w') as f:
    json.dump(optimization_results, f, indent=2)

print("\n✓ Optimization results saved to: optuna_optimization_results.json")

# Optional: Create submission file
# submission = pd.DataFrame({
#     'id': test['id'],
#     'accident_risk': final_test_predictions
# })
# submission.to_csv('submission_optuna_optimized.csv', index=False)
# print("✓ Submission file saved: submission_optuna_optimized.csv")


# from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# from sklearn.metrics import mean_squared_error

# # LightGBM model (same as before)
# # model = LGBMRegressor(
# #     n_estimators=2000,
# #     learning_rate=0.05,
# #     max_depth=8,
# #     num_leaves=255,
# #     subsample=0.9,
# #     colsample_bytree=0.9,
# #     random_state=42,
# #     metric='mse'
# # )

# model = LGBMRegressor(
#     n_estimators=50000,
#     learning_rate=0.05,
#     max_depth=8,
#     num_leaves=255,
#     subsample=0.9,
#     subsample_freq=1,
#     colsample_bytree=0.9,
#     reg_alpha=0.01,
#     reg_lambda=0.01,
#     min_child_samples=20,
#     min_split_gain=0.01,
#     n_jobs=-1,
#     random_state=42,
#     objective='rmse',
#     metric='rmse',
#     verbose=-1
# )

# #best score on this till now
# # model = LGBMRegressor(
# #     n_estimators=5000,
# #     learning_rate=0.05,
# #     max_depth=8,
# #     num_leaves=512,
# #     subsample=0.9,
# #     colsample_bytree=0.9,
# #     random_state=42
# # )

# # Define callbacks
# # early_stopping: stops training if the metric on the eval_set doesn't improve for 50 rounds
# es_callback = early_stopping(stopping_rounds=50, verbose=False)
# # log_evaluation: prints evaluation results every 100 boosting rounds
# log_callback = log_evaluation(period=100)

# # Train using the 'callbacks' parameter
# model.fit(
#     X_train, y_train,
#     eval_set=[(X_val, y_val)],
#     callbacks=[es_callback, log_callback] # Pass callbacks here
#     # early_stopping_rounds=50, # <--- REMOVE THESE
#     # verbose=100                 # <--- REMOVE THESE
# )

# # Predict
# y_pred = model.predict(X_val)

# # Evaluate (competition uses Mean Absolute Error)
# rmse = np.sqrt(mean_squared_error(y_val, y_pred))
# print(f'Validation RMSE: {rmse}')


# from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# from sklearn.metrics import mean_absolute_error
# import numpy as np

# # Improved LightGBM model with better hyperparameters
# model = LGBMRegressor(
#     # Boosting parameters
#     n_estimators=2000,              # Increased from 1000 (early stopping will prevent overfitting)
#     learning_rate=0.1,             # Lower learning rate for better generalization
#     max_depth=6,                    # Reduced from 8 to prevent overfitting
#     num_leaves=31,                  # Reduced from 512 (rule of thumb: 2^max_depth - 1)
    
#     # Sampling parameters
#     subsample=0.8,                  # Reduced from 0.9
#     subsample_freq=1,               # Apply subsample at every iteration
#     colsample_bytree=0.8,           # Reduced from 0.9
    
#     # Regularization
#     reg_alpha=0.1,                  # L1 regularization
#     reg_lambda=0.1,                 # L2 regularization
#     min_child_samples=20,           # Minimum data in leaf
#     min_split_gain=0.01,            # Minimum gain to make split
    
#     # Performance
#     n_jobs=-1,                      # Use all CPU cores
#     random_state=42,
    
#     # Additional improvements
#     boosting_type='gbdt',           # Can try 'dart' or 'goss' for variety
#     objective='mae',                # Since you're optimizing MAE
#     metric='mae',                   # Evaluation metric
#     verbose=-1                      # Suppress warnings
# )

# # Enhanced callbacks with better monitoring
# es_callback = early_stopping(stopping_rounds=100, verbose=True)  # Increased patience
# log_callback = log_evaluation(period=50)  # More frequent logging

# # Train with validation
# model.fit(
#     X_train, y_train,
#     eval_set=[(X_train, y_train), (X_val, y_val)],  # Monitor both train & val
#     eval_names=['train', 'valid'],
#     callbacks=[es_callback, log_callback]
# )

# # Predictions
# y_pred = model.predict(X_val)
# y_train_pred = model.predict(X_train)

# # Comprehensive evaluation
# mae_val = mean_absolute_error(y_val, y_pred)
# mae_train = mean_absolute_error(y_train, y_train_pred)
# rmae_val = np.sqrt(mae_val)
# rmae_train = np.sqrt(mae_train)

# print(f'\n{"="*50}')
# print(f'Training MAE: {mae_train:.6f} | RMAE: {rmae_train:.6f}')
# print(f'Validation MAE: {mae_val:.6f} | RMAE: {rmae_val:.6f}')
# print(f'Overfit ratio (train/val): {mae_train/mae_val:.3f}')
# print(f'Best iteration: {model.best_iteration_}')
# print(f'{"="*50}\n')

# # Feature importance analysis
# feature_importance = model.feature_importances_
# if hasattr(X_train, 'columns'):
#     feature_names = X_train.columns
#     importance_df = pd.DataFrame({
#         'feature': feature_names,
#         'importance': feature_importance
#     }).sort_values('importance', ascending=False)
    
#     print("Top 10 Most Important Features:")
#     print(importance_df.head(10))


# from catboost import CatBoostRegressor
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # CatBoost model
# model = CatBoostRegressor(
#     iterations=2000,
#     learning_rate=0.05,
#     depth=8,
#     subsample=0.9,
#     colsample_bylevel=0.9,  # CatBoost uses colsample_bylevel instead of colsample_bytree
#     random_state=42,
#     verbose=100,  # Print every 100 iterations
#     early_stopping_rounds=50,
#     eval_metric='RMSE'  # Since you're using MAE for evaluation
# )

# # Train the model
# # CatBoost uses eval_set parameter similar to LightGBM
# model.fit(
#     X_train, y_train,
#     eval_set=(X_val, y_val),  # Note: single tuple, not list of tuples
#     use_best_model=True  # Use the best model from early stopping
# )

# # Predict
# y_pred = model.predict(X_val)

# # Evaluate
# rmse = np.sqrt(mean_squared_error(y_val, y_pred))
# print(f'Validation RMAE: {rmse}')


# from lightgbm import LGBMRegressor, early_stopping, log_evaluation
# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import numpy as np
# import pandas as pd

# # ===============================================
# # 1. TRAIN LIGHTGBM MODEL
# # ===============================================
# print("="*60)
# print("Training LightGBM Model...")
# print("="*60)

# lgbm = LGBMRegressor(
#     n_estimators=2000,
#     learning_rate=0.03,
#     max_depth=6,
#     num_leaves=31,
#     subsample=0.8,
#     subsample_freq=1,
#     colsample_bytree=0.8,
#     reg_alpha=0.1,
#     reg_lambda=0.1,
#     min_child_samples=20,
#     min_split_gain=0.01,
#     n_jobs=-1,
#     random_state=42,
#     objective='regression',
#     metric='rmse',
#     verbose=-1
# )

# # Callbacks for early stopping and logging
# es_callback = early_stopping(stopping_rounds=100, verbose=True)
# log_callback = log_evaluation(period=100)

# # Train LightGBM
# lgbm.fit(
#     X_train, y_train,
#     eval_set=[(X_train, y_train), (X_val, y_val)],
#     eval_names=['train', 'valid'],
#     callbacks=[es_callback, log_callback]
# )

# print(f"\nLightGBM training completed.")
# print(f"Best iteration: {lgbm.best_iteration_}")
# print(f"Best score: {lgbm.best_score_['valid']['rmse']:.6f}")

# # ===============================================
# # 2. TRAIN EXTRATREES MODEL (ENSEMBLE COMPONENT)
# # ===============================================
# print("\n" + "="*60)
# print("Training ExtraTrees Model...")
# print("="*60)

# et = ExtraTreesRegressor(
#     n_estimators=400,
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     max_features='sqrt',
#     n_jobs=-1,
#     random_state=42,
#     verbose=1
# )

# et.fit(X_train, y_train)
# print(f"ExtraTrees training completed.")

# # ===============================================
# # 3. GET VALIDATION PREDICTIONS FROM BOTH MODELS
# # ===============================================
# print("\n" + "="*60)
# print("Getting validation predictions...")
# print("="*60)

# lgbm_val_pred = lgbm.predict(X_val)
# et_val_pred = et.predict(X_val)

# # Individual model performance
# lgbm_rmse = np.sqrt(mean_squared_error(y_val, lgbm_val_pred))
# lgbm_mae = mean_absolute_error(y_val, lgbm_val_pred)
# lgbm_rmae = np.sqrt(lgbm_mae)

# et_rmse = np.sqrt(mean_squared_error(y_val, et_val_pred))
# et_mae = mean_absolute_error(y_val, et_val_pred)
# et_rmae = np.sqrt(et_mae)

# print(f"\nLightGBM Validation - RMSE: {lgbm_rmse:.6f} | MAE: {lgbm_mae:.6f} | RMAE: {lgbm_rmae:.6f}")
# print(f"ExtraTrees Validation - RMSE: {et_rmse:.6f} | MAE: {et_mae:.6f} | RMAE: {et_rmae:.6f}")

# # ===============================================
# # 4. FIND OPTIMAL BLENDING WEIGHT
# # ===============================================
# print("\n" + "="*60)
# print("Finding optimal blend weight...")
# print("="*60)

# best_alpha = 0.5
# best_rmse = float('inf')
# results = []

# # Search for best alpha (weight for LightGBM)
# for alpha in np.linspace(0.0, 1.0, 21):
#     blend_pred = alpha * lgbm_val_pred + (1.0 - alpha) * et_val_pred
#     rmse = np.sqrt(mean_squared_error(y_val, blend_pred))
#     mae = mean_absolute_error(y_val, blend_pred)
#     rmae = np.sqrt(mae)
    
#     results.append({
#         'alpha': alpha,
#         'rmse': rmse,
#         'mae': mae,
#         'rmae': rmae
#     })
    
#     if rmse < best_rmse:
#         best_rmse = rmse
#         best_alpha = alpha

# results_df = pd.DataFrame(results)
# print(f"\nOptimal blend weight (alpha): {best_alpha:.2f}")
# print(f"Best validation RMSE: {best_rmse:.6f}")
# print(f"\nTop 5 blending configurations:")
# print(results_df.nsmallest(5, 'rmse')[['alpha', 'rmse', 'mae', 'rmae']])

# # ===============================================
# # 5. FINAL BLENDED PREDICTIONS
# # ===============================================
# print("\n" + "="*60)
# print("Creating final blended predictions...")
# print("="*60)

# # Final validation predictions with optimal blend
# final_val_pred = best_alpha * lgbm_val_pred + (1.0 - best_alpha) * et_val_pred

# # Final metrics
# final_rmse = np.sqrt(mean_squared_error(y_val, final_val_pred))
# final_mae = mean_absolute_error(y_val, final_val_pred)
# final_rmae = np.sqrt(final_mae)

# print(f"\nFinal Blended Model Performance:")
# print(f"RMSE: {final_rmse:.6f}")
# print(f"MAE: {final_mae:.6f}")
# print(f"RMAE: {final_rmae:.6f}")

# # Improvement metrics
# rmse_improvement = ((lgbm_rmse - final_rmse) / lgbm_rmse) * 100
# print(f"\nImprovement over LightGBM: {rmse_improvement:.2f}%")

# # ===============================================
# # 6. TEST SET PREDICTIONS (IF AVAILABLE)
# # ===============================================
# # if 'X_test' in globals():


# # ===============================================
# # 7. FEATURE IMPORTANCE ANALYSIS
# # ===============================================
# # print("\n" + "="*60)
# # print("Feature Importance Analysis")
# # print("="*60)

# # if hasattr(X_train, 'columns'):
# #     # LightGBM feature importance
# #     lgbm_importance = pd.DataFrame({
# #         'feature': X_train.columns,
# #         'importance': lgbm.feature_importances_
# #     }).sort_values('importance', ascending=False)
    
# #     print("\nTop 15 Features (LightGBM):")
# #     print(lgbm_importance.head(15).to_string(index=False))
    
#     # ExtraTrees feature importance
# #     et_importance = pd.DataFrame({
# #         'feature': X_train.columns,
# #         'importance': et.feature_importances_
# #     }).sort_values('importance', ascending=False)
    
# #     print("\nTop 15 Features (ExtraTrees):")
# #     print(et_importance.head(15).to_string(index=False))

# # # ===============================================
# # # 8. SUMMARY
# # # ===============================================
# # print("\n" + "="*60)
# # print("ENSEMBLE SUMMARY")
# # print("="*60)
# # print(f"LightGBM Weight: {best_alpha:.2f}")
# # print(f"ExtraTrees Weight: {1.0 - best_alpha:.2f}")
# # print(f"\nLightGBM alone:  RMSE={lgbm_rmse:.6f}")
# # print(f"ExtraTrees alone: RMSE={et_rmse:.6f}")
# # print(f"Blended Model:    RMSE={final_rmse:.6f}")
# # print("="*60)


# test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

# test['road_type_v'] = test['road_type'].map(road_type_dt)
# test['lighting_v'] = test['lighting'].map(lighting_dt)
# test['weather_v'] = test['weather'].map(weather_dt)
# test['time_of_day_dt'] = test['time_of_day'].map(time_of_day_dt)
# test.drop(['road_type', 'lighting', 'weather', 'time_of_day'], axis = 1, inplace = True)

# subm = test.copy()
# subm.drop(features, axis = 1, inplace = True)

# test.drop(['id'], axis = 1, inplace = True)


# test['road_type_v'] = test['road_type'].map(road_type_dt)
# test['lighting_v'] = test['lighting'].map(lighting_dt)
# test['weather_v'] = test['weather'].map(weather_dt)
# test['time_of_day_dt'] = test['time_of_day'].map(time_of_day_dt)
# test.drop(['road_type', 'lighting', 'weather', 'time_of_day'], axis = 1, inplace = True)


# subm = test.copy()
# subm.drop(features, axis = 1, inplace = True)


# subm.sample(5)


# test.sample(5)


# test.drop(['id'], axis = 1, inplace = True)


# Y_pred = model.predict(test)


# print("\n" + "="*60)
# print("Generating test predictions...")
# print("="*60)

# lgbm_test_pred = lgbm.predict(test)
# et_test_pred = et.predict(test)

# # Blended test predictions using optimal alpha
# test_pred = best_alpha * lgbm_test_pred + (1.0 - best_alpha) * et_test_pred

# # Ensure no negative predictions (if applicable)
# test_pred = np.maximum(test_pred, 0.0)

# print(f"Test predictions generated with alpha={best_alpha:.2f}")
# print(f"Test predictions range: [{test_pred.min():.2f}, {test_pred.max():.2f}]")


subm['accident_risk'] = final_test_predictions


subm.to_csv('submission_ensemble.csv', index = False)


# subm.sample(5)







