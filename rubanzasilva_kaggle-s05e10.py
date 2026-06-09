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


%%time
!uv pip install catboost
#%pip install optuna
#%pip install optuna_distributed
#%pip install openfe
!uv pip install seaborn
!uv pip install xgboost
!uv pip install lightgbm
#%pip install fastkaggle
#%pip install h2o
#%pip install -Uqq fastbook
#%pip install polars
#%pip install -q -U autogluon.tabular
#%pip install autogluon
#%pip install --upgrade pip
!uv pip install tqdm
#%pip install wandb
#%pip install sweetviz
#%pip install xlearn


!pip install -U scikit-learn


import sklearn
print(sklearn.__version__)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import random
from tqdm import tqdm
from pathlib import Path

from fastai.tabular.all import *
from ipywidgets import interact

from fastai.imports import *
np.set_printoptions(linewidth=130)
import gc

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import xgboost as xgb
from xgboost import plot_importance

import lightgbm as lgb
import catboost as cat


path = Path('/kaggle/input/playground-series-s5e10')
path


!ls /kaggle/input/playground-series-s5e10


train_df = pd.read_csv(path/'train.csv',index_col='id')
test_df = pd.read_csv(path/'test.csv',index_col='id')
sub_df = pd.read_csv(path/'sample_submission.csv')


train_df


train_df.head().T


train_df.columns


train_df.info()


train_df.shape, test_df.shape


train_df.describe().T


train_df.hist(figsize=(20,15),edgecolor='black');


train_df.describe(include=[object]).T


def plot_top_categories(data, column, top_n=20, figsize=(10,15)):
    # Get value counts and take top N
    top_values = data[column].value_counts().head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(x=top_values.values, y=top_values.index)
    plt.title(f'Top {top_n} {column} Categories')
    plt.xlabel('Count')
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()



plot_top_categories(train_df, 'road_type', top_n=5)


plot_top_categories(train_df, 'time_of_day', top_n=5)


plot_top_categories(train_df, 'weather', top_n=5)


plot_top_categories(train_df, 'lighting', top_n=5)


def cross_val_predict(model, X_train, y_train, X_test, n_splits=5, random_state=42, 
                       return_proba=True, scoring='roc_auc', verbose=True):
    """
    Generic cross-validation function that works with ANY sklearn-compatible model.
    
    Parameters:
    - model: INSTANTIATED model object (e.g., xgb.XGBClassifier(n_estimators=100))
    - X_train, y_train: training data and labels
    - X_test: test data for final predictions
    - n_splits: number of CV folds
    - random_state: random seed for reproducibility
    - return_proba: if True, return probabilities; if False, return class predictions
    - scoring: metric to use ('roc_auc', 'accuracy')
    - verbose: print progress
    
    Returns:
    - oof_predictions: out-of-fold predictions on training set
    - test_predictions: predictions on test set (averaged across folds)
    - mean_score: mean score across folds
    - fold_scores: list of scores for each fold
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.base import clone
    import numpy as np
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize arrays to store results
    if return_proba:
        oof_predictions = np.zeros(len(X_train))
        test_predictions = np.zeros(len(X_test))
    else:
        oof_predictions = np.zeros(len(X_train), dtype=int)
        test_predictions = np.zeros(len(X_test))
    
    fold_scores = []
    
    # Perform cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
        if verbose:
            print(f"Training fold {fold}/{n_splits}...")
        
        # Split data for current fold
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        
        # Clone the model to avoid interference between folds
        model_fold = clone(model)
        
        # Train model on current fold
        model_fold.fit(X_fold_train, y_fold_train)
        
        # Get predictions based on return_proba setting
        if return_proba:
            # Get probabilities for positive class
            y_pred_proba = model_fold.predict_proba(X_fold_val)[:, 1]
            oof_predictions[val_index] = y_pred_proba
            test_predictions += model_fold.predict_proba(X_test)[:, 1] / n_splits
            
            # Calculate score
            if scoring == 'roc_auc':
                cv_score = roc_auc_score(y_fold_val, y_pred_proba)
            else:  # accuracy
                y_pred_fold = (y_pred_proba > 0.5).astype(int)
                cv_score = accuracy_score(y_fold_val, y_pred_fold)
        else:
            # Get class predictions
            y_pred_fold = model_fold.predict(X_fold_val)
            oof_predictions[val_index] = y_pred_fold
            test_predictions += model_fold.predict(X_test) / n_splits
            cv_score = accuracy_score(y_fold_val, y_pred_fold)
        
        fold_scores.append(cv_score)
        if verbose:
            print(f"Fold {fold} {scoring}: {cv_score:.6f}")
    
    # Round test predictions if using hard predictions
    if not return_proba:
        test_predictions = np.round(test_predictions).astype(int)
    
    # Calculate mean score
    mean_score = np.mean(fold_scores)
    if verbose:
        print(f"\nMean {scoring}: {mean_score:.6f}")
    
    return oof_predictions, test_predictions, mean_score, fold_scores


# =============================================================================
# USAGE EXAMPLES WITH DIFFERENT MODELS
# =============================================================================

# Example: CatBoost (if installed)
"""
cat_model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=False
)

cat_oof, cat_test, cat_auc, cat_scores = cross_val_predict(
    model=cat_model,
    X_train=X_train,
    y_train=y_train, 
    X_test=X_test,
    return_proba=True,
    scoring='roc_auc'
)
"""

# =============================================================================
# ENSEMBLE ALL MODELS
# =============================================================================
"""
print("Individual Model Performance:")
print(f"XGBoost AUC: {xgb_auc:.6f}")
print(f"Random Forest AUC: {rf_auc:.6f}")
print(f"LightGBM AUC: {lgb_auc:.6f}")

# Simple ensemble
ensemble_oof = (xgb_oof + rf_oof + lgb_oof) / 3
ensemble_test = (xgb_test + rf_test + lgb_test) / 3

ensemble_auc = roc_auc_score(y_train, ensemble_oof)
print(f"\nEnsemble AUC: {ensemble_auc:.6f}")

# Create submission
submission = pd.DataFrame({
    'id': test_df.index,
    'Depression': ensemble_test
})
submission.to_csv('ensemble_submission.csv', index=False)

print(f"\nShapes:")
print(f"OOF predictions: {ensemble_oof.shape} (training data)")
print(f"Test predictions: {ensemble_test.shape} (test data)")
print("Different shapes = different datasets!")
"""


#splits = RandomSplitter(valid_pct=0.2)(range_of(original_df))
#train_df = pd.concat([train_df, original_df], ignore_index=True) *

cont_names,cat_names = cont_cat_split(train_df, dep_var='accident_risk')


splits = RandomSplitter(valid_pct=0.2)(range_of(train_df))


to = TabularPandas(train_df, procs=[Categorify, FillMissing,Normalize],
                   cat_names = cat_names,
                   cont_names = cont_names,
                   y_names='accident_risk',
                   y_block=RegressionBlock(),
                   splits=splits)


dls = to.dataloaders(bs=64)
#dls = to.dataloaders(bs=102



test_dl = dls.test_dl(test_df)
test_dl


test_dl.xs


X_train, y_train = to.train.xs, to.train.ys.values.ravel()
X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()


cat_names, len(cat_names)


cont_names,len(cont_names)


#rf_regr = RandomForestRegressor(n_estimators= 100, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
#                                max_features='sqrt', bootstrap=True, random_state=42, n_jobs=1)
rf_regr = RandomForestRegressor(100, min_samples_leaf=3)
rf_regr.fit(X_train,y_train)


rf_preds = rf_regr.predict(test_dl.xs)
rf_sc_preds = rf_regr.predict(X_test)


rf_sc = root_mean_squared_error(y_test, rf_sc_preds)
rf_sc


rf_preds,rf_preds.shape


def rf_feat_importance(m, train_subset):
    return pd.DataFrame({'cols':train_subset.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


fi = rf_feat_importance(rf_regr, X_train)
#fi[:10]

fi


def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

#plot_fi(fi[:30]);
plot_fi(fi);


sub_df


!rm submission.csv
sub_df['accident_risk'] = rf_preds
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub


%%time
xgb_regr = xgb.XGBRegressor()
xgb_regr


xgb_regr = xgb_regr.fit(X_train, y_train)
xgb_regr


xgb_preds = xgb_regr.predict(test_dl.xs)
xgb_sc_preds = xgb_regr.predict(X_test)


xgb_sc = root_mean_squared_error(y_test, xgb_sc_preds)
xgb_sc


xgb_preds,xgb_preds.shape


with open('xgb_regr.pkl', 'wb') as f:
    pickle.dump(xgb_regr, f)


!ls





plot_importance(xgb_regr)


!rm submission.csv
sub_df = pd.read_csv(path/'sample_submission.csv')
sub_df['accident_risk'] = xgb_preds
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub


lgb.LGBMRegressor?


lgbm_regr = lgb.LGBMRegressor()
lgbm_regr


lgbm_regr = lgbm_regr.fit(X_train,y_train)
lgbm_regr


lgbm_preds = lgbm_regr.predict(test_dl.xs)
lgbm_sc_preds = lgbm_regr.predict(X_test)


lgbm_sc = root_mean_squared_error(y_test, lgbm_sc_preds)
lgbm_sc


lgbm_preds,lgbm_preds.shape


!rm submission.csv
sub_df = pd.read_csv(path/'sample_submission.csv')
sub_df['accident_risk'] = lgbm_preds
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub


import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Fix the TabularPandas setup for REGRESSION (remove CategoryBlock)
cont_names, cat_names = cont_cat_split(train_df, dep_var='accident_risk')

to = TabularPandas(train_df, 
                   procs=[Categorify, FillMissing, Normalize],
                   cat_names=cat_names,
                   cont_names=cont_names,
                   y_names='accident_risk',
                   # Remove y_block=CategoryBlock() for regression
                   # Or explicitly use: y_block=RegressionBlock()
                   splits=splits)

# Extract preprocessed data
X_train, y_train = to.train.xs, to.train.ys.values.ravel()
X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# LightGBM parameters for regression
params = {
    'objective': 'regression',  # For regression task
    'metric': 'rmse',           # Root Mean Squared Error
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}

# Train the model
print("Training LightGBM model...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
print("\n" + "="*50)
print("Model Evaluation Results")
print("="*50)
print("\nTraining Set:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"R2 Score: {r2_score(y_train, y_pred_train):.4f}")

print("\nValidation Set:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred_test):.4f}")

# Feature importance
print("\n" + "="*50)
print("Top 10 Most Important Features")
print("="*50)
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(importance_df.head(10))

# Predict on test_df (the one without labels)
# First preprocess test_df using the same TabularPandas object
test_dl = dls.test_dl(test_df)
X_test_final = test_dl.xs
test_predictions = model.predict(X_test_final)

# Save predictions if needed
# test_df['predicted_accident_risk'] = test_predictions
# test_df.to_csv('predictions.csv', index=False)


test_predictions.shape


!rm submission.csv
sub_df = pd.read_csv(path/'sample_submission.csv')
sub_df['accident_risk'] = test_predictions
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub


import catboost as cat


cat.CatBoostRegressor??


cat_regr = cat.CatBoostRegressor(loss_function='RMSE')
cat_regr


cat_regr = cat_regr.fit(X_train, y_train)
cat_regr


cat_preds = cat_regr.predict(test_dl.xs)
cat_sc_preds = cat_regr.predict(X_test)


!rm submission.csv
sub_df = pd.read_csv(path/'sample_submission.csv')
sub_df['accident_risk'] = cat_preds
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub


def gbm_mdl_trng(model,metric_func, my_test_dl=test_dl, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):
  model = model.fit(X_train, y_train)
  model_preds = tensor(model.predict(my_test_dl.xs))
  model_sc_preds = tensor(model.predict(X_test))
  model_score = metric_func(y_test,model_sc_preds)
  return model, model_score, model_preds, model_preds.shape


cat_params = {'loss_function':'RMSE'}
cat_model = cat.CatBoostRegressor(**cat_params)
metric_func = root_mean_squared_error


cat_model, cat_score, cat_preds, cat_shape = gbm_mdl_trng(cat_model,metric_func)
cat_score, cat_preds, cat_shape


cat_preds


!ls


!rm submission.csv
sub_df = pd.read_csv(path/'sample_submission.csv')
sub_df['accident_risk'] = cat_preds
sub_df.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv', index_col='id')
sub




