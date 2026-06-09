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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from xgboost import XGBClassifier, DMatrix, train as xgb_train

# RAPIDS cuML imports
import cuml
from cuml.preprocessing import OneHotEncoder, StandardScaler
from cuml.ensemble import RandomForestClassifier
import cudf

# Load data
df = pd.read_csv("/kaggle/input/s5e11-5foldcv/trains_fold.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

features = [c for c in df.columns if c not in ('id','loan_paid_back','kfold')] 
num_cols = df[features].select_dtypes(exclude='object').columns.tolist()  
cat_cols = df[features].select_dtypes(include='object').columns.tolist()                                                             
df_test = df_test[features]

# Convert to cudf for GPU acceleration
df_gpu = cudf.from_pandas(df)
df_test_gpu = cudf.from_pandas(df_test)

# Initialize arrays for predictions
oof_xgb = np.zeros(len(df))
oof_rf = np.zeros(len(df))
test_preds_xgb = np.zeros(len(df_test))
test_preds_rf = np.zeros(len(df_test))

# XGBoost parameters with UPDATED GPU support
params_xgb = {
    'learning_rate': 0.14961834067692198,
    'reg_lambda': 74.24088221333685,
    'reg_alpha': 2.5002524143060135e-06,
    'subsample': 0.8527120022478751,
    'colsample_bytree': 0.5205468485450351,
    'max_depth': 3,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42
}

for fold in range(5):
    print(f"===== Fold {fold + 1} / {5} =====")

    # Get the actual indices for this fold
    train_idx = df_gpu[df_gpu.kfold != fold].index.to_pandas()
    valid_idx = df_gpu[df_gpu.kfold == fold].index.to_pandas()
    
    xtrain = df_gpu[df_gpu.kfold != fold]
    xvalid = df_gpu[df_gpu.kfold == fold]
    xtest = df_test_gpu.copy()

    ytrain = xtrain.loan_paid_back.values
    yvalid = xvalid.loan_paid_back.values

    xtrain = xtrain[features]
    xvalid = xvalid[features]

    # GPU Preprocessing - MUCH faster
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    xtrain_ohe = ohe.fit_transform(xtrain[cat_cols])
    xvalid_ohe = ohe.transform(xvalid[cat_cols])
    xtest_ohe = ohe.transform(xtest[cat_cols])

    # Convert to cudf DataFrames
    xtrain_ohe = cudf.DataFrame(xtrain_ohe, columns=[f'ohe_{i}' for i in range(xtrain_ohe.shape[1])])
    xvalid_ohe = cudf.DataFrame(xvalid_ohe, columns=[f'ohe_{i}' for i in range(xvalid_ohe.shape[1])])
    xtest_ohe = cudf.DataFrame(xtest_ohe, columns=[f'ohe_{i}' for i in range(xtest_ohe.shape[1])])

    # Concatenate and drop categorical columns
    xtrain = cudf.concat([xtrain.reset_index(drop=True), xtrain_ohe], axis=1).drop(cat_cols, axis=1)
    xvalid = cudf.concat([xvalid.reset_index(drop=True), xvalid_ohe], axis=1).drop(cat_cols, axis=1)
    xtest = cudf.concat([xtest.reset_index(drop=True), xtest_ohe], axis=1).drop(cat_cols, axis=1)

    # GPU Scaling
    scaler = StandardScaler()
    xtrain[num_cols] = scaler.fit_transform(xtrain[num_cols])
    xvalid[num_cols] = scaler.transform(xvalid[num_cols])
    xtest[num_cols] = scaler.transform(xtest[num_cols])

    # Convert to pandas for XGBoost (XGBoost needs pandas for DMatrix)
    xtrain_pd = xtrain.to_pandas()
    xvalid_pd = xvalid.to_pandas()
    xtest_pd = xtest.to_pandas()

    # Train XGBoost with UPDATED GPU parameters
    dtrain = DMatrix(xtrain_pd, label=ytrain.get())
    dvalid = DMatrix(xvalid_pd, label=yvalid.get())
    dtest = DMatrix(xtest_pd)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    xgb_model = xgb_train(
        params_xgb,
        dtrain,
        num_boost_round=7000,
        evals=watchlist,
        early_stopping_rounds=300,
        verbose_eval=False
    )

    # Store XGBoost predictions
    oof_xgb[valid_idx] = xgb_model.predict(DMatrix(xvalid_pd))
    test_preds_xgb += xgb_model.predict(dtest) / 5

    # Train cuML Random Forest (GPU accelerated)
    rf_model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=8,
        random_state=42,
        n_streams=1
    )
    
    # Train directly on GPU data - no conversion needed!
    rf_model.fit(xtrain, ytrain)
    
    # FIXED: Get predictions - handle cuML output format correctly
    rf_preds_valid = rf_model.predict_proba(xvalid)
    rf_preds_test = rf_model.predict_proba(xtest)
    
    # Convert cuML predictions to numpy arrays - FIXED indexing
    if hasattr(rf_preds_valid, 'to_numpy'):
        rf_valid_proba = rf_preds_valid.to_numpy()
        rf_test_proba = rf_preds_test.to_numpy()
    else:
        rf_valid_proba = rf_preds_valid
        rf_test_proba = rf_preds_test
    
    # Store predictions - use column 1 for positive class probability
    oof_rf[valid_idx] = rf_valid_proba[:, 1]  # Positive class probabilities
    test_preds_rf += rf_test_proba[:, 1] / 5  # Average across folds

    # Calculate fold AUC
    fold_auc_xgb = metrics.roc_auc_score(yvalid.get(), oof_xgb[valid_idx])
    fold_auc_rf = metrics.roc_auc_score(yvalid.get(), oof_rf[valid_idx])
    print(f"Fold {fold + 1} | XGB AUC: {fold_auc_xgb:.4f} | RF AUC: {fold_auc_rf:.4f}")

# Final blending
oof_blend = (oof_xgb + oof_rf) / 2
test_preds_blend = (test_preds_xgb + test_preds_rf) / 2

# Verify all folds have predictions
print(f"\nFinal verification:")
print(f"OOF XGB - zeros: {np.sum(oof_xgb == 0)}, non-zeros: {np.sum(oof_xgb != 0)}")
print(f"OOF RF - zeros: {np.sum(oof_rf == 0)}, non-zeros: {np.sum(oof_rf != 0)}")

overall_auc = metrics.roc_auc_score(df.loan_paid_back, oof_blend)
print(f"Overall Blended OOF AUC: {overall_auc:.4f}")

# Find optimal weights
from scipy.optimize import minimize

def objective(weights):
    blended = weights[0] * oof_xgb + weights[1] * oof_rf
    return -metrics.roc_auc_score(df.loan_paid_back, blended)

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1), (0, 1)]
initial_weights = [0.5, 0.5]

result = minimize(objective, initial_weights, method='SLSQP', 
                 bounds=bounds, constraints=constraints)

optimal_weights = result.x
optimal_auc = -result.fun

print(f"\nOptimal weights - XGB: {optimal_weights[0]:.4f}, RF: {optimal_weights[1]:.4f}")
print(f"Optimized AUC: {optimal_auc:.4f}")

# Apply optimal weights to test predictions
test_preds_optimal = optimal_weights[0] * test_preds_xgb + optimal_weights[1] * test_preds_rf

# Create submission
submission_optimal = sample_submission.copy()
submission_optimal['loan_paid_back'] = test_preds_optimal
submission_optimal.to_csv('submission_rapids_optimal.csv', index=False)

print("Submission file created: submission_rapids_optimal.csv")




