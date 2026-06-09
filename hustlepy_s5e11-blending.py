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
from sklearn import preprocessing,model_selection,linear_model,metrics,ensemble
from xgboost import XGBClassifier,DMatrix,train as xgb_train
from catboost import CatBoostClassifier



df = pd.read_csv("/kaggle/input/s5e11-5foldcv/trains_fold.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


features=[c for c in df.columns if c not in ('id','loan_paid_back','kfold')] 
num_cols=df[features].select_dtypes(exclude='object').columns.tolist()  
cat_cols=df[features].select_dtypes(include='object').columns.tolist() 


df_test=df_test[features] 


# Test with just one fold to debug
fold = 0  # Test with fold 0
print(f"===== Testing Fold {fold + 1} / {5} =====")

# Get the actual indices for this fold
train_idx = df[df.kfold != fold].index
valid_idx = df[df.kfold == fold].index

xtrain = df[df.kfold != fold]
xvalid = df[df.kfold == fold]
xtest = df_test.copy()

ytrain = xtrain.loan_paid_back.values
yvalid = xvalid.loan_paid_back.values

xtrain = xtrain[features]
xvalid = xvalid[features]

# Preprocessing
ohe = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
xtrain_ohe = ohe.fit_transform(xtrain[cat_cols])
xvalid_ohe = ohe.transform(xvalid[cat_cols])
xtest_ohe = ohe.transform(xtest[cat_cols])

xtrain_ohe = pd.DataFrame(xtrain_ohe, columns=[f'ohe_{i}' for i in range(xtrain_ohe.shape[1])])
xvalid_ohe = pd.DataFrame(xvalid_ohe, columns=[f'ohe_{i}' for i in range(xvalid_ohe.shape[1])])
xtest_ohe = pd.DataFrame(xtest_ohe, columns=[f'ohe_{i}' for i in range(xtest_ohe.shape[1])])

xtrain = pd.concat([xtrain.reset_index(drop=True), xtrain_ohe], axis=1).drop(cat_cols, axis=1)
xvalid = pd.concat([xvalid.reset_index(drop=True), xvalid_ohe], axis=1).drop(cat_cols, axis=1)
xtest = pd.concat([xtest.reset_index(drop=True), xtest_ohe], axis=1).drop(cat_cols, axis=1)

scaler = preprocessing.StandardScaler()
xtrain[num_cols] = scaler.fit_transform(xtrain[num_cols])
xvalid[num_cols] = scaler.transform(xvalid[num_cols])
xtest[num_cols] = scaler.transform(xtest[num_cols])

dtrain = DMatrix(xtrain, label=ytrain)
dvalid = DMatrix(xvalid, label=yvalid)
dtest = DMatrix(xtest)

# Train XGBoost
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
xgb_model = xgb_train(
    params_xgb,
    dtrain,
    num_boost_round=7000,
    evals=watchlist,
    early_stopping_rounds=300,
    verbose_eval=False
)

# Store predictions using the actual indices
oof_xgb_single = np.zeros(len(df))
oof_xgb_single[valid_idx] = xgb_model.predict(DMatrix(xvalid))

# Train CatBoost
cat_model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.05,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    task_type="GPU",
    verbose=False
)

cat_model.fit(xtrain, ytrain, eval_set=(xvalid, yvalid))
oof_cat_single = np.zeros(len(df))
oof_cat_single[valid_idx] = cat_model.predict_proba(xvalid)[:, 1]

# Calculate fold AUC using the stored predictions for this fold
fold_auc_xgb = metrics.roc_auc_score(yvalid, oof_xgb_single[valid_idx])
fold_auc_cat = metrics.roc_auc_score(yvalid, oof_cat_single[valid_idx])
print(f"Fold {fold + 1} | XGB AUC: {fold_auc_xgb:.4f} | CAT AUC: {fold_auc_cat:.4f}")

# Test the blending for just this fold
oof_blend_single = (oof_xgb_single + oof_cat_single) / 2
single_fold_auc = metrics.roc_auc_score(yvalid, oof_blend_single[valid_idx])
print(f"Single Fold Blended AUC: {single_fold_auc:.4f}")

# Debugging information
print(f"\nDebugging info for fold {fold + 1}:")
print(f"Validation indices range: {valid_idx.min()} to {valid_idx.max()}")
print(f"Validation set size: {len(valid_idx)}")
print(f"OOF XGB predictions for this fold: {oof_xgb_single[valid_idx][:5]}")  # First 5 predictions
print(f"OOF CAT predictions for this fold: {oof_cat_single[valid_idx][:5]}")  # First 5 predictions
print(f"Actual yvalid values: {yvalid[:5]}")  # First 5 actual values
print(f"Any zeros in OOF XGB for this fold: {np.sum(oof_xgb_single[valid_idx] == 0)}")
print(f"Any zeros in OOF CAT for this fold: {np.sum(oof_cat_single[valid_idx] == 0)}")


# Initialize fresh arrays for each run
oof_xgb = np.zeros(len(df))
oof_cat = np.zeros(len(df))
test_preds_xgb = np.zeros(len(df_test))
test_preds_cat = np.zeros(len(df_test))

for fold in range(5):
    print(f"===== Fold {fold + 1} / {5} =====")

    # Get the actual indices for this fold
    train_idx = df[df.kfold != fold].index
    valid_idx = df[df.kfold == fold].index
    
    xtrain = df[df.kfold != fold]
    xvalid = df[df.kfold == fold]
    xtest = df_test.copy()

    ytrain = xtrain.loan_paid_back.values
    yvalid = xvalid.loan_paid_back.values

    xtrain = xtrain[features]
    xvalid = xvalid[features]

    # Preprocessing
    ohe = preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    xtrain_ohe = ohe.fit_transform(xtrain[cat_cols])
    xvalid_ohe = ohe.transform(xvalid[cat_cols])
    xtest_ohe = ohe.transform(xtest[cat_cols])

    xtrain_ohe = pd.DataFrame(xtrain_ohe, columns=[f'ohe_{i}' for i in range(xtrain_ohe.shape[1])])
    xvalid_ohe = pd.DataFrame(xvalid_ohe, columns=[f'ohe_{i}' for i in range(xvalid_ohe.shape[1])])
    xtest_ohe = pd.DataFrame(xtest_ohe, columns=[f'ohe_{i}' for i in range(xtest_ohe.shape[1])])

    xtrain = pd.concat([xtrain.reset_index(drop=True), xtrain_ohe], axis=1).drop(cat_cols, axis=1)
    xvalid = pd.concat([xvalid.reset_index(drop=True), xvalid_ohe], axis=1).drop(cat_cols, axis=1)
    xtest = pd.concat([xtest.reset_index(drop=True), xtest_ohe], axis=1).drop(cat_cols, axis=1)

    scaler = preprocessing.StandardScaler()
    xtrain[num_cols] = scaler.fit_transform(xtrain[num_cols])
    xvalid[num_cols] = scaler.transform(xvalid[num_cols])
    xtest[num_cols] = scaler.transform(xtest[num_cols])

    dtrain = DMatrix(xtrain, label=ytrain)
    dvalid = DMatrix(xvalid, label=yvalid)
    dtest = DMatrix(xtest)

    # Train XGBoost
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    xgb_model = xgb_train(
        params_xgb,
        dtrain,
        num_boost_round=7000,
        evals=watchlist,
        early_stopping_rounds=300,
        verbose_eval=False
    )

    # Store predictions using the actual indices
    oof_xgb[valid_idx] = xgb_model.predict(DMatrix(xvalid))
    test_preds_xgb += xgb_model.predict(dtest) / 5

    # Train CatBoost
    cat_model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.05,
        depth=8,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        task_type="GPU",
        verbose=False
    )

    cat_model.fit(xtrain, ytrain, eval_set=(xvalid, yvalid))
    oof_cat[valid_idx] = cat_model.predict_proba(xvalid)[:, 1]
    test_preds_cat += cat_model.predict_proba(xtest)[:, 1] / 5

    # Calculate fold AUC using the stored predictions for this fold
    fold_auc_xgb = metrics.roc_auc_score(yvalid, oof_xgb[valid_idx])
    fold_auc_cat = metrics.roc_auc_score(yvalid, oof_cat[valid_idx])
    print(f"Fold {fold + 1} | XGB AUC: {fold_auc_xgb:.4f} | CAT AUC: {fold_auc_cat:.4f}")

# Final blending
oof_blend = (oof_xgb + oof_cat) / 2
test_preds_blend = (test_preds_xgb + test_preds_cat) / 2

# Verify all folds have predictions
print(f"\nFinal verification:")
print(f"OOF XGB - zeros: {np.sum(oof_xgb == 0)}, non-zeros: {np.sum(oof_xgb != 0)}")
print(f"OOF CAT - zeros: {np.sum(oof_cat == 0)}, non-zeros: {np.sum(oof_cat != 0)}")

overall_auc = metrics.roc_auc_score(df.loan_paid_back, oof_blend)
print(f"Overall Blended OOF AUC: {overall_auc:.4f}")


sample_submission.loan_paid_back = test_preds_blend
sample_submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved as submission.csv")

