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


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models and evaluation tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

# For gradient boosting – you can switch or try both
import lightgbm as lgb

# Load the datasets (paths as provided in the competition)
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

# Quick exploration
print("Training shape:", train.shape)
print("Test shape:", test.shape)
print(train.head())
print(train.info())
print("Missing values in train:\n", train.isnull().sum())



# Remove identifier and separate out the target variable
X = train.drop(['id', 'rainfall'], axis=1)
y = train['rainfall']

# For the test set (make sure you drop 'id' later when predicting)
test_ids = test['id']
X_test = test.drop(['id'], axis=1)

# Fill missing values if any found
X = X.fillna(X.median())
X_test = X_test.fillna(X_test.median())

# Optional: If you have categorical features, consider one-hot encoding or label encoding here.



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Train a baseline Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Validation predictions and AUC
y_val_pred_rf = rf_model.predict_proba(X_val)[:, 1]
roc_auc_rf = roc_auc_score(y_val, y_val_pred_rf)
print(f'Random Forest Validation ROC AUC: {roc_auc_rf:.4f}')



# Prepare LightGBM dataset
d_train = lgb.Dataset(X_train, label=y_train)

# Set parameters
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
    'seed': 42
}

# Train LightGBM model
lgb_model = lgb.train(lgb_params, d_train, num_boost_round=200)
y_val_pred_lgb = lgb_model.predict(X_val)
roc_auc_lgb = roc_auc_score(y_val, y_val_pred_lgb)
print(f'LightGBM Validation ROC AUC: {roc_auc_lgb:.4f}')



# Define parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 400],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Grid Search CV to optimize ROC AUC
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                       param_grid,
                       cv=3,
                       scoring='roc_auc',
                       n_jobs=-1,
                       verbose=1)

grid_rf.fit(X_train, y_train)
print("Best RF parameters:", grid_rf.best_params_)

# Evaluate tuned model
y_val_pred_grid = grid_rf.predict_proba(X_val)[:, 1]
roc_auc_grid = roc_auc_score(y_val, y_val_pred_grid)
print(f'Tuned Random Forest Validation ROC AUC: {roc_auc_grid:.4f}')



importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Feature Importance (RF Baseline)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()



def get_oof(model, X_train, y, X_test, n_splits=5):
    # Out-Of-Fold predictions arrays
    oof_train = np.zeros(X_train.shape[0])
    oof_test = np.zeros(X_test.shape[0])
    oof_test_skf = np.empty((n_splits, X_test.shape[0]))
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        
        model.fit(X_tr, y_tr)
        oof_train[val_idx] = model.predict_proba(X_val)[:, 1]
        oof_test_skf[i, :] = model.predict_proba(X_test)[:, 1]
    
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Base models
rf_base = RandomForestClassifier(n_estimators=200, random_state=42)
lgb_base = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=200, random_state=42)

# Generate out-of-fold predictions
rf_oof_train, rf_oof_test = get_oof(rf_base, X_train, y_train, X_test)
lgb_oof_train, lgb_oof_test = get_oof(lgb_base, X_train, y_train, X_test)

# Stack predictions as new features for meta-model
stacked_train = np.concatenate((rf_oof_train, lgb_oof_train), axis=1)
stacked_test = np.concatenate((rf_oof_test, lgb_oof_test), axis=1)

# Train meta-model (Logistic Regression)
meta_model = LogisticRegression(random_state=42)
meta_model.fit(stacked_train, y_train)

# Final stacked predictions for submission
stacked_pred = meta_model.predict_proba(stacked_test)[:, 1]

# Write to submission file
submission_stack = pd.DataFrame({
    'id': test_ids,
    'rainfall': stacked_pred
})
submission_stack.to_csv('stacked_submission.csv', index=False)
print("Stacked submission file 'stacked_submission.csv' generated!")



# For example, assume tuned RF (grid_rf) performed best on validation:
final_model = grid_rf  # Change this based on your best result

# Predict on test set; note that test set features should be processed exactly as training data.
test_pred = final_model.predict_proba(X_test)[:, 1]

# Create submission file in required format
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")



import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Model and evaluation libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from xgboost import XGBClassifier
import catboost as cb

# ---------------------------
# 1. Load Data
# ---------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

# Preprocessing: Drop the identifier and fill missing values
X = train.drop(['id', 'rainfall'], axis=1).fillna(train.median())
y = train['rainfall']
X_test = test.drop('id', axis=1).fillna(test.median())
test_ids = test['id']

# ---------------------------
# 2. Define Out-Of-Fold Function for Stacking
# ---------------------------
def get_oof(model, X, y, X_test, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_train = np.zeros(X.shape[0])
    oof_test = np.zeros(X_test.shape[0])
    oof_test_folds = np.zeros((n_splits, X_test.shape[0]))
    
    for i, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
        
        model.fit(X_tr, y_tr)
        oof_train[valid_idx] = model.predict_proba(X_val)[:, 1]
        oof_test_folds[i, :] = model.predict_proba(X_test)[:, 1]
    
    oof_test[:] = oof_test_folds.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# ---------------------------
# 3. Define Base Models
# ---------------------------
# Base Model 1: Random Forest
rf = RandomForestClassifier(n_estimators=250, random_state=42)

# Base Model 2: XGBoost
xgb_model = XGBClassifier(n_estimators=250, random_state=42, 
                          use_label_encoder=False, eval_metric='logloss')

# Base Model 3: LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=250, random_state=42)

# Base Model 4: CatBoost
cat_model = cb.CatBoostClassifier(iterations=250, random_seed=42, verbose=0)

# ---------------------------
# 4. Generate Out-Of-Fold Predictions for Each Model
# ---------------------------
oof_rf_train, oof_rf_test = get_oof(rf, X, y, X_test)
oof_xgb_train, oof_xgb_test = get_oof(xgb_model, X, y, X_test)
oof_lgb_train, oof_lgb_test = get_oof(lgb_model, X, y, X_test)
oof_cat_train, oof_cat_test = get_oof(cat_model, X, y, X_test)

# ---------------------------
# 5. Stack Base Model Predictions as New Features
# ---------------------------
X_stack_train = np.concatenate((oof_rf_train, oof_xgb_train, oof_lgb_train, oof_cat_train), axis=1)
X_stack_test  = np.concatenate((oof_rf_test, oof_xgb_test, oof_lgb_test, oof_cat_test), axis=1)

# Optional: Evaluate the stacked features on a local holdout set
X_t, X_val_stack, y_t, y_val_stack = train_test_split(X_stack_train, y, test_size=0.2, random_state=42)
meta_model_eval = LogisticRegression(random_state=42)
meta_model_eval.fit(X_t, y_t)
val_pred = meta_model_eval.predict_proba(X_val_stack)[:, 1]
print("Local Stacked Ensemble ROC AUC:", roc_auc_score(y_val_stack, val_pred))

# ---------------------------
# 6. Train Meta-Model and Generate Final Predictions
# ---------------------------
meta_model = LogisticRegression(random_state=42)
meta_model.fit(X_stack_train, y)
final_pred = meta_model.predict_proba(X_stack_test)[:, 1]

# ---------------------------
# 7. Create Submission File
# ---------------------------
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': final_pred
})
submission.to_csv('submission.csv', index=False)
print("Advanced stacked submission generated!")



import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Model and Evaluation libraries
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from xgboost import XGBClassifier
import catboost as cb

# ---------------------------
# 1. Load and Preprocess the Data
# ---------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Drop the identifier column and fill missing values with median
X = train.drop(['id','rainfall'], axis=1).fillna(train.median())
y = train['rainfall']
X_test = test.drop('id', axis=1).fillna(test.median())
test_ids = test['id']

# ---------------------------
# 2. Revised Out-of-Fold Function Using StratifiedKFold
# ---------------------------
def get_oof(model, X, y, X_test, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_train = np.zeros(X.shape[0])
    oof_test = np.zeros(X_test.shape[0])
    oof_test_folds = np.zeros((n_splits, X_test.shape[0]))
    
    for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_tr, y_tr)
        oof_train[valid_idx] = model.predict_proba(X_val)[:, 1]
        oof_test_folds[i, :] = model.predict_proba(X_test)[:, 1]
    
    oof_test[:] = oof_test_folds.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# ---------------------------
# 3. Define and Tune Base Models
# ---------------------------
# Base Model 1: Random Forest (with tuned hyperparameters)
rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=10, random_state=42)

# Base Model 2: XGBoost
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                          use_label_encoder=False, eval_metric='logloss', random_state=42)

# Base Model 3: LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42)

# Base Model 4: CatBoost
cat_model = cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, 
                                  random_seed=42, verbose=0)

# Base Model 5: Extra Trees for additional diversity
et_model = ExtraTreesClassifier(n_estimators=300, max_depth=15, min_samples_leaf=10, random_state=42)

# ---------------------------
# 4. Generate Out-of-Fold Predictions for Each Base Model
# ---------------------------
oof_rf_train, oof_rf_test   = get_oof(rf, X, y, X_test)
oof_xgb_train, oof_xgb_test = get_oof(xgb_model, X, y, X_test)
oof_lgb_train, oof_lgb_test = get_oof(lgb_model, X, y, X_test)
oof_cat_train, oof_cat_test = get_oof(cat_model, X, y, X_test)
oof_et_train, oof_et_test   = get_oof(et_model, X, y, X_test)

# ---------------------------
# 5. Stack Base Model Predictions as New Features
# ---------------------------
# Combine all out-of-fold predictions to create the stacking features
X_stack_train = np.concatenate((oof_rf_train, oof_xgb_train, oof_lgb_train, oof_cat_train, oof_et_train), axis=1)
X_stack_test  = np.concatenate((oof_rf_test,  oof_xgb_test,  oof_lgb_test,  oof_cat_test,  oof_et_test), axis=1)

# ---------------------------
# 6. Validate the Stacked Predictions with a Meta-Learner
# ---------------------------
X_t, X_val_stack, y_t, y_val_stack = train_test_split(X_stack_train, y, test_size=0.2, random_state=42)

# Use LightGBM as a meta-learner for a potentially improved fit
meta_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=42)
meta_model.fit(X_t, y_t)
val_pred = meta_model.predict_proba(X_val_stack)[:, 1]
print("Improved Stacked Ensemble Local ROC AUC:", roc_auc_score(y_val_stack, val_pred))

# ---------------------------
# 7. Train the Meta-Learner on the Full Stacking Dataset and Make Final Predictions
# ---------------------------
meta_model.fit(X_stack_train, y)
stacked_pred = meta_model.predict_proba(X_stack_test)[:, 1]

# ---------------------------
# 8. Create the Submission File
# ---------------------------
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': stacked_pred
})
submission.to_csv('submission_improved.csv', index=False)
print("Improved advanced stacked submission generated!")



import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import catboost as cb

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Drop identifier column and fill missing values using median
X = train.drop(['id','rainfall'], axis=1).fillna(train.median())
y = train['rainfall']
X_test = test.drop('id', axis=1).fillna(test.median())
test_ids = test['id']

# ---------------------------
# 2. Out-of-Fold Function Using StratifiedKFold (n_splits=10)
# ---------------------------
def get_oof(model, X, y, X_test, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_train = np.zeros(X.shape[0])
    oof_test_folds = np.zeros((n_splits, X_test.shape[0]))
    
    for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
        model.fit(X_tr, y_tr)
        oof_train[valid_idx] = model.predict_proba(X_val)[:, 1]
        oof_test_folds[i, :] = model.predict_proba(X_test)[:, 1]
    
    oof_test = oof_test_folds.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# ---------------------------
# 3. Define and Tune Base Models
# ---------------------------
# Base Model 1: Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=10, random_state=42)

# Base Model 2: XGBoost
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                          use_label_encoder=False, eval_metric='logloss', random_state=42)

# Base Model 3: LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42)

# Base Model 4: CatBoost
cat_model = cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, 
                                  random_seed=42, verbose=0)

# Base Model 5: ExtraTrees for added diversity
et_model = ExtraTreesClassifier(n_estimators=300, max_depth=15, min_samples_leaf=10, random_state=42)

# ---------------------------
# 4. Generate Out-of-Fold Predictions for Each Base Model
# ---------------------------
oof_rf_train, oof_rf_test   = get_oof(rf, X, y, X_test, n_splits=10)
oof_xgb_train, oof_xgb_test = get_oof(xgb_model, X, y, X_test, n_splits=10)
oof_lgb_train, oof_lgb_test = get_oof(lgb_model, X, y, X_test, n_splits=10)
oof_cat_train, oof_cat_test = get_oof(cat_model, X, y, X_test, n_splits=10)
oof_et_train, oof_et_test   = get_oof(et_model, X, y, X_test, n_splits=10)

# ---------------------------
# 5. Stack Base Model Predictions as Meta-Features
# ---------------------------
X_stack_train = np.concatenate((oof_rf_train, oof_xgb_train, oof_lgb_train, oof_cat_train, oof_et_train), axis=1)
X_stack_test  = np.concatenate((oof_rf_test,  oof_xgb_test,  oof_lgb_test,  oof_cat_test,  oof_et_test), axis=1)

# Optionally, scale the stacked features:
scaler = StandardScaler()
X_stack_train_scaled = scaler.fit_transform(X_stack_train)
X_stack_test_scaled = scaler.transform(X_stack_test)

# ---------------------------
# 6. Validate with a Meta Learner (Using an MLP)
# ---------------------------
X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(X_stack_train_scaled, y, test_size=0.2, random_state=42)

meta_model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam',
                           random_state=42, max_iter=500)
meta_model.fit(X_meta_train, y_meta_train)
meta_val_pred = meta_model.predict_proba(X_meta_val)[:, 1]
print("Local Meta Learner (MLP) ROC AUC:", roc_auc_score(y_meta_val, meta_val_pred))

# ---------------------------
# 7. Train the Meta Learner on Full Stacking Data and Generate Final Predictions
# ---------------------------
meta_model.fit(X_stack_train_scaled, y)
final_pred = meta_model.predict_proba(X_stack_test_scaled)[:, 1]

# ---------------------------
# 8. Create the Final Submission File
# ---------------------------
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': final_pred
})
submission.to_csv('submission_final_improved.csv', index=False)
print("Final submission file 'submission_final_improved.csv' generated!")



import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Modeling and utility libraries
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import catboost as cb

# For feature selection from original features
from sklearn.feature_selection import SelectKBest, f_classif

# For neural network meta learner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Basic preprocessing: remove identifier and impute missing values with median
X = train.drop(['id','rainfall'], axis=1).fillna(train.median())
y = train['rainfall']
X_test = test.drop('id', axis=1).fillna(test.median())
test_ids = test['id']

# ---------------------------
# 2. Out-of-Fold Function with 10-Fold StratifiedKFold
# ---------------------------
def get_oof(model, X, y, X_test, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_train = np.zeros(X.shape[0])
    oof_test_folds = np.zeros((n_splits, X_test.shape[0]))
    
    for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
        model.fit(X_tr, y_tr)
        oof_train[valid_idx] = model.predict_proba(X_val)[:, 1]
        oof_test_folds[i, :] = model.predict_proba(X_test)[:, 1]
    
    oof_test = oof_test_folds.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# ---------------------------
# 3. Define and Tune Base Models
# ---------------------------
# Base Model 1: RandomForest
rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=10, random_state=42)

# Base Model 2: XGBoost
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                          use_label_encoder=False, eval_metric='logloss', random_state=42)

# Base Model 3: LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42)

# Base Model 4: CatBoost
cat_model = cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, 
                                  random_seed=42, verbose=0)

# Base Model 5: ExtraTrees for model diversity
et_model = ExtraTreesClassifier(n_estimators=300, max_depth=15, min_samples_leaf=10, random_state=42)

# ---------------------------
# 4. Generate Out-of-Fold Predictions (Stacking Features)
# ---------------------------
oof_rf_train, oof_rf_test   = get_oof(rf, X, y, X_test, n_splits=10)
oof_xgb_train, oof_xgb_test = get_oof(xgb_model, X, y, X_test, n_splits=10)
oof_lgb_train, oof_lgb_test = get_oof(lgb_model, X, y, X_test, n_splits=10)
oof_cat_train, oof_cat_test = get_oof(cat_model, X, y, X_test, n_splits=10)
oof_et_train, oof_et_test   = get_oof(et_model, X, y, X_test, n_splits=10)

# Stack predictions horizontally
X_stack_train = np.concatenate((oof_rf_train, oof_xgb_train, oof_lgb_train, oof_cat_train, oof_et_train), axis=1)
X_stack_test  = np.concatenate((oof_rf_test, oof_xgb_test, oof_lgb_test, oof_cat_test, oof_et_test), axis=1)

# Scale the stacking predictions (meta-features)
scaler = StandardScaler()
X_stack_train_scaled = scaler.fit_transform(X_stack_train)
X_stack_test_scaled  = scaler.transform(X_stack_test)

# ---------------------------
# 5. Select a Few Original Features
# ---------------------------
# Sometimes, the original features may carry additional signal.
# Here we use SelectKBest to choose the top 5 features.
selector = SelectKBest(score_func=f_classif, k=5)
X_orig_selected = selector.fit_transform(X, y)
X_test_orig_selected = selector.transform(X_test)

# ---------------------------
# 6. Create Hybrid Meta-Features (Stacking + Original)
# ---------------------------
# Combine the scaled stacking predictions with selected original features.
X_hybrid_train = np.concatenate((X_stack_train_scaled, X_orig_selected), axis=1)
X_hybrid_test  = np.concatenate((X_stack_test_scaled, X_test_orig_selected), axis=1)

# ---------------------------
# 7. Define and Train a Deeper Neural Network Meta Learner
# ---------------------------
# Using Keras to define a penalized, deeper model with dropout.
input_dim = X_hybrid_train.shape[1]

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.AUC(name='auc')])

# Optional: split a holdout set to monitor progress (here we train on the full stacking data)
# X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(X_hybrid_train, y, test_size=0.2, random_state=42)

# Train the neural network
# (Increase epochs or adjust batch_size as needed—be wary of overfitting)
history = model.fit(X_hybrid_train, y, epochs=50, batch_size=32, verbose=1)

# Evaluate on training data (note: this is not a true holdout)
train_auc = model.evaluate(X_hybrid_train, y, verbose=0)[1]
print("Hybrid Neural Net Meta Learner Training AUC:", train_auc)

# ---------------------------
# 8. Generate Final Predictions and Create Submission
# ---------------------------
final_pred = model.predict(X_hybrid_test).reshape(-1)

submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': final_pred
})
submission.to_csv('submission_hybrid_final.csv', index=False)
print("Final submission file 'submission_hybrid_final.csv' generated!")



import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
import catboost as cb

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Drop the identifier column; fill missing values with the median
X = train.drop(['id', 'rainfall'], axis=1).fillna(train.median())
y = train['rainfall']
X_test = test.drop('id', axis=1).fillna(test.median())
test_ids = test['id']

# ---------------------------
# 2. Define OOF Function Using 10-Fold StratifiedKFold
# ---------------------------
def get_oof(model, X, y, X_test, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_train = np.zeros(X.shape[0])
    oof_test_folds = np.zeros((n_splits, X_test.shape[0]))
    
    for i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        oof_train[val_idx] = model.predict_proba(X_val)[:, 1]
        oof_test_folds[i, :] = model.predict_proba(X_test)[:, 1]
    
    oof_test = oof_test_folds.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# ---------------------------
# 3. Define and Tune Base Models
# ---------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=10, random_state=42)
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                          use_label_encoder=False, eval_metric='logloss', random_state=42)
lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42)
cat_model = cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, 
                                  random_seed=42, verbose=0)
et_model = ExtraTreesClassifier(n_estimators=300, max_depth=15, min_samples_leaf=10, random_state=42)

# ---------------------------
# 4. Generate Out-of-Fold Predictions for Base Models
# ---------------------------
print("Generating OOF predictions for base models...")
oof_rf_train, oof_rf_test   = get_oof(rf, X, y, X_test, n_splits=10)
oof_xgb_train, oof_xgb_test = get_oof(xgb_model, X, y, X_test, n_splits=10)
oof_lgb_train, oof_lgb_test = get_oof(lgb_model, X, y, X_test, n_splits=10)
oof_cat_train, oof_cat_test = get_oof(cat_model, X, y, X_test, n_splits=10)
oof_et_train, oof_et_test   = get_oof(et_model, X, y, X_test, n_splits=10)

# Stack predictions horizontally
X_stack_train = np.concatenate((oof_rf_train, oof_xgb_train, oof_lgb_train, oof_cat_train, oof_et_train), axis=1)
X_stack_test  = np.concatenate((oof_rf_test,  oof_xgb_test,  oof_lgb_test,  oof_cat_test,  oof_et_test), axis=1)

# Optionally scale the stacked features
scaler = StandardScaler()
X_stack_train_scaled = scaler.fit_transform(X_stack_train)
X_stack_test_scaled  = scaler.transform(X_stack_test)

# ---------------------------
# 5. Split Meta Data for Local Validation of Meta Models
# ---------------------------
X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(X_stack_train_scaled, y, test_size=0.2, random_state=42)

# ---------------------------
# 6A. Train Meta Model 1: Logistic Regression
# ---------------------------
meta_lr = LogisticRegression(C=1.0, random_state=42, solver='liblinear')
meta_lr.fit(X_meta_train, y_meta_train)
pred_meta_lr = meta_lr.predict_proba(X_meta_val)[:, 1]
auc_lr = roc_auc_score(y_meta_val, pred_meta_lr)
print("Meta Logistic Regression ROC AUC: {:.5f}".format(auc_lr))

# ---------------------------
# 6B. Train Meta Model 2: LightGBM
# ---------------------------
meta_lgb = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=42)
meta_lgb.fit(X_meta_train, y_meta_train)
pred_meta_lgb = meta_lgb.predict_proba(X_meta_val)[:, 1]
auc_lgb = roc_auc_score(y_meta_val, pred_meta_lgb)
print("Meta LightGBM ROC AUC: {:.5f}".format(auc_lgb))

# ---------------------------
# 7. Blend Meta Model Predictions on the Holdout Meta Set
# ---------------------------
# For simplicity, weight each prediction equally (you can also try optimizing weights)
pred_meta_combined_val = 0.5*pred_meta_lr + 0.5*pred_meta_lgb
auc_combined = roc_auc_score(y_meta_val, pred_meta_combined_val)
print("Combined Meta Prediction ROC AUC on Validation: {:.5f}".format(auc_combined))

# ---------------------------
# 8. Retrain Meta Models on Full Stacked Train Set and Generate Final Test Predictions
# ---------------------------
meta_lr.fit(X_stack_train_scaled, y)
meta_lgb.fit(X_stack_train_scaled, y)

pred_test_lr = meta_lr.predict_proba(X_stack_test_scaled)[:, 1]
pred_test_lgb = meta_lgb.predict_proba(X_stack_test_scaled)[:, 1]

# Final blended prediction (adjust weights if desired)
final_pred = 0.5 * pred_test_lr + 0.5 * pred_test_lgb

# ---------------------------
# 9. Create the Submission File
# ---------------------------
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': final_pred
})
submission.to_csv('submission_blended.csv', index=False)
print("Final submission file 'submission_blended.csv' generated!")



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import optuna

# Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Handle Missing Values
train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)

# Feature Engineering
train['feature_sum'] = train.drop(columns=['id', 'rainfall']).sum(axis=1)
test['feature_sum'] = test.drop(columns=['id']).sum(axis=1)
train['feature_mean'] = train.drop(columns=['id', 'rainfall']).mean(axis=1)
test['feature_mean'] = test.drop(columns=['id']).mean(axis=1)
train['feature_std'] = train.drop(columns=['id', 'rainfall']).std(axis=1)
test['feature_std'] = test.drop(columns=['id']).std(axis=1)

# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
train_poly = poly.fit_transform(train.drop(columns=['id', 'rainfall']))
test_poly = poly.transform(test.drop(columns=['id']))

# Data Preprocessing
for col in train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

scaler = StandardScaler()
X_train = scaler.fit_transform(train_poly)
X_test = scaler.transform(test_poly)
y_train = train['rainfall']

# Handling Class Imbalance
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Hyperparameter Tuning for XGBoost
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.002, 0.15, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True)
    }
    model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, y_pred))
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Train Stacking Model
xgb_model = XGBClassifier(**best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
lgb_model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=42)
cat_model = CatBoostClassifier(n_estimators=1000, learning_rate=0.05, verbose=0, random_state=42)
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_features = np.zeros((X_train.shape[0], 4))
test_meta_features = np.zeros((X_test.shape[0], 4))

for i, model in enumerate([xgb_model, lgb_model, cat_model, mlp_model]):
    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
    meta_features[:, i] = oof_preds
    test_meta_features[:, i] = test_preds

# Train Meta-Model
meta_model = LogisticRegression()
meta_model.fit(meta_features, y_train)
y_pred_meta = meta_model.predict_proba(test_meta_features)[:, 1]

# Final Submission
test['rainfall'] = y_pred_meta
submission = test[['id', 'rainfall']]
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")





