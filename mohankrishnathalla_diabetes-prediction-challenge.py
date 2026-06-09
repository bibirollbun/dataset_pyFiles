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


!pip install imbalanced-learn==0.11.0



import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer  # ← FIX: Imputer for RF
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
print("V29 – Multi-Seed + RF/XGB/LGBM + SMOTE + Advanced Ratio FE + Platt Calibration")


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sub   = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
orig  = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')

TARGET = 'diagnosed_diabetes'


base_cols = [c for c in train.columns if c not in ['id', TARGET]]

for col in base_cols:
    mean_map = orig.groupby(col)[TARGET].mean()
    train[f"enc_mean_{col}"] = train[col].map(mean_map)
    test[f"enc_mean_{col}"]  = test[col].map(mean_map)
    
    cnt = orig.groupby(col).size()
    train[f"enc_cnt_{col}"] = np.log1p(train[col].map(cnt).fillna(0))
    test[f"enc_cnt_{col}"]  = np.log1p(test[col].map(cnt).fillna(0))


train['bmi_cat'] = pd.cut(train['bmi'], bins=[0,18.5,25,30,999], labels=[0,1,2,3]).astype(int)
test['bmi_cat']  = pd.cut(test['bmi'],  bins=[0,18.5,25,30,999], labels=[0,1,2,3]).astype(int)

train['bp_cat'] = 0
train.loc[(train['systolic_bp']>=140)|(train['diastolic_bp']>=90), 'bp_cat'] = 2
train.loc[((train['systolic_bp']>=120)&(train['systolic_bp']<140))|
          ((train['diastolic_bp']>=80)&(train['diastolic_bp']<90)), 'bp_cat'] = 1
test['bp_cat'] = train['bp_cat'].copy()

train['non_hdl'] = train['cholesterol_total'] - train['hdl_cholesterol']
test['non_hdl']  = test['cholesterol_total'] - test['hdl_cholesterol']

train['ldl_hdl_ratio'] = train['ldl_cholesterol'] / (train['hdl_cholesterol'] + 1)
test['ldl_hdl_ratio']  = test['ldl_cholesterol']  / (test['hdl_cholesterol']  + 1)
train['bmi_age_ratio'] = train['bmi'] / (train['age'] + 1)
test['bmi_age_ratio']  = test['bmi']  / (test['age']  + 1)
train['bp_mean'] = (train['systolic_bp'] + train['diastolic_bp']) / 2
test['bp_mean']  = (test['systolic_bp']  + test['diastolic_bp'])  / 2


features = base_cols + ['bmi_cat','bp_cat','non_hdl','ldl_hdl_ratio','bmi_age_ratio','bp_mean'] + \
           [c for c in train.columns if c.startswith('enc_')]

# Fill NaNs for all models
imputer = SimpleImputer(strategy='median')
train[features] = imputer.fit_transform(train[features])
test[features]  = imputer.transform(test[features])

# Label encode categoricals
cat_cols = ['bmi_cat','bp_cat'] + train.select_dtypes('object').columns.tolist()
for col in cat_cols:
    if col in train.columns:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

X = train[features]
y = train[TARGET]
X_test = test[features]

print(f"Total features: {len(features)}")


smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"After SMOTE: {X_res.shape[0]} samples")


seeds = [42, 50, 100]
oof = np.zeros(len(X_res))
test_xgb = np.zeros(len(X_test))
test_lgb = np.zeros(len(X_test))
test_cb  = np.zeros(len(X_test))
test_rf  = np.zeros(len(X_test))

print(f"\nStarting Multi-Seed 10-Fold training...\n")

for seed in seeds:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_res, y_res), 1):
        print(f"Seed {seed} Fold {fold}/10 → ", end="")
        
        X_trn, X_val = X_res.iloc[trn_idx], X_res.iloc[val_idx]
        y_trn, y_val = y_res.iloc[trn_idx], y_res.iloc[val_idx]
        
        # XGB
        m1 = xgb.XGBClassifier(n_estimators=2000, max_depth=4, learning_rate=0.008,
                               subsample=0.7, colsample_bytree=0.6,
                               reg_alpha=3.0, reg_lambda=3.5,
                               random_state=seed, tree_method='hist', n_jobs=-1, verbosity=0)
        m1.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], early_stopping_rounds=200, verbose=False)
        
        # LGBM
        m2 = lgb.LGBMClassifier(n_estimators=2000, max_depth=4, learning_rate=0.008,
                                num_leaves=20, subsample=0.7, colsample_bytree=0.6,
                                reg_alpha=3.0, reg_lambda=3.5,
                                random_state=seed, n_jobs=-1, verbose=-1)
        m2.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(200)])
        
        # CatBoost
        m3 = cb.CatBoostClassifier(iterations=2000, depth=4, learning_rate=0.008,
                                   l2_leaf_reg=10.0, random_seed=seed,
                                   verbose=False, early_stopping_rounds=200)
        m3.fit(X_trn, y_trn, eval_set=(X_val, y_val), verbose=False)
        
        # Random Forest (with imputed data)
        m4 = RandomForestClassifier(n_estimators=600, max_depth=8,
                                    min_samples_split=20, random_state=seed, n_jobs=-1)
        m4.fit(X_trn, y_trn)
        
        # OOF & Test Blend
        val_pred = (m1.predict_proba(X_val)[:,1] * 0.35 +
                    m2.predict_proba(X_val)[:,1] * 0.30 +
                    m3.predict_proba(X_val)[:,1] * 0.25 +
                    m4.predict_proba(X_val)[:,1] * 0.10)
        
        oof[val_idx] = val_pred
        print(f"AUC = {roc_auc_score(y_val, val_pred):.6f}")
        
        test_xgb += m1.predict_proba(X_test)[:,1] / (len(seeds)*10)
        test_lgb += m2.predict_proba(X_test)[:,1] / (len(seeds)*10)
        test_cb  += m3.predict_proba(X_test)[:,1] / (len(seeds)*10)
        test_rf  += m4.predict_proba(X_test)[:,1] / (len(seeds)*10)

print(f"\nFinal CV AUC: {roc_auc_score(y_res, oof):.6f}")


final_pred = (test_xgb * 0.35 + test_lgb * 0.30 + test_cb * 0.25 + test_rf * 0.10)


calibrator = CalibratedClassifierCV(m1, method='sigmoid', cv='prefit')
calibrator.fit(X, y)
final_pred = calibrator.predict_proba(X_test)[:,1]


sub[TARGET] = final_pred
sub.to_csv('submission_v28_fixed.csv', index=False)

print("\nsubmission_v28_fixed.csv saved!")
print(f"Mean prediction: {final_pred.mean():.5f}")

sub.head()

