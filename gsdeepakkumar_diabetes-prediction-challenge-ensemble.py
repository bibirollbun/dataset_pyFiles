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
import os
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


N_SPLITS = 5
RANDOM_STATE = 42


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


train.head(2)


test.head(2)


def feature_engineering(df):
    # 1. BMI categories (Underweight, Normal, Overweight, Obese)
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
    
    # 2. Blood Pressure features
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['mean_arterial_pressure'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
    
    # 3. Cholesterol ratios (Traditional risk indicators)
    df['chol_ratio'] = df['cholesterol_total'] / df['hdl_cholesterol']
    df['non_hdl_chol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    # 4. Lifestyle indicators
    df['sedentary_ratio'] = df['screen_time_hours_per_day'] / (df['physical_activity_minutes_per_week'] / 60 + 1)
    
    return df


def preprocess_data(train, test):
    """
    Handle categorical encoding and feature selection.
    Ensure target column is not in the combined set for encoding.
    """
    # Identify target and common features
    target_col = 'diagnosed_diabetes'
    features = [c for c in train.columns if c != target_col]
    
    # Combine only features for uniform categorical encoding
    combined = pd.concat([train[features], test[features]], axis=0).reset_index(drop=True)
    
    # Categorical columns detected in train.csv
    cat_cols = [
        'gender', 'ethnicity', 'education_level', 'income_level', 
        'smoking_status', 'employment_status'
    ]
    
    # Simple Label Encoding for GBDTs
    for col in cat_cols:
        if col in combined.columns:
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))
    
    # Split back and re-attach target to train
    X_train_processed = combined[:len(train)].copy()
    X_test_processed = combined[len(train):].copy()
    X_train_processed[target_col] = train[target_col].values
    
    return X_train_processed, X_test_processed



def run_fold(X, y, X_test, model_type='lgb'):
    """
    Run 5-Fold Stratified CV for a specific model type.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Training {model_type} | Fold {fold+1} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if model_type == 'lgb':
            model = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=31,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=RANDOM_STATE,
                device="gpu", 
                importance_type='gain',
                verbose=-1
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      eval_metric='auc', callbacks=[lgb.early_stopping(50)])
            
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='auc',
                random_state=RANDOM_STATE,
                tree_method='hist' ,
                device='gpu',
              #  verbosity=0,
                early_stopping_rounds=50
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                       verbose=False)
            
        elif model_type == 'cat':
            model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.03,
                depth=6,
                random_seed=RANDOM_STATE,
                verbose=100,
                eval_metric='AUC',
                early_stopping_rounds=50,
                task_type='GPU'
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        
        gc.collect()
        
    print(f"\n{model_type} OOF Score: {roc_auc_score(y, oof_preds):.5f}")
    return oof_preds, test_preds


 train = feature_engineering(train)


train.head(2)


test = feature_engineering(test)


test.head(2)


train, test = preprocess_data(train, test)


train.head(2)


test.head(2)


X = train.drop(['id', 'diagnosed_diabetes'], axis=1)


y = train['diagnosed_diabetes']


X_test = test.drop(['id'], axis=1)


%%time
lgb_oof, lgb_test = run_fold(X, y, X_test, 'lgb')


%%time
xgb_oof, xgb_test = run_fold(X, y, X_test, 'xgb')


%%time
cat_oof,cat_test=run_fold(X,y,X_test,'cat')


submission=pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


#weighted avg ensemble,
final_test_preds = (lgb_test * 0.4) + (xgb_test * 0.3) + (cat_test * 0.3)


final_test_preds


submission['diagnosed_diabetes']=final_test_preds


submission.to_csv('submission.csv',index=False)




