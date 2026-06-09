import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations 
import optuna

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier as CBC, Pool
from lightgbm import LGBMClassifier as lgbm
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.simplefilter('ignore')


sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
orig = pd.read_csv('/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv')
print(f'Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}')


TARGET = 'diagnosed_diabetes'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS_BASE = ['gender','ethnicity','education_level','income_level','smoking_status', 'employment_status']
NUMS_BASE = ['age','alcohol_consumption_per_week','physical_activity_minutes_per_week','diet_score',
             'sleep_hours_per_day', 'screen_time_hours_per_day','bmi','waist_to_hip_ratio',
             'systolic_bp','diastolic_bp','heart_rate','cholesterol_total','hdl_cholesterol',
             'ldl_cholesterol','triglycerides','family_history_diabetes','hypertension_history',
             'cardiovascular_history']


train.head()


train.info()


ORIG = []
for col in BASE:
    #MEAN
    mean_map = orig.groupby(col)[TARGET].mean()
    new_col_name_mean = f'orig_mean_{col}'
    mean_map.name = new_col_name_mean
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_col_name_mean)

     # COUNT
    new_count_col_name = f"orig_count_{col}"
    count_map = orig.groupby(col).size().reset_index(name=new_count_col_name)
    
    train = train.merge(count_map, on=col, how='left')
    test = test.merge(count_map, on=col, how='left')
    ORIG.append(new_count_col_name)
print(f'{len(ORIG)} ORIG Features created.')


CATS = CATS_BASE.copy()
FEATURES = BASE + ORIG
X = train[FEATURES]
y = train[TARGET]


FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)


xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'colsample_bytree': 0.5928070106039101,
    'subsample': 0.5424734486551225,
    'n_estimators': 1978, 'learning_rate': 0.016868364653916668,
    'early_stopping_rounds': 200,
    'random_state': 42,
    'n_jobs': -1,
    'enable_categorical': True,
    'min_child_weight': 0.00016097129325809228,
    'device': 'cuda',
}


oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_ind, val_ind) in enumerate(skf.split(X, y), 1):
    print(f"Fold: {fold}/{FOLDS}")
    X_train, X_val = X.iloc[train_ind], X.iloc[val_ind]
    y_train, y_val = y.iloc[train_ind], y.iloc[val_ind]
    X_test = test[FEATURES].copy()
    
    X_train[CATS] = X_train[CATS].astype('category')
    X_val[CATS] = X_val[CATS].astype('category')
    X_test[CATS] = X_test[CATS].astype('category')

    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1000)

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_ind] = val_preds

    fold_auc = roc_auc_score(y_val, val_preds)
    print(f"Fold: {fold}, AUC: {fold_auc:.4f}")

    test_preds += model.predict_proba(X_test)[:, 1] / FOLDS
xgb_overall_auc = roc_auc_score(y, oof_preds)
print(f'====================')
print(f'XGBOOST: Overall OOF AUC: {xgb_overall_auc:.4f}')
print(f'====================')


submission = pd.DataFrame({'id': sub['id'], TARGET: test_preds})
submission.to_csv('submission.csv', index=False)
submission.head()




