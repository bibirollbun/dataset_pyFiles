import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from lightgbm import early_stopping
from lightgbm import log_evaluation
from xgboost.callback import EarlyStopping
import numpy as np


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


data_test= pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


df['MAP'] = (df['systolic_bp'] + 2*df['diastolic_bp']) / 3

df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']

df['chol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-6)

df['bmi_age'] = df['bmi'] * df['age']

df['activity_intensity'] = df['physical_activity_minutes_per_week'] / (df['bmi'] + 1)




df_test['MAP'] = (df_test['systolic_bp'] + 2*df_test['diastolic_bp']) / 3

df_test['pulse_pressure'] = df_test['systolic_bp'] - df_test['diastolic_bp']

df_test['chol_ratio'] = df_test['ldl_cholesterol'] / (df_test['hdl_cholesterol'] + 1e-6)

df_test['bmi_age'] = df_test['bmi'] * df_test['age']

df_test['activity_intensity'] = df_test['physical_activity_minutes_per_week'] / (df_test['bmi'] + 1)



log_cols = [
    'bmi',
    'physical_activity_minutes_per_week',
    'ldl_cholesterol',
    'triglycerides',
    'total_cholesterol',
]
for col in log_cols:
    if col in df.columns:
        df[f'log_{col}'] = np.log1p(df[col])
for col in log_cols:
    if col in df_test.columns:
        df_test[f'log_{col}'] = np.log1p(df_test[col])



df_test = df_test.drop(['id'],axis=1)
x3 = pd.get_dummies(df_test,drop_first=True)


x = df.drop(['id','diagnosed_diabetes'],axis = 1)
y = df['diagnosed_diabetes']


x_encode = pd.get_dummies(x,drop_first=True)


X = x_encode  # get_dummies yapılmış

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_pred_xgb = np.zeros(len(x3))
test_pred_lgb = np.zeros(len(x3))
test_pred_cat = np.zeros(len(x3))


xgb_params = dict(
    objective = 'binary:logistic',eval_metric = 'auc',
        n_estimators = 3000,
        learning_rate = 0.08,
        max_depth = 4,
        subsample = 0.8,
        colsample_bytree = 0.9,
        reg_lambda = 1.5,
        scale_pos_weight = 263693/436307,
        enable_categorical = False,
        random_state = 42,
        n_jobs = -1,
        early_stopping_rounds = 150
)



lgb_params = dict(
    objective = 'binary',
        metric = 'auc',
        n_estimators = 5000,
        learning_rate = 0.08,
        num_leaves = 64,
        max_depth = 3,
        min_child_samples = 40,
        subsample = 0.9,
        colsample_bytree = 0.9,
        scale_pos_weight = 263693/436307,
        random_state = 42,
        n_jobs = -1,
        verbose = -1
)



cat_params = dict(
    iterations=3000, learning_rate=0.07,
        depth=7, l2_leaf_reg=5,
        loss_function='Logloss', eval_metric='AUC',
        scale_pos_weight=263693/436307,
        verbose=0, random_seed=42, allow_writing_files=False
)



for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr,eval_set=[(X_va, y_va)],verbose=False)
    oof_xgb[va_idx] = xgb_model.predict_proba(X_va)[:,1]
    test_pred_xgb += xgb_model.predict_proba(x3)[:,1] / n_splits

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_tr, y_tr,eval_set=[(X_va, y_va)],callbacks=[early_stopping(stopping_rounds=150), log_evaluation(0)])
    oof_lgb[va_idx] = lgb_model.predict_proba(X_va)[:,1]
    test_pred_lgb += lgb_model.predict_proba(x3)[:,1] / n_splits

    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_tr, y_tr,eval_set=[(X_va, y_va)],verbose=False,early_stopping_rounds=150)
    oof_cat[va_idx] = cat_model.predict_proba(X_va)[:,1]
    test_pred_cat += cat_model.predict_proba(x3)[:,1] / n_splits



print("XGB AUC:", roc_auc_score(y, oof_xgb))
print("LGB AUC:", roc_auc_score(y, oof_lgb))
print("CAT AUC:", roc_auc_score(y, oof_cat))


final_pred = (
    0.23 * test_pred_xgb +
    0.47 * test_pred_lgb +
    0.30 * test_pred_cat
)



result=pd.DataFrame()
result['id']=data_test['id']
result['diagnosed_diabetes']=final_pred


result=result.set_index('id')


result

