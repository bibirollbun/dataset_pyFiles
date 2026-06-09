import pandas as pd
from pandas import DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as w
pd.set_option("display.max_columns", None)
w.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib

class pathing_init_dawg():
    def __init__(self):
        self.train_path = '/kaggle/input/playground-series-s5e12/train.csv'
        self.test_path = '/kaggle/input/playground-series-s5e12/test.csv'
        self.submission_path = '/kaggle/input/playground-series-s5e12/sample_submission.csv'
        self.target = 'diagnosed_diabetes'
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        self.submission_df = pd.read_csv(self.submission_path)
        print('Datasets loaded successfully!')

dawg_set = pathing_init_dawg()

def feature_engg_function(df: DataFrame) -> DataFrame:
    gender_map = {'Female': 0,'Male': 1,'Other': 2}
    ethnicity_map = {'Asian': 0,'Black': 1,'Hispanic': 2,'White': 3,'Other': 4}
    education_map = {'No formal': 0,'Highschool': 1,'Graduate': 2,'Postgraduate': 3}
    income_map = {'Low': 0,'Lower-Middle': 1,'Middle': 2,'Upper-Middle': 3,'High': 4}
    smoking_map = {'Never': 0,'Former': 1,'Current': 2}
    employment_map = {'Unemployed': 0,'Student': 1,'Employed': 2,'Retired': 3}

    df['gender'] = df['gender'].map(gender_map)
    df['ethnicity'] = df['ethnicity'].map(ethnicity_map)
    df['education_level'] = df['education_level'].map(education_map)
    df['income_level'] = df['income_level'].map(income_map)
    df['smoking_status'] = df['smoking_status'].map(smoking_map)
    df['employment_status'] = df['employment_status'].map(employment_map)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['bmi_waist_ratio'] = df['bmi'] / df['waist_to_hip_ratio']
    df['cholesterol_ratio'] = df['cholesterol_total'] / df['hdl_cholesterol']
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / df['hdl_cholesterol']
    df['activity_to_screen_ratio'] = (df['physical_activity_minutes_per_week'] / (df['screen_time_hours_per_day'] + 1))
    df['sleep_efficiency_score'] = (df['sleep_hours_per_day'] / (df['screen_time_hours_per_day'] + 1))
    df['bp_risk_score'] = ((df['systolic_bp'] / 120) * (df['diastolic_bp'] / 80))
    return df

def build_preprocessing(df):
    num = df.select_dtypes(include=['number']).columns.tolist()
    cat = df.select_dtypes(include=['object']).columns.tolist()
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                             ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                             ('scaler', StandardScaler())])
    preprocessing = ColumnTransformer([('pipe1', cat_pipeline, cat),
                                       ('pipe2', num_pipeline, num)])
    return preprocessing

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 10000,
    'learning_rate': 0.005,
    'num_leaves': 95,
    'max_depth': 8, 
    'min_child_samples': 25, 
    'subsample': 0.8,
    'colsample_bytree': 0.7, 
    'subsample_freq': 1,
    'min_gain_to_split': 0.02,
    'reg_alpha': 3,
    'reg_lambda': 1.2, 
    'scale_pos_weight': None, 
    'max_bin': 255,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'early_stopping_rounds': 100
}

xgb_params = {
    'n_estimators': 10000,
    'max_leaves': 95,
    'min_child_weight': 2.0,
    'max_depth': 8,
    'grow_policy': 'lossguide',
    'learning_rate': 0.005, 
    'tree_method': 'hist',
    'subsample': 0.8,
    'colsample_bylevel': 0.65, 
    'colsample_bytree': 0.7, 
    'colsample_bynode': 0.8, 
    'reg_alpha': 3.0,
    'reg_lambda': 1.2, 
    'gamma': 0.1, 
    'max_bin': 256,
    'enable_categorical': False,
    'max_cat_to_onehot': 1,
    'device': 'cuda',
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': 0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc', 
    'early_stopping_rounds': 100
}

cb_params = {
    'iterations': 10000,
    'learning_rate': 0.005,
    'depth': 8,
    'l2_leaf_reg': 3,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'verbose': 0,
    'early_stopping_rounds': 100,
    'allow_writing_files': False
}

def train_ensemble_with_auc_and_submission(
        X_train_split, y_train_split, test, 
        lgb_params, xgb_params, cb_params, preprocessing,
        weights={'lgb': 0.34, 'xgb': 0.33, 'cb': 0.33}):

    num = X_train_split.select_dtypes(include=['number']).columns.tolist()
    cat = X_train_split.select_dtypes(include=['object']).columns.tolist()

    oof_pred_lgb = np.zeros(len(X_train_split))
    oof_pred_xgb = np.zeros(len(X_train_split))
    oof_pred_cb  = np.zeros(len(X_train_split))
    
    test_pred_lgb = np.zeros(len(test))
    test_pred_xgb = np.zeros(len(test))
    test_pred_cb  = np.zeros(len(test))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr, val) in enumerate(skf.split(X_train_split, y_train_split)):
        X_tr, X_val = X_train_split.iloc[tr], X_train_split.iloc[val]
        y_tr, y_val = y_train_split.iloc[tr], y_train_split.iloc[val]

        preprocessing.fit(X_tr, y_tr)
        X_tr_pre = preprocessing.transform(X_tr)
        X_val_pre = preprocessing.transform(X_val)
        X_test_pre = preprocessing.transform(test)

        dtrain_lgb = lgb.Dataset(X_tr_pre, label=y_tr)
        dval_lgb = lgb.Dataset(X_val_pre, label=y_val)

        model_lgb = lgb.train(
            params=lgb_params,
            train_set=dtrain_lgb,
            valid_sets=[dtrain_lgb, dval_lgb],
            callbacks=[lgb.log_evaluation(0)]
        )
        oof_pred_lgb[val] = model_lgb.predict(X_val_pre)
        test_pred_lgb += model_lgb.predict(X_test_pre) / skf.n_splits

        dtrain_xgb = xgb.DMatrix(X_tr_pre, label=y_tr)
        dval_xgb = xgb.DMatrix(X_val_pre, label=y_val)
        dtest_xgb = xgb.DMatrix(X_test_pre)

        model_xgb = xgb.train(
            params=xgb_params,
            dtrain=dtrain_xgb,
            evals=[(dtrain_xgb, 'train'), (dval_xgb, 'valid')],
            verbose_eval=False
        )
        oof_pred_xgb[val] = model_xgb.predict(dval_xgb)
        test_pred_xgb += model_xgb.predict(dtest_xgb) / skf.n_splits

        model_cb = CatBoostClassifier(**cb_params)
        model_cb.fit(
            X_tr_pre, y_tr,
            eval_set=(X_val_pre, y_val),
            verbose=False,
            early_stopping_rounds=100
        )
        oof_pred_cb[val] = model_cb.predict_proba(X_val_pre)[:, 1]
        test_pred_cb += model_cb.predict_proba(X_test_pre)[:, 1] / skf.n_splits

    auc_lgb = roc_auc_score(y_train_split, oof_pred_lgb)
    auc_xgb = roc_auc_score(y_train_split, oof_pred_xgb)
    auc_cb  = roc_auc_score(y_train_split, oof_pred_cb)

    y_pred_ensemble = (weights['lgb'] * test_pred_lgb + 
                       weights['xgb'] * test_pred_xgb + 
                       weights['cb'] * test_pred_cb)

    submission = pd.DataFrame({
        'id': test['id'],
        'diagnosed_diabetes': y_pred_ensemble
    })

    submission.to_csv('submission.csv', index=False)

    return {
        'auc_lgb': auc_lgb,
        'auc_xgb': auc_xgb,
        'auc_cb': auc_cb,
        'submission_path': 'submission.csv'
    }

df = dawg_set.train_df
df = feature_engg_function(df)
dawg_set.test_df = feature_engg_function(dawg_set.test_df)
X = df.drop(['diagnosed_diabetes'], axis=1)
y = df['diagnosed_diabetes']

X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
preprocessing = build_preprocessing(X_train_split)

result = train_ensemble_with_auc_and_submission(
    X_train_split,
    y_train_split,
    dawg_set.test_df,
    lgb_params,
    xgb_params,
    cb_params,
    preprocessing,
    weights={'lgb': 0.34, 'xgb': 0.33, 'cb': 0.33})

print(result['auc_lgb'])
print(result['auc_xgb'])
print(result['auc_cb'])
print(result['submission_path'])




