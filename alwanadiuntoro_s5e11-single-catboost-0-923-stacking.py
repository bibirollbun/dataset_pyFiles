import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


df_train.head()


df_train.info()


def fetch_cols(df):
    numerical = df.columns[df.dtypes != object]
    categorical = df.columns[df.dtypes == object]

    return numerical, categorical

target = 'loan_paid_back'


numerical, categorical = fetch_cols(df_train)
plt.figure(figsize=(15,8))
df_train[numerical].hist(figsize=(20,20), edgecolor='lightblue', bins=20)
plt.show()


plt.figure(figsize=(20,10))
corr = df_train[numerical].corr()
sns.heatmap(corr, cmap='coolwarm', fmt='.2f', annot=True)
plt.show()


df = df_train.copy()


tmp=df_train['grade_subgrade'].unique()
np.sort(tmp)


df['grade'] = df['grade_subgrade'].str.get(0)
df['subgrade'] = df['grade_subgrade'].str.get(1)


df = df.drop('grade_subgrade', axis=1)


# Ordinal mapping
grade_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5}
subgrade_map = {'1':0, '2':1, '3':2, '4':3, '5':4}


df['grade'] = df['grade'].map(grade_map)


df['subgrade'] = df['subgrade'].map(subgrade_map)


numerical, categorical = fetch_cols(df)


plt.figure(figsize=(20,10))
corr = df[numerical].corr()
sns.heatmap(corr, cmap='coolwarm', fmt='.2f', annot=True)
plt.show()


for col_name, col_data in df.items():
    val, counts = np.unique(df[col_name], return_counts=True)
    print(f"{col_name}: {len(counts)}")


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)

encoded = ohe.fit_transform(df[categorical])
encoded_cols = ohe.get_feature_names_out(categorical)


encoded_df = pd.DataFrame(encoded, columns=encoded_cols)


df_processed = df.drop(columns=categorical)
df_final = pd.concat([df_processed, encoded_df], axis=1)


from lightgbm import LGBMClassifier


lgb = LGBMClassifier()


y = df_final[target]
X_train = df_final.drop([target, 'id'], axis=1)


from sklearn.model_selection import StratifiedKFold, cross_validate

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv = cross_validate(lgb, X_train, y, cv=skf, scoring='roc_auc', n_jobs=-1, return_estimator=True)


cv


cv['test_score'].mean()


# Check the feature importances in the highest score cv
    
features = pd.DataFrame({
    'feature': cv['estimator'][0].feature_name_,
    'importance': cv['estimator'][0].feature_importances_
}).sort_values('importance', ascending=False)

features


def preprocess_gradesubgrade(df):
    df_all = df.copy()
    df_all['grade'] = df_all['grade_subgrade'].str.get(0)
    df_all['subgrade'] = df_all['grade_subgrade'].str.get(1)
    
    grade_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5}
    subgrade_map = {'1':0, '2':1, '3':2, '4':3, '5':4}

    df_all['grade'] = df_all['grade'].map(grade_map)
    df_all['subgrade'] = df_all['subgrade'].map(subgrade_map)

    df_all = df_all.drop('grade_subgrade', axis=1)

    return df_all


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

numerical_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline ([
    ('ohe', OneHotEncoder(sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, make_column_selector(dtype_include=np.number)),
        ('cat', categorical_pipeline, make_column_selector(dtype_include=object))
    ],
    remainder='passthrough'
)

pipeline = Pipeline ([
    ('preprocess', preprocess)
])


# Wrap the code
def x_train_test(df_train, df_test, pipeline):
    X_train = df_train.copy()
    X_test = df_test.copy()

    X_train = pd.DataFrame(preprocess_gradesubgrade(X_train))
    X_test = pd.DataFrame(preprocess_gradesubgrade(X_test))

    X_train = X_train.drop(target, axis=1)
    
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    feature_names = pipeline.get_feature_names_out()

    X_train_final = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_final = pd.DataFrame(X_test_processed, columns=feature_names)
    
    return X_train_final, X_test_final


y = df_train[target]


X_train, X_test = x_train_test(df_train, df_test, pipeline)


X_train


lgb.fit(X_train, y)


y_prob = lgb.predict_proba(X_test)


df_test_id = df_test['id']
y_prob = y_prob[:, 1]


df_test_id, y_prob


submission = pd.DataFrame({
    'id': df_test_id,
    'loan_paid_back': y_prob
})

submission


submission.to_csv('submission_v1.csv', index=False)
print("Base model submission saved.\nScore: 0.91977")


def feature_engineering(df):
    df_all = df.drop('id', axis=1)
    
    df_all['loan_to_income'] = df_all['loan_amount'] / df_all['annual_income']
    df_all['disposable_income'] = df_all['annual_income'] - df_all['loan_amount']
    df_all['interest_burden'] = df_all['loan_amount'] * df_all['interest_rate']
    
    df_all['risk_interaction'] = df_all['debt_to_income_ratio'] * (1/df_all['credit_score'])
    df_all['existing_monthly_debt'] = df_all['debt_to_income_ratio'] * (df_all['annual_income']/12)
    df_all['disposable_monthly_income'] = (df_all['annual_income']/12) - df_all['existing_monthly_debt']

    return df_all


df_train_fe = feature_engineering(df_train)
df_test_fe = feature_engineering(df_test)
y = df_train[target]


X_train_fe, X_test_fe = x_train_test(df_train_fe, df_test_fe, pipeline)


from xgboost import XGBClassifier

xgb = XGBClassifier(device='cuda', verbosity=0)
lgb = LGBMClassifier(device='gpu', verbosity=-1)


cv_xgb = cross_validate(estimator=xgb, X=X_train_fe, y=y, scoring='roc_auc', cv=skf, n_jobs=-1, return_estimator=True)
cv_xgb['test_score'].mean()


cv_lgb = cross_validate(estimator=lgb, X=X_train_fe, y=y, scoring='roc_auc', cv=skf, n_jobs=-1, return_estimator=True)
cv_lgb['test_score'].mean()


from catboost import cv, Pool

pool = Pool(X_train_fe, y)

params = {
    'iterations': 200,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'task_type': 'GPU'
}

cv_cat = cv(pool, params, fold_count=5, return_models=True, verbose=0)


features_lgb = pd.DataFrame({
    'features': cv_lgb['estimator'][0].feature_name_,
    'importances': cv_lgb['estimator'][0].feature_importances_,
}).sort_values('importances', ascending=False)

features_lgb


features_xgb = cv_xgb['estimator'][0].get_booster().get_score(importance_type='gain')
features_xgb = dict(sorted(features_xgb.items(), key=lambda item: item[1]))

features_xgb = pd.DataFrame.from_dict(features_xgb, orient='index')
features_xgb


catboost = cv_cat[1][4]
get_features_cat = catboost.get_feature_importance(pool)

features_cat = pd.DataFrame({
    'features': X_train_fe.columns,
    'importances': get_features_cat
}).sort_values('importances', ascending=False)

features_cat


from catboost import CatBoostClassifier

xgb = XGBClassifier(device='cuda', verbosity=0)
lgb = LGBMClassifier(device='gpu', verbosity=-1)
cat = CatBoostClassifier(verbose=False, task_type='GPU')


y = df_train[target]


xgb.fit(X_train_fe, y)
lgb.fit(X_train_fe, y)
cat.fit(X_train_fe, y)


y_prob_xgb = xgb.predict_proba(X_test_fe)
y_prob_lgb = lgb.predict_proba(X_test_fe)
y_prob_cat = cat.predict_proba(X_test_fe)


# XGB Score: 0.92003
pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_prob_xgb[:,1]
}).to_csv('submission_xgb.csv', index=False)

# LGBM Score: 0.91964
pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_prob_lgb[:,1]
}).to_csv('submission_lgb.csv', index=False)

# CatBoost Score: 0.92287
pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_prob_cat[:,1]
}).to_csv('submission_cat.csv', index=False)


import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

## XGB
def objective_xgb(trial):
    X_train, X_test, y_train, y_test = train_test_split(X_train_fe, y, test_size=0.2, random_state=8)

    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',

        'verbosity': 0,

        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_split_loss': trial.suggest_float('min_split_loss', 1e-3, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2, 6),
        'subsample': trial.suggest_float('subsample', 0.6, 1),
        'colsample_bytree': trial.suggest_float('colsabmple_bytree', 0.6, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True)
    }

    model = XGBClassifier(**param)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )

    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)

    return auc


## LGBM
def objective_lgbm(trial):
    X_train, X_test, y_train, y_test = train_test_split(X_train_fe, y, test_size=0.2, random_state=8)

    param = {
        'objective': 'binary',
        'eval_metric': 'auc',
        'boosting_type': 'gbdt',
        'device': 'gpu',

        'verbosity': -1,

        'n_estimators': trial.suggest_int('n_estimators', 300, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-2, 0.3),
        
        'num_leaves': trial.suggest_int('num_leaves', 20, 100), 
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2, 6),
    }

    model = lgb.LGBMClassifier(**param)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)

    return auc


## CatBoost
def objective_cat(trial):
    X_train, X_test, y_train, y_test = train_test_split(X_train_fe, y, test_size=0.2, random_state=8)

    param = {
        'objective': 'Logloss',
        'eval_metric': 'AUC',
        'task_type': 'GPU',
        'verbose': False,

        'iterations': trial.suggest_int('iterations', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),

        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 6.0),

        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
    }

    if param['bootstrap_type'] == 'Bayesian':
        param['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    elif param['bootstrap_type'] == 'Bernoulli':
        param['subsample'] = trial.suggest_float('subsample', 0.1, 1.0)

    model = CatBoostClassifier(**param)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )

    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)

    return auc


## Wrap it all
objectives = {
    'XGB': objective_xgb,
    'LGBM': objective_lgbm,
    'Cat': objective_cat
}


MLA_compare = pd.DataFrame(columns = ['MLA Name', 'MLA Params', 'MLA RMSE'])

def study_objective(objective, n_trials=20):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=600, n_jobs=1)

    MLA_compare.loc[len(MLA_compare)] = [objective.__name__, study.best_params, study.best_value]


study_objective(objectives['Cat'], 15)
study_objective(objectives['XGB'])
study_objective(objectives['LGBM'])


for index, row in MLA_compare.iterrows():
    print(row['MLA Name'])
    print(row['MLA Params'])
    print(row['MLA RMSE'])
    print()


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

xgb_param = {'n_estimators': 756, 'learning_rate': 0.17516134534336256, 'max_depth': 4, 'min_split_loss': 3.5050987201731374, 'min_child_weight': 4, 'scale_pos_weight': 2.016502708517104, 'subsample': 0.6981818128930247, 'colsabmple_bytree': 0.6419165516267493, 'reg_alpha': 8.64207967389963e-06, 'reg_lambda': 4.920854074591851, 'verbosity': 0, 'tree_method': 'hist', 'device': 'cuda'}
lgbm_param = {'n_estimators': 1800, 'learning_rate': 0.04938937458184534, 'num_leaves': 46, 'max_depth': 4, 'min_child_samples': 36, 'subsample': 0.786408084548339, 'colsample_bytree': 0.7888662156142533, 'reg_alpha': 5.27437827174014e-06, 'reg_lambda': 1.813521540314814, 'scale_pos_weight': 2.7335147307789898, 'verbosity': -1, 'device': 'gpu'}
cat_param = {'iterations': 761, 'learning_rate': 0.24801043078777377, 'depth': 5, 'l2_leaf_reg': 0.34965324215736376, 'random_strength': 1.3171681031281477e-06, 'scale_pos_weight': 1.8156013831339937, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 0.33254098526009246, 'verbose': False, 'task_type': 'GPU'}

xgb_hyper = XGBClassifier(**xgb_param)
lgbm_hyper = LGBMClassifier(**lgbm_param)
cat_hyper = CatBoostClassifier(**cat_param)


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimators = [
    ('xgb', xgb_hyper),
    ('lgbm', lgbm_hyper),
    ('cat', cat_hyper),
]

clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=1, cv=3, passthrough=False)

clf.fit(X_train_fe, y)


y_pred_train = clf.predict_proba(X_train_fe)[:, 1]
roc_auc_score(y, y_pred_train)


y_pred = clf.predict_proba(X_test_fe)


y_pred = y_pred[:, 1]
y_pred


## Score: 0.92128
submission_stack = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_pred
})

submission_stack.to_csv('submission_stack.csv', index=False)


xgb_hyper.fit(X_train_fe, y)
lgbm_hyper.fit(X_train_fe, y)
cat_hyper.fit(X_train_fe, y)


y_pred_xgb = xgb_hyper.predict_proba(X_train_fe)[:, 1]
y_pred_lgbm = lgbm_hyper.predict_proba(X_train_fe)[:, 1]
y_pred_cat = cat_hyper.predict_proba(X_train_fe)[:, 1]


blend_pred = (0.6 * y_pred_cat) + (0.3 * y_pred_lgbm) + (0.1 * y_pred_xgb)
roc_auc_score(y, blend_pred)


y_pred_xgb = xgb_hyper.predict_proba(X_test_fe)[:, 1]
y_pred_lgbm = lgbm_hyper.predict_proba(X_test_fe)[:, 1]
y_pred_cat = cat_hyper.predict_proba(X_test_fe)[:, 1]


submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_pred_xgb
})

submission.to_csv('hyper_xgb.csv', index=False)

submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_pred_lgbm
})

submission.to_csv('hyper_lgbm.csv', index=False)

submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': y_pred_cat
})

submission.to_csv('hyper_cat.csv', index=False)


blend_pred = (0.6 * y_pred_cat) + (0.3 * y_pred_lgbm) + (0.1 * y_pred_xgb)


## Score: 0.92113
submission_blend = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': blend_pred
})

submission_blend.to_csv('submission_blend.csv', index=False)

