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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder,KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression as MetaLR
from catboost import CatBoostClassifier  

import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)
print("\nTrain Info:")
print(train_df.info())
print("\nMissing Values in Train:")
print(train_df.isnull().sum())
print("\nTarget Distribution:")
print(train_df['diagnosed_diabetes'].value_counts(normalize=True))
print("\nTrain Head:")
print(train_df.head())
print("\nTest Head:")
print(test_df.head())


cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
num_cols = [col for col in train_df.columns if col not in cat_cols + ['id', 'diagnosed_diabetes']]
ohe_cols = ['gender']
te_cols = [col for col in cat_cols if col not in ohe_cols]


preprocessor = ColumnTransformer(
    transformers=[
        ('te', TargetEncoder(), te_cols),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False), ohe_cols),
        ('num', 'passthrough', num_cols)
    ],
    remainder='drop'
)


X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train_proc = preprocessor.fit_transform(X_train, y_train)
X_val_proc = preprocessor.transform(X_val)


print("Processed Train Shape:", X_train_proc.shape)
print("Processed Val Shape:", X_val_proc.shape)
print("\nProcessed Feature Names (first 10):")
print(preprocessor.get_feature_names_out()[:10])
print("\nTarget Split Proportions:")
print(pd.Series(y_train).value_counts(normalize=True))
print(pd.Series(y_val).value_counts(normalize=True))
print("\nSample Processed Train Row:")
print(pd.DataFrame(X_train_proc, columns=preprocessor.get_feature_names_out()).iloc[0])


lr_model = LogisticRegression(random_state=42, max_iter=2000, C=1.0)
lr_model.fit(X_train_proc, y_train)
lr_proba = lr_model.predict_proba(X_val_proc)[:, 1]
lr_auc = roc_auc_score(y_val, lr_proba)
print(f"Logistic Regression OOF AUC: {lr_auc:.4f}")


lgb_model = lgb.LGBMClassifier(
    random_state=42, n_estimators=100, learning_rate=0.1, 
    max_depth=6, verbose=-1, force_col_wise=True
)
lgb_model.fit(X_train_proc, y_train)
lgb_proba = lgb_model.predict_proba(X_val_proc)[:, 1]
lgb_auc = roc_auc_score(y_val, lgb_proba)
print(f"LightGBM OOF AUC: {lgb_auc:.4f}")


feat_names = preprocessor.get_feature_names_out()
importances_df = pd.DataFrame({
    'feature': feat_names, 
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)
print("\nTop 10 LGBM Feature Importances:")
print(importances_df)


num_cols_with_target = num_cols + ['diagnosed_diabetes']
corr_to_target = train_df[num_cols_with_target].corr()['diagnosed_diabetes'].sort_values(ascending=False)
print("\nTop 10 Numeric Correlations to Target:")
print(corr_to_target.head(10))


X_train_eng = pd.DataFrame(X_train_proc, columns=feat_names).copy()
X_val_eng = pd.DataFrame(X_val_proc, columns=feat_names).copy()


age_col = 'num__age'
bmi_col = 'num__bmi'
act_col = 'num__physical_activity_minutes_per_week'
diet_col = 'num__diet_score'
hist_col = 'num__family_history_diabetes'  # Proxy for histories
trig_col = 'num__triglycerides'

X_train_eng['inter_age_bmi'] = X_train_eng[age_col] * X_train_eng[bmi_col]
X_train_eng['inter_act_diet'] = X_train_eng[act_col] * X_train_eng[diet_col]
X_train_eng['inter_hist_trig'] = X_train_eng[hist_col] * X_train_eng[trig_col]
X_val_eng['inter_age_bmi'] = X_val_eng[age_col] * X_val_eng[bmi_col]
X_val_eng['inter_act_diet'] = X_val_eng[act_col] * X_val_eng[diet_col]
X_val_eng['inter_hist_trig'] = X_val_eng[hist_col] * X_val_eng[trig_col]


binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_train_eng['bin_age'] = binner.fit_transform(X_train_eng[[age_col]])[:, 0]
X_val_eng['bin_age'] = binner.transform(X_val_eng[[age_col]])[:, 0]


X_train_eng_arr = X_train_eng.values
X_val_eng_arr = X_val_eng.values
eng_feat_names = X_train_eng.columns.tolist()  


sub_idx = np.random.choice(len(X_train_eng_arr), size=int(0.1 * len(X_train_eng_arr)), replace=False)
X_sub = X_train_eng_arr[sub_idx]
y_sub = y_train.iloc[sub_idx].values


def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'force_col_wise': True
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_sub, y_sub)
    proba = model.predict_proba(X_val_eng_arr)[:, 1]
    return roc_auc_score(y_val, proba)


optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, show_progress_bar=False)
best_params = study.best_params
print(f"Best LGBM Params: {best_params}")
print(f"Best OOF AUC from Opt: {study.best_value:.4f}")


tuned_lgb = lgb.LGBMClassifier(**best_params, random_state=42, verbosity=-1, force_col_wise=True)
tuned_lgb.fit(X_train_eng_arr, y_train)
tuned_lgb_proba = tuned_lgb.predict_proba(X_val_eng_arr)[:, 1]
tuned_lgb_auc = roc_auc_score(y_val, tuned_lgb_proba)
print(f"Tuned LGBM OOF AUC: {tuned_lgb_auc:.4f}")


xgb_params = best_params.copy()
xgb_params.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'random_state': 42})
tuned_xgb = xgb.XGBClassifier(**xgb_params)
tuned_xgb.fit(X_train_eng_arr, y_train)
tuned_xgb_proba = tuned_xgb.predict_proba(X_val_eng_arr)[:, 1]
tuned_xgb_auc = roc_auc_score(y_val, tuned_xgb_proba)
print(f"Tuned XGBoost OOF AUC: {tuned_xgb_auc:.4f}")


ensemble = VotingClassifier(
    estimators=[('lgb', tuned_lgb), ('xgb', tuned_xgb)],
    voting='soft'
)
ensemble.fit(X_train_eng_arr, y_train)
ens_proba = ensemble.predict_proba(X_val_eng_arr)[:, 1]
ens_auc = roc_auc_score(y_val, ens_proba)
print(f"Ensemble OOF AUC: {ens_auc:.4f}")

eng_importances_df = pd.DataFrame({
    'feature': eng_feat_names,
    'importance': tuned_lgb.feature_importances_
}).sort_values('importance', ascending=False).head(10)
print("\nTop 10 Tuned LGBM Importances (Engineered):")
print(eng_importances_df)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


if 'cat_cols' not in locals():
    cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
    num_cols = [col for col in train_df.columns if col not in cat_cols + ['id', 'diagnosed_diabetes']]
    ohe_cols = ['gender']
    te_cols = [col for col in cat_cols if col not in ohe_cols]
    from category_encoders import TargetEncoder
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('te', TargetEncoder(), te_cols),
            ('ohe', OneHotEncoder(drop='first', sparse_output=False), ohe_cols),
            ('num', 'passthrough', num_cols)
        ],
        remainder='drop'
    )

X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']
test_ids = test_df['id']
X_test = test_df.drop('id', axis=1)


lgb_params = {'n_estimators': 383, 'learning_rate': 0.15907339767519801, 'max_depth': 3, 
              'num_leaves': 35, 'subsample': 0.6404682278872474, 'colsample_bytree': 0.6913310429699284, 
              'reg_alpha': 7.2965841787230765, 'reg_lambda': 5.7697760113121666, 'objective': 'binary', 
              'metric': 'auc', 'random_state': 42, 'verbosity': -1, 'force_col_wise': True}
xgb_params = lgb_params.copy()
xgb_params.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'random_state': 42, 'verbosity': 0})
cb_params = {'iterations': 383, 'learning_rate': 0.159, 'depth': 3, 'l2_leaf_reg': 5.77, 
             'random_seed': 42, 'verbose': False, 'loss_function': 'Logloss', 'eval_metric': 'AUC'}


def engineer_features(df_proc, feat_names):
    df_eng = pd.DataFrame(df_proc, columns=feat_names).copy()
    age_col, bmi_col = 'num__age', 'num__bmi'
    act_col, diet_col = 'num__physical_activity_minutes_per_week', 'num__diet_score'
    hist_col, trig_col = 'num__family_history_diabetes', 'num__triglycerides'
    df_eng['inter_age_bmi'] = df_eng[age_col] * df_eng[bmi_col]
    df_eng['inter_act_diet'] = df_eng[act_col] * df_eng[diet_col]
    df_eng['inter_hist_trig'] = df_eng[hist_col] * df_eng[trig_col]
    
    if 'binner' not in globals():
        from sklearn.preprocessing import KBinsDiscretizer
        binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        binner.fit(train_df[[age_col]].values)  
        globals()['binner'] = binner
    df_eng['bin_age'] = globals()['binner'].transform(df_eng[[age_col]].values)[:, 0]
    return df_eng.values, df_eng.columns.tolist()


skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_probas = np.zeros(len(X))
test_probas = np.zeros(len(X_test))
fold_aucs = []

meta_lgb, meta_xgb, meta_cb = [], [], []
test_lgb, test_xgb, test_cb = np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
    
    # Preprocess fold
    X_tr_proc = preprocessor.fit_transform(X_tr, y_tr)
    X_vl_proc = preprocessor.transform(X_vl)
    feat_names = preprocessor.get_feature_names_out()
    
    # Engineer
    X_tr_eng, eng_names = engineer_features(X_tr_proc, feat_names)
    X_vl_eng, _ = engineer_features(X_vl_proc, feat_names)
    
    # Fold models
    lgb_fold = lgb.LGBMClassifier(**lgb_params)
    xgb_fold = xgb.XGBClassifier(**xgb_params)
    cb_fold = CatBoostClassifier(**cb_params)
    
    lgb_fold.fit(X_tr_eng, y_tr); xgb_fold.fit(X_tr_eng, y_tr); cb_fold.fit(X_tr_eng, y_tr)
    
    lgb_p = lgb_fold.predict_proba(X_vl_eng)[:, 1]
    xgb_p = xgb_fold.predict_proba(X_vl_eng)[:, 1]
    cb_p = cb_fold.predict_proba(X_vl_eng)[:, 1]
    fold_probas = np.mean([lgb_p, xgb_p, cb_p], axis=0)  # Fold ensemble
    oof_probas[val_idx] = fold_probas
    fold_auc = roc_auc_score(y_vl, fold_probas)
    fold_aucs.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")
    
    # Meta-features for stacking (train on fold probs)
    meta_lgb.append(lgb_p); meta_xgb.append(xgb_p); meta_cb.append(cb_p)
    
    # Test preds
    X_test_proc = preprocessor.transform(X_test)  # Consistent transform
    X_test_eng, _ = engineer_features(X_test_proc, feat_names)
    test_lgb += lgb_fold.predict_proba(X_test_eng)[:, 1]
    test_xgb += xgb_fold.predict_proba(X_test_eng)[:, 1]
    test_cb += cb_fold.predict_proba(X_test_eng)[:, 1]

# Average test preds across folds
test_probas = (test_lgb + test_xgb + test_cb) / (3 * 3)  # 3 models x 3 folds

# Stacking: Fit meta-LR on OOF meta-features
meta_features = np.column_stack([np.concatenate(meta_lgb), np.concatenate(meta_xgb), np.concatenate(meta_cb)])
meta_model = MetaLR(random_state=42)
meta_model.fit(meta_features, y)
stacked_probas = meta_model.predict_proba(meta_features)[:, 1]
cv_auc = roc_auc_score(y, stacked_probas)
print(f"\nCV OOF AUC (Stacked): {cv_auc:.4f}")
print(f"Mean Fold AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")


sub_df = pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': test_probas})
sub_df.to_csv('submission.csv', index=False)
print("\nSubmission Shape:", sub_df.shape)
print("Submission Head:")
print(sub_df.head())
print("\nProba Distro (Submission):")
print(pd.Series(test_probas).describe())

