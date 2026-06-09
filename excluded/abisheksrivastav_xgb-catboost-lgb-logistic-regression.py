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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import joblib

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
original = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")

# Merge original dataset (assumes same structure as train)
data = pd.concat([train.drop(columns=['Personality']), test, original.drop(columns=['Personality'])], axis=0).reset_index(drop=True)

# Feature engineering: Add more features
for df in [train, test, original]:
    df['Alone_to_Social_ratio'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)
    df['Outside_x_Drained'] = df['Going_outside'] * df['Drained_after_socializing'].map({"Yes": 1, "No": 0, "missing": 0}).fillna(0)

# Merge updated original again with new features
data = pd.concat([train.drop(columns=['Personality']), test, original.drop(columns=['Personality'])], axis=0).reset_index(drop=True)

# Fill missing values
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].fillna("missing")
for col in data.select_dtypes(include='number').columns:
    data[col] = data[col].fillna(data[col].mean())

# Label encode categorical
cat_cols = data.select_dtypes(include='object').columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

# Scale numeric features
scaler = StandardScaler()
num_cols = data.select_dtypes(include='number').columns.drop('id')
data[num_cols] = scaler.fit_transform(data[num_cols])

# Split back to X_train, X_test
X_train = data.iloc[:len(train)].drop(columns=['id'])
X_test = data.iloc[len(train):len(train)+len(test)].drop(columns=['id'])
y = LabelEncoder().fit_transform(train['Personality'])
num_classes = len(np.unique(y))

# Initialize prediction lists
oof_preds_list = []
test_preds_list = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. XGBoost
xgb_oof = np.zeros((len(X_train), num_classes))
xgb_test = np.zeros((len(X_test), num_classes))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    model = xgb.XGBClassifier(tree_method='hist', eval_metric='mlogloss', use_label_encoder=False)
    model.fit(X_train.iloc[train_idx], y[train_idx])
    xgb_oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])
    xgb_test += model.predict_proba(X_test) / skf.n_splits
oof_preds_list.append(xgb_oof)
test_preds_list.append(xgb_test)

# 2. LightGBM with Optuna tuning
def lgb_objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': 1,
        'verbosity': -1
    }
    model = lgb.LGBMClassifier(**params)
    scores = []
    for train_idx, val_idx in skf.split(X_train, y):
        model.fit(X_train.iloc[train_idx], y[train_idx])
        preds = model.predict(X_train.iloc[val_idx])
        scores.append(accuracy_score(y[val_idx], preds))
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(lgb_objective, n_trials=20)
best_params = study.best_params
best_params.update({'objective': 'multiclass', 'num_class': num_classes})

lgb_opt_oof = np.zeros((len(X_train), num_classes))
lgb_opt_test = np.zeros((len(X_test), num_classes))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train.iloc[train_idx], y[train_idx])
    lgb_opt_oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])
    lgb_opt_test += model.predict_proba(X_test) / skf.n_splits
oof_preds_list.append(lgb_opt_oof)
test_preds_list.append(lgb_opt_test)

# 3. Tuned CatBoost
cat_oof = np.zeros((len(X_train), num_classes))
cat_test = np.zeros((len(X_test), num_classes))
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    model = CatBoostClassifier(verbose=0, task_type='CPU', loss_function='MultiClass',
                               depth=6, learning_rate=0.03, iterations=500)
    model.fit(X_train.iloc[train_idx], y[train_idx])
    cat_oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])
    cat_test += model.predict_proba(X_test) / skf.n_splits
oof_preds_list.append(cat_oof)
test_preds_list.append(cat_test)

# === Logistic Regression or GBDT as meta-ensemble ===
stacked_oof = np.hstack(oof_preds_list)
stacked_test = np.hstack(test_preds_list)

# Logistic Regression CV
meta_model = LogisticRegressionCV(cv=5, max_iter=1000, multi_class='multinomial')
# Alternatively use: meta_model = GradientBoostingClassifier(n_estimators=200)
meta_model.fit(stacked_oof, y)
final_oof = meta_model.predict(stacked_oof)
final_test = meta_model.predict(stacked_test)

# Accuracy
print("OOF Accuracy (LogReg Ensemble):", accuracy_score(y, final_oof))

# Prepare submission
label_decoder = LabelEncoder().fit(train['Personality'])
final_labels = label_decoder.inverse_transform(final_test)
submission = pd.DataFrame({
    "id": test["id"],
    "Personality": final_labels
})
submission.to_csv("ensemble_optuna_submission.csv", index=False)
print("Submission saved as ensemble_optuna_submission.csv")


