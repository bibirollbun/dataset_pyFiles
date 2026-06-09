import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
df_train=df_train.replace('NaN', np.nan)
df_test=df_test.replace('NaN', np.nan)
submission_df=submission_df.replace('NaN', np.nan)


print(df_train.shape)
print(df_test.shape)
print(submission_df.shape)


import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, accuracy_score, make_scorer


X = df_train.drop(columns=['Personality'])
y = df_train['Personality']
X_test = df_test.copy()

le = LabelEncoder()
y = le.fit_transform(y)

cat_features = X.select_dtypes(include='object').columns.tolist()

X_xgb = pd.get_dummies(X, columns=cat_features)
X_test_xgb = pd.get_dummies(X_test, columns=cat_features)
X_test_xgb = X_test_xgb.reindex(columns=X_xgb.columns, fill_value=0)
X_xgb.fillna(0, inplace=True)
X_test_xgb.fillna(0, inplace=True)

scaler = StandardScaler()
X_svm = scaler.fit_transform(X_xgb)
X_test_svm = scaler.transform(X_test_xgb)

X_cat = X.copy()
X_test_cat = X_test.copy()
X_cat[cat_features] = X_cat[cat_features].astype(str).fillna("nan")
X_test_cat[cat_features] = X_test_cat[cat_features].astype(str).fillna("nan")

train_meta = np.zeros((X.shape[0], 3))
test_meta = np.zeros((X_test.shape[0], 3))


def run_optuna_tuning(model_name, X, y):
    def catboost_objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1000),
            'depth': trial.suggest_int('depth', 5, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 4, 8),
            'verbose': 0,
            'random_seed': 42
        }
        model = CatBoostClassifier(**params)
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        X_train[cat_features] = X_train[cat_features].astype(str).fillna("nan")
        X_val[cat_features] = X_val[cat_features].astype(str).fillna("nan")
        model.fit(X_train, y_train, cat_features=cat_features)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average='macro')

    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'use_label_encoder': False,
            'eval_metric': 'mlogloss',
            'random_state': 42
        }
        model = XGBClassifier(**params)
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average='macro')

    def svm_objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 5),
            'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42
        }
        model = SVC(**params)
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return f1_score(y_val, preds, average='macro')

    study = optuna.create_study(direction='maximize')
    if model_name == 'catboost':
        study.optimize(catboost_objective, n_trials=10)
        return CatBoostClassifier(**study.best_params, verbose=0, random_seed=42)
    elif model_name == 'xgboost':
        study.optimize(xgb_objective, n_trials=10)
        return XGBClassifier(**study.best_params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    elif model_name == 'svm':
        study.optimize(svm_objective, n_trials=10)
        return SVC(**study.best_params, probability=True, random_state=42)



base_models = ['catboost', 'xgboost', 'svm']
model_data_map = {
    'catboost': (X_cat, X_test_cat),
    'xgboost': (X_xgb, X_test_xgb),
    'svm': (X_svm, X_test_svm)
}


for i, model_name in enumerate(base_models):
    print(f"\nTraining base model: {model_name.upper()}")
    train_X, test_X = model_data_map[model_name]
    model = run_optuna_tuning(model_name, train_X, y)

    if model_name == 'catboost':
        model.fit(train_X, y, cat_features=cat_features)
    else:
        model.fit(train_X, y)

    train_preds = model.predict(train_X)
    f1 = f1_score(y, train_preds, average='macro')
    acc = accuracy_score(y, train_preds)
    print(f"{model_name.upper()} Final Training F1 Score: {f1:.4f}")
    print(f"{model_name.upper()} Final Training Accuracy: {acc:.4f}")

    train_meta[:, i] = train_preds
    test_meta[:, i] = model.predict(test_X)



meta_model = LogisticRegression(max_iter=100)
meta_model.fit(train_meta, y)
final_preds = meta_model.predict(test_meta)

meta_train_preds = meta_model.predict(train_meta)
print(f"Meta-Model Training Accuracy: {accuracy_score(y, meta_train_preds):.4f}")
print(f"Meta-Model Training F1 Score (macro): {f1_score(y, meta_train_preds, average='macro'):.4f}")


submission_df = pd.DataFrame({
    'id': df_test['id'],
    'Personality': le.inverse_transform(final_preds)
})

submission_df.to_csv("stacked_submission_v2.csv", index=False)


submission_df.head(5)

