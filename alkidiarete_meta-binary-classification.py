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
from sklearn.preprocessing import LabelEncoder
from numpy import hstack, vstack, asarray
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

test_id = test['id']
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)



numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

def handle_outliers(df):
    for col in numerical_cols:
        lower = df[col].quantile(0.005)
        upper = df[col].quantile(0.995)
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df

def add_binary_features(df):
    df['balance_positive'] = (df['balance'] > 0).astype(int)
    df['has_previous'] = (df['previous'] > 0).astype(int)
    df['duration_long'] = (df['duration'] > 300).astype(int)
    df['campaign_multiple'] = (df['campaign'] > 2).astype(int)
    df['contacted_before'] = (df['pdays'] != -1).astype(int)
    return df

def add_numeric_transformations(df):
    df['sqrt_age'] = np.sqrt(df['age'])
    df['duration_log'] = np.log(df['duration'])
    df['campaign_log'] = np.log(df['campaign'])
    df['pdays_log'] = np.log(df['pdays'] + 2)
    df['previous_log'] = np.log(df['previous'] + 1)
    df['balance_log'] = np.log1p(df['balance'].clip(lower=0))
    df['age_squared'] = df['age'] ** 2
    return df

def cyclical_encode_duration(df):
    df['duration_sin'] = np.sin(2 * np.pi * df['duration'] / 400)
    df['duration_cos'] = np.cos(2 * np.pi * df['duration'] / 400)
    return df

def cyclical_encode_month(df):
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_num'] = df['month'].map(month_map).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    df.drop('month_num', axis=1, inplace=True)
    return df

def cast_categoricals(df):
    for feature in cat_cols:
        df[feature] = df[feature].astype('category')
    return df

def preprocess(df):
    df = handle_outliers(df)
    df = add_binary_features(df)
    df = add_numeric_transformations(df)
    df = cyclical_encode_duration(df)
    df = cyclical_encode_month(df)
    df = cast_categoricals(df)
    return df

train = preprocess(train)
test = preprocess(test)


label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]], axis=0))
    
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])


def get_models():
    models = []
    models.append(XGBClassifier(tree_method='gpu_hist',predictor='gpu_predictor',use_label_encoder=False,eval_metric='logloss',random_state=42))
    models.append(LGBMClassifier(device='gpu',gpu_platform_id=0,gpu_device_id=0,verbose=-1,random_state=42))
    models.append(CatBoostClassifier(task_type='GPU',devices='0',verbose=0,random_state=42 ))
    models.append(RandomForestClassifier(n_estimators=100,n_jobs=-1))
    return models

def get_out_of_fold_predictions(X, y, models):
    meta_X, meta_y = [], []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_ix, test_ix in kfold.split(X):
        fold_yhats = []
        train_X, test_X = X.iloc[train_ix], X.iloc[test_ix]
        train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
        meta_y.extend(test_y)
        for model in models:
            model.fit(train_X, train_y)
            yhat = model.predict_proba(test_X)
            fold_yhats.append(yhat)
        meta_X.append(hstack(fold_yhats))
    return vstack(meta_X), asarray(meta_y)

def fit_base_models(X, y, models):
    for model in models:
        model.fit(X, y)

def fit_meta_model(X, y):
    meta_model = LogisticRegression(solver='liblinear')
    meta_model.fit(X, y)
    return meta_model

def evaluate_models(X, y, models):
    for model in models:
        yhat_labels = model.predict(X)
        yhat_proba = model.predict_proba(X)[:, 1]
        acc = accuracy_score(y, yhat_labels)
        auc = roc_auc_score(y, yhat_proba)
        print(f"{model.__class__.__name__} → Accuracy: {acc*100:.3f} | AUC: {auc:.4f}")

def super_learner_predictions(X, models, meta_model):
    meta_X = []
    for model in models:
        yhat = model.predict_proba(X)
        meta_X.append(yhat)
    meta_X = hstack(meta_X)
    return meta_model.predict(meta_X), meta_model.predict_proba(meta_X)[:, 1]

if __name__ == "__main__":
    
    X = train.drop(columns=['y'])
    y = train['y']
    X_test = test.copy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Train', X_train.shape, y_train.shape, 'Validation', X_val.shape, y_val.shape)

    models = get_models()

    meta_X, meta_y = get_out_of_fold_predictions(X_train, y_train, models)
    print('Meta ', meta_X.shape, meta_y.shape)

    fit_base_models(X_train, y_train, models)

    meta_model = fit_meta_model(meta_X, meta_y)

    evaluate_models(X_val, y_val, models)

    yhat_labels, yhat_proba = super_learner_predictions(X_val, models, meta_model)
    acc = accuracy_score(y_val, yhat_labels)
    auc = roc_auc_score(y_val, yhat_proba)
    print(f"Super Learner → Accuracy: {acc*100:.3f} | AUC: {auc:.4f}")



yhat_test_labels, yhat_test_proba = super_learner_predictions(X_test, models, meta_model)

submission = pd.DataFrame({
    "id": test_id, 
    "y": yhat_test_proba
})

submission.to_csv("submission.csv", index=False)

submission.head()

