# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train.describe()


# setting up columns
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome']
target_col = 'y'
id_col = 'id'


# preprocessing

def clip_outliers(df):
    for col in numerical_cols:
        low = df[col].quantile(0.005)
        high = df[col].quantile(0.995)
        df[col] = df[col].clip(lower=low, upper=high)
    return df

def add_log_features(df):
    df['duration_log'] = np.log(df['duration'] + 1)
    df['campaign_log'] = np.log(df['campaign'] + 1)
    df['pdays_log'] = np.log(df['pdays'] + 2)
    df['previous_log'] = np.log(df['previous'] + 1)
    return df

def encode_categories(train, test, cat_cols):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        encoders[col] = le

        # Transform test using the same encoder
        test[col] = test[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    return train, test, encoders


# Full preprocessing pipeline
train = clip_outliers(train)
test = clip_outliers(test)

train = add_log_features(train)
test = add_log_features(test)

train, test, encoders = encode_categories(train, test, cat_cols)


# Train-validation split
X = train.drop(columns=[id_col, target_col])
y = train[target_col]
X_test = test.drop(columns=[id_col])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# Train LightGBM model
model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    random_state=42,
    n_estimators=1000,
    learning_rate=0.05
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)


# Validation ROC AUC
y_val_pred = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC AUC: {roc_auc:.4f}")


# predicting & saving submission
y_test_pred = model.predict_proba(X_test)[:, 1]
submission['y'] = y_test_pred
submission.to_csv("submission.csv", index=False)


submission.head(10)

