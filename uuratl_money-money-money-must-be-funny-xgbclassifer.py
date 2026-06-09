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
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
import optuna
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv(os.path.join('/kaggle/input/playground-series-s5e8/train.csv'))
train_df = train.drop(columns='id', axis=1).copy()
train_df.head()


train_df.info()


train_df.describe().T


cat_cols = train_df.select_dtypes(include=['object']).columns
num_cols = train_df.select_dtypes(exclude=['object']).columns


for col in num_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(y=train_df[col])
    plt.title(f'Box Plot of {col}')
    plt.show()


for col in num_cols:
    plt.figure(figsize=(10, 5))
    sns.histplot(train_df[col], bins=50)
    plt.title(f'Histogram of {col}')
    plt.show()


colors = sns.color_palette('bright')

for col in cat_cols:
    plt.pie(train_df[col].value_counts(), labels=train_df[col].unique(), colors=colors, autopct='%.0f%%')
    plt.title(f'Pie Chart of {col}')
    plt.show()



def prepare_data(data):
    df = data.copy()
    df['p_contact'] = df['pdays'].apply(lambda x: 'no' if x == -1 else 'yes')
    df['pdays'] = df['pdays'].apply(lambda x: np.inf if x == -1 else x)
    df['age'] = pd.cut(df['age'], bins=[0, 18, 30, 40, 60, 100], labels=['0', '1', '2', '3', '4']).astype(str)
    df['pdays'] = pd.cut(df['pdays'], bins=[-1, 30, 90, 180, 360, np.inf], labels=['0', '1', '2', '3', '4']).astype(str)
    df['day'] = pd.cut(df['day'], bins=[0, 10, 20, 31], labels=['0', '1', '2']).astype(str)

    return df


train_df = prepare_data(train_df)


cat_cols = train_df.select_dtypes(include=['object']).columns
num_cols = train_df.select_dtypes(exclude=['object']).columns


x = train_df.drop(columns=['y'], axis=1)
y = train_df['y']


class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
weights = dict(zip(np.unique(y), class_weights))
weights


scale_class = y.value_counts()[0] / y.value_counts()[1]
scale_class


encode_dict = {}
for col in cat_cols:
    encode_dict[col] = x[col].value_counts(normalize=True).to_dict()
    x[col] = x[col].map(encode_dict[col])


x.head()


x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


xgb_classifier = XGBClassifier()
xgb_classifier.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=10)

y_pred = xgb_classifier.predict_proba(x_valid)[:, 1]
roc_auc_score(y_valid, y_pred)


def objective(trial):
    
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'use_label_encoder': False,
        'eval_metric': 'auc'
    }
    model = XGBClassifier(**params, scale_pos_weight=scale_class, random_state=42)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    y_pred_opt = model.predict_proba(x_valid)[:, 1]
    return roc_auc_score(y_valid, y_pred_opt)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best trial:")
print(study.best_trial)


best_xgb_model = XGBClassifier(**study.best_params, scale_pos_weight=scale_class, random_state=42)
best_xgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)


pred_y = best_xgb_model.predict(x_valid)


sns.heatmap(confusion_matrix(y_valid, pred_y), annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes']);


print(classification_report(y_valid, pred_y))


test = pd.read_csv(os.path.join('/kaggle/input/playground-series-s5e8/test.csv'))
test_df = test.copy()
test_df


sample_submission = pd.read_csv(os.path.join('/kaggle/input/playground-series-s5e8/sample_submission.csv'))
sample_submission


def prepare_test_data(test_df, encode_dict):
    test_df['p_contact'] = test_df['pdays'].apply(lambda x: 'no' if x == -1 else 'yes')
    test_df['pdays'] = test_df['pdays'].apply(lambda x: np.inf if x == -1 else x)
    test_df['age'] = pd.cut(test_df['age'], bins=[0, 18, 30, 40, 60, 100], labels=['0', '1', '2', '3', '4']).astype(str)
    test_df['pdays'] = pd.cut(test_df['pdays'], bins=[-1, 30, 90, 180, 360, np.inf], labels=['0', '1', '2', '3', '4']).astype(str)
    test_df['day'] = pd.cut(test_df['day'], bins=[0, 10, 20, 31], labels=['0', '1', '2']).astype(str)
    for col in cat_cols:
        test_df[col] = test_df[col].map(encode_dict[col])
    return test_df


test_df = prepare_test_data(test_df, encode_dict)
test_df


predicts = best_xgb_model.predict_proba(test_df.drop(columns='id'))[:, 1]

submission = sample_submission.copy()
submission['y'] = predicts
submission.to_csv('submission.csv', index=False)

