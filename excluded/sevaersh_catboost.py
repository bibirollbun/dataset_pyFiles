!pip install optuna-integration[sklearn]


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import optuna
from catboost import CatBoostClassifier
import catboost as cb


data = pd.read_csv(
    '/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv'
).set_index('id')


data.describe()


data.info()


data['CustomerId'].unique().shape


data.select_dtypes(exclude='object').corr()


unique_customers = data['CustomerId'].unique()
train_customers, test_customers = train_test_split(
    unique_customers, test_size=0.1, random_state=42
)

train = data[data['CustomerId'].isin(train_customers)]\
    .drop(['CustomerId', 'Surname'], axis=1)
val = data[data['CustomerId'].isin(test_customers)]\
    .drop(['CustomerId', 'Surname'], axis=1)

X_train, y_train = train.drop('Exited', axis=1), train['Exited']
X_val, y_val = val.drop('Exited', axis=1), val['Exited']


clf = CatBoostClassifier(
    eval_metric='AUC',
    cat_features=['Geography', 'Gender'],
    random_state=42,
    max_depth=3,
    iterations=500,
    verbose=1
)


X_train['Balance_Salary'] = X_train['Balance'] / X_train['EstimatedSalary']
X_train['Products_Balance'] = X_train['NumOfProducts'] / X_train['Balance']

X_val['Balance_Salary'] = X_val['Balance'] / X_val['EstimatedSalary']
X_val['Products_Balance'] = X_val['NumOfProducts'] / X_val['Balance']


clf.fit(X_train, y_train)


roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])


all_data = data.drop(['CustomerId', 'Surname'], axis=1)
all_data['Balance_Salary'] = all_data['Balance'] / all_data['EstimatedSalary']
all_data['Products_Balance'] = all_data['NumOfProducts'] / all_data['Balance']

X_data, y_data = all_data.drop('Exited', axis=1), all_data['Exited']


clf.fit(X_data, y_data)


test = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv')

test['Balance_Salary'] = test['Balance'] / test['EstimatedSalary']
test['Products_Balance'] = test['NumOfProducts'] / test['Balance']

features = list(X_data.columns)


y_pred_proba = clf.predict_proba(test[features])[:, 1]

df_id = test["id"]
submission_df = pd.DataFrame({
    "id": df_id,
    "Exited": y_pred_proba
})

submission_df.to_csv("submission.csv", index=False)


clf = CatBoostClassifier(
    eval_metric='AUC',
    cat_features=['Geography', 'Gender'],
    random_state=42,
    verbose=0
)

params = {
        "iterations": optuna.distributions.IntDistribution(100, 700),
        "depth": optuna.distributions.IntDistribution(2, 6),
        "l2_leaf_reg": optuna.distributions.FloatDistribution(1e-2, 10.0, log=True),
        "border_count": optuna.distributions.IntDistribution(32, 255),
        "random_strength": optuna.distributions.FloatDistribution(0.1, 10.0),
        "bagging_temperature": optuna.distributions.FloatDistribution(0.0, 1.0)
    }

optuna_search = optuna.integration.OptunaSearchCV(
    clf,
    params,
    n_trials=30,
    scoring='roc_auc'
)


optuna_search.fit(X_data, y_data)


y_pred_proba = optuna_search.predict_proba(test[features])[:, 1]

df_id = test["id"]
submission_df = pd.DataFrame({
    "id": df_id,
    "Exited": y_pred_proba
})

submission_df.to_csv("submission.csv", index=False)




