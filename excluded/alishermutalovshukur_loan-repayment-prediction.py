# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/train.csv")
test_df = pd.read_csv("/kaggle/input/binaryclassificationwithabankchurndataset/test.csv")


train_df.head()


test_df.head()


train_df.isnull().sum()


test_df.isnull().sum()


train_df = train_df.drop('Surname', axis=1)
test_df = test_df.drop('Surname', axis=1)


train_df.head()


print(train_df['Geography'].unique())
print(train_df['Gender'].unique())


train_df['Gender'] = train_df['Gender'].map({'Male': 1, 'Female': 0})
test_df['Gender'] = test_df['Gender'].map({'Male': 1, 'Female': 0})


print(train_df.head(10))
print(test_df.head(10))


one_hot_encoder = OneHotEncoder(drop='first', sparse=False)

geo_encoded_train = one_hot_encoder.fit_transform(train_df[['Geography']])
geo_encoded_test = one_hot_encoder.fit_transform(test_df[['Geography']])

geo_cols = one_hot_encoder.get_feature_names_out(["Geography"])

geo_train_df = pd.DataFrame(geo_encoded_train, columns=geo_cols, index=train_df.index)
geo_test_df = pd.DataFrame(geo_encoded_test, columns=geo_cols, index=test_df.index)

train_df_encoded = pd.concat([train_df.drop('Geography', axis=1), geo_train_df], axis=1)
test_df_encoded = pd.concat([test_df.drop('Geography', axis=1), geo_test_df], axis=1)



train_df_encoded.head()


test_df_encoded.head()


train_df_encoded.corrwith(train_df_encoded['Exited'])


# Modelga kiritilmaydigan ustunlar
drop_cols = ['id', 'CustomerId']

X_train = train_df_encoded.drop(columns=drop_cols + ['Exited'])
y_train = train_df_encoded['Exited']

X_test = test_df_encoded.drop(columns=drop_cols)


from sklearn.model_selection import train_test_split

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_split, y_train_split)
    y_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)
    results[name] = auc
    print(f"{name} ROC AUC: {auc:.4f}")



best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n✅ Eng yaxshi model: {best_model_name} (ROC AUC = {results[best_model_name]:.4f})")

test_preds = best_model.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    'id': test_df['id'],  
    'Exited': test_preds
})

submission.to_csv('submission.csv', index=False)
print("submission.csv fayl saqlandi.")





