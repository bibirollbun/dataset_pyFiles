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

df = pd.read_csv(
    "/kaggle/input/home-credit-default-risk/application_train.csv"
)

df.head()


# Check what TARGET means
df['TARGET'].value_counts()


# Percentage of defaulters vs non-defaulters
df['TARGET'].value_counts(normalize=True) * 100


df.shape


df.info()


df.describe()


features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CNT_CHILDREN",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CODE_GENDER"
]

df_model = df[features + ["TARGET"]].copy()
df_model.head()



# Separate features
num_cols = df_model.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df_model.select_dtypes(include=["object"]).columns

# Fill missing values
df_model[num_cols] = df_model[num_cols].fillna(df_model[num_cols].median())
df_model[cat_cols] = df_model[cat_cols].fillna(df_model[cat_cols].mode().iloc[0])


df_model = pd.get_dummies(df_model, drop_first=True)
df_model.head()


from sklearn.model_selection import train_test_split

X = df_model.drop("TARGET", axis=1)
y = df_model["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)



from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))



from xgboost import XGBClassifier


xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=11,  # ~92/8 imbalance ratio
    eval_metric="auc",
    random_state=42
)

xgb_model.fit(X_train, y_train)



y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

