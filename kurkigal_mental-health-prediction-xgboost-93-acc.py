import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv')
sam_sub = pd.read_csv('/kaggle/input/playground-series-s4e11/sample_submission.csv')


train.head()


test.head()


sam_sub.head()


train.shape


train.info()


train.columns


train.isnull().sum()


test.isnull().sum()


# Fill missing values for categorical columns
categorical_cols = train.select_dtypes(include=['object']).columns.intersection(test.columns)
for col in categorical_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)


numeric_cols = [col for col in train.select_dtypes(include=['float64', 'int64']).columns if col in test.columns]

# Fill missing values for numeric columns
train[numeric_cols] = train[numeric_cols].fillna(train[numeric_cols].mean())
test[numeric_cols] = test[numeric_cols].fillna(test[numeric_cols].mean())


test.isnull().sum()


train.isnull().sum()


train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

test = test.reindex(columns=train.columns, fill_value=0)

X = train.drop('Depression', axis=1)
y = train['Depression']



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')


xgb_model.fit(X_train, y_train)


y_pred_xgb = xgb_model.predict(X_val)


print("Accuracy:", accuracy_score(y_val, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_val, y_pred_xgb))


