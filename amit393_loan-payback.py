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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
print(df_train.head())
print("Shape:", df_train.shape)


df_train.describe()


df_train.duplicated().sum()


for col in df_train.select_dtypes(include=['object', 'category']):
    print(f"{col}: {df_train[col].unique()}")


df_train['loan_to_income'] = df_train['loan_amount'] / df_train['annual_income']
df_test['loan_to_income'] = df_test['loan_amount'] / df_test['annual_income']


df_train['interest_burden'] = df_train['interest_rate'] * df_train['loan_amount'] / df_train['annual_income']
df_test['interest_burden'] = df_test['interest_rate'] * df_test['loan_amount'] / df_test['annual_income']


df_train['credit_bucket'] = pd.cut(
    df_train['credit_score'],
    bins=[0, 580, 670, 740, 800, 900],
    labels=['Poor','Fair','Good','VeryGood','Excellent']
)

df_test['credit_bucket'] = pd.cut(
    df_test['credit_score'],
    bins=[0, 580, 670, 740, 800, 900],
    labels=['Poor','Fair','Good','VeryGood','Excellent']
)


for col in df_train.select_dtypes(include='number'):
    print(f"{col}  skew: {df_train[col].skew()}")


cols = ['annual_income', 'debt_to_income_ratio', 'loan_to_income', 'interest_burden']

for col in cols:
    df_train[col] = np.log1p(df_train[col])
    df_test[col] = np.log1p(df_test[col])


for col in df_train.select_dtypes(include='number'):
    print(f"{col}  skew: {df_train[col].skew()}")


import matplotlib.pyplot as plt
import seaborn as sns
corr = df_train.select_dtypes(include='number').corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.show()


for col in df_train.select_dtypes(include=['object', 'category']):
    plt.figure(figsize=(10,6))
    sns.countplot(x=col, data=df_train)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()


for col in df_train.select_dtypes(include=['object', 'category']).columns:
    df_train[col] = df_train[col].astype('category')
    df_test[col] = df_test[col].astype('category')
df_train.drop('id',axis=1)
df_test.drop("id",axis=1)


from sklearn.model_selection import train_test_split
X = df_train.drop('loan_paid_back', axis=1)
y = df_train['loan_paid_back']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_features = X_train.select_dtypes(include=['int', 'float']).columns.tolist()

# Convert categorical columns to category dtype
for col in cat_features:
    X_train[col] = X_train[col].astype("category")
    X_valid[col] = X_valid[col].astype("category")


# One-hot encode for XGBoost
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Identify categorical and numerical features
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_features = X_train.select_dtypes(include=['int', 'float']).columns.tolist()

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
    ]
)

# Transform training and validation data
X_train_encoded = preprocessor.fit_transform(X_train)
X_valid_encoded = preprocessor.transform(X_valid)
X_test_encoded = preprocessor.transform(df_test)

print(f"Original shape: {X_train.shape}")
print(f"Encoded shape: {X_train_encoded.shape}")


# XGBoost - Reduced search
xgb_model = XGBClassifier(
    tree_method='hist',
    device='cuda:0',
    random_state=42,
    eval_metric='auc'
)

xgb_param_grid = {
    'n_estimators': [500, 800],
    'max_depth': [5, 7],
    'learning_rate': [0.03, 0.05],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0, 0.1]
}

xgb_random = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_grid,
    n_iter=10,  # Reduced from 30
    scoring='roc_auc',
    cv=2,  # Reduced from 3
    n_jobs=1,
    random_state=42,
    verbose=1
)

xgb_random.fit(X_train_encoded, y_train)

print("Best XGBoost params:", xgb_random.best_params_)
print("Best XGBoost score:", xgb_random.best_score_)


lgb_base = LGBMClassifier(
    learning_rate=0.0356,
    n_estimators=200,
    max_depth=6,
    num_leaves=40,
    subsample=0.59,
    colsample_bytree=0.55,
    device='gpu',
    random_state=42
)

param_grid = {
    'n_estimators': [180, 200, 220],
    'max_depth': [5, 6, 7],
    'num_leaves': [35, 40, 45],
    'colsample_bytree': [0.54, 0.55, 0.56],
    'subsample': [0.58, 0.59, 0.60],
    'reg_alpha': [2.28, 2.4],
    'reg_lambda': [4.28, 4.3],
    'learning_rate': [0.034, 0.0356],
    'min_gain_to_split': [0.88, 0.8898]
}
xgb_random = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_grid,
    n_iter=10,  # Reduced from 30
    scoring='roc_auc',
    cv=2,  # Reduced from 3
    n_jobs=1,
    random_state=42,
    verbose=1
)

xgb_random.fit(X_train_encoded, y_train)

print("Best XGBoost params:", xgb_random.best_params_)
print("Best XGBoost score:", xgb_random.best_score_)




