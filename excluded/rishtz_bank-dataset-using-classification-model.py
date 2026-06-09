# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df


df.columns


df.info()


df.isnull().sum()


for col in df.columns:
    uni_value = df[col].unique()
    print(f"{col} has {len(uni_value)} values")
    if len(uni_value) <=15:
        print(f"    values are : {uni_value}")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

X = df.drop(columns=['y'], axis=1)
y = df['y']

num_col = X.select_dtypes(exclude = 'object').columns
cat_col = X.select_dtypes(include = 'object').columns

preprocessor = ColumnTransformer(
    [
        ("numerical", StandardScaler(), num_col),
        ("categorical", OneHotEncoder(), cat_col)
    ]
)

X = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import xgboost as xgb 
import lightgbm as lgb
from sklearn.metrics import accuracy_score

models = {
    "LogisticRegression" : LogisticRegression(max_iter=1000),
    "RandomForestClassifier" : RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost" : xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM" : lgb.LGBMClassifier(random_state=42),
    "CatBoost" : CatBoostClassifier(verbose=0, random_state=42)
}

result = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    result[name] = acc_score

for classifier , acc_score in result.items():
    print(f"{classifier} has predicted with an accuracy score of {acc_score :.4f}")




from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import xgboost

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# CatBoost parameter grid
cat_params = {
    'iterations': [200, 400, 600],
    'depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

# XGBoost parameter grid
xgb_params = {
    'n_estimators': [200, 500, 800],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

cat = CatBoostClassifier(verbose=0, random_state=42)
xgb_clf = xgboost.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)

cat_random = RandomizedSearchCV(
    estimator=cat,
    param_distributions=cat_params,
    n_iter=20,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

xgb_random = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=xgb_params,
    n_iter=20,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Fit models separately
cat_random.fit(X_train, y_train)
xgb_random.fit(X_train, y_train)

# Print best results
print("\nCatBoost :")
print("     Best params: ", cat_random.best_params_)
print("     Best score: ", cat_random.best_score_)

print("\nXGBoost :")
print("     Best params: ", xgb_random.best_params_)
print("     Best score: ", xgb_random.best_score_)


from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import  accuracy_score, classification_report

cat_model = CatBoostClassifier( learning_rate=0.1, l2_leaf_reg=3, iterations=600, depth=7, verbose=0, random_state=42 )

xgb_model = XGBClassifier( subsample=1.0, n_estimators=800, max_depth=9, learning_rate=0.05, colsample_bytree=0.8, random_state=42,
                           eval_metric='logloss', use_label_encoder=False )


xgb_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)

cat_preds = cat_model.predict_proba(X_test)[:, 1]
xgb_preds = xgb_model.predict_proba(X_test)[:, 1]

cat_weight = 0.5
xgb_weight = 0.5
ensemble_preds = (cat_weight * cat_preds) + (xgb_weight * xgb_preds)

ensemble_class_preds = (ensemble_preds >= 0.5).astype(int)

print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_class_preds))


test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
X_test_data = test_df  # no need to drop y

print(test_df.columns) 

print("Transforming test data...")
X_test_transformed = preprocessor.transform(X_test_data)

print("Making predictions...")
cat_preds_test = cat_model.predict_proba(X_test_transformed)[:, 1]
xgb_preds_test = xgb_model.predict_proba(X_test_transformed)[:, 1]

ensemble_preds_test = (cat_weight * cat_preds_test) + (xgb_weight * xgb_preds_test)
ensemble_class_preds_test = (ensemble_preds_test >= 0.5).astype(int)

# Create submission dataframe
submission_df = pd.DataFrame({
    'id': test_df['id'] if 'id' in test_df.columns else range(len(ensemble_class_preds_test)),
    'y': ensemble_class_preds_test
})


submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print(f"\nSubmission file created with {len(submission_df)} predictions")
print("\nFirst few predictions:")
print(submission_df.head())

print("\nPrediction distribution:\n", submission_df['y'].value_counts())




