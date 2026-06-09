import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')


test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


categorical_cols = train.select_dtypes(include=['object']).columns
train = pd.get_dummies(train, columns=categorical_cols)
test = pd.get_dummies(test, columns=categorical_cols)
train, test = train.align(test, join='left', axis=1, fill_value=0)


X = train.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train['diagnosed_diabetes']


X_test = test.drop(['id'], axis=1)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)


model.fit(X_train, y_train)


import xgboost as xgb


model_xgb = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    tree_method='hist',
    eval_metric='auc',
    random_state=42
)


model_xgb.fit(X_train, y_train)


val_probs_xgb = model_xgb.predict_proba(X_val)[:, 1]
print(f"Skor ROC AUC XGBoost: {roc_auc_score(y_val, val_probs_xgb)}")


X_test_final = X_test[X_train.columns]
test_probs_xgb = model_xgb.predict_proba(X_test_final)[:, 1]


submission = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': test_probs_xgb
})
submission.to_csv('submission.csv', index=False)
print("File submission.csv berhasil dibuat dengan XGBoost!")

