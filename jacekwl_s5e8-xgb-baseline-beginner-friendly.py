import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb

path = "/kaggle/input/playground-series-s5e8/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
submission = pd.read_csv(path + "sample_submission.csv")


TARGET = 'y'

# encode categorical columns using LabelEncoder
for c in ['job', 'marital', 'education', 'default',
          'housing', 'loan', 'contact', 'month', 'poutcome']:
    le = LabelEncoder()
    le.fit(train[c])
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])


X = train.drop(['id', TARGET], axis=1)
y = train[TARGET]
test = test.drop('id', axis=1)


model = xgb.XGBClassifier(
    n_estimators=1000,
    eval_metric='logloss'
)

# 5-fold stratified cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

auc_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)
    # predict probabilities for the positive class on validation data
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    # calculate ROC AUC for the current fold
    auc = roc_auc_score(y_val, y_pred_proba)
    auc_scores.append(auc)
    print(f'Fold {fold} ROC AUC: {auc:.4f}')

print(f'\nMean ROC AUC: {np.mean(auc_scores):.4f}')


model.fit(X, y)
submission[TARGET] = model.predict_proba(test)[:, 1]
submission.to_csv("submission.csv", index=False)

