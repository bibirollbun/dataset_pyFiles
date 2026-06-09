import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.head(3)


X = train.drop(columns=["y", "id"])
y = train["y"]
X_test = test.drop(columns=["id"]).copy()

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0)
        le.fit(combined.astype(str))
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

kf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
n_splits = 9
oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_splits}")
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=20000,
        learning_rate=0.06,
        num_leaves=100,
        max_depth=10,
        min_child_samples=9,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.79,
        reg_lambda=3.0,
        max_bin=4523,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(period=100)]
    )
    
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    print(f"Fold {fold + 1} AUC: {roc_auc_score(y_val, val_preds)}")

    test_preds += model.predict_proba(X_test)[:, 1] / n_splits
print(f"Overall AUC score: {roc_auc_score(y, oof_preds)}")


submission = pd.DataFrame({'id': test['id'], 'y': test_preds })
submission.to_csv('submission.csv', index=False)
print("submission")




