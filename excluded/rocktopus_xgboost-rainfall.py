import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
train.head

oof_pred = np.zeros(len(train))
test_pred = np.zeros(len(test))

X_train = train.drop(columns=['rainfall'], axis=1)
y_train = train['rainfall']
X_test = test

k=5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):

    train_fold_X, val_fold_X = X_train.iloc[train_idx], X_train.iloc[test_idx]
    train_fold_y, val_fold_y = y_train.iloc[train_idx], y_train.iloc[test_idx]
    
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.2, objective='binary:logistic')
    model.fit(train_fold_X, train_fold_y)
    oof_pred[test_idx] = model.predict_proba(val_fold_X)[:, 1]

    pred = model.predict_proba(X_test)[:, 1]
    test_pred += pred

test_pred /= k

submission_df = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
submission_df['rainfall'] = test_pred
submission_df.to_csv("/kaggle/working/submission.csv", index=False)

submission_df.head()

