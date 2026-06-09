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


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



TARGET = "loan_paid_back"
ID_COL = "id"

y = train[TARGET]
X = train.drop(columns=[TARGET, ID_COL])
X_test = test.drop(columns=[ID_COL])



from sklearn.preprocessing import OrdinalEncoder

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns:", cat_cols)

if cat_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_cols] = enc.fit_transform(X[cat_cols])
    X_test[cat_cols] = enc.transform(X_test[cat_cols])



from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

params = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "objective": "binary",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1
}

model = LGBMClassifier(**params)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc")
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
    
    print(f"Fold {fold} AUC:", roc_auc_score(y_val, oof_preds[val_idx]))

print("\nCV AUC:", roc_auc_score(y, oof_preds))



submission = pd.DataFrame({
    "id": test[ID_COL],
    "loan_paid_back": test_preds
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
submission.head()


