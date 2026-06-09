import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print(train.shape)
print(test.shape)



train.head()



test.head()


# Re-create X, y, X_test cleanly
X = train.drop(columns=["id", "diagnosed_diabetes"])
y = train["diagnosed_diabetes"]

X_test = test.drop(columns=["id"])




from sklearn.preprocessing import LabelEncoder

cat_cols = X.select_dtypes(include="object").columns
print("Categorical columns:", cat_cols)

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))



X.dtypes



cat_cols = X.select_dtypes(include="object").columns
cat_cols



skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



lgb_auc = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=42 + fold,
        verbosity=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc"
    )

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)

    lgb_auc.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.5f}")
 
print("Mean CV AUC:", np.mean(lgb_auc))



final_model = lgb.LGBMClassifier(
    n_estimators=2500,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    verbosity=-1
)

final_model.fit(X, y)



test_preds = final_model.predict_proba(X_test)[:, 1]



submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)



submission.head()


submission.tail()

