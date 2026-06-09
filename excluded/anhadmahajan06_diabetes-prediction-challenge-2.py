import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

SEED = 42



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

TARGET = "diagnosed_diabetes"
ID_COL = "id"

print(train.shape, test.shape)
train.head()


X = train.drop(columns=[TARGET, ID_COL])
y = train[TARGET]

X_test = test.drop(columns=[ID_COL])



cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical features:", cat_features)



for col in X.columns:
    if col in cat_features:
        X[col].fillna("Unknown", inplace=True)
        X_test[col].fillna("Unknown", inplace=True)
    else:
        med = X[col].median()
        X[col].fillna(med, inplace=True)
        X_test[col].fillna(med, inplace=True)



def feature_engineering(df):
    df = df.copy()

    if "age" in df.columns and "bmi" in df.columns:
        df["age_bmi"] = df["age"] * df["bmi"]

    if "glucose" in df.columns:
        df["glucose_sq"] = df["glucose"] ** 2
        df["log_glucose"] = np.log1p(df["glucose"])

    if "blood_pressure" in df.columns and "bmi" in df.columns:
        df["bp_bmi_ratio"] = df["blood_pressure"] / (df["bmi"] + 1)

    if "insulin" in df.columns:
        df["log_insulin"] = np.log1p(df["insulin"])

    return df

X = feature_engineering(X)
X_test = feature_engineering(X_test)



skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))



for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}")

    model = CatBoostClassifier(
        iterations=4500,
        learning_rate=0.015,
        depth=8,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False
    )

    model.fit(
        X.iloc[tr_idx],
        y.iloc[tr_idx],
        cat_features=cat_features
    )

    oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits



cv_auc = roc_auc_score(y, oof_preds)
print("CV ROC-AUC:", cv_auc)



submission = pd.DataFrame({
    "id": test[ID_COL],
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()





