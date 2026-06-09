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


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


import pandas as pd

df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")



df.head()


import pandas as pd

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

# List all unwanted columns here
drop_cols = [
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "diet_score",
    "waist_to_hip_ratio",
    "systolic_bp",
    "diastolic_bp",
    "alcohol_consumption_per_week"
]

# Drop from train and test
train = train.drop(columns=drop_cols)
test  = test.drop(columns=drop_cols)

display(train.head())



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

X = train.drop("diagnosed_diabetes", axis=1)
y = train["diagnosed_diabetes"]



# FACTORIZE CATEGORICAL FEATURES

categorical_cols = X.select_dtypes(include="object").columns.tolist()

for col in categorical_cols:
    X[col], _ = pd.factorize(X[col])
    test[col] = pd.factorize(test[col])[0]  # convert test also


# 5-FOLD STRATIFIED CROSS VALIDATION

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))
fold_num = 1

for train_idx, val_idx in kf.split(X, y):
    print(f"Training fold {fold_num}...")

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",     # FASTEST
        enable_categorical=False,
        random_state=42
    )

    model.fit(X_tr, y_tr)

    # OOF predictions
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Test predictions (averaged)
    test_preds += model.predict_proba(test)[:, 1] / 5

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold_num} AUC: {fold_auc:.5f}")

    fold_num += 1


# FINAL VALIDATION SCORE
# -------------------------------
final_auc = roc_auc_score(y, oof_preds)
print("\n===============================")
print(f"Final 5-Fold AUC: {final_auc:.6f}")
print("===============================")


submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": np.clip(test_preds, 0, 1)
})

submission.to_csv("submission_xgb_cv.csv", index=False)
print("Submission saved as submission_xgb_cv.csv")


print(submission.columns)



print(len(submission), len(test))



print((submission["id"] == test["id"]).all())



submission.isna().sum()



print(submission.columns.tolist())



sample = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

submission = pd.DataFrame({
    "id": sample["id"],   # force exact IDs
    "diagnosed_diabetes": test_preds[:len(sample)]
})

# clip
submission["diagnosed_diabetes"] = submission["diagnosed_diabetes"].clip(0, 1)

submission.to_csv("submission.csv", index=False)



X[col], _ = pd.factorize(X[col])
test[col] = pd.factorize(test[col])[0]



for col in categorical_cols:
    full = pd.concat([train[col], test[col]], axis=0).astype("category")
    train[col] = full[:len(train)].cat.codes
    test[col] = full[len(train):].cat.codes



tree_method="gpu_hist",
predictor="gpu_predictor",



model = XGBClassifier(
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    early_stopping_rounds=100,   # ← FIXED
    random_state=42
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)




from sklearn.preprocessing import StandardScaler

num_cols = X.select_dtypes(exclude="object").columns
scaler = StandardScaler()

X[num_cols] = scaler.fit_transform(X[num_cols])
test[num_cols] = scaler.transform(test[num_cols])









