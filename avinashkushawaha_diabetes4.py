import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

print(train.shape)
print(test.shape)
print(sample.shape)



TARGET = "diagnosed_diabetes"

X = train.drop(columns=["id", TARGET])
y = train[TARGET]

X_test = test.drop(columns=["id"])

print(X.shape, y.shape, X_test.shape)



X = X.copy()
X_test = X_test.copy()

# Obesity risk increases with age
X["bmi_age"] = X["bmi"] * X["age"]
X_test["bmi_age"] = X_test["bmi"] * X_test["age"]

# Central obesity stress
X["central_obesity"] = X["bmi"] * X["waist_to_hip_ratio"]
X_test["central_obesity"] = X_test["bmi"] * X_test["waist_to_hip_ratio"]

# Diabetic lipid signature
X["lipid_ratio"] = X["triglycerides"] / (X["hdl_cholesterol"] + 1)
X_test["lipid_ratio"] = X_test["triglycerides"] / (X_test["hdl_cholesterol"] + 1)

# Activity-adjusted obesity
X["activity_burden"] = X["physical_activity_minutes_per_week"] / (X["bmi"] + 1)
X_test["activity_burden"] = X_test["physical_activity_minutes_per_week"] / (X_test["bmi"] + 1)

print("Domain features added")



all_data = pd.concat([X, X_test], axis=0)

all_data_encoded = pd.get_dummies(all_data, drop_first=True)

X_encoded = all_data_encoded.iloc[:len(X)]
X_test_encoded = all_data_encoded.iloc[len(X):]

print(X_encoded.shape, X_test_encoded.shape)
print("Columns match:", X_encoded.columns.equals(X_test_encoded.columns))



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_encoded, y)):
    
    X_train, X_val = X_encoded.iloc[tr_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc"
    )
    
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    
    auc_scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.5f}")

print("Mean CV AUC:", np.mean(auc_scores))



final_model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=900,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_encoded, y)

final_preds = final_model.predict_proba(X_test_encoded)[:, 1]

print(final_preds.min(), final_preds.max())



final_submission = sample.copy()
final_submission["diagnosed_diabetes"] = final_preds

final_submission.to_csv("final_domain_lgbm.csv", index=False)

print(final_submission.shape)
print(final_submission.columns)
print(
    final_submission["diagnosed_diabetes"].min(),
    final_submission["diagnosed_diabetes"].max()
)

final_submission.head()





