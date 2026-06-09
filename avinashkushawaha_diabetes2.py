# Cell 1: Imports

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier



# Cell 2: Load data

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

print(train.shape)
print(test.shape)
print(sample.shape)



# Cell 3: Split features and target

TARGET = "diagnosed_diabetes"

X = train.drop(columns=["id", TARGET])
y = train[TARGET]

X_test = test.drop(columns=["id"])

print(X.shape, y.shape, X_test.shape)



# Cell 4: One-hot encode categorical variables

all_data = pd.concat([X, X_test], axis=0)

all_data_encoded = pd.get_dummies(all_data, drop_first=True)

X_encoded = all_data_encoded.iloc[:len(X), :]
X_test_encoded = all_data_encoded.iloc[len(X):, :]

print(X_encoded.shape, X_test_encoded.shape)
print("Columns match:", X_encoded.columns.equals(X_test_encoded.columns))



# Cell 5: Random Forest with Stratified CV

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_auc_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_encoded, y)):
    
    X_train, X_val = X_encoded.iloc[tr_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    val_preds = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    
    rf_auc_scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.5f}")

print("Mean RF CV AUC:", np.mean(rf_auc_scores))



# Cell 6: Train RF on full data

final_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

final_rf.fit(X_encoded, y)

rf_test_preds = final_rf.predict_proba(X_test_encoded)[:, 1]

print(rf_test_preds.min(), rf_test_preds.max())



# Cell 7: Create submission

rf_submission = sample.copy()
rf_submission["diagnosed_diabetes"] = rf_test_preds

rf_submission.to_csv("rf_submission.csv", index=False)
rf_submission.head()



# Cell 8: Verify submission

print(rf_submission.shape)
print(rf_submission.columns)
print(
    rf_submission["diagnosed_diabetes"].min(),
    rf_submission["diagnosed_diabetes"].max()
)



# RF-1: Train Random Forest on full training data

final_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

final_rf.fit(X_encoded, y)

rf_test_preds = final_rf.predict_proba(X_test_encoded)[:, 1]

print("RF predictions ready")
print("Min:", rf_test_preds.min())
print("Max:", rf_test_preds.max())



# RF-2: Create Random Forest submission

rf_submission = sample.copy()
rf_submission["diagnosed_diabetes"] = rf_test_preds

rf_submission.to_csv("rf_submission.csv", index=False)
rf_submission.head()



# RF-3: Verify RF submission

print(rf_submission.shape)
print(rf_submission.columns)
print(
    rf_submission["diagnosed_diabetes"].min(),
    rf_submission["diagnosed_diabetes"].max()
)





