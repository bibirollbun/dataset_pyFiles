import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


train_id = train["id"]
test_id = test["id"]

X = train.drop(columns=["id", "loan_paid_back"])
y = train["loan_paid_back"]
X_test = test.drop(columns=["id"])


X = X.fillna(-999)
X_test = X_test.fillna(-999)


# Ordinal encoding for grade_subgrade
def grade_to_num(g):
    if pd.isna(g): return -1
    letter = g[0]
    number = int(g[1])
    base = (ord(letter) - ord('A')) * 5
    return base + number

X["grade_subgrade"] = X["grade_subgrade"].apply(grade_to_num)
X_test["grade_subgrade"] = X_test["grade_subgrade"].apply(grade_to_num)


# Encode other categoricals
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))


# Feature engineering
X["income_to_loan_ratio"] = X["annual_income"] / (X["loan_amount"] + 1)
X_test["income_to_loan_ratio"] = X_test["annual_income"] / (X_test["loan_amount"] + 1)

X["loan_to_income_ratio"] = X["loan_amount"] / (X["annual_income"] + 1)
X_test["loan_to_income_ratio"] = X_test["loan_amount"] / (X_test["annual_income"] + 1)

X["credit_to_interest_ratio"] = X["credit_score"] / (X["interest_rate"] + 1)
X_test["credit_to_interest_ratio"] = X_test["credit_score"] / (X_test["interest_rate"] + 1)



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


model = XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    use_label_encoder=False
)



model.fit(X_train, y_train)


val_preds = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_preds)
print(f"Validation ROC-AUC: {auc:.4f}")


test_preds = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test_id,
    "loan_paid_back": test_preds
})


submission.to_csv("submission.csv", index=False)
print("submission.csv saved successfully!")
submission.head()




