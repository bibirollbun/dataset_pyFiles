# =====================================================
# PREDICTING LOAN PAYBACK - SINGLE FULL NOTEBOOK FILE
# =====================================================

# 1. IMPORT LIBRARIES
# -----------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


# 2. LOAD DATA
# -----------------------------------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print(train.head())


# 3. SEPARATE TARGET AND FEATURES
# -----------------------------------------------------
y = train["loan_paid_back"]
X = train.drop("loan_paid_back", axis=1)


# 4. ENCODE CATEGORICAL COLUMNS
# -----------------------------------------------------
cat_cols = X.select_dtypes(include=["object"]).columns
le = LabelEncoder()

for col in cat_cols:
    X[col] = le.fit_transform(X[col])
    test[col] = le.transform(test[col])


# 5. TRAIN/VALIDATION SPLIT
# -----------------------------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape)
print("Valid:", X_valid.shape)


# 6. TRAIN LIGHTGBM BASELINE MODEL
# -----------------------------------------------------
model = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc"
)

# VALIDATION PREDICTION
valid_pred = model.predict_proba(X_valid)[:, 1]
roc = roc_auc_score(y_valid, valid_pred)
print("Validation ROC-AUC:", roc)


# 7. TRAIN FULL MODEL AND PREDICT TEST SET
# -----------------------------------------------------
model.fit(X, y)
test_pred = model.predict_proba(test)[:, 1]


# 8. CREATE SUBMISSION FILE
# -----------------------------------------------------
submission = sample.copy()
submission["loan_paid_back"] = test_pred
submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")
print(submission.head())


