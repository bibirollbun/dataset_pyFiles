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


# ===============================================================
# ðŸ“˜ Predicting Loan Payback - Kaggle Playground Series S5E11
# ===============================================================

# Step 1: Imports
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# ===============================================================
# Step 2: Load Data
# ===============================================================
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

print("âœ… Data Loaded")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ===============================================================
# Step 3: Prepare Features and Target
# ===============================================================
TARGET = "loan_paid_back"
ID = "id"

X = train.drop(columns=[TARGET, ID])
y = train[TARGET]
X_test = test.drop(columns=[ID])

# Encode categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns
for col in cat_cols:
    le = LabelEncoder()
    all_data = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(all_data)
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# ===============================================================
# Step 4: Train/Validation Split
# ===============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================================================
# Step 5: LightGBM Model
# ===============================================================
model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    objective="binary"
)

# âœ… Fixed training call (no 'verbose' argument)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping(100), log_evaluation(100)]
)

# ===============================================================
# Step 6: Evaluate
# ===============================================================
val_pred = model.predict_proba(X_val)[:, 1]
roc = roc_auc_score(y_val, val_pred)
print(f"\nðŸŽ¯ Validation ROC-AUC: {roc:.5f}")

# ===============================================================
# Step 7: Predict on Test
# ===============================================================
test_pred = model.predict_proba(X_test)[:, 1]

# ===============================================================
# Step 8: Create Submission
# ===============================================================
submission = sample.copy()
submission["loan_paid_back"] = test_pred
submission.to_csv("submission.csv", index=False)

print("\nâœ… Submission file saved as submission.csv")
submission.head()


