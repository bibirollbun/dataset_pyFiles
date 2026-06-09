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


# ===============================
# 1. IMPORTS
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier

# ===============================
# 2. LOAD DATA
# ===============================
train_df = pd.read_csv("/kaggle/input/mle-ese-mock/train (5).csv")
test_df = pd.read_csv("/kaggle/input/mle-ese-mock/test (4).csv")
sample_sub = pd.read_csv("/kaggle/input/mle-ese-mock/submission (6).csv")


TARGET = "quality_grade"

# ===============================
# 3. FIX CLASS MISMATCH (MOST IMPORTANT)
# ===============================
# Keep only those classes which Kaggle expects
valid_classes = sample_sub.columns[1:].tolist()

train_df = train_df[train_df[TARGET].isin(valid_classes)]

print("Classes after filtering:", train_df[TARGET].nunique())
# MUST be 10

# ===============================
# 4. SPLIT FEATURES & TARGET
# ===============================
X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]

# ===============================
# 5. ENCODE TARGET
# ===============================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ===============================
# 6. FIND CATEGORICAL COLUMNS
# ===============================
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
cat_features = [X.columns.get_loc(col) for col in cat_cols]

print("Categorical columns:", cat_cols)

# ===============================
# 7. TRAIN / VALIDATION SPLIT
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# ===============================
# 8. MODEL (BEST FOR LOG LOSS)
# ===============================
model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    iterations=800,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    verbose=100
)

model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    use_best_model=True
)




# ===============================
# 9. CHECK LOG LOSS
# ===============================
val_preds = model.predict_proba(X_val)
print("Validation Log Loss:", log_loss(y_val, val_preds))

# ===============================
# 10. TEST PREDICTION
# ===============================
test_preds = model.predict_proba(test_df)

print("Test prediction shape:", test_preds.shape)
# MUST be (rows, 10)

# ===============================
# 11. CREATE SUBMISSION (NO ERROR)
# ===============================
submission_cols = sample_sub.columns[1:]

submission = sample_sub.copy()
submission.iloc[:, 1:] = test_preds

submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully ✅")


