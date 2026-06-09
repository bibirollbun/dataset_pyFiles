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


# =========================
# 1. Imports
# =========================
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier

# =========================
# 2. Load Data
# =========================
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

X = train.drop(columns=["diagnosed_diabetes", "id"])
y = train["diagnosed_diabetes"]
X_test = test.drop(columns=["id"])

# =========================
# 3. Column Types
# =========================
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

# =========================
# 4. Preprocessing Pipelines
# =========================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# =========================
# 5. Stratified K-Fold CV
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
val_scores = np.zeros(skf.get_n_splits())
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"===== Fold {fold} =====")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=64,
            colsample_bytree=0.7,
            subsample=0.7,
            random_state=42
        ))
    ])
    
    clf.fit(X_train, y_train)  # no early_stopping_rounds
    
    val_pred = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold} ROC-AUC: {auc:.5f}")
    
    val_scores[fold-1] = auc
    test_preds += clf.predict_proba(X_test)[:, 1] / skf.get_n_splits()

print(f"\nMean CV ROC-AUC: {val_scores.mean():.5f}")

# =========================
# 6. Submission
# =========================
submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully!")


