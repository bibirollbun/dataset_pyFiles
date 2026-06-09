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


# ğŸ“¦ Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# ğŸ“¥ Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# ğŸ�¯ Encode Target
target_encoder = LabelEncoder()
train["Personality"] = target_encoder.fit_transform(train["Personality"])  # Extrovert=0, Introvert=1

# ğŸ§¾ Separate Features/Target
X = train.drop(columns=["id", "Personality"])
y = train["Personality"]
X_test = test.drop(columns=["id"])

# ğŸ”  Encode Categorical Features
for col in X.columns:
    if X[col].dtype == 'object':
        le_col = LabelEncoder()
        X[col] = le_col.fit_transform(X[col].astype(str))
        X_test[col] = le_col.transform(X_test[col].astype(str))

# ğŸ§¼ Fill Missing Values
X = X.fillna(X.mean())
X_test = X_test.fillna(X.mean())

# âœ… Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ğŸ¤– Logistic Regression
logreg = LogisticRegression(max_iter=1000)
log_scores = cross_val_score(logreg, X, y, cv=cv, scoring="accuracy")
print(f"ğŸ“Š Logistic Regression CV Accuracy: {log_scores.mean():.5f}")

# ğŸŒ² RandomForest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
print(f"ğŸŒ² RandomForest CV Accuracy: {rf_scores.mean():.5f}")

# âš¡ XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb_scores = cross_val_score(xgb, X, y, cv=cv, scoring="accuracy")
print(f"âš¡ XGBoost CV Accuracy: {xgb_scores.mean():.5f}")

# ğŸ�† Choose best model
best_model = rf  # â†� change to logreg or xgb if they perform better

# ğŸ“ˆ Fit on full training data
best_model.fit(X, y)
test_preds = best_model.predict(X_test)

# ğŸ”„ Decode predictions
submission = sample_submission.copy()
submission["Personality"] = ["Extrovert" if p == 0 else "Introvert" for p in test_preds]

# ğŸ’¾ Save to file
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as 'submission.csv'")

# ğŸ‘€ Preview
print("\nğŸ“‹ Submission preview:")
print(submission.head())





