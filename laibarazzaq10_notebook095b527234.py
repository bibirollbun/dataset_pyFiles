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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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

# ğŸ“Š Logistic Regression
logreg = LogisticRegression(max_iter=1000)
log_scores = cross_val_score(logreg, X, y, cv=cv, scoring="accuracy")
print(f"ğŸ“Š Logistic Regression CV Accuracy: {log_scores.mean():.5f}")

# ğŸŒ² Tuned RandomForest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
print(f"ğŸŒ² Tuned RandomForest CV Accuracy: {rf_scores.mean():.5f}")

# âš¡ XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgb_scores = cross_val_score(xgb, X, y, cv=cv, scoring="accuracy")
print(f"âš¡ XGBoost CV Accuracy: {xgb_scores.mean():.5f}")

# ğŸ§ƒ Voting Ensemble (soft voting)
blend = VotingClassifier(
    estimators=[("rf", rf), ("logreg", logreg), ("xgb", xgb)],
    voting='soft',
    n_jobs=-1
)
blend_scores = cross_val_score(blend, X, y, cv=cv, scoring="accuracy")
print(f"ğŸ§ƒ Voting Ensemble CV Accuracy: {blend_scores.mean():.5f}")

# ğŸ�† Pick the best model based on CV scores
model_scores = {
    "logreg": log_scores.mean(),
    "rf": rf_scores.mean(),
    "xgb": xgb_scores.mean(),
    "blend": blend_scores.mean()
}
best_model_name = max(model_scores, key=model_scores.get)
print(f"\nğŸ¥‡ Best model: {best_model_name} with CV accuracy {model_scores[best_model_name]:.5f}")

# ğŸš€ Train best model on full data
if best_model_name == "logreg":
    best_model = logreg
elif best_model_name == "rf":
    best_model = rf
elif best_model_name == "xgb":
    best_model = xgb
else:
    best_model = blend

best_model.fit(X, y)

# ğŸ�¯ Predict with Probabilities
probs = best_model.predict_proba(X_test)[:, 1]

# ğŸ”§ Try a custom threshold (ADJUST THIS!)
threshold = 0.52  # Try 0.48, 0.49, 0.50, 0.51, 0.53, etc.
test_preds = (probs > threshold).astype(int)

# ğŸ”„ Decode predictions
submission = sample_submission.copy()
submission["Personality"] = ["Extrovert" if p == 0 else "Introvert" for p in test_preds]

# ğŸ’¾ Save with threshold in filename
filename = f"submission_threshold_{threshold:.2f}.csv"
submission.to_csv(filename, index=False)

print(f"\nâœ… Submission file saved as '{filename}'")
print("ğŸ“‹ Preview:")
print(submission.head())





