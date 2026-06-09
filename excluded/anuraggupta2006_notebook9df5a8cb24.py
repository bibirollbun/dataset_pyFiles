# ================================================================
# Diabetes Prediction Challenge (TPS S5E12)
# One-Hot Encoding + RandomForest + Submission
# ================================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# 1ï¸�âƒ£ Load Data
# --------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

print(train.shape, test.shape, sample.shape)

# --------------------------
# 2ï¸�âƒ£ Target & ID
# --------------------------
TARGET = "diagnosed_diabetes"
ID_COL = "id"

# --------------------------
# 3ï¸�âƒ£ Features
# --------------------------
feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]
X = train[feature_cols]
y = train[TARGET]
X_test = test[feature_cols]

# --------------------------
# 4ï¸�âƒ£ Separate numeric & categorical columns
# --------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns

print("Numeric features:", list(numeric_features))
print("Categorical features:", list(categorical_features))

# --------------------------
# 5ï¸�âƒ£ Preprocessor: scale numeric + one-hot encode categorical
# --------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# --------------------------
# 6ï¸�âƒ£ Model
# --------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# --------------------------
# 7ï¸�âƒ£ Pipeline = preprocessing + model
# --------------------------
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# --------------------------
# 8ï¸�âƒ£ Fit on full training data
# --------------------------
clf.fit(X, y)

# --------------------------
# 9ï¸�âƒ£ Predict probabilities for test set
# --------------------------
test_preds = clf.predict_proba(X_test)[:, 1]

# --------------------------
# ğŸ”Ÿ Build submission
# --------------------------
submission = sample.copy()
# sample_submission has columns: ['id', 'diagnosed_diabetes']
submission[TARGET] = test_preds

submission.to_csv("submission.csv", index=False)
submission.head()


