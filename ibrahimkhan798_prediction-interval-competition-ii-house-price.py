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


# -----------------------------
# 1. Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 2. Load Data
# -----------------------------
train_df = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test_df = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")
sample_submission = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# -----------------------------
# 3. Combine for Preprocessing
# -----------------------------
target = "sale_price"
test_df[target] = np.nan  # Add dummy column for consistent structure

combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# -----------------------------
# 4. Encode Categorical Features
# -----------------------------
categorical_cols = combined.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# -----------------------------
# 5. Separate Back
# -----------------------------
train_df = combined[~combined[target].isna()].copy()
test_df = combined[combined[target].isna()].drop(columns=[target]).copy()

X = train_df.drop(columns=[target])
y = train_df[target]

# -----------------------------
# 6. Handle Missing Values
# -----------------------------
X = X.fillna(X.median(numeric_only=True))
test_df = test_df.fillna(X.median(numeric_only=True))

# -----------------------------
# 7. Train/Validation Split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 8. Train Quantile Models (Using Callbacks for Early Stopping)
# -----------------------------
def train_quantile_model(alpha):
    model = lgb.LGBMRegressor(objective='quantile', alpha=alpha, n_estimators=1000)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='quantile',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    return model

model_lower = train_quantile_model(0.05)
model_upper = train_quantile_model(0.95)

# -----------------------------
# 9. Winkler Score Evaluation
# -----------------------------
def winkler_score(y_true, lower, upper, alpha=0.1):
    score = []
    for yt, l, u in zip(y_true, lower, upper):
        if yt < l:
            score.append(u - l + (2 / alpha) * (l - yt))
        elif yt > u:
            score.append(u - l + (2 / alpha) * (yt - u))
        else:
            score.append(u - l)
    return np.mean(score)

val_lower = model_lower.predict(X_val)
val_upper = model_upper.predict(X_val)

winkler = winkler_score(y_val, val_lower, val_upper)
print(f"Winkler Score on validation set: {winkler:.2f}")

# -----------------------------
# 10. Predict on Test Set
# -----------------------------
test_lower = model_lower.predict(test_df)
test_upper = model_upper.predict(test_df)

# -----------------------------
# 11. Generate Submission File
# -----------------------------
submission = pd.DataFrame({
    "id": test_df["id"],
    "pi_lower": test_lower,
    "pi_upper": test_upper
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved for upload!")

