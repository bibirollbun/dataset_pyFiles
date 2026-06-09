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


# ===============================================
# ğŸ�¦ Binary Classification - Bank Dataset (S5E8)
# With XGBoost + EDA + Graphs + Submission
# ===============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from xgboost import XGBClassifier

# ------------------------------------------------
# ğŸ”¹ Load dataset (auto or manual)
# ------------------------------------------------
if os.path.exists("/kaggle/input/playground-series-s5e8/train.csv"):
    train_path = "/kaggle/input/playground-series-s5e8/train.csv"
    test_path = "/kaggle/input/playground-series-s5e8/test.csv"
else:
    # ğŸ§© Local testing fallback
    train_path = "train.csv"
    test_path = "test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Train shape: {train.shape}")
print(f"âœ… Test shape: {test.shape}")
print(train.head())

# ------------------------------------------------
# ğŸ”¹ Prepare data
# ------------------------------------------------
TARGET = "y"
ID_COL = "id"

X = train.drop(columns=[TARGET])
y = train[TARGET].astype(int)
X_test = test.copy()

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols and c != ID_COL]

print(f"\nCategorical columns: {len(cat_cols)} | Numerical columns: {len(num_cols)}")

# ------------------------------------------------
# ğŸ”¹ Train/Validation Split
# ------------------------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------
# ğŸ”¹ Preprocessing
# ------------------------------------------------
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", "passthrough", num_cols)
])

# ------------------------------------------------
# ğŸ”¹ Model
# ------------------------------------------------
model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("model", model)
])

# ------------------------------------------------
# ğŸ”¹ Train
# ------------------------------------------------
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_valid)

acc = accuracy_score(y_valid, y_pred)
print(f"\nâœ… Validation Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_valid, y_pred))

# ------------------------------------------------
# ğŸ“Š Graphs: Actual vs Predicted
# ------------------------------------------------
y_pred_proba = pipe.predict_proba(X_valid)[:, 1]
y_pred_class = (y_pred_proba >= 0.5).astype(int)

viz_df = pd.DataFrame({
    'Actual': y_valid.values,
    'Predicted': y_pred_class,
    'Prob': y_pred_proba
})

# Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(pd.crosstab(viz_df['Actual'], viz_df['Predicted']), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Actual vs Predicted)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Probability Distribution
plt.figure(figsize=(6,4))
sns.kdeplot(viz_df[viz_df['Actual']==0]['Prob'], label='Actual=0', fill=True)
sns.kdeplot(viz_df[viz_df['Actual']==1]['Prob'], label='Actual=1', fill=True)
plt.title("Predicted Probability Distribution by Class")
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# Scatter Plot (sample)
sample_viz = viz_df.sample(500, random_state=42)
plt.figure(figsize=(6,4))
plt.scatter(range(len(sample_viz)), sample_viz['Prob'], 
            c=sample_viz['Actual'], cmap='coolwarm', alpha=0.7)
plt.axhline(0.5, color='black', linestyle='--')
plt.title("Actual vs Predicted Probability (sample)")
plt.ylabel("Predicted Probability")
plt.xlabel("Sample index")
plt.colorbar(label='Actual Class (0=No, 1=Yes)')
plt.tight_layout()
plt.show()

# ------------------------------------------------
# ğŸ”¹ Predict Test + Submission
# ------------------------------------------------
test_pred = pipe.predict(X_test)
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: test_pred
})
submission.to_csv("submission.csv", index=False)
print("\nğŸ“� submission.csv saved successfully!")
print(submission.head())


