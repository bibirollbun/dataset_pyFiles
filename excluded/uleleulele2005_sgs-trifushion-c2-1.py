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


# ================================================================
# RMIT GenAI & Cybersecurity Hackathon 2025
# Challenge 1 â€” Fundamentals (20%)
# Baseline model: TF-IDF + Logistic Regression
# ================================================================

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# ================================================================
# 1ï¸�âƒ£ Explore input directory
# ================================================================
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ================================================================
# 2ï¸�âƒ£ Import libraries for modeling
# ================================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ================================================================
# 3ï¸�âƒ£ Load datasets (verified path)
# ================================================================
train = pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")
test  = pd.read_csv("/kaggle/input/rmit-hackathon-2025/test.csv")

print("âœ… Train shape:", train.shape)
print("âœ… Test shape:", test.shape)
print("âœ… Columns:", list(train.columns))

# ================================================================
# 4ï¸�âƒ£ Data preparation
# ================================================================
# Handle potential column naming inconsistencies
text_col = "text" if "text" in train.columns else train.columns[1]
label_col = "label" if "label" in train.columns else train.columns[-1]

train[text_col] = train[text_col].astype(str)
test[text_col]  = test[text_col].astype(str)

print("Label distribution:\n", train[label_col].value_counts())

# ================================================================
# 5ï¸�âƒ£ TF-IDF Vectorization
# ================================================================
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = tfidf.fit_transform(train[text_col])
y = train[label_col]
X_test = tfidf.transform(test[text_col])

# ================================================================
# 6ï¸�âƒ£ Train-validation split (for local check)
# ================================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================================================
# 7ï¸�âƒ£ Model training
# ================================================================
model = LogisticRegression(max_iter=300, solver='lbfgs')
model.fit(X_train, y_train)

# ================================================================
# 8ï¸�âƒ£ Validation (optional)
# ================================================================
val_preds = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_preds)
print(f"Validation ROC-AUC: {auc:.4f}")

# ================================================================
# 9ï¸�âƒ£ Predictions on test data
# ================================================================
test_preds = model.predict_proba(X_test)[:, 1]

# ================================================================
# ğŸ”Ÿ Create submission file
# ================================================================
# Detect ID column (it could be 'id', 'Id', or 'ID')
id_col = None
for possible_id in ["id", "Id", "ID"]:
    if possible_id in test.columns:
        id_col = possible_id
        break

if id_col is None:
    raise KeyError("â�Œ No ID column found in test.csv. Please check your dataset.")

submission = pd.DataFrame({
    id_col: test[id_col],
    "target": test_preds
})

submission.to_csv("submission.csv", index=False)
print("\nâœ… submission.csv created successfully!")
print(submission.head())

# ================================================================
# ğŸ�¯ Final Notes
# ================================================================
# â€¢ Submit this file to Kaggle for your Challenge 1 score.
# â€¢ This baseline automatically earns 20% marks upon valid submission.
# â€¢ Next: upgrade to a BERT or ensemble model for Challenge 2 (Protector).


