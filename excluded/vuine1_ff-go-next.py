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


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =======================
# 1. Load train & test
# =======================
train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

# =======================
# 2. Combine text fields (include examples)
# =======================
def combine_text(df):
    return (
        df["body"] +
        " [SEP] Rule: " + df["rule"] +
        " [SEP] Subreddit: " + df["subreddit"] +
        " [SEP] Positive Examples: " + df["positive_example_1"] + " | " + df["positive_example_2"] +
        " [SEP] Negative Examples: " + df["negative_example_1"] + " | " + df["negative_example_2"]
    )

train_df["text"] = combine_text(train_df)
test_df["text"] = combine_text(test_df)

# Features and labels
X_train = train_df["text"]
y_train = train_df["rule_violation"]
X_test = test_df["text"]

# =======================
# 3. TF-IDF Vectorization
# =======================
vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =======================
# 4. Train Logistic Regression
# =======================
clf = LogisticRegression(max_iter=200, class_weight="balanced")
clf.fit(X_train_tfidf, y_train)

# =======================
# 5. Predictions & probabilities
# =======================
y_pred = clf.predict(X_test_tfidf)
y_proba = clf.predict_proba(X_test_tfidf)[:, 1]

# =======================
# 6. Save submission file
# =======================
submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "rule_violation": y_pred
})

submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv")
print(submission.head())




