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


train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
train.head()
test.head()


pip show transformers


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# 1. Load data
train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
sample_submission = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")

# 2. Columns
TEXT_COL = "body"
LABEL_COL = "rule_violation"

# 3. Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df[TEXT_COL], train_df[LABEL_COL], test_size=0.1, random_state=42
)

# 4. Create pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ("clf", LogisticRegression(C=1.0, max_iter=1000))
])

# 5. Train model
pipeline.fit(train_texts, train_labels)

# 6. Evaluate on validation set
val_preds = pipeline.predict(val_texts)
print("Validation Classification Report:")
print(classification_report(val_labels, val_preds))

# 7. Predict on test set
test_preds = pipeline.predict(test_df[TEXT_COL])

# 8. Prepare submission
submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "rule_violation": test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission saved to 'submission.csv'")








