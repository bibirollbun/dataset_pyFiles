# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

files = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
        print(files[-1])

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_path = next((f for f in files if f.lower().endswith('train.csv')), None)
test_path  = next((f for f in files if f.lower().endswith('test.csv')), None)
sample_path = next((f for f in files if 'sample' in f.lower() and f.lower().endswith('.csv')), None)

print("train:", train_path)
print("test: ", test_path)
print("sample:", sample_path)

if not train_path or not test_path:
    raise FileNotFoundError("Could not find train.csv/test.csv in attached dataset. Rename accordingly or update paths.")


# Read
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

# Inspect basic columns
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())
display(train.head())


# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

TEXT_COL = 'text'
ID_COL = 'Id'
TARGET_COL = 'label'

# Fill missing text
train[TEXT_COL] = train[TEXT_COL].fillna("")
test[TEXT_COL]  = test[TEXT_COL].fillna("")

# Basic pipeline: TF-IDF -> LogisticRegression
tfidf = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1,2),
    min_df=2,
    stop_words='english'
)
clf = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000, class_weight='balanced')

pipeline = make_pipeline(tfidf, clf)

# Quick cross-validation on train
X = train[TEXT_COL].values
y = train[TARGET_COL].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print("Cross-val ROC AUC (5-fold):", scores, "mean:", scores.mean())

# Fit on full train
pipeline.fit(X, y)

# Predict probabilities on test
probs = pipeline.predict_proba(test[TEXT_COL].values)[:, 1]

# Build submission
submission = pd.DataFrame({
    ID_COL: test[ID_COL].values,
    'target': probs
})

# Clip probabilities to [0,1] just in case
submission['target'] = submission['target'].clip(0,1)

# Save
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv")
display(submission.head())

