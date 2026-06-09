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


# %% [markdown]
# Challenge 1 — Fundamentals (20%)
# TF-IDF + Logistic Regression Baseline
# -------------------------------------
# Output: submission.csv (Id, TARGET) – exactly 1000 rows

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



# %%
train = pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")
test = pd.read_csv("/kaggle/input/rmit-hackathon-2025/test.csv")
sample_sub = pd.read_csv("/kaggle/input/rmit-hackathon-2025/sample_submission.csv")

# Normalize column names (lowercase all)
train.columns = train.columns.str.lower()
test.columns = test.columns.str.lower()
sample_sub.columns = sample_sub.columns.str.lower()

print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())
print("Sample submission columns:", sample_sub.columns.tolist())



# %%
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove urls
    text = re.sub(r"[^a-z0-9\s]", " ", text)             # keep alphanumeric
    text = re.sub(r"\s+", " ", text).strip()             # remove extra spaces
    return text

train["clean_text"] = train["text"].apply(clean_text)
test["clean_text"] = test["text"].apply(clean_text)



# %%
# TF-IDF with both word and character features
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1,2),
    max_features=20000
)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(2,5),
    max_features=30000
)

X_word = word_vectorizer.fit_transform(train["clean_text"])
X_char = char_vectorizer.fit_transform(train["clean_text"])
from scipy.sparse import hstack
X = hstack([X_word, X_char])

y = train["label"]

# Vectorize test data using same vectorizers
X_test_word = word_vectorizer.transform(test["clean_text"])
X_test_char = char_vectorizer.transform(test["clean_text"])
X_test = hstack([X_test_word, X_test_char])



# %%
from sklearn.linear_model import LogisticRegression

# Train logistic regression for probability output
model = LogisticRegression(max_iter=1000, C=2.0, solver="liblinear", random_state=RANDOM_STATE)
model.fit(X, y)

print("✅ Model trained successfully.")



# %%
# Predict probabilities instead of class labels
pred_probs = model.predict_proba(X_test)[:, 1]  # column 1 = probability of class '1' (jailbreak)

submission = pd.DataFrame({
    "Id": test.iloc[:, 0],    # automatically take first column if uncertain name
    "TARGET": pred_probs
})

# Ensure correct shape
assert submission.shape[0] == 1000, f"❌ Submission must have exactly 1000 rows, found {submission.shape[0]}."

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved with probabilities, shape:", submission.shape)
submission.head()


