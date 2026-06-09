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
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score







# ---------------- Load data ----------------
train_df = pd.read_csv("/kaggle/input/comments-classification/Dataset/train.csv")
test_df = pd.read_csv("/kaggle/input/comments-classification/Dataset/test.csv")

train_df = train_df.rename(columns={"comment_text": "text", "psychotic_depression": "label"})
test_df = test_df.rename(columns={"comment_text": "text"})

# ---------------- Preprocessing ----------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

train_df["cleaned_text"] = train_df["text"].apply(clean_text)
test_df["cleaned_text"] = test_df["text"].apply(clean_text)

# ---------------- TF-IDF with bigrams ----------------
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X = vectorizer.fit_transform(train_df["cleaned_text"])
y = train_df["label"]

X_test_final = vectorizer.transform(test_df["cleaned_text"])

# ---------------- Train/Validation split ----------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Improved Logistic Regression ----------------
model = LogisticRegression(max_iter=300, class_weight="balanced")  
model.fit(X_train, y_train)

# ---------------- Validation ----------------
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# ---------------- Predict on test.csv ----------------
test_predictions = model.predict(X_test_final)

# ---------------- Save submission ----------------
submission = pd.DataFrame({
    "ID": test_df["id"],
    "psychotic_depression": test_predictions
})

submission.to_csv("submission.csv", index=False)
print(submission.head())


