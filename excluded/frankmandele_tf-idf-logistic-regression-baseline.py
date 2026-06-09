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


# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")


# Load data
train = pd.read_csv("/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip", sep="\t")
test = pd.read_csv("/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip", sep="\t")

train.head()


# Sentiment count distribution
print(train['Sentiment'].value_counts())


# Cleaning the data
train['Phrase'] = train['Phrase'].fillna("").str.lower().str.strip()
test['Phrase']  = test['Phrase'].fillna("").str.lower().str.strip()


# Process & Split
X = train.Phrase.values
y = train.Sentiment.values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Vectorization
tfidf = TfidfVectorizer(
    ngram_range=(1,2),      # unigrams + bigrams
    max_features=50000,     # cap vocabulary size
    stop_words="english"    # remove common stopwords
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(test['Phrase'])


# Train Model (Baseline)
model = LogisticRegression(
    max_iter=200, 
    class_weight="balanced",  # handle class imbalance
    solver="lbfgs",
    multi_class="multinomial"
)
model.fit(X_train_tfidf, y_train)

# Validation Evaluation
y_pred = model.predict(X_val_tfidf)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))


# # Train on Full Data & Predict Test
# X_full_tfidf = tfidf.fit_transform(X)   # retrain vectorizer on all train data
# model.fit(X_full_tfidf, y)

# X_test_tfidf = tfidf.transform(test['Phrase'])
# test_preds = model.predict(X_test_tfidf)

# # Submission file
# submission = pd.DataFrame({
#     "PhraseId": test['PhraseId'],
#     "Sentiment": test_preds
# })

# submission.to_csv("submission.csv", index=False)
# print("Submission file created: submission.csv")


# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression


# # --- Pipeline: TF-IDF -> Logistic Regression ---
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer()),
#     ('lr', LogisticRegression(max_iter=300, solver='lbfgs', multi_class='multinomial'))
# ])

# # --- Hyperparameter Grid ---
# param_grid = {
#     'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
#     'tfidf__max_features': [20000, 50000, 100000],
#     'tfidf__stop_words': [None, 'english'],
#     'lr__C': [0.01, 0.1, 1, 10, 100]
# }

# # --- Grid Search ---
# grid = GridSearchCV(
#     pipeline,
#     param_grid,
#     cv=3,              # 3-fold cross-validation
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=2
# )

# grid.fit(train['Phrase'], train['Sentiment'])

# print("Best Parameters:", grid.best_params_)
# print("Best CV Accuracy:", grid.best_score_)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform
import numpy as np



# --- Pipeline: TF-IDF -> Logistic Regression ---
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression(
        max_iter=300, 
        solver='lbfgs', 
        multi_class='multinomial'
    ))
])

# --- Hyperparameter Distributions ---
param_dist = {
    'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
    'tfidf__max_features': [20000, 50000, 100000],
    'tfidf__stop_words': [None, 'english'],
    'lr__C': loguniform(1e-2, 1e2)  # random samples from 0.01 to 100
}


# --- Randomized Search ---
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,          # number of random combos to try
    cv=3,               # 3-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(train['Phrase'], train['Sentiment'])

print("Best Parameters:", random_search.best_params_)
print("Best CV Accuracy:", random_search.best_score_)


best_model = random_search.best_estimator_

# --- Clean test data ---
test['Phrase'] = test['Phrase'].fillna("").str.lower().str.strip()

# --- Predict with the tuned model ---
test_preds = best_model.predict(test['Phrase'])

# --- Submission file ---
submission = pd.DataFrame({
    "PhraseId": test['PhraseId'],
    "Sentiment": test_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")





