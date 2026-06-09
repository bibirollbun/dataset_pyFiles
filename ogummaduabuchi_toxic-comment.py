# Step 1: Load and Explore the Data

import numpy as np
import pandas as pd

# Load the data (CSV files are inside Kaggle input zip folder)
train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
sample_submission = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')

# Define the target label columns
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Show dataset shapes
print("Training Data Shape:", train.shape)
print("Test Data Shape:", test.shape)
print("Sample Submission Shape:", sample_submission.shape)

# Preview the training data
train.head()



# Step 2: Preprocess the Data

import re

# Define text cleaning function
def clean_text(text):
    text = str(text).lower()                                # lowercase
    text = re.sub(r"[^a-z\s]", "", text)                    # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()                # remove extra spaces
    return text

# Apply cleaning to train and test
train["clean_text"] = train["comment_text"].apply(clean_text)
test["clean_text"] = test["comment_text"].apply(clean_text)

# Preview cleaned text
train[["comment_text", "clean_text"]].head()



# Step 3: Vectorization (TF-IDF)

from sklearn.feature_extraction.text import TfidfVectorizer

# Define TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=50000,        # limit to top 50,000 features
    stop_words="english",      # remove common stopwords
    ngram_range=(1,2)          # include unigrams and bigrams
)

# Fit on train and transform both train/test
X_train = tfidf.fit_transform(train["clean_text"])
X_test = tfidf.transform(test["clean_text"])

# Target labels
y_train = train[labels]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)



from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Split into training and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Define Logistic Regression model (with regularization)
log_reg = OneVsRestClassifier(
    LogisticRegression(solver='sag', C=1, max_iter=1000)
)

# Train
log_reg.fit(X_tr, y_tr)

# Predict probabilities on validation set
y_val_pred = log_reg.predict_proba(X_val)

# Evaluate using ROC-AUC for each label
roc_auc_scores = {}
for i, label in enumerate(labels):
    score = roc_auc_score(y_val[label], y_val_pred[:, i])
    roc_auc_scores[label] = score

print("Validation ROC-AUC per label:")
for label, score in roc_auc_scores.items():
    print(f"{label}: {score:.4f}")

print("\nMean ROC-AUC:", np.mean(list(roc_auc_scores.values())))



# -------------------------------
# Step B: Naive Bayes Classifier
# -------------------------------

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

# -------------------------------
# 1. Load Data
# -------------------------------
train = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
sample_submission_df = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# -------------------------------
# 2. Clean Text
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

train["clean_text"] = train["comment_text"].apply(clean_text)
test["clean_text"] = test["comment_text"].apply(clean_text)

# -------------------------------
# 3. Vectorization (TF-IDF)
# -------------------------------
tfidf = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1,2)
)

X = tfidf.fit_transform(train["clean_text"])
X_test = tfidf.transform(test["clean_text"])
y = train[labels]

# -------------------------------
# 4. Train / Validation Split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Train Naive Bayes
# -------------------------------
nb = OneVsRestClassifier(MultinomialNB(alpha=0.1))
nb.fit(X_train, y_train)

# -------------------------------
# 6. Validation Evaluation
# -------------------------------
y_val_pred = nb.predict_proba(X_val)

for i, label in enumerate(labels):
    auc = roc_auc_score(y_val[label], y_val_pred[:, i])
    print(f"{label}: {auc:.4f}")

mean_auc = roc_auc_score(y_val, y_val_pred, average="macro")
print("\nMean ROC-AUC:", mean_auc)

# -------------------------------
# 7. Make Predictions on Test
# -------------------------------
y_test_pred = nb.predict_proba(X_test)

sample_submission_df.iloc[:, 1:] = y_test_pred
sample_submission_df.to_csv("submission_nb.csv", index=False)

print("\n✅ Naive Bayes submission file saved as submission_nb.csv")



# -------------------------------
# Regenerate Logistic Regression Predictions
# -------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Train Logistic Regression again
lr = OneVsRestClassifier(LogisticRegression(solver="liblinear", max_iter=200))
lr.fit(X_train, y_train)

# Predict on test
y_test_pred_lr = lr.predict_proba(X_test)

# Save LR submission
sub_lr = sample_submission_df.copy()
sub_lr.iloc[:, 1:] = y_test_pred_lr
sub_lr.to_csv("submission.csv", index=False)


# -------------------------------
# Regenerate Naive Bayes Predictions
# -------------------------------
from sklearn.naive_bayes import MultinomialNB

nb = OneVsRestClassifier(MultinomialNB(alpha=0.1))
nb.fit(X_train, y_train)

y_test_pred_nb = nb.predict_proba(X_test)

# Save NB submission
sub_nb = sample_submission_df.copy()
sub_nb.iloc[:, 1:] = y_test_pred_nb
sub_nb.to_csv("submission_nb.csv", index=False)


# -------------------------------
# Blend Predictions
# -------------------------------
sub_blend = sub_lr.copy()
sub_blend.iloc[:, 1:] = (sub_lr.iloc[:, 1:] + sub_nb.iloc[:, 1:]) / 2
sub_blend.to_csv("submission_blend.csv", index=False)

print("✅ Blended submission saved as submission_blend.csv")



!kaggle competitions submit -c jigsaw-toxic-comment-classification-challenge -f submission_blend.csv -m "Blended Logistic Regression + Naive Bayes"


# Display a clickable download link in the notebook
from IPython.display import FileLink

FileLink("/kaggle/working/submission_blend.csv")


