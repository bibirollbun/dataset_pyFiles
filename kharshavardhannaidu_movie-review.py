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


# ============================================================
# Sentiment Analysis on Movie Reviews (Kaggle)
# TF-IDF (1–2 grams) + Logistic Regression / Multinomial NB
# Confusion matrix + validation bar + submission.csv
# ============================================================

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os, re, string, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use seaborn if available
try:
    import seaborn as sns
    sns.set(style="whitegrid")
    HAS_SNS = True
except Exception:
    HAS_SNS = False

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# ----------------------------
# 1) Paths
# ----------------------------
DATA_DIR = "/kaggle/input/sentiment-analysis-on-movie-reviews"
TRAIN_TSV = os.path.join(DATA_DIR, "train.tsv")
TEST_TSV  = os.path.join(DATA_DIR, "test.tsv")
TRAIN_ZIP = os.path.join(DATA_DIR, "train.tsv.zip")
TEST_ZIP  = os.path.join(DATA_DIR, "test.tsv.zip")
SAMPLE    = os.path.join(DATA_DIR, "sampleSubmission.csv")

# ----------------------------
# 2) Robust loaders (ZIP/TSV)
# ----------------------------
def read_tsv_or_zip(tsv_path, zip_path):
    if os.path.exists(tsv_path):
        return pd.read_csv(tsv_path, sep="\t")
    elif os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            tsv_members = [n for n in zf.namelist() if n.lower().endswith(".tsv")]
            if not tsv_members:
                raise FileNotFoundError(f"No .tsv found inside {zip_path}")
            with zf.open(tsv_members[0]) as f:
                return pd.read_csv(f, sep="\t")
    else:
        raise FileNotFoundError(f"Neither {tsv_path} nor {zip_path} found.")

def read_sample(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample submission not found at {path}")
    return pd.read_csv(path)

# ----------------------------
# 3) Load data
# ----------------------------
train = read_tsv_or_zip(TRAIN_TSV, TRAIN_ZIP)
test  = read_tsv_or_zip(TEST_TSV,  TEST_ZIP)
sample_sub = read_sample(SAMPLE)

print("Train shape:", train.shape, "| Test shape:", test.shape)
print("Train head:\n", train.head(3))

# ----------------------------
# 4) EDA - Sentiment Distribution
# ----------------------------
plt.figure(figsize=(6,4))
if HAS_SNS:
    sns.countplot(x=train["Sentiment"], palette="Blues")
else:
    cnt = train["Sentiment"].value_counts().sort_index()
    plt.bar(cnt.index.astype(str), cnt.values)
plt.title("Sentiment Class Distribution")
plt.xlabel("Sentiment"); plt.ylabel("Count")
plt.tight_layout(); plt.show()

# ----------------------------
# 5) Text Cleaning
# ----------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train["clean"] = train["Phrase"].apply(clean_text)
test["clean"]  = test["Phrase"].apply(clean_text)

# ----------------------------
# 6) TF-IDF Features
# ----------------------------
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words="english",
    max_features=40000,
    ngram_range=(1,2)
)
X = vectorizer.fit_transform(train["clean"])
y = train["Sentiment"].values
X_test = vectorizer.transform(test["clean"])

print("TF-IDF shapes:", X.shape, X_test.shape)

# ----------------------------
# 7) Train / Validation Split
# ----------------------------
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "MultinomialNB": MultinomialNB(),
}

val_scores = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    acc = accuracy_score(y_va, pred)
    val_scores[name] = acc
    print(f"\n{name} | Validation accuracy: {acc:.4f}")
    print(classification_report(y_va, pred, digits=4))

# ----------------------------
# 8) Validation Accuracy Bar
# ----------------------------
plt.figure(figsize=(6,4))
plt.bar(list(val_scores.keys()), list(val_scores.values()), color=["#4C72B0", "#55A868"])
plt.ylim(0,1)
plt.title("Validation Accuracy by Model")
plt.tight_layout(); plt.show()

best_model_name = max(val_scores, key=val_scores.get)
best_model = models[best_model_name]
print(f"\n✅ Best model: {best_model_name} ({val_scores[best_model_name]:.4f})")

# ----------------------------
# 9) Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_va, best_model.predict(X_va))
plt.figure(figsize=(6,5))
if HAS_SNS:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
else:
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
plt.title(f"Confusion Matrix — {best_model_name}")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()

# ----------------------------
# 10) Actual vs Predicted Distribution
# ----------------------------
best_preds = best_model.predict(X_va)
compare_df = pd.DataFrame({"Actual": y_va, "Predicted": best_preds})

plt.figure(figsize=(7,4))
if HAS_SNS:
    sns.countplot(x="Actual", hue="Predicted", data=compare_df, palette="Blues")
else:
    cross_tab = pd.crosstab(compare_df["Actual"], compare_df["Predicted"])
    cross_tab.plot(kind="bar", stacked=True)
plt.title(f"Actual vs Predicted Sentiment — {best_model_name}")
plt.xlabel("Actual Sentiment")
plt.ylabel("Count")
plt.legend(title="Predicted")
plt.tight_layout()
plt.show()

# ----------------------------
# 11) Cross Validation
# ----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")
print(f"{best_model_name} 5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ----------------------------
# 12) Final Training + Submission
# ----------------------------
best_model.fit(X, y)
test_preds = best_model.predict(X_test)

sub = sample_sub.copy()
cols_lower = [c.lower() for c in sub.columns]
phraseid_col = sample_sub.columns[[c.lower()=="phraseid" for c in sample_sub.columns]][0]
sentiment_col = sample_sub.columns[[c.lower()=="sentiment" for c in sample_sub.columns]][0]

sub[sentiment_col] = test_preds
sub.to_csv("submission.csv", index=False)
print("\n✅ Saved submission.csv. Preview:")
print(sub.head())

