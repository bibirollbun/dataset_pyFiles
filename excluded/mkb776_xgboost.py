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


# Cell 1: Imports and load
import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

print("Loading data...")
df_train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv', index_col='row_id')
df_test  = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv', index_col='row_id')
print("Train:", df_train.shape, "Test:", df_test.shape)
df_train.head(2)


# Cell 2: Combine text columns
text_cols = [c for c in df_train.columns if c != 'rule_violation']
print("Text columns to combine:", text_cols)

def combine_row_text(row, cols):
    parts = []
    for c in cols:
        v = row.get(c, "")
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            # small separator to avoid accidental word joins
            parts.append(f"[{c}] {s}")
    return " ".join(parts)

# Create full_text for train and test
df_train['full_text'] = df_train.apply(lambda r: combine_row_text(r, text_cols), axis=1)
# For test, ensure we use same columns (test may have same columns)
test_text_cols = [c for c in df_test.columns if c != 'rule_violation'] if 'rule_violation' in df_test.columns else df_test.columns.tolist()
df_test['full_text']  = df_test.apply(lambda r: combine_row_text(r, test_text_cols), axis=1)

# quick preview
df_train[['full_text','rule_violation']].head(3)



# Cell 3: Text cleaning
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove urls
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    # remove html entities
    text = re.sub(r'&\w+;', ' ', text)
    # remove punctuation (keep internal dashes/spaces removed)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    # remove numbers
    text = re.sub(r'\d+', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # remove short words and stopwords
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

# Apply
df_train['clean_text'] = df_train['full_text'].map(clean_text)
df_test['clean_text']  = df_test['full_text'].map(clean_text)

# Quick check
print("Sample cleaned text:")
print(df_train['clean_text'].iloc[0][:500])



# Cell 4: Prepare and balance
X_all = df_train['clean_text']
y_all = df_train['rule_violation'].astype(int)

print("Original label distribution:\n", y_all.value_counts())

# If minority class much smaller, upsample it (simple reproducible approach)
df_major = df_train[df_train.rule_violation == 0].copy()
df_minor = df_train[df_train.rule_violation == 1].copy()

ratio = len(df_minor) / max(1, len(df_major))
print("Minor/major ratio:", ratio)

if ratio < 0.4:
    df_minor_up = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
    df_bal = pd.concat([df_major, df_minor_up]).sample(frac=1, random_state=42).reset_index()
else:
    df_bal = df_train.reset_index()

print("Balanced shape:", df_bal.shape)
X = df_bal['clean_text']
y = df_bal['rule_violation'].astype(int)



# Cell 5: Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, "Valid:", X_valid.shape)


# Cell 6: TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_valid_tfidf = tfidf.transform(X_valid)
X_test_tfidf  = tfidf.transform(df_test['clean_text'])

print("Shapes -> train:", X_train_tfidf.shape, "valid:", X_valid_tfidf.shape, "test:", X_test_tfidf.shape)



# Cell 7: Train models and evaluate
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_valid_tfidf)
    probs = model.predict_proba(X_valid_tfidf)[:,1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_valid, preds)
    f1 = f1_score(y_valid, preds)
    roc = roc_auc_score(y_valid, probs) if probs is not None else np.nan

    print(f"{name} -> Acc: {acc:.4f}, F1: {f1:.4f}, ROC_AUC: {roc if not np.isnan(roc) else 'N/A'}")
    print(classification_report(y_valid, preds, zero_division=0))

    results.append({"Model": name, "Accuracy": acc, "F1": f1, "ROC_AUC": roc})
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values(by='F1', ascending=False).reset_index(drop=True)
display(results_df)



# Cell 8: Save best model (by F1 on validation), predict test probs, and save submission
best_name = results_df.loc[0, 'Model']
best_model = trained_models[best_name]
print("Best model selected:", best_name)

# Save model
os.makedirs('/kaggle/working/models', exist_ok=True)
model_path = f"/kaggle/working/models/{best_name}_tfidf.joblib"
joblib.dump(best_model, model_path)
print("Saved model to:", model_path)

# Predict on test set (probabilities)
if hasattr(best_model, "predict_proba"):
    test_probs = best_model.predict_proba(X_test_tfidf)[:,1]
else:
    # If no predict_proba, try decision_function -> sigmoid to probability
    dec = best_model.decision_function(X_test_tfidf)
    test_probs = 1 / (1 + np.exp(-dec))

# Build submission dataframe (row_id from df_test.index)
df_submission = pd.DataFrame({
    'row_id': df_test.index.astype(int),
    'rule_violation': np.round(test_probs, 6)
}).sort_values('row_id').reset_index(drop=True)

out_path = '/kaggle/working/submission.csv'
df_submission.to_csv(out_path, index=False)
print("Saved submission to:", out_path)
df_submission.head(10)




