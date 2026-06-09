


import pandas as pd
import numpy as np
import re, string

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore")

# Download only if not already present (offline safe)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)



train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



print("Columns:", train.columns.tolist())
print("\nMissing Values:\n", train.isnull().sum())

sns.countplot(x='rule_violation', data=train)
plt.title("Target Distribution")
plt.show()



train['body_len'] = train['body'].fillna('').apply(len)
train['rule_len'] = train['rule'].fillna('').apply(len)
test['body_len'] = test['body'].fillna('').apply(len)
test['rule_len'] = test['rule'].fillna('').apply(len)

sns.histplot(train['body_len'], bins=50)
plt.title("Body text length")
plt.show()



stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text): return ''
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)  # remove punctuation
    text = re.sub("\d+", " ", text)  # remove numbers
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Apply cleaning
for col in ['body', 'rule', 'subreddit']:
    train[col] = train[col].astype(str).apply(clean_text)
    test[col] = test[col].astype(str).apply(clean_text)



# Combine body + rule + subreddit as single text feature
train['text'] = train['body'] + " " + train['rule'] + " " + train['subreddit']
test['text'] = test['body'] + " " + test['rule'] + " " + test['subreddit']



# TF-IDF Vectorizer: unigrams + bigrams, min_df=3 for noise reduction
tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=3)

X_tfidf = tfidf.fit_transform(train['text'])
X_test_tfidf = tfidf.transform(test['text'])

print("TF-IDF shape:", X_tfidf.shape)



# Numeric features: body_len, rule_len
num_features = ['body_len', 'rule_len']

# Scale numeric features
scaler = StandardScaler()
X_num = scaler.fit_transform(train[num_features])
X_test_num = scaler.transform(test[num_features])

# Merge TF-IDF + Numeric features (horizontal stack)
from scipy.sparse import hstack

X = hstack([X_tfidf, X_num])
X_test_final = hstack([X_test_tfidf, X_test_num])

y = train['rule_violation'].values

print("Final feature matrix shape:", X.shape)



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Validation shape:", X_val.shape)



log_reg = LogisticRegression(
    C=4.0,                 # higher C = less regularization
    max_iter=200,          # more iterations for convergence
    class_weight='balanced', # handle imbalance
    solver='saga',         # handles large sparse data
    n_jobs=-1
)

log_reg.fit(X_train, y_train)

val_preds_lr = log_reg.predict(X_val)
val_probs_lr = log_reg.predict_proba(X_val)[:, 1]

print("Logistic Regression Validation F1:", f1_score(y_val, val_preds_lr))
print("Logistic Regression Validation Accuracy:", accuracy_score(y_val, val_preds_lr))



print("Logistic Regression Report:\n")
print(classification_report(y_val, val_preds_lr))



rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

val_preds_rf = rf.predict(X_val)
val_probs_rf = rf.predict_proba(X_val)[:, 1]

print("RandomForest Validation F1:", f1_score(y_val, val_preds_rf))
print("RandomForest Validation Accuracy:", accuracy_score(y_val, val_preds_rf))



print("Random Forest Report:\n")
print(classification_report(y_val, val_preds_rf))



f1_lr = f1_score(y_val, val_preds_lr)
f1_rf = f1_score(y_val, val_preds_rf)

print(f"F1 Logistic Regression: {f1_lr:.4f}")
print(f"F1 Random Forest: {f1_rf:.4f}")

plt.bar(["Logistic Regression", "Random Forest"], [f1_lr, f1_rf])
plt.ylabel("F1 Score")
plt.title("Model Comparison")
plt.show()



# Weight ensemble: more weight to LR as it's usually more stable
ensemble_probs = 0.6 * val_probs_lr + 0.4 * val_probs_rf
ensemble_preds = (ensemble_probs > 0.5).astype(int)

print("Ensemble F1:", f1_score(y_val, ensemble_preds))
print("Ensemble Accuracy:", accuracy_score(y_val, ensemble_preds))



print("Ensemble Report:\n")
print(classification_report(y_val, ensemble_preds))



# Retrain on full data for final predictions
log_reg.fit(X, y)
rf.fit(X, y)



test_probs_lr = log_reg.predict_proba(X_test_final)[:, 1]
test_probs_rf = rf.predict_proba(X_test_final)[:, 1]

final_probs = 0.6 * test_probs_lr + 0.4 * test_probs_rf
final_preds = (final_probs > 0.5).astype(int)

print("Sample predictions:", final_preds[:10])



submission = pd.DataFrame({
    "row_id": test['row_id'],
    "rule_violation": final_preds
})

submission.head()



submission.to_csv("submission.csv", index=False)
print("âœ… Submission file created successfully!")



print("Preview of Submission File:")
print(submission.sample(10))


