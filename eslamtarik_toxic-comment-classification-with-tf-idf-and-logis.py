!pip install scikit-learn==1.7.1 -q


import re
import string
import zipfile
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def load_csv_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            return pd.read_csv(f)



train_df = load_csv_from_zip("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test_df  = load_csv_from_zip("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")



train_df.head()


print("Training shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nData types:")
print(train_df.dtypes)



print("\nMissing values:")
print(train_df.isnull().sum())


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


label_counts = train_df[label_cols].sum().sort_values(ascending=False)
plt.figure(figsize=(8,4))
sns.barplot(x=label_counts.index, y=label_counts.values, palette="Blues_d")
plt.title("Number of Positive Samples per Toxicity Label")
plt.ylabel("Count")
plt.xlabel("Toxicity Type")
plt.show()

(label_counts / len(train_df) * 100).round(2)



train_df["label_count"] = train_df[label_cols].sum(axis=1)
plt.figure(figsize=(7,4))
sns.countplot(x="label_count", data=train_df, palette="mako")
plt.title("Number of Toxic Categories per Comment")
plt.xlabel("Toxic Labels per Comment")
plt.ylabel("Count")
plt.show()



plt.figure(figsize=(6,5))
sns.heatmap(train_df[label_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Toxicity Labels")
plt.show()



train_df["char_len"] = train_df["comment_text"].apply(len)
train_df["word_len"] = train_df["comment_text"].apply(lambda x: len(x.split()))

plt.figure(figsize=(8,4))
sns.histplot(train_df["word_len"], bins=100, color="steelblue")
plt.xlim(0, 300)
plt.title("Distribution of Comment Lengths (Words)")
plt.xlabel("Words per Comment")
plt.ylabel("Frequency")
plt.show()


train_df[["char_len", "word_len"]].describe(percentiles=[.25, .5, .75, .9, .95, .99])



def get_top_words(texts, n=20):
    vec = CountVectorizer(stop_words='english', max_features=5000)
    bag = vec.fit_transform(texts)
    freqs = dict(zip(vec.get_feature_names_out(), bag.toarray().sum(axis=0)))
    top = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:n]
    return pd.DataFrame(top, columns=["word", "count"])

toxic_words = get_top_words(train_df.loc[train_df["toxic"]==1, "comment_text"])
non_toxic_words = get_top_words(train_df.loc[train_df["toxic"]==0, "comment_text"])

fig, axes = plt.subplots(1, 2, figsize=(14,5))
sns.barplot(y="word", x="count", data=toxic_words, ax=axes[0], color="firebrick")
axes[0].set_title("Top Words in Toxic Comments")
sns.barplot(y="word", x="count", data=non_toxic_words, ax=axes[1], color="steelblue")
axes[1].set_title("Top Words in Non-Toxic Comments")
plt.tight_layout()
plt.show()



co_occurrence = train_df[label_cols].T.dot(train_df[label_cols])
plt.figure(figsize=(6,5))
sns.heatmap(co_occurrence, annot=True, fmt='d', cmap="YlGnBu")
plt.title("Co-Occurrence of Toxic Labels")
plt.show()



print("=== Toxic Examples ===")
for i, row in train_df[train_df["toxic"]==1].sample(3, random_state=42).iterrows():
    print("-"*80)
    print(row["comment_text"])
    print({col:int(row[col]) for col in label_cols if row[col]==1})

print("\n=== Non-Toxic Examples ===")
for i, row in train_df[train_df["toxic"]==0].sample(3, random_state=24).iterrows():
    print("-"*80)
    print(row["comment_text"][:400], "...")



def _normalize_repeated_chars(text):
    return re.sub(r'(.)\1{3,}', r'\1\1\1', text)



def _remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', ' ', text)


def _remove_html(text):
    return re.sub(r'<.*?>', ' ', text)


_punc_table = str.maketrans(string.punctuation, " " * len(string.punctuation))

def _remove_punctuation(text):
    return text.translate(_punc_table)


def _remove_numbers(text):
    return re.sub(r'\d+', ' ', text)



def _fix_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()



def _filter_short_tokens(text, min_len=3, keep_tokens=('i',)):
    tokens = text.split()
    cleaned_tokens = []
    for tok in tokens:
        if tok in keep_tokens:
            cleaned_tokens.append(tok)
        elif len(tok) >= min_len:
            cleaned_tokens.append(tok)
    return " ".join(cleaned_tokens)


def _filter_long_tokens(text, max_len=1000, keep_tokens=('i',)):
    tokens = text.split()
    cleaned_tokens = []
    for tok in tokens:
        if tok in keep_tokens:
            cleaned_tokens.append(tok)
        elif len(tok) <= max_len:
            cleaned_tokens.append(tok)
    return " ".join(cleaned_tokens)


def clean_comment(text: str) -> str:
    text = text.lower()
    text = _remove_urls(text)
    text = _remove_html(text)
    text = _remove_numbers(text)
    text = _normalize_repeated_chars(text)
    text = _remove_punctuation(text)
    text = _fix_whitespace(text)
    text = _filter_short_tokens(text, min_len=3, keep_tokens=('i', 'im'))
    text = _filter_long_tokens(text, max_len=1000, keep_tokens=('i', 'im'))
    text = _fix_whitespace(text)
    return text



train_df["clean_text"] = train_df["comment_text"].astype(str).apply(clean_comment)
test_df["clean_text"]  = test_df["comment_text"].astype(str).apply(clean_comment)


train_df[["comment_text", "clean_text"]].head(5)


train_df["raw_word_len"]   = train_df["comment_text"].apply(lambda x: len(str(x).split()))
train_df["clean_word_len"] = train_df["clean_text"].apply(lambda x: len(str(x).split()))
length_summary = train_df[["raw_word_len", "clean_word_len"]].describe(percentiles=[0.5,0.75,0.9,0.95,0.99])
length_summary



plt.figure(figsize=(8,4))
sns.kdeplot(train_df["raw_word_len"], label="raw", bw_adjust=1)
sns.kdeplot(train_df["clean_word_len"], label="clean", bw_adjust=1)
plt.xlim(0, 150)
plt.xlabel("Words per comment")
plt.ylabel("Density")
plt.title("Effect of Cleaning on Comment Length")
plt.legend()
plt.show()



train_df["is_empty_after_clean"] = train_df["clean_text"].str.len() == 0
empty_count = train_df["is_empty_after_clean"].sum()
print(f"Comments that became empty after cleaning: {empty_count}")

# We generally drop these from training
train_df = train_df[~train_df["is_empty_after_clean"]].copy()

print("New train_df shape after dropping empty rows:", train_df.shape)



label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("=== Cleaned toxic examples ===")
for i, row in train_df[train_df["toxic"]==1].sample(3, random_state=0).iterrows():
    print("-"*80)
    print("RAW:", row["comment_text"])
    print("CLEAN:", row["clean_text"])
    print({c:int(row[c]) for c in label_cols if row[c]==1})

print("\n=== Cleaned non-toxic examples ===")
for i, row in train_df[train_df["toxic"]==0].sample(3, random_state=1).iterrows():
    print("-"*80)
    print("RAW:", row["comment_text"][:300])
    print("CLEAN:", row["clean_text"][:300])



X_text = train_df["clean_text"]
y = train_df[label_cols].astype(int)


print("X_text shape:", X_text.shape)
print("y shape:", y.shape)
print("Labels:", y.columns.tolist())


X_train, X_val, y_train, y_val = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=train_df["toxic"]
)



print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)



word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1,2),
    max_features=50000
)



char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(2,6),
    max_features=30000
)


word_vectorizer.fit(X_train)
char_vectorizer.fit(X_train)


X_train_word = word_vectorizer.transform(X_train)
X_val_word   = word_vectorizer.transform(X_val)

X_train_char = char_vectorizer.transform(X_train)
X_val_char   = char_vectorizer.transform(X_val)


X_train_tfidf = hstack([X_train_word, X_train_char])
X_val_tfidf   = hstack([X_val_word, X_val_char])


print("TF-IDF shape (train):", X_train_tfidf.shape)
print("TF-IDF shape (val):  ", X_val_tfidf.shape)



def evaluate_model(model, X_val, y_val, label_cols):
    y_pred_proba = model.predict_proba(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    results = {}
    for i, col in enumerate(label_cols):
        roc = roc_auc_score(y_val[col], y_pred_proba[:, i])
        f1  = f1_score(y_val[col], y_pred[:, i])
        results[col] = {"ROC-AUC": roc, "F1": f1}
    return pd.DataFrame(results).T



log_reg = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs', n_jobs=-1)
)

log_reg.fit(X_train_tfidf, y_train)
log_results = evaluate_model(log_reg, X_val_tfidf, y_val, y_train.columns)

print("=== Logistic Regression Performance ===")
display(log_results)
print("Mean ROC-AUC:", log_results["ROC-AUC"].mean().round(4))



svm_clf = OneVsRestClassifier(
    CalibratedClassifierCV(LinearSVC(), method='sigmoid', cv=3)
)

svm_clf.fit(X_train_tfidf, y_train)
svm_results = evaluate_model(svm_clf, X_val_tfidf, y_val, y_train.columns)

print("=== Linear SVM Performance ===")
display(svm_results)
print("Mean ROC-AUC:", svm_results["ROC-AUC"].mean().round(4))



model_scores = pd.DataFrame({
    "Logistic Regression": log_results["ROC-AUC"],
    "Linear SVM": svm_results["ROC-AUC"],
})
model_scores.plot(kind='bar', figsize=(10,4), title="ROC-AUC per Label")
plt.ylabel("ROC-AUC Score")
plt.show()

print("Mean ROC-AUCs:")
print(model_scores.mean().round(4))



class TfidfLinearPipeline:
    def __init__(self, word_vectorizer, char_vectorizer, clf, label_cols):
        self.word_vectorizer = word_vectorizer
        self.char_vectorizer = char_vectorizer
        self.clf = clf
        self.label_cols = label_cols

    def _vectorize(self, texts):
        X_word = self.word_vectorizer.transform(texts)
        X_char = self.char_vectorizer.transform(texts)
        return hstack([X_word, X_char])

    def predict_proba(self, texts):
        X = self._vectorize(texts)
        return self.clf.predict_proba(X)

    def predict_labels(self, texts, thresholds=None):
        """
        thresholds: optional dict {label: threshold}
                    if None, defaults to 0.5 for all labels.
        returns DataFrame with binary predictions for each label
        """
        probas = self.predict_proba(texts)
        preds = []
        for i, label in enumerate(self.label_cols):
            thr = 0.5 if thresholds is None else thresholds.get(label, 0.5)
            preds.append((probas[:, i] >= thr).astype(int))
        preds = np.stack(preds, axis=1)
        return pd.DataFrame(preds, columns=self.label_cols)




tfidf_logreg_pipeline = TfidfLinearPipeline(
    word_vectorizer=word_vectorizer,
    char_vectorizer=char_vectorizer,
    clf=log_reg,
    label_cols=label_cols
)


joblib.dump(tfidf_logreg_pipeline, "tfidf_logreg_pipeline.pkl")
print("Saved tfidf_logreg_pipeline.pkl")


