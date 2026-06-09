import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import gc
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb


train_path = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
test_path  = "/kaggle/input/jigsaw-agile-community-rules/test.csv"
train_data=pd.read_csv(train_path)
test_data=pd.read_csv(test_path)

TARGET = "rule_violation"
assert TARGET in train_data.columns, f"Target column '{TARGET}' not found!"

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)


text_cols = ["body", "rule", "subreddit",
             "positive_example_1", "positive_example_2",
             "negative_example_1", "negative_example_2"]
text_cols = [c for c in text_cols if c in train_data.columns]

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+|\#\w+", "", text)           # remove mentions and hashtags
    text = re.sub(r"[^a-z\s]", " ", text)            # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

for col in text_cols:
    train_data[col] = train_data[col].fillna("").apply(clean_text)
    test_data[col] = test_data[col].fillna("").apply(clean_text)


import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemm = WordNetLemmatizer()

def lemmatize_text(text):
    return " ".join([lemm.lemmatize(w) for w in text.split()])

for col in text_cols:
    train_data[col] = train_data[col].apply(lemmatize_text)
    test_data[col] = test_data[col].apply(lemmatize_text)



def add_length_features(df):
    df["text_all"] = df[text_cols].fillna("").apply(lambda r: " ".join(r.values.astype(str)), axis=1)
    df["len_chars"] = df["text_all"].str.len()
    df["len_words"] = df["text_all"].str.split().apply(len)
    df["num_exclaims"] = df["text_all"].str.count("!")
    df["num_questions"] = df["text_all"].str.count(r"\?")
    df["num_urls"] = df["text_all"].str.count("http|www")
    df["num_caps"] = df["text_all"].apply(lambda x: sum(1 for c in x if c.isupper()))
    df["chars_per_word"] = df["len_chars"] / (df["len_words"] + 1)
    df["unique_words"] = df["text_all"].apply(lambda x: len(set(x.split())))
    df["word_diversity"] = df["unique_words"] / (df["len_words"] + 1)
    return df

train_data = add_length_features(train_data)
test_data = add_length_features(test_data)


tfidf_word = TfidfVectorizer(
    ngram_range=(1, 3),          # unigrams, bigrams, trigrams
    stop_words="english",
    max_features=80000,
    min_df=3
)
X_tfidf_word = tfidf_word.fit_transform(train_data["text_all"])
X_test_tfidf_word = tfidf_word.transform(test_data["text_all"])

tfidf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=30000
)
X_tfidf_char = tfidf_char.fit_transform(train_data["text_all"])
X_test_tfidf_char = tfidf_char.transform(test_data["text_all"])


num_features = [
    "len_chars", "len_words", "num_exclaims", "num_questions",
    "num_urls", "num_caps", "chars_per_word", "unique_words", "word_diversity"
]
X_num = train_data[num_features].fillna(0).values
X_test_num = test_data[num_features].fillna(0).values

scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)
X_test_num = scaler.transform(X_test_num)

# Combine all
X_sparse = hstack([csr_matrix(X_tfidf_word), csr_matrix(X_tfidf_char), csr_matrix(X_num)]).tocsr()
X_test_sparse = hstack([csr_matrix(X_test_tfidf_word), csr_matrix(X_test_tfidf_char), csr_matrix(X_test_num)]).tocsr()
print("Combined features shape:", X_sparse.shape, X_test_sparse.shape)

del X_tfidf_word, X_tfidf_char, X_num, X_test_tfidf_word, X_test_tfidf_char, X_test_num
gc.collect()


y = train_data[TARGET].values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression
print("\nTraining Logistic Regression...")
lr_oof = np.zeros(X_sparse.shape[0])
lr_test_pred = np.zeros(X_test_sparse.shape[0])
lr_auc_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sparse, y), 1):
    X_tr, X_val = X_sparse[tr_idx], X_sparse[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    model = LogisticRegression(C=2.0, solver='saga', penalty="l2", max_iter=3000, n_jobs=-1)
    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    lr_oof[val_idx] = val_pred
    lr_test_pred += model.predict_proba(X_test_sparse)[:, 1] / skf.n_splits
    auc = roc_auc_score(y_val, val_pred)
    lr_auc_scores.append(auc)
    print(f"Fold {fold}: AUC={auc:.4f}")

print("LR CV mean AUC:", np.mean(lr_auc_scores))


# LightGBM
print("\nTraining LightGBM...")
lgb_oof = np.zeros(X_sparse.shape[0])
lgb_test_pred = np.zeros(X_test_sparse.shape[0])
lgb_auc_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sparse, y), 1):
    X_tr, X_val = X_sparse[tr_idx], X_sparse[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_val, y_val)
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42
    }
    clf = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)]
    )
    val_pred = clf.predict(X_val, num_iteration=clf.best_iteration)
    lgb_oof[val_idx] = val_pred
    lgb_test_pred += clf.predict(X_test_sparse, num_iteration=clf.best_iteration) / skf.n_splits
    auc = roc_auc_score(y_val, val_pred)
    lgb_auc_scores.append(auc)
    print(f"Fold {fold}: AUC={auc:.4f}")

print("LGB CV mean AUC:", np.mean(lgb_auc_scores))


lr_mean = np.mean(lr_auc_scores)
lgb_mean = np.mean(lgb_auc_scores)
total = lr_mean + lgb_mean
w_lr = lr_mean / total
w_lgb = lgb_mean / total
final_oof = w_lr * lr_oof + w_lgb * lgb_oof
final_test_pred = w_lr * lr_test_pred + w_lgb * lgb_test_pred

print("\nENSEMBLE CV AUC:", roc_auc_score(y, final_oof))

# ================== SAVE SUBMISSION ====================
id_col = "row_id" if "row_id" in test_data.columns else "id"
submission = pd.DataFrame({id_col: test_data.get(id_col, test_data.index), TARGET: final_test_pred})
submission.to_csv("submission.csv", index=False)
print("\n✅ Saved submission.csv")

