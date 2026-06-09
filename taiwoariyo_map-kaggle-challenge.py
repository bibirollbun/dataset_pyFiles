# Import required Libraries
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, vstack
from IPython.display import display

RANDOM_SEED = 200
np.random.seed(RANDOM_SEED)

# Load the datasets
df_train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
df_test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

# Rename Columns
TEXT_COL = "StudentExplanation"
CAT_COL  = "Category"
MIS_COL  = "Misconception"
ID_COL   = "row_id"
Q_COL    = "QuestionText"
MC_COL   = "MC_Answer"

# Retrive first 5 rows
df_train.head(5)


# Target construction on training datasets
# Rebuild label using NA for missing misconception
df_train = df_train.copy()
df_train["label"] = (
    df_train["Category"].astype(str).str.strip()
    + ":" +
    df_train["Misconception"].fillna("NA").astype(str).str.strip()
)

assert not df_train["label"].str.endswith(":nan").any(), "Found ':nan' in labels"

# Rebuild _text/TF-IDF, resplit, retrain, and recreate submission.csv

print("Unique labels in TRAIN:", df_train["label"].nunique())
df_train[["StudentExplanation", "Category", "Misconception", "label"]].head(5)


# View the statistical summary and data info
df_train.info()
print(df_train.shape)

# Fail fast if any core columns are missing
assert all(c in df_train.columns for c in ["StudentExplanation","Category","Misconception"]), \
       "Missing core columns in df_train"

# Quick EDA preview
from IPython.display import display
display(df_train[["StudentExplanation", "Category", "Misconception", "label"]].head(5))
print("Unique labels:", df_train["label"].nunique())
display(df_train["label"].value_counts().head(10))




# View the statistical summary and data info
df_train.info()
print(df_train.shape)

# EDA preview
from IPython.display import display
display(df_train[["StudentExplanation", "Category", "Misconception", "label"]].head(5))
print("Unique labels:", df_train["label"].nunique())
display(df_train["label"].value_counts().head(5))

# Approx text length (words) to inform n-grams
lens = df_train[TEXT_COL].astype(str).str.split().str.len()
print("Text length (words) — mean/median/min/max:",
      round(lens.mean(), 2), int(lens.median()), int(lens.min()), int(lens.max()))



# Build enriched text (QuestionText + MC_Answer + StudentExplanation)
def build_text(df):
    return (
        df[Q_COL].fillna("")
        + " [MC] " + df[MC_COL].fillna("")
        + " [EXP] " + df[TEXT_COL].fillna("")
    ).astype(str)

df_train["_text"] = build_text(df_train)
df_test["_text"]  = build_text(df_test)

from IPython.display import display
display(df_train[["_text", "label"]].head(5))



# Trimmed settings
tfidf_word = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=3,
    max_features=200_000,
    sublinear_tf=True
)
tfidf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,4),
    min_df=3,
    max_features=80_000
)

# Fit on TRAIN; transform TRAIN and TEST
Xw_tr = tfidf_word.fit_transform(df_train["_text"])
Xc_tr = tfidf_char.fit_transform(df_train["_text"])
X_trn = hstack([Xw_tr, Xc_tr]).tocsr()

Xw_te = tfidf_word.transform(df_test["_text"])
Xc_te = tfidf_char.transform(df_test["_text"])
X_tst = hstack([Xw_te, Xc_te]).tocsr()

print("TF-IDF shapes → Train:", X_trn.shape, "| Test:", X_tst.shape)


# Handle labels that appear only once: keep them in TRAIN so stratified split is valid.
y_all = df_train["label"].values
vc = pd.Series(y_all).value_counts()
singletons = vc[vc == 1].index

mask_single = np.isin(y_all, singletons)
X_single, y_single = X_trn[mask_single], y_all[mask_single]
X_pool,   y_pool   = X_trn[~mask_single], y_all[~mask_single]

X_tr_core, X_va, y_tr_core, y_va = train_test_split(
    X_pool, y_pool, test_size=0.2, random_state=RANDOM_SEED, stratify=y_pool
)

X_tr = vstack([X_tr_core, X_single])
y_tr = np.concatenate([y_tr_core, y_single])

# Build label maps from TRAIN classes
classes = np.unique(y_tr)
lab2id  = {lab:i for i, lab in enumerate(classes)}
id2lab  = {i:lab for lab, i in lab2id.items()}

y_tr_enc = np.array([lab2id[v] for v in y_tr], dtype=np.int32)
y_va_enc = np.array([lab2id[v] for v in y_va], dtype=np.int32)

print("Split OK — Train:", X_tr.shape, "| Val:", X_va.shape, "| #classes:", len(classes))



# Tiny MAP@3 scorer (single truth per row)
def map_at_3(true_labels, proba_matrix, id_to_label):
    top3 = np.argsort(-proba_matrix, axis=1)[:, :3]
    labs = np.vectorize(id_to_label.get)(top3)
    s = 0.0
    for t, preds in zip(true_labels, labs):
        got = 0.0
        for r, p in enumerate(preds, 1):
            if p == t:
                got = 1.0 / r
                break
        s += got
    return s / len(true_labels)

# Sparse-friendly Logistic Regression
C_VAL = 38.0
CW    = None          # or "balanced"
clf = LogisticRegression(
    solver="liblinear", multi_class="ovr",
    C=C_VAL, class_weight=CW, max_iter=1000, tol=1e-3, dual=True
)

t0 = time.time()
clf.fit(X_tr, y_tr_enc)
va_proba = clf.predict_proba(X_va)
val_map3 = map_at_3(y_va, va_proba, id2lab)
print(f"Validation MAP@3: {val_map3:.4f}  (trained in {time.time()-t0:.1f}s)")



BEST_C  = 38.0
BEST_CW = None

clf_full = LogisticRegression(
    solver="liblinear", multi_class="ovr",
    C=BEST_C, class_weight=BEST_CW, max_iter=1000, tol=1e-3, dual=True
)
y_all = df_train["label"].values
clf_full.fit(X_trn, y_all)

# Predict top-3 for test
test_proba = clf_full.predict_proba(X_tst)
top3_idx   = np.argsort(-test_proba, axis=1)[:, :3]
top3_labs  = clf_full.classes_[top3_idx]
pred_strings = [" ".join(row) for row in top3_labs]

# Build submission in the exact required format
submission = pd.DataFrame({
    "row_id": df_test[ID_COL],
    "Category:Misconception": pred_strings
})
display(submission.head(10))


# Save Submission
OUT_PATH = "submission.csv"
submission.to_csv(OUT_PATH, index=False)
print(f"Saved {OUT_PATH}")

