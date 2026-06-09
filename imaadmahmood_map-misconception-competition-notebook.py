from IPython.display import Image, display

img_path = "/kaggle/input/map-header/MAP Logo.png"

display(Image(filename=img_path))


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


# =========================================================================================
# SECTION 1: INSTALL & IMPORTS
# =========================================================================================
# Here, we'll import all the necessary libraries for our workflow.
# We need standard data handling tools (pandas, numpy), GPU-accelerated libraries from
# RAPIDS (cudf, cuml), and our new additions: 'transformers' for the language model
# and 'xgboost' for our classifier.

import os, gc, re, warnings, random
from pathlib import Path
import numpy as np
import pandas as pd
import cupy as cp
import cudf, cuml
from cuml.feature_extraction.text import TfidfVectorizer as cuTfidf
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
from sklearn.metrics import f1_score

# Import libraries for Transformers and XGBoost
import torch
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb

warnings.filterwarnings("ignore")

# Set a universal seed for reproducibility across all libraries
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
cp.random.seed(SEED)
torch.manual_seed(SEED)

print("RAPIDS Version:", cuml.__version__)


# =========================================================================================
# SECTION 2: LOAD DATA & INITIAL PREPROCESSING
# =========================================================================================

DATA_DIR = Path("/kaggle/input/map-charting-student-math-misunderstandings")
train = pd.read_csv(DATA_DIR/"train.csv")
test  = pd.read_csv(DATA_DIR/"test.csv")

# Standardize the 'Misconception' column by filling nulls
train["Misconception"].fillna("NA", inplace=True)
train["Misconception"] = train["Misconception"].astype(str)

# Create a combined target for easy evaluation later
train["target_cat"] = train["Category"] + ":" + train["Misconception"]
print("Train rows:", len(train), " Test rows:", len(test))

# =========================================================================================
# SECTION 3: LABEL ENCODING
# =========================================================================================
# Machine learning models require numerical inputs, so we map our text-based
# category and misconception labels to integer IDs. We also create reverse
# mappings (id2cat, id2mis) to convert predictions back to strings later.

# Category map
cat2id = {c:i for i,c in enumerate(train["Category"].value_counts().index)}
mis2id = {m:i for i,m in enumerate(train["Misconception"].value_counts().index)}
id2cat = {v:k for k,v in cat2id.items()}
id2mis = {v:k for k,v in mis2id.items()}
train["cat_id"] = train["Category"].map(cat2id)
train["mis_id"] = train["Misconception"].map(mis2id)

# =========================================================================================
# SECTION 4: TEXT CLEANING & SENTENCE CONSTRUCTION
# =========================================================================================
# We define a simple, fast cleaning function to normalize our text by removing
# punctuation, extra spaces, and newlines. Then, we apply this to create our
# unified 'sentence' column, which will be the input for all our NLP models.

clean_newlines = re.compile(r"\n+")
clean_spaces   = re.compile(r"\s+")
clean_punct    = re.compile(r"[^a-zA-Z0-9\s_]")

def fast_clean(text: str) -> str:
    text = clean_newlines.sub(" ", text)
    text = clean_spaces.sub(" ", text)
    text = clean_punct.sub("", text)
    return text.lower().strip()

for df in (train, test):
    df["sentence"] = (
        "Question: " + df["QuestionText"].astype(str) +
        " Answer: "  + df["MC_Answer"].astype(str) +
        " Explanation: " + df["StudentExplanation"].astype(str)
    ).apply(fast_clean)


# =========================================================================================
# SECTION 5: TF-IDF FEATURE EXTRACTION (CHAR + WORD)
# =========================================================================================

print("Fitting TF-IDF vectorisers …")
# Character-level TF-IDF
char_vec = cuTfidf(analyzer="char", ngram_range=(1,4), min_df=2, max_df=0.95,
                   dtype=np.float32)
char_vec.fit(cudf.Series(pd.concat([train.sentence, test.sentence])))
# Word-level TF-IDF
word_vec = cuTfidf(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.90,
                   stop_words="english", dtype=np.float32)
word_vec.fit(cudf.Series(pd.concat([train.sentence, test.sentence])))

# Transform the text data into TF-IDF features
train_char_tfidf = char_vec.transform(cudf.Series(train.sentence))
train_word_tfidf = word_vec.transform(cudf.Series(train.sentence))
test_char_tfidf  = char_vec.transform(cudf.Series(test.sentence))
test_word_tfidf  = word_vec.transform(cudf.Series(test.sentence))

# Stack the sparse matrices to combine them
train_tfidf_emb = sparse.hstack([train_char_tfidf.get(), train_word_tfidf.get()]).tocsr()
test_tfidf_emb  = sparse.hstack([test_char_tfidf.get(),  test_word_tfidf.get()]).tocsr()
print("TF-IDF feature dims:", train_tfidf_emb.shape)

# Free up GPU memory
del char_vec, word_vec, train_char_tfidf, train_word_tfidf, test_char_tfidf, test_word_tfidf
cp.get_default_memory_pool().free_all_blocks()
gc.collect()


# =========================================================================================
# SECTION 6: TRANSFORMER EMBEDDING EXTRACTION
# =========================================================================================

# Point to the local Kaggle dataset directory for the model
MODEL_NAME = '/kaggle/input/debertav3small'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device for Transformers: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
transformer_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

# This function processes text in batches to generate embeddings efficiently
def get_embeddings(texts, batch_size=32):
    all_embeddings = []
    transformer_model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = transformer_model(**inputs)
            # Mean pooling to get a single vector representation for each sentence
            attention_mask = inputs['attention_mask']
            last_hidden_state = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
        all_embeddings.append(mean_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

print("Generating transformer embeddings for train data...")
train_transformer_emb = get_embeddings(train['sentence'].tolist())
print("Generating transformer embeddings for test data...")
test_transformer_emb = get_embeddings(test['sentence'].tolist())
print("Transformer embedding dims:", train_transformer_emb.shape)


# =========================================================================================
# SECTION 7: COMBINE FEATURE SETS
# =========================================================================================
# Here we stack our two feature matrices to create the final input for our model.

train_emb = sparse.hstack([train_tfidf_emb, sparse.csr_matrix(train_transformer_emb)]).tocsr()
test_emb = sparse.hstack([test_tfidf_emb, sparse.csr_matrix(test_transformer_emb)]).tocsr()

print("Final combined feature dims:", train_emb.shape)

# =========================================================================================
# SECTION 8: CROSS-VALIDATED XGBOOST MODEL TRAINING
# =========================================================================================

N_FOLDS = 10
oof_cat = np.zeros((len(train), len(cat2id)))
pred_cat = np.zeros((len(test),  len(cat2id)))

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ---------- Train Category Model ----------
for fold,(tr_idx, val_idx) in enumerate(skf.split(train_emb, train.cat_id)):
    print(f"Category fold {fold+1}/{N_FOLDS}")
    # XGBoost classifier with GPU acceleration
    clf = xgb.XGBClassifier(objective='multi:softprob',
                            tree_method='gpu_hist',
                            n_estimators=1000,
                            learning_rate=0.05,
                            max_depth=4,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=SEED,
                            num_class=len(cat2id))
    
    eval_set = [(train_emb[val_idx], train.cat_id.iloc[val_idx])]
    clf.fit(train_emb[tr_idx], train.cat_id.iloc[tr_idx],
            eval_set=eval_set, eval_metric='mlogloss',
            early_stopping_rounds=50, verbose=False)
            
    oof_cat[val_idx] = clf.predict_proba(train_emb[val_idx])
    pred_cat += clf.predict_proba(test_emb) / N_FOLDS
    del clf; gc.collect(); torch.cuda.empty_cache()

print("Category F1 Score:", f1_score(train.cat_id, oof_cat.argmax(1), average="weighted"))

# ---------- Train Misconception Model ----------
oof_mis = np.zeros((len(train), len(mis2id)))
pred_mis = np.zeros((len(test),  len(mis2id)))

for fold,(tr_idx, val_idx) in enumerate(skf.split(train_emb, train.mis_id)):
    print(f"Misconception fold {fold+1}/{N_FOLDS}")
    clf = xgb.XGBClassifier(objective='multi:softprob',
                            tree_method='gpu_hist',
                            n_estimators=1000,
                            learning_rate=0.05,
                            max_depth=4,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=SEED,
                            num_class=len(mis2id))

    eval_set = [(train_emb[val_idx], train.mis_id.iloc[val_idx])]
    clf.fit(train_emb[tr_idx], train.mis_id.iloc[tr_idx],
            eval_set=eval_set, eval_metric='mlogloss',
            early_stopping_rounds=50, verbose=False)
            
    oof_mis[val_idx] = clf.predict_proba(train_emb[val_idx])
    pred_mis += clf.predict_proba(test_emb) / N_FOLDS
    del clf; gc.collect(); torch.cuda.empty_cache()

print("Misconception F1 Score:", f1_score(train.mis_id, oof_mis.argmax(1), average="weighted"))


# =========================================================================================
# SECTION 9: OOF EVALUATION & SUBMISSION FILE GENERATION
# =========================================================================================

na_index = mis2id["NA"]

def build_pred_matrix(cat_prob, mis_prob, topk=3):
    """Combines model outputs to generate final predictions."""
    mis_prob[:, na_index] = 0  # Never predict NA unless the category requires it
    cat_top = np.argsort(-cat_prob, axis=1)[:,:topk]
    mis_best = mis_prob.argmax(1)
    res = []
    for i in range(len(cat_prob)):
        row = []
        for j in range(topk):
            c = id2cat[cat_top[i,j]]
            # Intelligently pair misconception only when category implies one
            if "Misconception" in c:
                m = id2mis[mis_best[i]]
                row.append(f"{c}:{m}")
            else:
                row.append(f"{c}:NA")
        res.append(row)
    return res

def map3(truth, pred):
    """Calculates the MAP@3 competition metric."""
    score = 0.
    for t,p in zip(truth, pred):
        if t==p[0]: score+=1
        elif t==p[1]: score+=1/2
        elif t==p[2]: score+=1/3
    return score/len(truth)

# Calculate and print our local CV score
oof_pred_strings = build_pred_matrix(oof_cat, oof_mis)
print("Final OOF MAP@3 Score:", map3(train.target_cat.tolist(), oof_pred_strings))

# Generate the submission file
submission_strings = [" ".join(row) for row in build_pred_matrix(pred_cat, pred_mis)]
sub = pd.read_csv(DATA_DIR/"sample_submission.csv")
sub["Category:Misconception"] = submission_strings
sub.to_csv("submission.csv", index=False)

print("\nSubmission file created successfully!")
print(sub.head())

