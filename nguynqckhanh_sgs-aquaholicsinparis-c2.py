

# --- Import required modules ---
import os             # For directory and file path operations
import re             # For regular expressions (pattern matching)
import logging        # For logging events and information
import random         # For generating random numbers
import numpy as np    # For numerical operations and arrays
import pandas as pd   # For data manipulation and analysis
from datetime import datetime  # For handling dates and timestamps
    
# --- Seed control for reproducibility ---
SEED = 42              # Fixed seed value to make random operations reproducible
random.seed(SEED)      # Set seed for Python's built-in random module
np.random.seed(SEED)   # Set seed for NumPy's random number generator

# --- Logging setup ---
os.makedirs("logs", exist_ok=True)
log_path = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

print(f"Environment ready | SEED={SEED} | Logs saved to: {log_path}")


# --- Load datasets from input directory ---
train = pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")  # Load training data
test  = pd.read_csv("/kaggle/input/rmit-hackathon-2025/test.csv")   # Load test data

# --- Normalize column names ---
train.columns = train.columns.str.lower()   # Convert all training column names to lowercase
test.columns  = test.columns.str.lower()    # Convert all test column names to lowercase

# --- Display dataset information ---
print("Data loaded successfully.")           # Confirm that the data was loaded
print(f"Train shape: {train.shape} | Test shape: {test.shape}")  # Show dataset dimensions
print("\nLabel distribution:")               # Display class balance in the training set
print(train["label"].value_counts())

# --- Preview data ---
display(train.head())                        # Display the first few rows of the training dataset


# --- Import necessary libraries ---
import nltk, string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# --- Download required NLTK data packages ---
nltk.download("punkt", quiet=True)       # Tokenizer data
nltk.download("stopwords", quiet=True)   # Common English stop words

# --- Prepare preprocessing utilities ---
stemmer = PorterStemmer()                       # Initialize stemmer to reduce words to their root form
stop_words = set(stopwords.words("english"))    # Define list of stop words to remove
punct = set(string.punctuation)                 # Define punctuation characters to remove

# --- Define text cleaning function ---
def clean_advanced(text):
    if not isinstance(text, str): 
        return ""                               # Handle non-string inputs safely
    text = text.lower()                         # Convert text to lowercase
    text = re.sub(r"[^a-z\s]", " ", text)       # Keep only letters and spaces
    tokens = word_tokenize(text)                # Split text into individual words
    # Remove stop words and punctuation, then stem each token
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and t not in punct]
    return " ".join(tokens)                     # Rejoin tokens into a cleaned string

# --- Apply cleaning function to datasets ---
train["clean_text"] = train["text"].apply(clean_advanced)  # Clean training text
test["clean_text"]  = test["text"].apply(clean_advanced)   # Clean test text

# --- Confirmation message ---
print("Advanced text preprocessing complete.")             # Confirm successful preprocessing


def extract_features(df):
    f = pd.DataFrame()

    # --- Basic text metrics ---
    f["text_len"] = df["text"].str.len()                     # Total number of characters in text
    f["word_cnt"] = df["text"].str.split().str.len()         # Total number of words
    f["avg_word_len"] = f["text_len"] / f["word_cnt"].replace(0, 1)  # Average word length
    f["unique_ratio"] = df["text"].apply(                    # Ratio of unique words to total words
        lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0
    )
    f["special_cnt"] = df["text"].apply(                     # Count of special (non-alphanumeric) characters
        lambda x: len(re.findall(r"[^\w\s]", str(x)))
    )
    f["upper_ratio"] = df["text"].apply(                     # Ratio of uppercase letters to total characters
        lambda x: sum(c.isupper() for c in str(x)) / len(str(x)) if len(str(x)) > 0 else 0
    )

    # --- Keyword triggers often used in jailbreak prompts ---
    triggers = [
        "ignore", "bypass", "override", "jailbreak", "hack",
        "exploit", "filter", "restriction", "malicious"
    ]
    for kw in triggers:
        f[f"has_{kw}"] = df["text"].str.contains(kw, case=False, na=False).astype(int)
        # Add binary feature indicating presence (1) or absence (0) of each keyword

    return f

# --- Apply feature extraction to datasets ---
X_num = extract_features(train)       # Numeric features for training data
X_test_num = extract_features(test)   # Numeric features for test data

# --- Confirmation message ---
print(f"Numeric features created | Count = {X_num.shape[1]}")


from sklearn.feature_extraction.text import TfidfVectorizer

# --- Initialize TF-IDF vectorizer ---
tfidf = TfidfVectorizer(
    max_features=5000,         # Keep only the 5,000 most informative terms
    ngram_range=(1, 2),        # Include both unigrams (single words) and bigrams (two-word sequences)
    stop_words="english",      # Remove common English stop words
    min_df=2,                  # Ignore terms that appear in fewer than 2 documents
    max_df=0.8                 # Ignore terms that appear in more than 80% of documents
)

# --- Fit vectorizer on training data and transform both datasets ---
X_tfidf = tfidf.fit_transform(train["clean_text"])   # Learn vocabulary and weights from training text
X_test_tfidf = tfidf.transform(test["clean_text"])   # Apply the same transformation to test text

# --- Confirmation message ---
print("TF-IDF features ready.")
print("Train TF-IDF shape:", X_tfidf.shape, "| Test TF-IDF shape:", X_test_tfidf.shape)


from scipy.sparse import hstack, csr_matrix

# --- Combine text and numeric features into a single sparse matrix ---
X_train = hstack([X_tfidf, csr_matrix(X_num.values)])       # Combine training features
X_test  = hstack([X_test_tfidf, csr_matrix(X_test_num.values)])  # Combine test features

# --- Encode target labels ---
y_train = (train["label"] == "jailbreak").astype(int)        # Convert label to binary (1 = jailbreak, 0 = benign)

# --- Confirmation message ---
print("Combined feature matrix created.")
print("Train:", X_train.shape, "| Test:", X_test.shape)


from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --- Define candidate models with tuned hyperparameters ---
models = {
    "logreg": LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED
    ),
    "xgb": XGBClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric="logloss"
    ),
    "lgbm": LGBMClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1
    ),
}

# --- Stratified 5-fold cross-validation setup ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# --- Cross-validation performance evaluation ---
print("Performing 5-fold cross-validation...")
for name, model in models.items():
    feats = X_tfidf if name == "logreg" else X_train   # Logistic Regression uses TF-IDF only; others use combined features
    scores = cross_val_score(model, feats, y_train, cv=cv, scoring="roc_auc")
    print(f"{name:6} | Mean AUC = {scores.mean():.4f} Â± {scores.std()*2:.4f}")


from sklearn.metrics import roc_auc_score

# --- Choose best performer based on CV results (XGB) ---
best_model = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    eval_metric="logloss"
)

# --- Train model on entire training dataset ---
print("ğŸš€ Training final XGBoost model...")
best_model.fit(X_train, y_train)

# --- Predict probabilities on test data ---
probs = best_model.predict_proba(X_test)[:, 1]

# --- Prepare submission file ---
submission = pd.DataFrame({
    "Id": test["id"],
    "TARGET": np.clip(probs, 0, 1)  # Ensure valid probability range
})

# --- Save to CSV and preview ---
submission.to_csv("submission.csv", index=False)
print("  submission.csv saved successfully.")
print("TARGET probability range:", submission["TARGET"].min(), "â†’", submission["TARGET"].max())
display(submission.head())


# =========================================================
# BLOCK 9 â€” Model Blending / Stacking Ensemble
# Purpose: Combine multiple model outputs to capture complementary patterns
# =========================================================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# --- Re-train base models on full training set ---

# XGBoost
model_xgb = XGBClassifier(
    n_estimators=400, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, eval_metric="logloss"
)
model_xgb.fit(X_train, y_train)
pred_xgb_val = model_xgb.predict_proba(X_train)[:, 1]
pred_xgb_test = model_xgb.predict_proba(X_test)[:, 1]

# LightGBM
model_lgb = LGBMClassifier(
    n_estimators=400, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, verbose=-1
)
model_lgb.fit(X_train, y_train)
pred_lgb_val = model_lgb.predict_proba(X_train)[:, 1]
pred_lgb_test = model_lgb.predict_proba(X_test)[:, 1]

# Logistic Regression (TF-IDF only)
model_lr = LogisticRegression(
    C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED
)
model_lr.fit(X_tfidf, y_train)
pred_lr_val = model_lr.predict_proba(X_tfidf)[:, 1]
pred_lr_test = model_lr.predict_proba(X_test_tfidf)[:, 1]

print("Base model predictions prepared.")

# --- Create stacking input matrices ---
stack_train = np.vstack([pred_xgb_val, pred_lgb_val, pred_lr_val]).T
stack_test  = np.vstack([pred_xgb_test, pred_lgb_test, pred_lr_test]).T
print("Stacked feature matrix shapes:", stack_train.shape, stack_test.shape)

# --- Train meta-learner (second-level Logistic Regression) with cross-validation ---
meta_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
meta_oof = np.zeros(y_train.shape[0])
meta_test = np.zeros(stack_test.shape[0])

for fold, (trn_idx, val_idx) in enumerate(meta_cv.split(stack_train, y_train), 1):
    X_tr, X_val = stack_train[trn_idx], stack_train[val_idx]
    y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta.fit(X_tr, y_tr)
    meta_oof[val_idx] = meta.predict_proba(X_val)[:, 1]
    meta_test += meta.predict_proba(stack_test)[:, 1] / meta_cv.n_splits
    print(f"Fold {fold} AUC: {roc_auc_score(y_val, meta_oof[val_idx]):.4f}")

meta_auc = roc_auc_score(y_train, meta_oof)
print(f"\nMeta-model Out-of-Fold ROC-AUC: {meta_auc:.4f}")

# --- Final blended predictions ---
submission_blend = pd.DataFrame({
    "Id": test["id"],
    "TARGET": np.clip(meta_test, 0, 1)
})
submission_blend.to_csv("submission_blended.csv", index=False)
print("\nsubmission_blended.csv saved successfully.")
print("TARGET range:", submission_blend["TARGET"].min(), "â†’", submission_blend["TARGET"].max())
display(submission_blend.head())


import os

# --- Ensure only the final submission file remains ---
if os.path.exists("/kaggle/working/submission.csv"):
    os.remove("/kaggle/working/submission.csv")

os.rename("/kaggle/working/submission_blended.csv", "/kaggle/working/submission.csv")

print("Now only submission.csv remains.")
print(os.listdir("/kaggle/working"))

