import os, json, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path


RANDOM_STATE = 42 # Sets a fixed random seed so results are reproducible
N_SPLITS = 5 # Number of folds for K-Fold Cross Validation
MAX_FEATURES = 100_000 # Maximum number of features (words or tokens) the TF-IDF vectorizer will keep
LR_C = 4.0 # A hyperparameter for Logistic Regression that controls regularization strength.


train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
sample_submission = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")


train.shape


test.shape


train.columns


def build_text(df):
    return (df["body"].astype(str) + " [RULE] " + df["rule"].astype(str) + 
        "[positive_1] " + df['positive_example_1'].astype(str) + 
        "[positive_2] " + df['positive_example_2'].astype(str) + 
        "[negative_1] " + df['negative_example_1'].astype(str) +
        "[negative_2] " + df['negative_example_2'].astype(str)).str.strip()

train["comment_text"] = build_text(train)
test["comment_text"] = build_text(test)


target = train["rule_violation"].astype(int).values


train[["comment_text", "rule_violation"]].head()


vectorizer = TfidfVectorizer(
    ngram_range = (1,2),
    max_features = MAX_FEATURES,
    min_df = 2,
    strip_accents = "unicode",
    lowercase = True,
    sublinear_tf = True

)


# fit_transform --> Learn vocabulary + transform
# fit looks at all the training text and learns the vocabulary & statistics it needs.
# transform → Converts each training text into its TF-IDF vector using that learned vocabulary.
X = vectorizer.fit_transform(train["comment_text"].values)



# transform --> Reuse learned vocabulary; no leakage
# Use the vocabulary learned from training to convert test comments into vectors.
X_test = vectorizer.transform(test["comment_text"].values)
feature_names = np.array(vectorizer.get_feature_names_out()) 


X.shape


X_test.shape


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)



# Pre-allocate an array for "out-of-fold" predictions (OOF) for every train row.
# We'll fill the validation predictions for each fold at the appropriate indices.
oof = np.zeros(len(train), dtype=float)


# Lists to collect per-fold AUCs and ROC curves for later reporting/plotting
fold_aucs, roc_curves = [], []


# Loop over each fold - skf.split(X,y) yields (train_indices, valid_indices) for that fold.

for fold, (tr_idx, va_idx) in enumerate(skf.split(X,target), 1):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = target[tr_idx], target[va_idx]

    # Define a simple logistic Regression classifier for sparse data

    clf = LogisticRegression(
        solver="liblinear",
        C=LR_C,
        max_iter=200,
        random_state = RANDOM_STATE
    )

    # Fit the model on this fold's training split
    clf.fit(X_tr, y_tr)

    # Get predicted probabilities for the positive class on the validation split
    va_prob = clf.predict_proba(X_va)[:, 1]

    # Store the validation predictions into the OOF array at the proper indices
    # After the loop, oof contains a prediction for every training row 
    oof[va_idx] = va_prob

    # Compute ROC-AUC for this fold's validation set 
    auc = roc_auc_score(y_va, va_prob)

    # Also compute the ROC curve points (FPR, TPR) 
    fpr, tpr, _ = roc_curve(y_va, va_prob)

    # Keep the fold’s AUC and ROC points for reporting/plots
    fold_aucs.append(float(auc))
    roc_curves.append({"fold": fold, "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc)})
    
    # Print a quick fold-level score for visibility during training
    print(f"[Fold {fold}] AUC = {auc:.4f}")


cv_mean, cv_std = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
oof_auc = float(roc_auc_score(target, oof))
print(f"\nCV mean AUC = {cv_mean:.3f} ± {cv_std:.3f}")
print(f"OOF AUC     = {oof_auc:.3f}")



clf_full = LogisticRegression(
    solver="liblinear",
    C=LR_C,
    max_iter=200,
    random_state=RANDOM_STATE
)
clf_full.fit(X, target)

test_prob = clf_full.predict_proba(X_test)[:, 1]
test_prob[:5], test_prob.min(), test_prob.max()



# Create submission DataFrame

submission = pd.DataFrame({
    "row_id": test["row_id"],        # use row_id from test dataset
    "rule_violation": test_prob      # predictions from model
})


submission.shape


submission.to_csv("submission.csv", index=False)

