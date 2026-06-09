# ======================================================
# BLOCK 1 â€” Import & Load Data
# ======================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Load dataset
train = pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")
test  = pd.read_csv("/kaggle/input/rmit-hackathon-2025/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Chuáº©n hoÃ¡ tÃªn cá»™t
train.columns = [c.strip().lower() for c in train.columns]
test.columns  = [c.strip().lower() for c in test.columns]


# ======================================================
# BLOCK 2 â€” Clean Text
# ======================================================
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

train["text_clean"] = train["text"].apply(clean_text)
test["text_clean"]  = test["text"].apply(clean_text)

print("âœ… Text cleaning done.")


# ======================================================
# BLOCK 3 â€” TF-IDF Vectorization
# ======================================================
vectorizer = TfidfVectorizer(
    max_features=25000,      # giá»¯ á»Ÿ má»©c vá»«a pháº£i Ä‘á»ƒ train nhanh
    ngram_range=(1, 2),      # unigram + bigram
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)

X = vectorizer.fit_transform(train["text_clean"])
X_test = vectorizer.transform(test["text_clean"])
y = train["label"].map({"benign": 0, "jailbreak": 1})

print("TF-IDF shape:", X.shape, X_test.shape)


# ======================================================
# BLOCK 4 â€” Define Models
# ======================================================
lr_model = LogisticRegression(
    C=8,
    solver="saga",
    max_iter=5000,
    n_jobs=-1,
    random_state=42
)

lgb_model = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)


# ======================================================
# BLOCK 5 â€” 5-Fold Cross Validation Ensemble
# ======================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lr = np.zeros(len(train))
oof_lgb = np.zeros(len(train))
preds_lr = np.zeros(len(test))
preds_lgb = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"\n===== Fold {fold+1} =====")
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    
    # Logistic Regression
    lr_model.fit(X_tr, y_tr)
    oof_lr[va_idx] = lr_model.predict_proba(X_va)[:, 1]
    preds_lr += lr_model.predict_proba(X_test)[:, 1] / skf.n_splits
    
    # LightGBM
    lgb_model.fit(X_tr, y_tr)
    oof_lgb[va_idx] = lgb_model.predict_proba(X_va)[:, 1]
    preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / skf.n_splits


# ======================================================
# BLOCK 6 â€” Ensemble + Evaluation
# ======================================================
# Trá»�ng sá»‘ giá»¯a 2 mÃ´ hÃ¬nh
w_lr, w_lgb = 0.6, 0.4

oof_final = w_lr * oof_lr + w_lgb * oof_lgb
preds_final = w_lr * preds_lr + w_lgb * preds_lgb

auc_lr = roc_auc_score(y, oof_lr)
auc_lgb = roc_auc_score(y, oof_lgb)
auc_final = roc_auc_score(y, oof_final)

print("\nâœ… AUC Logistic Regression:", auc_lr)
print("âœ… AUC LightGBM:", auc_lgb)
print("âœ… AUC Ensemble:", auc_final)


# ======================================================
# BLOCK 7 â€” Save Submission
# ======================================================
submission = pd.DataFrame({
    "Id": test["id"],
    "TARGET": preds_final
})
submission.to_csv("submission.csv", index=False)
print("\nğŸ“� Saved: submission.csv")

