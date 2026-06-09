!ls /kaggle/input/jigsaw-agile-community-rules


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def simple_clean(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()

def main(train_path, test_path, out_path, text_col="body", label_col="rule_violation"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print("Train columns:", train.columns.tolist())
    print("Test columns:", test.columns.tolist())

    # clean text
    train[text_col] = train[text_col].apply(simple_clean)
    test[text_col] = test[text_col].apply(simple_clean)

    vect = TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=100000, min_df=3)
    all_text = pd.concat([train[text_col], test[text_col]], axis=0).astype(str)
    vect.fit(all_text)
    X_train = vect.transform(train[text_col])
    X_test = vect.transform(test[text_col])

    y = train[label_col].fillna(0).astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="saga", n_jobs=-1)
        clf.fit(X_tr, y_tr)
        oof[val_idx] = clf.predict_proba(X_val)[:,1]
        test_preds += clf.predict_proba(X_test)[:,1] / skf.n_splits
        print(f"Fold {fold+1} AUC:", roc_auc_score(y_val, oof[val_idx]))

    print("OOF AUC:", roc_auc_score(y, oof))

    final_clf = LogisticRegression(max_iter=2000, C=1.0, solver="saga", n_jobs=-1)
    final_clf.fit(X_train, y)
    final_preds = final_clf.predict_proba(X_test)[:,1]

    # save submission
    submission = pd.DataFrame({
        "row_id": test["row_id"],
        "rule_violation": final_preds
    })
    submission.to_csv(out_path, index=False)
    print("Saved to", out_path)

# ✅ run with Kaggle paths
main(
    "/kaggle/input/jigsaw-agile-community-rules/train.csv",
    "/kaggle/input/jigsaw-agile-community-rules/test.csv",
    "/kaggle/working/submission.csv",
    text_col="body",                 # <-- use body here
    label_col="rule_violation"       # label column is correct
)


