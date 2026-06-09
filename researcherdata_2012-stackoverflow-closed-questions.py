# Fast prototype: ~ tiny sample + hashing + SGD (probabilities)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

# ---- 1) Load (adjust paths if needed) ----
train = pd.read_csv("/kaggle/input/predict-closed-questions-on-stack-overflow/train.csv")
test  = pd.read_csv("/kaggle/input/predict-closed-questions-on-stack-overflow/public_leaderboard.csv")

# ---- 2) Quick stratified sample (keeps all classes) ----
classes = ["not a real question","not constructive","off topic","open","too localized"]
per_class = 4000  # reduce if you need even faster (e.g., 2000)
parts = []
for c in classes:
    parts.append(train[train["OpenStatus"] == c].sample(
        n=min(per_class, (train["OpenStatus"] == c).sum()),
        random_state=42
    ))
train_small = pd.concat(parts, axis=0, ignore_index=True)

# ---- 3) Text fields ----
Xtr_text = (train_small["Title"].fillna("") + " " + train_small["BodyMarkdown"].fillna(""))
Xte_text = (test["Title"].fillna("") + " " + test["BodyMarkdown"].fillna(""))

# ---- 4) Very fast features: HashingVectorizer (no fit) ----
hv = HashingVectorizer(n_features=2**18, ngram_range=(1,2), alternate_sign=False)  # ~262k dims, fast
Xtr = hv.transform(Xtr_text)
Xte = hv.transform(Xte_text)

# ---- 5) Fast linear model with probabilities ----
clf = SGDClassifier(loss="log_loss", max_iter=5, tol=1e-3, n_jobs=-1, random_state=42)
clf.fit(Xtr, train_small["OpenStatus"])

# ---- 6) Probabilities (ensure correct class order) ----
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

if hasattr(clf, "predict_proba"):
    proba_all = clf.predict_proba(Xte)
else:
    # Fallback if sklearn build doesn’t expose predict_proba for SGD
    scores = clf.decision_function(Xte)
    if scores.ndim == 1:  # rare binary-shape case, expand to 2 classes then map
        scores = np.column_stack([-scores, scores])
    proba_all = softmax(scores)

# Map columns to the required order
# clf.classes_ may be a subset ordering; reindex to expected classes
proba_df = pd.DataFrame(proba_all, columns=clf.classes_)
for c in classes:
    if c not in proba_df.columns:
        proba_df[c] = 0.0  # if a class never appeared in the sample
sub = proba_df.reindex(columns=classes, fill_value=0)

# ---- 7) Optional id column ----
if "id" in test.columns:
    sub.insert(0, "id", test["id"].values)

# ---- 8) Save submission ----
sub.to_csv("submission.csv", index=False)
print("submission.csv written (fast prototype).")


