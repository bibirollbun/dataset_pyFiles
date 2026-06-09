import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test  = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")



def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train["text"] = (
    train["QuestionText"].fillna("") + " " +
    train["MC_Answer"].fillna("") + " " +
    train["StudentExplanation"].fillna("")
).apply(clean_text)

test["text"] = (
    test["QuestionText"].fillna("") + " " +
    test["MC_Answer"].fillna("") + " " +
    test["StudentExplanation"].fillna("")
).apply(clean_text)

#  LABEL ENCODING 
le = LabelEncoder()
train["Label"] = le.fit_transform(train["Misconception"])


X_train, X_val, y_train, y_val = train_test_split(
    train["text"], train["Label"],
    test_size=0.2, random_state=42, stratify=train["Label"]
)


tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(test["text"])


xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    learning_rate=0.1,
    max_depth=6,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8
)

# Train
xgb.fit(X_train_tfidf, y_train)

# -----------VALIDATION PREDICTIONS -------

val_probs = xgb.predict_proba(X_val_tfidf)
top3_preds = np.argsort(val_probs, axis=1)[:, -3:][:, ::-1]

def mapk(y_true, y_pred, k=3):
    n = len(y_true)
    score = 0.0
    for i in range(n):
        for rank, p in enumerate(y_pred[i, :k]):
            if p == y_true[i]:
                score += 1.0 / (rank + 1)
                break
    return score / n

map3_score = mapk(y_val.to_numpy(), top3_preds, k=3)
print(f"✅ Validation MAP@3: {map3_score:.8f}")


test_probs = xgb.predict_proba(X_test_tfidf)

# Get top 3 predicted labels indices
test_top3_idx = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]

# Decode labels back to Misconception
test_top3_labels = le.inverse_transform(test_top3_idx.flatten()).reshape(test_top3_idx.shape)

# Map back to Category:Misconception
misconception_to_category = dict(zip(train["Misconception"], train["Category"]))
test_top3_catmis = np.vectorize(lambda x: f"{misconception_to_category[x]}:{x}")(test_top3_labels)

# Concatenate top 3 predictions space-delimited
submission_col = [" ".join(row) for row in test_top3_catmis]

# Create submission
submission = pd.DataFrame({
    "row_id": test["row_id"],
    "Category:Misconception": submission_col
})

submission.to_csv("submission.csv", index=False)
submission.head(5) 

