import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

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



le = LabelEncoder()
train["Label"] = le.fit_transform(train["Misconception"])


X_train, X_val, y_train, y_val = train_test_split(
    train["text"], train["Label"],
    test_size=0.2, random_state=42, stratify=train["Label"]
)

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    analyzer='word',
    stop_words="english"
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)
X_test_tfidf  = tfidf.transform(test["text"])


def mapk(y_true, y_pred, k=3):
    preds = np.argsort(y_pred, axis=1)[:, -k:][:, ::-1]
    score = 0.0
    for i, true_label in enumerate(y_true):
        for rank, p in enumerate(preds[i]):
            if p == true_label:
                score += 1.0 / (rank + 1)
                break
    return score / len(y_true)

def xgb_map3(preds, dtrain):
    """Custom MAP@3 metric for XGBoost"""
    labels = dtrain.get_label().astype(int)
    num_class = len(np.unique(labels))
    preds = preds.reshape(-1, num_class)
    score = mapk(labels, preds, k=3)
    return "MAP@3", score  # ✅ only two values (metric name, metric value)





dtrain = xgb.DMatrix(X_train_tfidf, label=y_train)
dval   = xgb.DMatrix(X_val_tfidf, label=y_val)
dtest  = xgb.DMatrix(X_test_tfidf)

params = {
    "objective": "multi:softprob",
    "num_class": len(le.classes_),
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 2.0,
    "min_child_weight": 2,
    "gamma": 0.2,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "random_state": 42
}

evallist = [(dtrain, "train"), (dval, "eval")]

xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=600,
    evals=evallist,
    custom_metric=xgb_map3,   # ✅ correct replacement for feval
    early_stopping_rounds=30,
    verbose_eval=50
)

log_reg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=200,
    C=3.0,
    n_jobs=-1,
    random_state=42
)
log_reg.fit(X_train_tfidf, y_train)


xgb_val_pred = xgb_model.predict(dval)
log_val_pred = log_reg.predict_proba(X_val_tfidf)

# Ensemble: average of both model probabilities
ensemble_val_pred = (xgb_val_pred + log_val_pred) / 2
val_score = mapk(y_val, ensemble_val_pred, k=3)
print(f"✅ Ensemble Validation MAP@3: {val_score:.4f}")


#  TEST PREDICTIONS 
xgb_test_pred = xgb_model.predict(dtest)
log_test_pred = log_reg.predict_proba(X_test_tfidf)
ensemble_test_pred = (xgb_test_pred + log_test_pred) / 2

# Get top 3 predicted labels
test_top3_idx = np.argsort(ensemble_test_pred, axis=1)[:, -3:][:, ::-1]
test_top3_labels = le.inverse_transform(test_top3_idx.flatten()).reshape(test_top3_idx.shape)


# SUBMISSION
misconception_to_category = dict(zip(train["Misconception"], train["Category"]))
test_top3_catmis = np.vectorize(lambda x: f"{misconception_to_category[x]}:{x}")(test_top3_labels)
submission_col = [" ".join(row) for row in test_top3_catmis]

submission = pd.DataFrame({
    "row_id": test["row_id"],
    "Category:Misconception": submission_col
})

submission.to_csv("submission.csv", index=False)
submission.head(5)

