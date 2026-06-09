import numpy as np 
import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

RANDOM_STATE = 42 # Sets a fixed random seed for reproducibility
N_FOLDS = 5 # Sets the number of cross-validation folds


train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test  = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")


print("Train shape:", train.shape, " Test shape:", test.shape)


train.isna().sum()


test.isna().sum()



for col in [
    "body", "rule", "subreddit",
    "positive_example_1", "positive_example_2",
    "negative_example_1", "negative_example_2"
]:
    if col in test.columns:
        test[col] = test[col].fillna("")


FEWSHOT_TEMPLATE = """r/{subreddit}
Rule: {rule}

Positive example 1:
{p1}
Label: Yes

Negative example 1:
{n1}
Label: No

Negative example 2:
{n2}
Label: No

Positive example 2:
{p2}
Label: Yes

Comment:
{body}
"""


def row_to_fewshot_text(row):
    return FEWSHOT_TEMPLATE.format(
        subreddit=row.get("subreddit",""),
        rule=row.get("rule",""),
        p1=row.get("positive_example_1",""),
        n1=row.get("negative_example_1",""),
        n2=row.get("negative_example_2",""),
        p2=row.get("positive_example_2",""),
        body=row.get("body",""),
    )


train_text = train.apply(row_to_fewshot_text, axis=1)
test_text  = test.apply(row_to_fewshot_text, axis=1)


y = train["rule_violation"].astype(int).values


word_vect = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=3,
    max_features=300_000,
    strip_accents="unicode",
    lowercase=True,
    sublinear_tf=True,
)

char_vect = TfidfVectorizer(
    analyzer="char_wb",           # within word boundaries
    ngram_range=(3,5),
    min_df=3,
    max_features=200_000,
    lowercase=True,
    sublinear_tf=True,
)

vectorizer = FeatureUnion([
    ("w", word_vect),
    ("c", char_vect)
])


pos = y.sum()
neg = len(y) - pos
scale_pos_weight = (neg / pos) if pos > 0 else 1.0
print(f"Class balance: pos={pos}, neg={neg}, scale_pos_weight={scale_pos_weight:.3f}")


def build_model():
    return XGBClassifier(
        n_estimators=1000,   # Number of trees (boosting rounds)
        learning_rate=0.05,  # Step size shrinkage after each boosting round
        max_depth=8,         # Maximum depth of each decision tre
        subsample=0.8,       # Fraction of training data sampled for each tree
        colsample_bytree=0.6, # Fraction of features sampled per tree
        reg_lambda=1.0,   # L2 regularization term on weights
        reg_alpha=0.0,    # L1 regularization term on weights
        min_child_weight=2.0,  # Minimum sum of instance weights (roughly number of samples) needed in a child node
        objective="binary:logistic",    # binary classification with probabilities as outputs
        eval_metric="auc",  # Evaluation metric used during training/validation
        random_state=RANDOM_STATE,
        n_jobs=-1, # Number of CPU threads to use.
    )


def build_pipeline():
    return Pipeline([
        ("tfidf", vectorizer),
        ("xgb",   build_model())
    ])


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


oof = np.zeros(len(train), dtype=float)


for fold, (tr_idx, va_idx) in enumerate(skf.split(train_text, y), 1):
    X_tr = train_text.iloc[tr_idx]
    y_tr = y[tr_idx]
    X_va = train_text.iloc[va_idx]
    y_va = y[va_idx]

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)

    oof[va_idx] = pipe.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, oof[va_idx])
    print(f"Fold {fold} AUC: {auc:.4f}")


final_pipe = build_pipeline()
final_pipe.fit(train_text, y)

test_pred = final_pipe.predict_proba(test_text)[:, 1]
print("Test predictions ready:", test_pred.shape)


submission = pd.DataFrame({
    "row_id": test["row_id"],
    "rule_violation": test_pred
})
submission.to_csv("submission.csv", index=False)
submission.head()

