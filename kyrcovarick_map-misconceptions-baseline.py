
import os, sys, gc, math
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from scipy import sparse
np.random.seed(42)

# MAP@K metric (K=3 per competition)
def apk(actual, predicted, k=3):
    """Average precision at k for a single observation.
    `actual` is a single-element list [true_label] (competition has one truth per row).
    `predicted` is an ordered list of labels (top-k predictions).
    """
    if not actual:
        return 0.0
    if k < len(predicted):
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p == actual[0]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            break  # only one relevant label per row
    return score

def mapk(actual_list, predicted_list, k=3):
    return np.mean([apk([a], p, k) for a, p in zip(actual_list, predicted_list)])

def topk_from_proba(class_labels, proba_row, k=3):
    idx = np.argsort(-proba_row)[:k]
    return [class_labels[i] for i in idx]

print("Libraries loaded.")



# Detect Kaggle input path
KAGGLE_INPUT = '/kaggle/input/map-charting-student-math-misunderstandings'
LOCAL_INPUT  = '../input/map-charting-student-math-misunderstandings'
HERE = os.getcwd()

if os.path.exists(KAGGLE_INPUT):
    DATA_DIR = KAGGLE_INPUT
elif os.path.exists(LOCAL_INPUT):
    DATA_DIR = LOCAL_INPUT
else:
    # Fallback to working directory (for local testing with provided CSVs)
    DATA_DIR = '.'

print('DATA_DIR =', DATA_DIR)

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
print(train.shape, test.shape)
train.head(2)



# Build single target label "Category:Misconception"
train['Misconception'] = train['Misconception'].fillna('NA')
train['target'] = train['Category'].astype(str) + ':' + train['Misconception'].astype(str)

# Text features: concatenate question, MC answer, and explanation
def combine_text(df):
    return (
        df['QuestionText'].fillna('') + ' [MC] ' +
        df['MC_Answer'].fillna('') + ' [EXPL] ' +
        df['StudentExplanation'].fillna('')
    )

X_text = combine_text(train)
X_test_text = combine_text(test)

y = train['target'].values
groups = train['QuestionId'].values

print('Unique classes:', len(np.unique(y)))
print('Example label:', y[0])
print('Text example:', X_text.iloc[0][:300])



# A simple but strong baseline:
# - Word-level TF-IDF (1-2 grams)
# - SGDClassifier with log_loss to get predict_proba for many classes efficiently
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        ngram_range=(1, 2),
        max_features=250_000,
        min_df=2
    )),
    ('clf', SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=1e-5,
        max_iter=10_000,
        tol=1e-4,
        random_state=42
    ))
])

pipeline



N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

oof_true = []
oof_pred_topk = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_text, y, groups), 1):
    X_tr, X_va = X_text.iloc[tr_idx], X_text.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = pipeline
    model.fit(X_tr, y_tr)

    # Predict probabilities for validation
    proba = model.predict_proba(X_va)
    classes = model.named_steps['clf'].classes_
    pred_topk = [topk_from_proba(classes, row, k=3) for row in proba]

    oof_true.extend(list(y_va))
    oof_pred_topk.extend(pred_topk)

    fold_map3 = mapk(list(y_va), pred_topk, k=3)
    print(f"Fold {fold} MAP@3: {fold_map3:.5f}")
    gc.collect()

cv_map3 = mapk(oof_true, oof_pred_topk, k=3)
print(f"\nCV MAP@3 (mean over out-of-fold): {cv_map3:.5f}")



final_model = pipeline
final_model.fit(X_text, y)

test_proba = final_model.predict_proba(X_test_text)
classes = final_model.named_steps['clf'].classes_

top3 = [topk_from_proba(classes, row, k=3) for row in test_proba]

# Build submission
sub = pd.DataFrame({
    'row_id': test.index + 36696,  # Kaggle's sample_submission starts at this ID; we re-index safely
    'Category:Misconception': [' '.join(t) for t in top3]
})
sub.head()



# Use the provided sample_submission to ensure exact row_id ordering
sample_path = os.path.join(DATA_DIR, 'sample_submission.csv')
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    if 'row_id' in sample.columns:
        sub = sample[['row_id']].merge(sub, on='row_id', how='left')
        # If any rows didn't merge (shouldn't happen), fill with a safe default
        default_label = 'True_Correct:NA'
        sub['Category:Misconception'] = sub['Category:Misconception'].fillna(default_label)
        print('Aligned with sample_submission.')
else:
    print('sample_submission.csv not found; using generated row_id sequence.')

sub.head(3)



SUB_PATH = 'submission.csv'
sub.to_csv(SUB_PATH, index=False)
print('Saved:', os.path.abspath(SUB_PATH))
sub.head()

