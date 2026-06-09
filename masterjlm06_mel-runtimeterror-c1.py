import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ------------------ Load data ------------------
train = pd.read_csv('/kaggle/input/rmit-hackathon-2025/train.csv')
test  = pd.read_csv('/kaggle/input/rmit-hackathon-2025/test.csv')

# ------------------ Meta-feature extractor ------------------
def extract_meta(X):
    feats = []
    for text in X:
        text = str(text)
        feats.append([
            len(text),
            np.mean([len(w) for w in text.split()]) if text.split() else 0,
            sum(c.isupper() for c in text),
            sum(c.isdigit() for c in text),
            sum(not c.isalnum() and not c.isspace() for c in text),
            int(bool(re.search(r'\bignore\b', text.lower()))),
            int(bool(re.search(r'\binstruction\b', text.lower()))),
            int(bool(re.search(r'\bpolicy\b', text.lower()))),
            int(bool(re.search(r'\bjailbreak\b', text.lower()))),
        ])
    return np.array(feats, dtype=float)

meta_transformer = FunctionTransformer(extract_meta, validate=False)

# ------------------ Vectorizers ------------------
word_tfidf = TfidfVectorizer(
    max_features=25000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    stop_words='english'
)
char_tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 6),
    max_features=30000,
    sublinear_tf=True
)

# ------------------ Combine all features ------------------
features = FeatureUnion([
    ('word', word_tfidf),
    ('char', char_tfidf),
    ('meta', Pipeline([
        ('meta_feats', meta_transformer),
        ('scale', StandardScaler(with_mean=False))
    ]))
])

# ------------------ Base models ------------------
lr = LogisticRegression(
    solver='saga',
    penalty='l2',
    C=1.5,
    class_weight='balanced',
    max_iter=2000,
    n_jobs=-1,
    random_state=42
)

svc = CalibratedClassifierCV(
    LinearSVC(C=1.0, class_weight='balanced', max_iter=3000),
    method='sigmoid',
    cv=5
)

sgd = SGDClassifier(
    loss='modified_huber',      # smooth hinge, probabilistic
    alpha=1e-4,
    class_weight='balanced',
    random_state=42,
    max_iter=2000,
    tol=1e-3
)

# ------------------ Pipelines ------------------
pipeline_lr = Pipeline([('features', features), ('clf', lr)])
pipeline_svc = Pipeline([('features', features), ('clf', svc)])
pipeline_sgd = Pipeline([('features', features), ('clf', sgd)])

# ------------------ Cross-validation ------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in [('LogReg', pipeline_lr), ('SVC', pipeline_svc), ('SGD', pipeline_sgd)]:
    auc = cross_val_score(pipe, train['text'], train['label'],
                          cv=cv, scoring='roc_auc',
                          n_jobs=-1, error_score='raise').mean()
    print(f"{name} mean AUC: {auc:.4f}")

# ------------------ Fit all models ------------------
pipeline_lr.fit(train['text'], train['label'])
pipeline_svc.fit(train['text'], train['label'])
pipeline_sgd.fit(train['text'], train['label'])

# ------------------ Predict probabilities ------------------
pred_lr = pipeline_lr.predict_proba(test['text'])[:, 1]
pred_svc = pipeline_svc.predict_proba(test['text'])[:, 1]
pred_sgd = pipeline_sgd.decision_function(test['text'])
pred_sgd = (pred_sgd - pred_sgd.min()) / (pred_sgd.max() - pred_sgd.min())

# Weighted ensemble (tuned weights)
#preds = 0.5 * pred_lr + 0.3 * pred_svc + 0.2 * pred_sgd
preds = pred_svc

# ------------------ Save submission ------------------
submission = pd.DataFrame({'Id': test['Id'], 'target': preds})
submission.to_csv('submission.csv', index=False)
print("✅ High-AUC submission saved successfully!")


