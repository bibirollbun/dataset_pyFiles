# ------------------------- Imports -------------------------
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer


# ------------------------- File Paths -------------------------
train_path = '/kaggle/input/jigsaw-agile-community-rules/train.csv'
test_path = '/kaggle/input/jigsaw-agile-community-rules/test.csv'
sample_submission_path = '/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv'
submission_output_path = 'submission.csv'

sbert_local_path = '/kaggle/input/sbert_model/pytorch/default/1'  # <- Upload this folder to Kaggle


# ------------------------- Load Data -------------------------
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
subm_df = pd.read_csv(sample_submission_path)

assert 'body' in train_df.columns and 'rule_violation' in train_df.columns
assert 'body' in test_df.columns and 'row_id' in test_df.columns


# ------------------------- Feature Engineering -------------------------
def generate_features(df, trial=None, fit=True, tfidf_vectorizer=None, bert_model=None):
    corpus = df['body'].fillna('')

    word_max_features = trial.suggest_int('word_max_features', 1000, 5000) if trial else 2000
    ngram_range = trial.suggest_categorical('ngram_range', [(1, 1), (1, 2), (1, 3)]) if trial else (1, 2)

    if fit:
        tfidf_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=ngram_range,
            max_features=word_max_features,
            sublinear_tf=True,
            stop_words='english'
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    else:
        tfidf_matrix = tfidf_vectorizer.transform(corpus)

    if bert_model is None:
        bert_model = SentenceTransformer(sbert_local_path)
    bert_matrix = bert_model.encode(corpus.tolist(), show_progress_bar=fit)

    return hstack([tfidf_matrix, bert_matrix]), tfidf_vectorizer, bert_model


# ------------------------- Optuna Objective -------------------------
def objective(trial):
    X, _, _ = generate_features(train_df, trial)
    y = train_df['rule_violation']

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'verbosity': -1
    }

    clf = lgb.LGBMClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=3, scoring='roc_auc')
    return scores.mean()


# ------------------------- Run Optimization -------------------------
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("Best Parameters Found:", best_params)


# ------------------------- Train Final Models -------------------------
X_train, tfidf_vectorizer, bert_model = generate_features(train_df)
y_train = train_df['rule_violation']

lgb_model = lgb.LGBMClassifier(**best_params)
lgb_model.fit(X_train, y_train)

lr_model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=500))
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train.toarray(), y_train)


# ------------------------- Predict on Test Set -------------------------
X_test, _, _ = generate_features(test_df, fit=False, tfidf_vectorizer=tfidf_vectorizer, bert_model=bert_model)

lgb_preds = lgb_model.predict_proba(X_test)[:, 1]
lr_preds = lr_model.predict_proba(X_test)[:, 1]
rf_preds = rf_model.predict_proba(X_test.toarray())[:, 1]

ensemble_preds = (lgb_preds + lr_preds + rf_preds) / 3


# ------------------------- Save Submission -------------------------
subm_df['rule_violation'] = ensemble_preds
subm_df.to_csv(submission_output_path, index=False)
print(f"\n✅ submission.csv saved successfully → {submission_output_path}")





