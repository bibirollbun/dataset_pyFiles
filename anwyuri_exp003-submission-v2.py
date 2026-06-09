import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
print(f"Train: {train_df.shape}, Test: {test_df.shape}")


def create_combined_text(df):
    combined = []
    for idx, row in df.iterrows():
        text_parts = [
            f"Body: {row['body']}",
            f"Rule: {row['rule']}",
            f"Positive: {row['positive_example_1']} {row['positive_example_2']}",
            f"Negative: {row['negative_example_1']} {row['negative_example_2']}"
        ]
        combined.append(" ".join(text_parts))
    return combined

train_combined = create_combined_text(train_df)
test_combined = create_combined_text(test_df)


tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
X_train = tfidf.fit_transform(train_combined)
X_test = tfidf.transform(test_combined)
y_train = train_df['rule_violation'].values
print(f"Features: {X_train.shape}")


model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


model.fit(X_train, y_train)
train_preds = model.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, train_preds)
print(f"Training AUC: {train_auc:.4f}")


test_preds = model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': test_preds})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(f"Submission saved! Shape: {submission.shape}")
print(submission.head(10))

