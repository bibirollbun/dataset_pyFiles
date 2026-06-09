import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Load competition data (directly from /kaggle/input/)
train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
sample_sub = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")

print("✅ Data Loaded")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# 2. Prepare text and labels
X_train = train_df["body"].fillna("")
y_train = train_df["rule_violation"]
X_test = test_df["body"].fillna("")

# 3. Build pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=200, n_jobs=-1))
])

# 4. Train model
print("Training model...")
pipeline.fit(X_train, y_train)

# 5. Predict on test
test_preds = pipeline.predict(X_test)

# 6. Make submission
submission = sample_sub.copy()
submission["rule_violation"] = test_preds
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("✅ Submission file saved as submission.csv")

