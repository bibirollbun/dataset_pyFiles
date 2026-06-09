
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test  = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

# Combine important text fields
train["text"] = train["body"].fillna("") + " " + train["rule"].fillna("") + " " + train["subreddit"].fillna("")
test["text"]  = test["body"].fillna("")  + " " + test["rule"].fillna("")  + " " + test["subreddit"].fillna("")

# Target
y = train["rule_violation"]


tfidf = TfidfVectorizer(
    max_features=50000,   # reduce if memory still an issue (e.g., 20000)
    ngram_range=(1,2),
    stop_words="english"
)

X = tfidf.fit_transform(train["text"])
X_test = tfidf.transform(test["text"])

print("TF-IDF shapes:", X.shape, X_test.shape)


logreg = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    solver="saga",      # efficient for large sparse matrices
    verbose=1
)

logreg.fit(X, y)


pred = logreg.predict(X_test)

submission = pd.DataFrame({
    "row_id": test["row_id"],
    "rule_violation": pred
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ submission.csv ready!")
print(submission.head())


