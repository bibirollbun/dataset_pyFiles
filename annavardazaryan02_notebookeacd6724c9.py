import pandas as pd
from joblib import load

model = load("/kaggle/input/llm-detect-12/logistic_model.joblib")
vectorizer = load("/kaggle/input/llm-detect-12/logistic_vectorizer.joblib")

test_df = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")

X_test = test_df["text"].astype(str)
X_vec = vectorizer.transform(X_test)

preds = model.predict_proba(X_vec)[:, 1]

submission = pd.DataFrame({
    "id": test_df["id"],
    "generated": preds
})
submission.to_csv("submission.csv", index=False)

