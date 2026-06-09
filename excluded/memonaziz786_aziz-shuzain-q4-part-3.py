import pandas as pd
import zipfile
import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def load_json_from_zip(zip_path, json_filename):
    with zipfile.ZipFile(zip_path,'r') as z:
        with z.open(json_filename,'r') as f:
            return json.load(f)

train_data = load_json_from_zip('/kaggle/input/whats-cooking/train.json.zip', "train.json")
test_data = load_json_from_zip('/kaggle/input/whats-cooking/test.json.zip', "test.json")

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_df["text"] = train_df["ingredients"].apply(lambda x: " ".join(x))
test_df["text"] = test_df["ingredients"].apply(lambda x: " ".join(x))

X_train = train_df["text"]
y_train = train_df["cuisine"]
X_test = test_df["text"]

vectorizer = TfidfVectorizer()
XTrainVec = vectorizer.fit_transform(X_train)
XTestVec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=3000)
model.fit(XTrainVec, y_train)


pred = model.predict(XTestVec)

submission = pd.DataFrame({
    "id": test_df["id"],
    "cuisine": pred
})
print("Predictions are saved to submission.csv")
submission.to_csv("submission.csv", index=False)

