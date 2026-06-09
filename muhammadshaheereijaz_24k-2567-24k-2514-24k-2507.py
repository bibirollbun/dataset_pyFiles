import os

for root, dirs, files in os.walk("/kaggle/input"):
    print(root)



import os

for file in os.listdir("/kaggle/input/whats-cooking"):
    print(file)



import pandas as pd
import json
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train_zip = "/kaggle/input/whats-cooking/train.json.zip"
test_zip = "/kaggle/input/whats-cooking/test.json.zip"

with zipfile.ZipFile(train_zip) as z:
    with z.open("train.json") as f:
        train_data = json.load(f)

with zipfile.ZipFile(test_zip) as z:
    with z.open("test.json") as f:
        test_data = json.load(f)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_df["ingredients_joined"] = train_df["ingredients"].apply(lambda x: " ".join(x))
test_df["ingredients_joined"] = test_df["ingredients"].apply(lambda x: " ".join(x))

tfidf = TfidfVectorizer(stop_words="english")
X_train = tfidf.fit_transform(train_df["ingredients_joined"])
X_test = tfidf.transform(test_df["ingredients_joined"])

model = LogisticRegression(max_iter=2000)
model.fit(X_train, train_df["cuisine"])

preds = model.predict(X_test)

submission = pd.DataFrame({
    "id": test_df["id"],
    "cuisine": preds
})

submission.to_csv("submission.csv", index=False)
submission.head()


