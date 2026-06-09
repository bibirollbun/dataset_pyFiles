import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix,f1_score
import matplotlib.pyplot as plt
import seaborn as sns


train_path = "/kaggle/input/comments-classification/Dataset/train.csv"   
test_path = "/kaggle/input/comments-classification/Dataset/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df.head())


X_train = train_df["comment_text"]
y_train = train_df["psychotic_depression"]
X_test = test_df["comment_text"]


y_train.value_counts()


model = make_pipeline(
    TfidfVectorizer(stop_words="english", max_features=10000),
    MultinomialNB(class_prior=[0.5, 0.5])
)


model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)



print("\nClassification Report (Train):\n")
print(f1_score(y_train, y_train_pred))


test_preds = model.predict(X_test)
ids = list(range(1, len(test_preds) + 1))
results = pd.DataFrame({
    "ID": ids,
    "psychotic_depression": test_preds
})


results = results.sort_values(by="ID")

results.to_csv("test_predictions.csv", index=False)
results.head()

