# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score




file_path = "/kaggle/input/cure-bench/curebench_testset_phase1.jsonl"


df = pd.read_json(file_path, lines=True)


print(df.head())



df.shape


df.info()


print(df["options"].iloc[0])   



print(df.isnull().sum())


print(df["question_type"].value_counts())


# ========================
plt.figure(figsize=(6,4))
sns.countplot(y=df["question_type"])
plt.title("Distribution of Question Types")
plt.show()


df["question_length"] = df["question"].apply(lambda x: len(str(x)))
sns.histplot(df["question_length"], bins=30, kde=True)
plt.title("Distribution of Question Lengths")
plt.show()


df.groupby("question_type")["question_length"].mean().plot(kind="bar", figsize=(6,4))
plt.title("Avg Question Length by Type")
plt.show()


options_df = df["options"].apply(pd.Series)
df_expanded = pd.concat([df.drop(columns=["options"]), options_df], axis=1)
print(df_expanded.head())


df["num_options"] = df["options"].apply(lambda x: len(x) if isinstance(x, dict) else 0)
sns.countplot(x=df["num_options"])
plt.title("Number of Options per Question")
plt.show()


text = " ".join(df["question"].astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


le = LabelEncoder()
df["question_type_encoded"] = le.fit_transform(df["question_type"])
print(le.classes_)


X = df["question"]
y = df["question_type_encoded"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=200, random_state=42)


model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)


print("Accuracy:", accuracy_score(y_test, y_pred))


print(classification_report(y_test, y_pred, target_names=le.classes_))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
top_features = np.argsort(coefs)[-10:]
print("Top 10 important words:", feature_names[top_features])



df.to_csv("processed_curebench.csv", index=False)



import joblib
joblib.dump(model, "logistic_model.pkl")


joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


print("✅ EDA, Visualization, Preprocessing & Model Training Done Successfully!")

