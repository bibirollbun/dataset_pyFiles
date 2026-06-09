# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from wordcloud import WordCloud, STOPWORDS
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


# Load datasets
train = pd.read_csv("/kaggle/input/predict-movie-genres-from-plot-summaries/train.csv")
test = pd.read_csv("/kaggle/input/predict-movie-genres-from-plot-summaries/test.csv")
genres = pd.read_csv("/kaggle/input/predict-movie-genres-from-plot-summaries/movies_genres.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Genre classes:", len(genres))


train.head()


train.info()
train.describe(include='all')
train.isnull().sum()


train['genre_ids'] = train['genre_ids'].apply(lambda x: ast.literal_eval(x))
all_genres = [g for sublist in train['genre_ids'] for g in sublist]

genre_counts = pd.Series(all_genres).value_counts().reset_index()
genre_counts.columns = ['genre_id', 'count']
genre_counts = genre_counts.merge(genres, left_on='genre_id', right_on='id', how='left')

plt.figure(figsize=(12,6))
sns.barplot(x='count', y='name', data=genre_counts.sort_values('count', ascending=False))
plt.title("Most Common Genres in Training Data")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")
plt.show()


train['overview_length'] = train['overview'].astype(str).apply(len)
plt.figure(figsize=(8,5))
sns.histplot(train['overview_length'], bins=40, kde=True)
plt.title("Distribution of Overview Lengths")
plt.xlabel("Overview Length (characters)")
plt.show()


text = " ".join(train['overview'].dropna().tolist())
wc = WordCloud(width=1200, height=600, background_color='white', stopwords=STOPWORDS).generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Movie Overviews")
plt.show()


# Clean Overview Text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train["overview"] = train["overview"].apply(clean_text)
test["overview"] = test["overview"].apply(clean_text)


# Prepare Multi-Label Targets
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(train["genre_ids"])

print("Number of genres:", len(mlb.classes_))


X_train, X_val, y_train, y_val = train_test_split(
    train["overview"], Y, test_size=0.2, random_state=42
)


#  TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    stop_words="english"
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(test["overview"])


# Model: One-vs-Rest Logistic Regression
model = OneVsRestClassifier(LogisticRegression(max_iter=200))
model.fit(X_train_tfidf, y_train)


# Validation Performance
y_val_pred = model.predict(X_val_tfidf)
macro_f1 = f1_score(y_val, y_val_pred, average="macro")
print(f"Validation Macro F1 Score: {macro_f1:.4f}")


# Predict on Test Set
y_test_pred = model.predict(X_test_tfidf)

# Convert back to genre IDs
pred_labels = mlb.inverse_transform(y_test_pred)

submission = pd.DataFrame({
    "movie_id": test["movie_id"],
    "genre_ids": [" ".join(map(str, labels)) for labels in pred_labels]
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv file created!")




