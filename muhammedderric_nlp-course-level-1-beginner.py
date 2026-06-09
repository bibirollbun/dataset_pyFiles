# Imports
import os
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# For reproducibility
random.seed(42)
np.random.seed(42)



import nltk
nltk.data.find('corpora/movie_reviews')

# Load movie_reviews from NLTK
from nltk.corpus import movie_reviews

# movie_reviews.fileids() gives file ids; categories() gives labels
fileids = movie_reviews.fileids()
len(fileids), movie_reviews.categories()[:2]  # show count and categories


# Build a pandas DataFrame: columns = text, label
documents = []
for fileid in movie_reviews.fileids():
    label = movie_reviews.categories(fileid)[0]  # 'pos' or 'neg'
    text = movie_reviews.raw(fileid)
    documents.append((text, label))

df = pd.DataFrame(documents, columns=['text', 'label'])
df.head()  


# Label distribution
label_counts = df['label'].value_counts()
print(label_counts)

# Plot label distribution
plt.figure(figsize=(6,4))
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.title('Distribution of labels (positive vs negative)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Add a review length column
df['length'] = df['text'].apply(len)
df['length'].describe()


# Show one positive and one negative review (shortened)
print('--- Positive example ---')
print(df[df['label']=='pos']['text'].iloc[0][:500], '...\n')
print('--- Negative example ---')
print(df[df['label']=='neg']['text'].iloc[0][:500], '...')


# Simple preprocessing helper
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, do_lemmatize=True, remove_stopwords=True):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove HTML tags (if any)
    text = re.sub(r'<.*?>', ' ', text)
    # 3. Remove non-letter characters (keep spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # 4. Tokenize into words
    tokens = word_tokenize(text)
    # 5. Optionally remove stopwords and very short tokens
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    else:
        tokens = [t for t in tokens if len(t) > 1]
    # 6. Optionally lemmatize tokens
    if do_lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # 7. Join back to string
    return ' '.join(tokens)

# Apply to a small sample to demonstrate
sample = df['text'].iloc[:3].apply(lambda x: preprocess_text(x)[:200])
sample


# Apply preprocessing to the full dataframe 
# We store the cleaned text in a new column 'clean_text'
df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, do_lemmatize=True, remove_stopwords=True))

# Show results
df[['text','clean_text']].head(2)


# Split into train and test
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print('Train size:', X_train.shape[0], 'Test size:', X_test.shape[0])


# Bag-of-Words (CountVectorizer) example
count_vec = CountVectorizer(max_features=5000)  # limit vocabulary for speed
X_train_counts = count_vec.fit_transform(X_train)
X_test_counts = count_vec.transform(X_test)

print('Count vector shape:', X_train_counts.shape)

# TF-IDF example
tfidf_vec = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vec.fit_transform(X_train)
X_test_tfidf = tfidf_vec.transform(X_test)

print('TF-IDF vector shape:', X_train_tfidf.shape)


# Train Multinomial Naive Bayes on counts
nb_count = MultinomialNB()
nb_count.fit(X_train_counts, y_train)

# Predict and evaluate
y_pred_nb_count = nb_count.predict(X_test_counts)
print('Naive Bayes (Counts) accuracy:', accuracy_score(y_test, y_pred_nb_count))


# Train Multinomial Naive Bayes on TF-IDF
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)

y_pred_nb_tfidf = nb_tfidf.predict(X_test_tfidf)
print('Naive Bayes (TF-IDF) accuracy:', accuracy_score(y_test, y_pred_nb_tfidf))


# Train Logistic Regression on TF-IDF
lr_tfidf = LogisticRegression(max_iter=1000)
lr_tfidf.fit(X_train_tfidf, y_train)

y_pred_lr_tfidf = lr_tfidf.predict(X_test_tfidf)
print('Logistic Regression (TF-IDF) accuracy:', accuracy_score(y_test, y_pred_lr_tfidf))


# Classification report for best model (Logistic Regression on TF-IDF)
print('Classification report (Logistic Regression, TF-IDF):\n')
print(classification_report(y_test, y_pred_lr_tfidf))


cm = confusion_matrix(y_test, y_pred_lr_tfidf, labels=['pos','neg'])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['pos','neg'], yticklabels=['pos','neg'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Logistic Regression, TF-IDF)')
plt.show()


# Create word clouds for positive and negative reviews using the cleaned text
pos_text = ' '.join(df[df['label']=='pos']['clean_text'])
neg_text = ' '.join(df[df['label']=='neg']['clean_text'])

# Generate word clouds
wordcloud_pos = WordCloud(width=600, height=400, background_color='white').generate(pos_text)
wordcloud_neg = WordCloud(width=600, height=400, background_color='white').generate(neg_text)

# Plot side by side
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud — Positive Reviews')

plt.subplot(1,2,2)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud — Negative Reviews')

plt.show()


# Get feature names and coefficients
feature_names = tfidf_vec.get_feature_names_out()
coefs = lr_tfidf.coef_[0]

# Top positive words
top_pos_idx = np.argsort(coefs)[-20:]
top_neg_idx = np.argsort(coefs)[:20]

print('Top words indicating positive sentiment:')
print(', '.join(feature_names[top_pos_idx][::-1]))

print('\nTop words indicating negative sentiment:')
print(', '.join(feature_names[top_neg_idx]))

