import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
import re
import warnings
warnings.filterwarnings("ignore")


# Load datasets
train_df = pd.read_csv("/kaggle/input/commonlitreadabilityprize/train.csv")
test_df = pd.read_csv("/kaggle/input/commonlitreadabilityprize/test.csv")
sample_submission_df = pd.read_csv("/kaggle/input/commonlitreadabilityprize/sample_submission.csv")


# Dataset shapes
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Column names
print("Train columns:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())

# First few rows of the training set
train_df.head()



# Preview the first few rows of the test set
test_df.head()



# Check for missing values in test set
test_df.isnull().sum()



# --- Target Variable Distribution ---
plt.figure(figsize=(10,6))
sns.histplot(train_df['target'], kde=True, bins=30)
plt.title('Distribution of Target Variable', fontsize=16)
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.show()

# --- Text Length Features ---
# Calculate text statistics
train_df['char_count'] = train_df['excerpt'].apply(len)
train_df['word_count'] = train_df['excerpt'].apply(lambda x: len(x.split()))
train_df['avg_word_length'] = train_df['char_count'] / train_df['word_count']

# Character count distribution
plt.figure(figsize=(10,6))
sns.histplot(train_df['char_count'], bins=30, kde=True)
plt.title('Character Count Distribution', fontsize=16)
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.show()

# Word count distribution
plt.figure(figsize=(10,6))
sns.histplot(train_df['word_count'], bins=30, kde=True)
plt.title('Word Count Distribution', fontsize=16)
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

# Average word length distribution
plt.figure(figsize=(10,6))
sns.histplot(train_df['avg_word_length'], bins=30, kde=True)
plt.title('Average Word Length Distribution', fontsize=16)
plt.xlabel('Average Word Length')
plt.ylabel('Frequency')
plt.show()



# Character count vs target
plt.figure(figsize=(8,6))
sns.scatterplot(x='char_count', y='target', data=train_df, alpha=0.5)
plt.title('Character Count vs Target', fontsize=14)
plt.xlabel('Character Count')
plt.ylabel('Target')
plt.show()

# Word count vs target
plt.figure(figsize=(8,6))
sns.scatterplot(x='word_count', y='target', data=train_df, alpha=0.5)
plt.title('Word Count vs Target', fontsize=14)
plt.xlabel('Word Count')
plt.ylabel('Target')
plt.show()

# Average word length vs target
plt.figure(figsize=(8,6))
sns.scatterplot(x='avg_word_length', y='target', data=train_df, alpha=0.5)
plt.title('Average Word Length vs Target', fontsize=14)
plt.xlabel('Average Word Length')
plt.ylabel('Target')
plt.show()



# Combine all excerpts into a single string
all_text = " ".join(train_df['excerpt'].astype(str))

# Simple cleaning: convert to lowercase
all_text = all_text.lower()

# Generate WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Plot WordCloud
plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words - WordCloud', fontsize=16)
plt.show()

# Word frequency (basic count)
from collections import Counter

words = all_text.split()
word_freq = Counter(words)

# Top 20 most frequent words
most_common_words = word_freq.most_common(20)
print("ğŸ“Œ Top 20 Most Frequent Words:")
for word, freq in most_common_words:
    print(f"{word}: {freq}")



import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Filter words: remove stopwords and non-alphabetic tokens
words_filtered = [word for word in all_text.split() 
                  if word not in stop_words and word.isalpha()]

# Generate WordCloud without stopwords
wordcloud_filtered = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words_filtered))

plt.figure(figsize=(15,7))
plt.imshow(wordcloud_filtered, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud after Stopword Removal', fontsize=16)
plt.show()

# Word frequency after stopword removal
from collections import Counter
word_freq_filtered = Counter(words_filtered)
most_common_filtered = word_freq_filtered.most_common(20)

print("ğŸ“Œ Top 20 Most Frequent Words (after stopword removal):")
for word, freq in most_common_filtered:
    print(f"{word}: {freq}")



from sklearn.feature_extraction.text import CountVectorizer

# --- Bigram Analysis ---
bigram_vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english').fit(train_df['excerpt'])
bigram_freq = bigram_vectorizer.transform(train_df['excerpt']).sum(axis=0)
bigram_freq = [(word, bigram_freq[0, idx]) for word, idx in bigram_vectorizer.vocabulary_.items()]
bigram_freq = sorted(bigram_freq, key=lambda x: x[1], reverse=True)

print("ğŸ“Œ Top 20 most frequent bigrams:")
for bigram, freq in bigram_freq[:20]:
    print(f"{bigram}: {freq}")

# --- Trigram Analysis ---
trigram_vectorizer = CountVectorizer(ngram_range=(3,3), stop_words='english').fit(train_df['excerpt'])
trigram_freq = trigram_vectorizer.transform(train_df['excerpt']).sum(axis=0)
trigram_freq = [(word, trigram_freq[0, idx]) for word, idx in trigram_vectorizer.vocabulary_.items()]
trigram_freq = sorted(trigram_freq, key=lambda x: x[1], reverse=True)

print("\nğŸ“Œ Top 20 most frequent trigrams:")
for trigram, freq in trigram_freq[:20]:
    print(f"{trigram}: {freq}")



# --- Character, word, and sentence based features (train) ---
train_df['char_count'] = train_df['excerpt'].apply(len)
train_df['word_count'] = train_df['excerpt'].apply(lambda x: len(x.split()))
train_df['sentence_count'] = train_df['excerpt'].apply(lambda x: len(x.split('.')))
train_df['avg_word_length'] = train_df['char_count'] / train_df['word_count']

# --- Apply the same to test data ---
test_df['char_count'] = test_df['excerpt'].apply(len)
test_df['word_count'] = test_df['excerpt'].apply(lambda x: len(x.split()))
test_df['sentence_count'] = test_df['excerpt'].apply(lambda x: len(x.split('.')))
test_df['avg_word_length'] = test_df['char_count'] / test_df['word_count']

# Preview with new features
train_df.head()



# After reading train_df, test_df
INTERNET_OFF = True  # Kaggle scoring always has no internet
USE_TEXTSTAT = False if INTERNET_OFF else True  # disable textstat

# Safe readability features
def add_readability_feats(df):
    if USE_TEXTSTAT:
        try:
            import textstat
            df['flesch_reading_ease']  = df['excerpt'].apply(textstat.flesch_reading_ease)
            df['flesch_kincaid_grade'] = df['excerpt'].apply(textstat.flesch_kincaid_grade)
            df['smog_index']           = df['excerpt'].apply(textstat.smog_index)
            return df
        except Exception:
            pass
    # fallback without internet
    df['flesch_reading_ease']  = 0.0
    df['flesch_kincaid_grade'] = 0.0
    df['smog_index']           = 0.0
    return df

train_df = add_readability_feats(train_df)
test_df  = add_readability_feats(test_df)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# --- TF-IDF Vectorization ---
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000,       # top 5000 most frequent terms
    ngram_range=(1,2)        # unigrams + bigrams
)

X_tfidf_train = tfidf.fit_transform(train_df['excerpt'])
X_tfidf_test = tfidf.transform(test_df['excerpt'])

# --- Dimensionality Reduction with SVD ---
svd = TruncatedSVD(n_components=100, random_state=42)
X_tfidf_train_svd = svd.fit_transform(X_tfidf_train)
X_tfidf_test_svd = svd.transform(X_tfidf_test)

# Convert SVD features into DataFrame
svd_cols = [f'svd_{i}' for i in range(1, 101)]
train_svd_df = pd.DataFrame(X_tfidf_train_svd, columns=svd_cols)
test_svd_df = pd.DataFrame(X_tfidf_test_svd, columns=svd_cols)

# Reset indices and merge with original features
train_svd_df.reset_index(drop=True, inplace=True)
test_svd_df.reset_index(drop=True, inplace=True)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

train_fe_df = pd.concat([train_df, train_svd_df], axis=1)
test_fe_df = pd.concat([test_df, test_svd_df], axis=1)

# Check final shapes
print("âœ… Train shape after TF-IDF + SVD:", train_fe_df.shape)
print("âœ… Test shape after TF-IDF + SVD:", test_fe_df.shape)

# Preview with new features
train_fe_df.head()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

# Features and target
X = train_fe_df.drop(['id','url_legal','license','excerpt','target','standard_error'], axis=1)
y = train_fe_df['target']

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression model
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)

# Predictions and RMSE
y_pred = ridge.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"âœ… Ridge RMSE: {rmse:.4f}")



from lightgbm import LGBMRegressor

# LightGBM model
lgbm = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

# Fit model with validation set
lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse'
)

# Predictions and RMSE
y_pred_lgbm = lgbm.predict(X_val)
rmse_lgbm = np.sqrt(mean_squared_error(y_val, y_pred_lgbm))
print(f"âœ… LightGBM RMSE: {rmse_lgbm:.4f}")



# Predict on test set
y_test_pred = lgbm.predict(test_fe_df.drop(['id','url_legal','license','excerpt'], axis=1))

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_fe_df['id'],
    'target': y_test_pred
})

# Save as CSV
submission.to_csv('submission.csv', index=False)
print("âœ… submission.csv created!")
submission.head()

import os
print("Working dir:", os.listdir("/kaggle/working"))
assert "submission.csv" in os.listdir("/kaggle/working"), "submission.csv not found!"
print("âœ… submission.csv is ready for Kaggle submission")



import joblib

# Save model
joblib.dump(lgbm, 'model.pkl')

# Save used feature columns
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

print("âœ… model.pkl and model_columns.pkl saved!")





