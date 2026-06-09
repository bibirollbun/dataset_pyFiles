# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Project Specific Imports

import missingno as msno
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression


df_train = pd.read_csv("/kaggle/input/learn-ai-bbc/BBC News Train.csv")
df_test = pd.read_csv("/kaggle/input/learn-ai-bbc/BBC News Test.csv")


df_train.info()
df_test.info()


df_train.head()


df_test.head()


msno.matrix(df_train, figsize=(4, 2))
msno.matrix(df_test, figsize=(4, 2))



print(f"Number of duplicate rows identified in Training Data: {len(df_train[df_train.duplicated()])}")

print(f"Number of duplicate rows identified in Testing Data: {len(df_train[df_train.duplicated()])}")


# Count plots for categorical column
plt.figure(figsize=(8,4))
sns.countplot(x='Category', data=df_train, order=df_train['Category'].value_counts().index)
plt.title('Distribution of Categories')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]

df_train['tokens'] = df_train['Text'].apply(preprocess)


categories = df_train['Category'].unique()

for category in categories:
    tokens = df_train[df_train['Category'] == category]['tokens'].sum()
    word_freq = Counter(tokens)
    
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(4, 2))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Unigram Word Cloud – {category}', fontsize=16)
    plt.show()


from sklearn.feature_extraction.text import CountVectorizer

def get_trigram_frequencies(texts, stop_words=None, top_k=100):
    vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(3,3))
    X = vectorizer.fit_transform(texts)
    freqs = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])
    sorted_freqs = sorted(freqs, key=lambda x: x[1], reverse=True)
    return dict(sorted_freqs[:top_k])  # return top_k most frequent

# For each category, join the text and extract top trigrams
for category in categories:
    texts = df_train[df_train['Category'] == category]['Text']
    trigram_freqs = get_trigram_frequencies(texts, stop_words='english', top_k=100)

    if trigram_freqs:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(trigram_freqs)

        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Trigram Word Cloud – {category}', fontsize=16)
        plt.show()
    else:
        print(f"No trigrams found for category: {category}")



df_train['cleaned_text'] = df_train['Text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
df_test['cleaned_text'] = df_test['Text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Let's encode the categories into id number 
label_encoder = LabelEncoder()
df_train['category_id'] = label_encoder.fit_transform(df_train['Category'])


tfidf = TfidfVectorizer(stop_words='english', max_features=10000) #Capping the features to ensure execution
X_train_tfidf = tfidf.fit_transform(df_train['cleaned_text'])
X_test_tfidf = tfidf.transform(df_test['cleaned_text'])

print(f'Train TF-IDF Shape: {X_train_tfidf.shape}')
print(f'Test  TF-IDF Shape: {X_test_tfidf.shape}')


# Function to evaluate NMF with different combinations of topic_count and max_features_count
def evaluate_nmf_unsup(topic_count, max_features_count):
    # Vectorize text with the chosen number of features
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features_count)
    X_train_tfidf = tfidf.fit_transform(df_train['cleaned_text'])

    # Apply NMF with the chose number of topics
    nmf = NMF(n_components=topic_count, random_state=21)
    W_train = nmf.fit_transform(X_train_tfidf)  # Topic distribution for the train set

    # Here get the topic with the highest weight for each document
    predicted_topics_id = np.argmax(W_train, axis=1)

    # Assess the model performance with Adjusted Rand Index and Normalized Mutual Information 
    ari = adjusted_rand_score(df_train['category_id'], predicted_topics_id)
    nmi = normalized_mutual_info_score(df_train['category_id'], predicted_topics_id)
    return ari, nmi

# Apply different values for topic_count and max_features_count
results = []
for topic_count in [3, 5, 10]:
    for max_features_count in [1000, 5000, 10000, 15000, None]:
        ari, nmi = evaluate_nmf_unsup(topic_count, max_features_count)
        results.append((topic_count, max_features_count, ari, nmi))

# Display results in a table
results_df = pd.DataFrame(results, columns=['Topic Count', 'Max Features', 'ARI Score', 'NMI Score'])
print(results_df)


# Take the best parameters based on ARI Score
best_result = results_df.loc[results_df['ARI Score'].idxmax()]
best_topic_count = int(best_result['Topic Count'])
best_max_features_count = int(best_result['Max Features'])

print(f"Best Topic Count: {best_topic_count}")
print(f"Best Max Features Count: {best_max_features_count}")


# Apply NMF with the best topic count
tfidf_best = TfidfVectorizer(stop_words='english', max_features=best_max_features_count)
X_train_tfidf_best = tfidf_best.fit_transform(df_train['cleaned_text'])
X_test_tfidf_best = tfidf_best.transform(df_test['cleaned_text'])

# Apply NMF to the best configuration
nmf_best = NMF(n_components=best_topic_count, random_state=21)
W_train_best = nmf_best.fit_transform(X_train_tfidf_best)
W_test_best = nmf_best.transform(X_test_tfidf_best)

# For each document in the test set we predict the topic with the highest score
predicted_category_id = np.argmax(W_test_best, axis=1)

# We need to ensure the id are inline with training
topic_to_category = {}
for topic_id in range(best_topic_count):
    assigned_docs = df_train[np.argmax(W_train_best, axis=1) == topic_id]
    if not assigned_docs.empty:
        topic_to_category[topic_id] = mode(assigned_docs['category_id'], keepdims=True)[0][0]
    else:
        topic_to_category[topic_id] = -1

# Inverse transform ID to Label of the Category
mapped_category_id = [topic_to_category.get(tid, -1) for tid in predicted_category_id]
predicted_category = label_encoder.inverse_transform(mapped_category_id)

# Add predictions to the test dataframe
df_test['Category'] = predicted_category

# Show the results for the first few test entries
print(df_test[['ArticleId', 'Text', 'Category']].head())


df_submit = df_test[['ArticleId', 'Category']]
df_submit.to_csv("submission.csv",index=False)
#0.92653


clf = LogisticRegression(max_iter=1000, random_state=21)
clf.fit(X_train_tfidf, df_train['category_id'])

predicted_category_id_sup = clf.predict(X_test_tfidf)
predicted_categories_sup = label_encoder.inverse_transform(predicted_category_id_sup)

df_test_sup = df_test.copy()
df_test_sup['Category'] = predicted_categories_sup

df_submit_sup = df_test_sup[['ArticleId', 'Category']]
df_submit_sup.to_csv("submission_sup.csv",index=False)
#df_submit_sup.to_csv("submission.csv",index=False)
#0.98231

# Show first few entries
print(df_test[['ArticleId', 'Text', 'Category']].head())

