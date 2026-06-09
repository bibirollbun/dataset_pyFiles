pip install -q transformers sentence-transformers pandas numpy textdistance Levenshtein


import os
import urllib.request
import zipfile

# Create the directory if it doesn't exist
os.makedirs('/root/nltk_data/corpora', exist_ok=True)

# Download the wordnet.zip file
url = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip'
urllib.request.urlretrieve(url, '/root/nltk_data/corpora/wordnet.zip')

# Unzip the file
with zipfile.ZipFile('/root/nltk_data/corpora/wordnet.zip', 'r') as zip_ref:
    zip_ref.extractall('/root/nltk_data/corpora/')

# Remove the zip file (optional)
os.remove('/root/nltk_data/corpora/wordnet.zip')

# Verify the directory exists
if os.path.exists('/root/nltk_data/corpora/wordnet'):
    print("WordNet successfully downloaded and extracted!")
else:
    print("Failed to extract WordNet.")


import zipfile

zip_path = '/kaggle/input/quora-question-pairs/train.csv.zip'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall()
    for file in zip_ref.namelist():
        print(file)

print("ZIP file has been extracted successfully!")


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK data (only need stopwords and punkt)
nltk.download('punkt')
nltk.download('stopwords')

# Load spacy model for lemmatization
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# Initialize stopwords
stop_words = set(stopwords.words('english'))
# Preprocessing functions
def preprocess_for_tfidf(text):
    text = '' if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [token.lemma_ for token in nlp(text) if token.text not in stop_words]
    return ' '.join(tokens)

def preprocess_for_levenshtein(text):
    text = '' if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

def preprocess_for_sbert(text):
    text = '' if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    return text


df = pd.read_csv('train.csv')
df.head()


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK data (stopwords, punkt, and averaged_perceptron_tagger for POS tagging)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load spacy model for lemmatization
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing functions
def preprocess_for_tfidf(text):
    text = '' if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [token.lemma_ for token in nlp(text) if token.text not in stop_words]
    return ' '.join(tokens)

def preprocess_for_levenshtein(text):
    text = '' if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

def preprocess_for_word_overlap(text):
    text = '' if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = set(word_tokenize(text))
    tokens = {token for token in tokens if token not in stop_words}
    return tokens

def preprocess_for_pos_tags(text):
    text = '' if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    # Get POS tags for the tokens
    pos_tags = nltk.pos_tag(tokens)
    # Return the list of POS tags
    return [tag for word, tag in pos_tags]


# Replace NaN with empty string and ensure all entries are strings
df['question1'] = df['question1'].fillna('').astype(str)
df['question2'] = df['question2'].fillna('').astype(str)

# Apply preprocessing
df['q1_tfidf'] = df['question1'].apply(preprocess_for_tfidf)
df['q2_tfidf'] = df['question2'].apply(preprocess_for_tfidf)
df['q1_levenshtein'] = df['question1'].apply(preprocess_for_levenshtein)
df['q2_levenshtein'] = df['question2'].apply(preprocess_for_levenshtein)
df['q1_word_overlap'] = df['question1'].apply(preprocess_for_word_overlap)
df['q2_word_overlap'] = df['question2'].apply(preprocess_for_word_overlap)
df['q1_pos_tags'] = df['question1'].apply(preprocess_for_pos_tags)
df['q2_pos_tags'] = df['question2'].apply(preprocess_for_pos_tags)

# Feature extraction
# 1. TF-IDF Cosine Similarity
all_questions_tfidf = pd.concat([df['q1_tfidf'], df['q2_tfidf']])
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_questions_tfidf)
tfidf_q1 = tfidf_matrix[:len(df)]
tfidf_q2 = tfidf_matrix[len(df):]
tfidf_similarities = [cosine_similarity(tfidf_q1[i], tfidf_q2[i])[0][0] for i in range(len(df))]
df['tfidf_similarity'] = tfidf_similarities

# 2. Levenshtein Similarity
def levenshtein_similarity(q1, q2):
    dist = levenshtein_distance(q1, q2)
    max_len = max(len(q1), len(q2))
    return 1 - (dist / max_len) if max_len > 0 else 1

df['levenshtein_similarity'] = df.apply(
    lambda row: levenshtein_similarity(row['q1_levenshtein'], row['q2_levenshtein']), axis=1
)

# 3. Word Overlap Ratio (Syntax-based feature)
def word_overlap_ratio(set1, set2):
    intersection = len(set1.intersection(set2))
    total_unique = len(set1.union(set2))
    return intersection / total_unique if total_unique > 0 else 0

df['word_overlap_ratio'] = df.apply(
    lambda row: word_overlap_ratio(row['q1_word_overlap'], row['q2_word_overlap']), axis=1
)

# 4. POS Tag Similarity (Syntax-based feature)
def pos_tag_similarity(tags1, tags2):
    set1 = set(tags1)
    set2 = set(tags2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

df['pos_tag_similarity'] = df.apply(
    lambda row: pos_tag_similarity(row['q1_pos_tags'], row['q2_pos_tags']), axis=1
)

# Features and labels
features = df[['tfidf_similarity', 'levenshtein_similarity', 'word_overlap_ratio', 'pos_tag_similarity']]
labels = df['is_duplicate']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Train a simple Logistic Regression model
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("Feature Coefficients:")
for feature, coef in zip(features.columns, logreg_model.coef_[0]):
    print(f"{feature}: {coef}")


new_ds = X_test
new_ds['y_pred'] = y_pred
new_ds['y_test'] = y_test

