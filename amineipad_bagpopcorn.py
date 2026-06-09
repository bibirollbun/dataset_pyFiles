import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- 1. Data Loading and Cleaning ---

# Load the datasets
print("Loading data...")
train = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3)

# Download NLTK's stopwords
try:
    stopwords.words("english")
except LookupError:
    nltk.download('stopwords')

# Function to clean and preprocess a single review
def review_to_wordlist(review, remove_stopwords=False):
    # 1. Remove HTML tags
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 2. Remove non-letters
    review_text = re.sub(r"[^a-zA-Z]", " ", review_text)
    # 3. Convert to lower case and split into individual words
    words = review_text.lower().split()
    # 4. (Optional) Remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words

# Function to process all reviews into a list of sentences for Word2Vec
def get_sentences(reviews):
    sentences = []
    for review in reviews["review"]:
        sentences.append(review_to_wordlist(review))
    return sentences

# --- 2. Training the Word2Vec Model ---

print("Training Word2Vec model...")
# Combine labeled and unlabeled reviews for more training data
all_sentences = get_sentences(train) + get_sentences(unlabeled_train)

# Set values for the Word2Vec model parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model
model = Word2Vec(all_sentences,
                 workers=num_workers,
                 vector_size=num_features,
                 min_count=min_word_count,
                 window=context,
                 sample=downsampling)

# It can be helpful to lock the model's vocabulary after training
model.init_sims(replace=True)


# --- 3. Creating Feature Vectors for Reviews ---

# Function to average all word vectors in a review
def make_feature_vec(words, model, num_features):
    feature_vec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    # Index2word is a list that contains the names of the words in the model's vocabulary.
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            feature_vec = np.add(feature_vec, model.wv[word])
    # Divide the result by the number of words to get the average
    if nwords > 0:
        feature_vec = np.divide(feature_vec, nwords)
    return feature_vec

# Function to create average feature vectors for all reviews
def get_avg_feature_vecs(reviews, model, num_features):
    counter = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews["review"]:
        if counter % 1000 == 0:
            print(f"Review {counter} of {len(reviews)}")
        review_feature_vecs[counter] = make_feature_vec(review_to_wordlist(review, remove_stopwords=True), model, num_features)
        counter = counter + 1
    return review_feature_vecs

print("Creating average feature vectors for training data...")
X_train = get_avg_feature_vecs(train, model, num_features)
y_train = train["sentiment"]

print("Creating average feature vectors for test data...")
X_test = get_avg_feature_vecs(test, model, num_features)


# --- 4. Classifier Training ---

print("Training Random Forest classifier...")
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
forest = forest.fit(X_train, y_train)


# --- 5. Prediction and Submission ---

print("Generating predictions...")
# Predict sentiment for the test set
result = forest.predict(X_test)

# Create the submission file
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("submission.csv", index=False, quoting=3)

print("Submission file created successfully!")
print("A preview of the submission file:")
print(output.head())

