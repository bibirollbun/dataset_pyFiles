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


import zipfile

with zipfile.ZipFile("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip","r") as z:
    z.extractall(".")

with zipfile.ZipFile("/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip", "r") as z1:
    z1.extractall(".")

with zipfile.ZipFile("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip", "r") as zt:
    zt.extractall(".")
    
train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
train.shape


train.columns.values


from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return(" ".join(meaningful_words))   


clean_review = review_to_words(train["review"][0])
print(clean_review)


# Get the number of reviews based on the dataframe column size
num_reviews = train["review"].size
clean_train_reviews = []
for i in range(0, num_reviews):
    if((i+1)%1000 == 0):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))


print("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object - Bag of Words Tool
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()


print(train_data_features.shape)


vocab = vectorizer.get_feature_names_out()

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print(count, tag)


#Implement Word Cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

word_frequencies = dict(zip(vocab, dist))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100, max_features = "sqrt", max_depth = 6, max_leaf_nodes = 6) 
train_sentiment = train['sentiment'].tolist()

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
forest = forest.fit(train_data_features, train["sentiment"])


# Read the test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3)

# Verify that there are 25,000 rows and 2 columns
print(test.shape)

num_reviews = len(test["review"])
clean_test_reviews = [] 

for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result_bag = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output_bag = pd.DataFrame(data={"id":test["id"], "sentiment":result_bag})

# Use pandas to write the comma-separated output file
output_bag.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)


# Sum up the counts of each vocabulary word
dist_test = np.sum(test_data_features, axis=0)

word_frequencies_test = dict(zip(vocab, dist_test))
wordcloud_test = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies_test)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_test, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()


import collections
def classified_correct(model, i):
    return (0 if int(int(model["id"][i].split("_")[1][:-1]) <= 5) else 1)

correct = np.array([classified_correct(test,i) for i in range(test.shape[0])])
correct


def a_correct(model, i):
    return int(int(model["id"][i].split("_")[1][:-1]))

summary_true_sentiments = np.array([a_correct(test, i) for i in range(test.shape[0])])
vals = np.unique(summary_true_sentiments, return_counts = True)[1]
indices = np.unique(summary_true_sentiments, return_counts = True)[0]
print(vals)
plt.bar(indices, vals)
plt.xlabel("Review Rating")
plt.ylabel("Number of Reviews")
plt.title("Movie Review Rating Distribution")


from sklearn.metrics import precision_score, recall_score, f1_score
conf_matrix = confusion_matrix(correct, result_bag)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(correct, result_bag)
print("Classification Report:\n", class_report)

# Accuracy Score
accuracy = accuracy_score(correct, result_bag)
print("Accuracy:", accuracy)

# Prediction Score
precision = precision_score(correct, result_bag)
print("Precision:", precision)

# Recall Score
recall = recall_score(correct, result_bag)
print("Recall:", recall)

# F1 Score
f1_score = f1_score(correct, result_bag)
print("F1 Score:", f1_score)


# Read data from files 
train = pd.read_csv("labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3)

# Verify the number of reviews that were read (100,000 in total)
print("Read %d labeled train reviews, %d labeled test reviews, " \
 "and %d unlabeled reviews\n" % (train["review"].size,  
 test["review"].size, unlabeled_train["review"].size))


def review_to_wordlist(raw_review, remove_stopwords = False):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    if remove_stopwords:
        stops = set(stopwords.words("english"))                  
        # 
        # 5. Remove stop words
        words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return(words)


# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence, \
              remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


#Word Cloud for Unlabeled Reviews
num_reviews = unlabeled_train["review"].size
unlabeled_train_reviews = []
for i in range(0, num_reviews):
    if((i+1)%1000 == 0):
        print("Review %d of %d\n" % (i+1, num_reviews))
    unlabeled_train_reviews.append(review_to_words(unlabeled_train["review"][i]))

unlabeled_train_data_features = vectorizer.fit_transform(unlabeled_train_reviews)
unlabeled_train_data_features = unlabeled_train_data_features.toarray()

dist_unlabeled = np.sum(unlabeled_train_data_features, axis=0)
    
word_frequencies_unlabeled = dict(zip(vocab, dist_unlabeled))
wordcloud_unlabeled = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies_unlabeled)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_unlabeled, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.show()


for tag, count in zip(vocab, dist_unlabeled):
    print(count, tag)


sentences = []  # Initialize an empty list of sentences

print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)


print(len(sentences))


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

from gensim.models import word2vec
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            vector_size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)


model.wv.doesnt_match("man woman child kitchen".split())


model.wv.doesnt_match("france england germany berlin".split())


model.wv.doesnt_match("paris berlin london austria".split())


model.wv.most_similar("good")


model.wv.most_similar("mediocre")


model.wv.most_similar("worth")


model.wv.most_similar("queen")


from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context")
type(model.wv.vectors) #numpy.ndarray
model.wv.vectors.shape


model.wv["glass"]


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in model.wv:
            featureVec += model.wv[word]
            nwords += 1

    if nwords == 0:
        return featureVec  # return zeros if no word is in vocab
    return featureVec / nwords


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for i, review in enumerate(reviews):
        if i % 1000 == 0:
            print(f"Review {i} of {len(reviews)}")
        reviewFeatureVecs[i] = makeFeatureVec(review, model, num_features)

    return reviewFeatureVecs


clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)


# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)

print("Fitting a random forest to labeled training data...")
forest = forest.fit(trainDataVecs, train["sentiment"])

# Test & extract results 
result_vec = forest.predict(testDataVecs)

# Write the test results 
output_vec = pd.DataFrame(data={"id":test["id"], "sentiment":result_vec})
output_vec.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)


conf_matrix = confusion_matrix(correct, result_vec)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(correct, result_vec)
print("Classification Report:\n", class_report)

# Accuracy Score
accuracy = accuracy_score(correct, result_vec)
print("Accuracy:", accuracy)

# Prediction Score
precision = precision_score(correct, result_vec)
print("Precision:", precision)

# Recall Score
recall = recall_score(correct, result_vec)
print("Recall:", recall)


print(result_vec)


from sklearn.cluster import KMeans
import time

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.vectors
num_clusters = word_vectors.shape[0] / 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters = int(num_clusters))
idx = kmeans_clustering.fit_predict(word_vectors)

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")


word_centroid_map = dict(zip(model.wv.index_to_key, idx))


for cluster in range(0,10):
    print(f"\nCluster {cluster}")
    words = []
    for i in range(0,len(word_centroid_map.values())):
        if(list(word_centroid_map.values())[i] == cluster):
            words.append(list(word_centroid_map.keys())[i])
    print(words)


def create_bag_of_centroids(wordlist, word_centroid_map):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros((train["review"].size, int(num_clusters)), \
    dtype="float32")

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review, \
        word_centroid_map)
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros((test["review"].size, int(num_clusters)), \
    dtype="float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review, \
        word_centroid_map)
    counter += 1


test_centroids


# Fit a random forest and extract predictions 
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_centroids,train["sentiment"])
result_cent = forest.predict(test_centroids)

# Write the test results 
output_cent = pd.DataFrame(data={"id":test["id"], "sentiment":result_cent})
output_cent.to_csv("BagOfCentroids.csv", index=False, quoting=3)


conf_matrix = confusion_matrix(correct, result_cent)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
class_report = classification_report(correct, result_cent)
print("Classification Report:\n", class_report)

# Accuracy Score
accuracy = accuracy_score(correct, result_cent)
print("Accuracy:", accuracy)

# Prediction Score
precision = precision_score(correct, result_cent)
print("Precision:", precision)

# Recall Score
recall = recall_score(correct, result_cent)
print("Recall:", recall)

