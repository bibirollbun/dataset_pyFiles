import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score

class FoldAveragedLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, n_folds=5, max_iter=2000, random_state=None, solver='saga'):
        self.n_folds = n_folds
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        
    def fit(self, X, y):
        # Validate inputs
        #X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        # Initialize Kolder 
        kf = KFold(n_splits=self.n_folds, shuffle=True, 
                  random_state=self.random_state)
        
        # Store all models and coefficients
        self.models_ = []
        all_coefs = []
        all_intercepts = []
        
        for train_idx, _ in kf.split(X):
            # Train individual model
            lr = LogisticRegression(max_iter=self.max_iter,
                                  multi_class='multinomial',
                                  solver=self.solver,
                                  random_state=self.random_state)
            lr.fit(X[train_idx], y[train_idx])
            
            # Store model and parameters
            self.models_.append(lr)
            all_coefs.append(lr.coef_)
            all_intercepts.append(lr.intercept_)
        
        # Create averaged model
        self.avg_model_ = LogisticRegression(max_iter=1,  # No actual fitting
                                           multi_class='multinomial',
                                           solver=self.solver)
        
        # Set averaged parameters
        self.avg_model_.coef_ = np.mean(all_coefs, axis=0)
        self.avg_model_.intercept_ = np.mean(all_intercepts, axis=0)
        self.avg_model_.classes_ = self.classes_
        
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        #X = check_array(X)
        return self.avg_model_.predict(X)
    
    def predict_proba(self, X):
        check_is_fitted(self)
        #X = check_array(X)
        return self.avg_model_.predict_proba(X)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    
    def get_individual_scores(self, X, y):
        """Returns accuracy scores for each individual fold model"""
        return [m.score(X, y) for m in self.models_]
    
    def get_ensemble_predict(self, X, method='average'):
        """
        Alternative prediction using all models
        method: 'average' (avg probabilities) or 'vote' (majority vote)
        """
        if method == 'average':
            probas = np.mean([m.predict_proba(X) for m in self.models_], axis=0)
            return self.classes_[np.argmax(probas, axis=1)]
        else:  # voting
            preds = np.array([m.predict(X) for m in self.models_])
            return np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=preds)

# === Load dataset ===
train = pd.read_csv('/kaggle/input/inst-414-spring-2025/train.csv')
test = pd.read_csv('/kaggle/input/inst-414-spring-2025/test.csv')
X, y = train.review, train.label.values

# === Load pre-trained Word2Vec (change path if needed) ===
# GoogleNews-vectors-negative300.bin.gz is ~1.5GB and available from Google
w2v_model = KeyedVectors.load_word2vec_format('/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin.gz', binary=True)

# === Define sentiment axis ===
positive_word = 'good'
negative_word = 'bad'
sentiment_axis = w2v_model[positive_word] - w2v_model[negative_word]
sentiment_axis = sentiment_axis / np.linalg.norm(sentiment_axis)

# === Simple tokenizer ===
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# === Feature enrichment ===
def sentiment_features(X_texts, model, sentiment_axis):
    features = []
    for text in X_texts:
        tokens = simple_tokenize(text)
        projections = []
        positions = []

        for i, token in enumerate(tokens):
            if token in model:
                vec = model[token]
                sentiment_score = np.dot(vec, sentiment_axis) / np.linalg.norm(vec)
                projections.append(sentiment_score)
                positions.append(i)

        if not projections:
            features.append([0.0] * 10)
            continue

        projections = np.array(projections)
        positions = np.array(positions)
        norm_positions = positions / max(len(tokens) - 1, 1)
        position_weights = norm_positions / norm_positions.sum()
        weighted_sentiment = np.sum(projections * position_weights)
        last3_sentiment = np.mean(projections[-3:]) if len(projections) >= 3 else np.mean(projections)

        num_pos = np.sum(projections > 0.3)
        num_neg = np.sum(projections < -0.3)

        features.append([
            float(np.mean(projections)),
            float(np.sum(projections)),
            float(weighted_sentiment),
            float(last3_sentiment),
            len(projections),
            int(num_pos),
            int(num_neg),
            int(num_pos - num_neg),
            float(num_pos / len(projections)),
            int(np.any(projections > 0.7)),
        ])

    return np.array(features)
import numpy as np
import re

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def text_to_sentiment_matrix(text, model, sentiment_axis, max_sentences=25, max_words=25):
    sentences = re.split(r'[.!?]', text)
    matrix = np.zeros((max_sentences, max_words))

    for i, sentence in enumerate(sentences[:max_sentences]):
        tokens = simple_tokenize(sentence)
        for j, token in enumerate(tokens[:max_words]):
            if token in model:
                vec = model[token]
                polarity = np.dot(vec, sentiment_axis) / np.linalg.norm(vec)
                matrix[i, j] = polarity
            else:
                matrix[i, j] = 0.0
    return matrix.flatten()

# === TF-IDF vectorization ===
vectorizer = TfidfVectorizer(max_features=40000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)
print(X_tfidf.shape)
# === Combine with sentiment enrichment ===
sent_feat = sentiment_features(X, w2v_model, sentiment_axis)
scaler = StandardScaler()
sent_feat_scaled = scaler.fit_transform(sent_feat)
print(sent_feat_scaled.shape)
# Apply to whole dataset
matrices = np.array([
    text_to_sentiment_matrix(text, w2v_model, sentiment_axis)
    for text in X
])
print(matrices.shape)
# You now have a (num_samples, 625) numpy array!

# You can hstack it to your TF-IDF or train alone!

X_combined = hstack([X_tfidf,csr_matrix(sent_feat_scaled),csr_matrix(matrices)])#,csr_matrix(sent_feat_scaled)])

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42)

# === Train classifier (use your custom FoldAveragedLogisticRegression) ===
# from your_module import FoldAveragedLogisticRegression
falr = FoldAveragedLogisticRegression(n_folds=5, random_state=42)
falr.fit(X_train, y_train)

# === Evaluate ===
print(f"Average Model Test Accuracy: {falr.score(X_test, y_test):.1%}")
individual_scores = falr.get_individual_scores(X_test, y_test)
print(f"Individual Model Accuracies: {[f'{s:.1%}' for s in individual_scores]}")
print(f"Mean Individual Accuracy: {np.mean(individual_scores):.1%}")

ensemble_pred = falr.get_ensemble_predict(X_test, method='average')
print(f"Ensemble (avg probs) Accuracy: {accuracy_score(y_test, ensemble_pred):.1%}")
#  pure tfidf 88%
#  tfidf + image 84.5%
# tfidf + feature 85.5%
#  tfidf + feature + image 84%
# combine 


# Predict class labels
y_pred = falr.predict(X_train)

# Predict class probabilities
y_proba = falr.predict_proba(X_train)
import numpy as np

# Calculate confidence scores
confidence_scores = np.max(y_proba, axis=1)
print(np.sort(confidence_scores)[:20])
# Get indices of the 20 least confident predictions
bottom_20_indices = np.argsort(confidence_scores)[:20]
# Identify misclassified samples
misclassified = y_pred != y_train

# Check which of the bottom 20 are misclassified
bottom_20_misclassified = misclassified[bottom_20_indices]
for bi in bottom_20_indices:
    print(y_proba[bi],y_train[bi],y_pred[bi],train.iloc[bi].label,train.iloc[bi].review)



import matplotlib.pyplot as plt
import numpy as np
import re

def split_sentences(text):
    return re.split(r'[.!?]', text)

def count_sentences(text):
    sentences = split_sentences(text)
    return len([s for s in sentences if s.strip()])

def count_words(sentence):
    return len(re.findall(r'\b\w+\b', sentence))

# === Analyze sentences per text ===
sentence_counts = [count_sentences(text) for text in train.review]

plt.hist(sentence_counts, bins=30, color='skyblue')
plt.title('Distribution of Number of Sentences per Text')
plt.xlabel('Number of Sentences')
plt.ylabel('Number of Reviews')
plt.show()

print(f"Mean sentences per text: {np.mean(sentence_counts):.2f}")
print(f"Median sentences per text: {np.median(sentence_counts)}")

# === Analyze words per sentence ===
all_sentences = []
for text in train.review:
    all_sentences.extend([s for s in split_sentences(text) if s.strip()])

word_counts = [count_words(s) for s in all_sentences]

plt.hist(word_counts, bins=30, color='lightcoral')
plt.title('Distribution of Number of Words per Sentence')
plt.xlabel('Number of Words')
plt.ylabel('Number of Sentences')
plt.show()

print(f"Mean words per sentence: {np.mean(word_counts):.2f}")
print(f"Median words per sentence: {np.median(word_counts)}")



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

# === Simple Tokenizer and Sentiment Projector ===
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def text_polarity(text, model, sentiment_axis):
    tokens = simple_tokenize(text)
    polarities = []
    for token in tokens:
        if token in model:
            vec = model[token]
            polarity = np.dot(vec, sentiment_axis) / np.linalg.norm(vec)
            polarities.append(polarity)
    return np.mean(polarities) if polarities else 0

def sentence_polarities(text, model, sentiment_axis):
    sentences = re.split(r'[.!?]', text)
    sentence_polarities = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = simple_tokenize(sentence)
        polarities = []
        for token in tokens:
            if token in model:
                vec = model[token]
                polarity = np.dot(vec, sentiment_axis) / np.linalg.norm(vec)
                polarities.append(polarity)
        if polarities:
            sentence_polarities.append(np.mean(polarities))
    return sentence_polarities

# === Calculate text and sentence polarities ===
text_polarities_pos = []
text_polarities_neg = []
sentence_polarities_pos = []
sentence_polarities_neg = []

for text, label in zip(train.review, train.label):
    text_polar = text_polarity(text, w2v_model, sentiment_axis)
    sentence_polars = sentence_polarities(text, w2v_model, sentiment_axis)
    
    if label == 1:  # positive
        text_polarities_pos.append(text_polar)
        sentence_polarities_pos.extend(sentence_polars)
    else:  # negative
        text_polarities_neg.append(text_polar)
        sentence_polarities_neg.extend(sentence_polars)

# === Visualize ===

# 1. Text-level polarity distributions
plt.figure(figsize=(10,5))
plt.hist(text_polarities_pos, bins=50, alpha=0.6, label='Positive Reviews', color='green', density=True)
plt.hist(text_polarities_neg, bins=50, alpha=0.6, label='Negative Reviews', color='red', density=True)
plt.title('Distribution of Average Polarity per Text')
plt.xlabel('Average Polarity')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# 2. Sentence-level polarity distributions
plt.figure(figsize=(10,5))
plt.hist(sentence_polarities_pos, bins=50, alpha=0.6, label='Positive Sentences', color='blue', density=True)
plt.hist(sentence_polarities_neg, bins=50, alpha=0.6, label='Negative Sentences', color='orange', density=True)
plt.title('Distribution of Average Polarity per Sentence')
plt.xlabel('Average Polarity')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()



import matplotlib.pyplot as plt
import numpy as np
import re
from textblob import TextBlob

# Sample lists of positive and negative reviews
positive_reviews = train[train['label']==1].review[:5000]

negative_reviews = train[train['label']==0].review[:5000]


# Function to split text into sentences
def split_into_sentences(text):
    return re.split(r'[.!?]', text)

# Function to compute polarity of a sentence
def compute_polarity(sentence):
    return TextBlob(sentence).sentiment.polarity

# Function to collect normalized positions and corresponding polarities
def collect_positions_and_polarities(reviews):
    positions = []
    polarities = []
    for review in reviews:
        sentences = [s.strip() for s in split_into_sentences(review) if s.strip()]
        num_sentences = len(sentences)
        for idx, sentence in enumerate(sentences):
            norm_position = idx / (num_sentences - 1) if num_sentences > 1 else 0
            polarity = compute_polarity(sentence)
            positions.append(norm_position)
            polarities.append(polarity)
    return positions, polarities

# Collect data for positive and negative reviews
pos_positions, pos_polarities = collect_positions_and_polarities(positive_reviews)
neg_positions, neg_polarities = collect_positions_and_polarities(negative_reviews)

# Bin the positions
bins = np.linspace(0, 1, 20)
bin_centers = (bins[:-1] + bins[1:]) / 2

def compute_bin_stats(positions, polarities, bins):
    bin_means = []
    bin_stds = []
    for i in range(len(bins) - 1):
        bin_polarities = [pol for pos, pol in zip(positions, polarities) if bins[i] <= pos < bins[i+1]]
        if bin_polarities:
            bin_means.append(np.mean(bin_polarities))
            bin_stds.append(np.std(bin_polarities))
        else:
            bin_means.append(0)
            bin_stds.append(0)
    return np.array(bin_means), np.array(bin_stds)

# Compute statistics for positive and negative reviews
pos_means, pos_stds = compute_bin_stats(pos_positions, pos_polarities, bins)
neg_means, neg_stds = compute_bin_stats(neg_positions, neg_polarities, bins)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, pos_means, label='Positive Reviews', color='green')
plt.fill_between(bin_centers, pos_means - pos_stds, pos_means + pos_stds, color='green', alpha=0.2)

plt.plot(bin_centers, neg_means, label='Negative Reviews', color='red')
plt.fill_between(bin_centers, neg_means - neg_stds, neg_means + neg_stds, color='red', alpha=0.2)

plt.xlabel('Normalized Sentence Position')
plt.ylabel('Average Sentence Polarity')
plt.title('Sentence Polarity by Position in Reviews with Â±1 Std Dev')
plt.legend()
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt
import numpy as np
import re
from textblob import TextBlob

# Sample lists of positive and negative reviews
positive_reviews = train[train['label']==1].review[:5000]

negative_reviews = train[train['label']==0].review[:5000]

# Function to split text into sentences
def split_into_sentences(text):
    return re.split(r'[.!?]', text)

# Function to compute polarity of a sentence
def compute_polarity(sentence):
    return TextBlob(sentence).sentiment.polarity

# Function to compute cumulative polarity for each review
def compute_cumulative_polarities(reviews):
    cumulative_polarities = []
    for review in reviews:
        sentences = [s.strip() for s in split_into_sentences(review) if s.strip()]
        polarities = [compute_polarity(sentence) for sentence in sentences]
        cumulative = np.cumsum(polarities)
        normalized = cumulative / (np.arange(1, len(cumulative) + 1))
        cumulative_polarities.append(normalized)
    return cumulative_polarities

# Compute cumulative polarities
pos_cumulative = compute_cumulative_polarities(positive_reviews)
neg_cumulative = compute_cumulative_polarities(negative_reviews)

# Determine the maximum length among all reviews
max_length = max(max(len(p) for p in pos_cumulative), max(len(n) for n in neg_cumulative))

# Function to pad sequences to the same length
def pad_sequences(sequences, max_len):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = np.pad(seq, (0, max_len - len(seq)), 'edge')
        else:
            padded_seq = seq[:max_len]
        padded.append(padded_seq)
    return np.array(padded)

# Pad the cumulative polarities
pos_padded = pad_sequences(pos_cumulative, max_length)
neg_padded = pad_sequences(neg_cumulative, max_length)

# Compute mean and standard deviation
pos_mean = np.mean(pos_padded, axis=0)
pos_std = np.std(pos_padded, axis=0)
neg_mean = np.mean(neg_padded, axis=0)
neg_std = np.std(neg_padded, axis=0)

# Plotting
x = np.linspace(0, 1, max_length)
plt.figure(figsize=(10, 6))
plt.plot(x, pos_mean, label='Positive Reviews', color='green')
plt.fill_between(x, pos_mean - pos_std, pos_mean + pos_std, color='green', alpha=0.2)

plt.plot(x, neg_mean, label='Negative Reviews', color='red')
plt.fill_between(x, neg_mean - neg_std, neg_mean + neg_std, color='red', alpha=0.2)

plt.xlabel('Normalized Sentence Position')
plt.ylabel('Cumulative Average Sentiment Polarity')
plt.title('Cumulative Sentiment Polarity Across Reviews')
plt.legend()
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt
import numpy as np
import re
from textblob import TextBlob

# Sample lists of positive and negative reviews
positive_reviews = train[train['label']==1].review[:500]

negative_reviews = train[train['label']==0].review[:500]

# Function to split text into sentences
def split_into_sentences(text):
    return re.split(r'[.!?]', text)

# Function to compute polarity of a sentence
def compute_polarity(sentence):
    return TextBlob(sentence).sentiment.polarity

# Function to collect normalized positions and corresponding polarities
def collect_positions_and_polarities(reviews):
    positions = []
    polarities = []
    for review in reviews:
        sentences = [s.strip() for s in split_into_sentences(review) if s.strip()]
        num_sentences = len(sentences)
        for idx, sentence in enumerate(sentences):
            norm_position = idx / (num_sentences - 1) if num_sentences > 1 else 0
            polarity = compute_polarity(sentence)
            positions.append(norm_position)
            polarities.append(polarity)
    return positions, polarities

# Collect data for positive and negative reviews
pos_positions, pos_polarities = collect_positions_and_polarities(positive_reviews)
neg_positions, neg_polarities = collect_positions_and_polarities(negative_reviews)

# Bin the positions
bins = np.linspace(0, 1, 20)
bin_indices_pos = np.digitize(pos_positions, bins) - 1
bin_indices_neg = np.digitize(neg_positions, bins) - 1

# Compute average polarity for each bin
bin_means_pos = [
    np.mean([pol for i, pol in enumerate(pos_polarities) if bin_indices_pos[i] == j]) if any(bin_indices_pos == j) else 0
    for j in range(len(bins) - 1)
]
bin_means_neg = [
    np.mean([pol for i, pol in enumerate(neg_polarities) if bin_indices_neg[i] == j]) if any(bin_indices_neg == j) else 0
    for j in range(len(bins) - 1)
]

# Plotting
bin_centers = (bins[:-1] + bins[1:]) / 2
plt.plot(bin_centers, bin_means_pos, label='Positive Reviews', color='green')
plt.plot(bin_centers, bin_means_neg, label='Negative Reviews', color='red')
plt.xlabel('Normalized Sentence Position')
plt.ylabel('Average Sentence Polarity')
plt.title('Sentence Polarity by Position in Reviews')
plt.legend()
plt.grid(True)
plt.show()



import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Load your data (train.csv) ===
train = pd.read_csv('/kaggle/input/inst-414-spring-2025/train.csv')
X_texts, y_texts = train.review, train.label.values

# === Load embedding model + sentiment axis ===
# model = ... 
# sentiment_axis = ...

# === Simple tokenizer ===
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# === Sentence -> Matrix ===
def sentence_to_sentiment_matrix(sentence, model, sentiment_axis, max_words=25):
    tokens = simple_tokenize(sentence)
    matrix = np.zeros((1, max_words))
    for j, token in enumerate(tokens[:max_words]):
        if token in model:
            vec = model[token]
            polarity = np.dot(vec, sentiment_axis) / np.linalg.norm(vec)
            matrix[0, j] = polarity
    return matrix.flatten()

# === Text -> List of sentence matrices ===
def text_to_sentences_matrices(text, model, sentiment_axis, max_sentences=25, max_words=25):
    sentences = re.split(r'[.!?]', text)
    matrices = []
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue  # skip empty
        matrix = sentence_to_sentiment_matrix(sentence, model, sentiment_axis, max_words)
        matrices.append(matrix)
        if len(matrices) >= max_sentences:
            break
    return matrices

# === Build sentence dataset ===
sentence_X = []
sentence_y = []

for text, label in zip(X_texts, y_texts):
    sentence_matrices = text_to_sentences_matrices(text, w2v_model, sentiment_axis)
    for matrix in sentence_matrices:
        sentence_X.append(matrix)
        sentence_y.append(label)  # propagate text label to all sentences

sentence_X = np.array(sentence_X)
sentence_y = np.array(sentence_y)

print(f"Total sentences extracted: {len(sentence_X)}")

# === Train a sentence-level Logistic Regression ===
sentence_clf = LogisticRegression(max_iter=1000)
sentence_clf.fit(sentence_X, sentence_y)

# === Now per TEXT, predict sentences, then aggregate ===
def predict_text(text):
    matrices = text_to_sentences_matrices(text, w2v_model, sentiment_axis)
    if not matrices:
        return 0  # default to negative
    matrices = np.array(matrices)
    sentence_preds = sentence_clf.predict_proba(matrices)[:, 1]  # prob of positive
    aggregated_score = np.mean(sentence_preds)  # simple mean aggregation
    return 1 if aggregated_score >= 0.5 else 0

# === Build text-level predictions ===
X_preds = np.array([predict_text(text) for text in X_texts])

# === Evaluate ===
print("Overall accuracy:", accuracy_score(y_texts, X_preds))



import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

# === Custom Classifier ===
class FoldAveragedLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, n_folds=5, max_iter=2000, random_state=None, solver='saga'):
        self.n_folds = n_folds
        self.max_iter = max_iter
        self.random_state = random_state
        self.solver = solver
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.models_ = []
        all_coefs = []
        all_intercepts = []
        
        for train_idx, _ in kf.split(X):
            lr = LogisticRegression(max_iter=self.max_iter, multi_class='multinomial', solver=self.solver, random_state=self.random_state)
            lr.fit(X[train_idx], y[train_idx])
            self.models_.append(lr)
            all_coefs.append(lr.coef_)
            all_intercepts.append(lr.intercept_)
        
        self.avg_model_ = LogisticRegression(max_iter=1, multi_class='multinomial', solver=self.solver)
        self.avg_model_.coef_ = np.mean(all_coefs, axis=0)
        self.avg_model_.intercept_ = np.mean(all_intercepts, axis=0)
        self.avg_model_.classes_ = self.classes_
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        return self.avg_model_.predict(X)
    
    def predict_proba(self, X):
        check_is_fitted(self)
        return self.avg_model_.predict_proba(X)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    
    def get_individual_scores(self, X, y):
        return [m.score(X, y) for m in self.models_]
    
    def get_ensemble_predict(self, X, method='average'):
        if method == 'average':
            probas = np.mean([m.predict_proba(X) for m in self.models_], axis=0)
            return self.classes_[np.argmax(probas, axis=1)]
        else:
            preds = np.array([m.predict(X) for m in self.models_])
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)

# === Load Dataset ===
train = pd.read_csv('/kaggle/input/inst-414-spring-2025/train.csv')
test = pd.read_csv('/kaggle/input/inst-414-spring-2025/test.csv')
X, y = train.review, train.label.values

# === Load Word2Vec model ===
w2v_model = KeyedVectors.load_word2vec_format('/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin.gz', binary=True)

# === Define Sentiment Axis ===
positive_word = 'good'
negative_word = 'bad'
sentiment_axis = w2v_model[positive_word] - w2v_model[negative_word]
sentiment_axis = sentiment_axis / np.linalg.norm(sentiment_axis)

# === Utility Functions ===
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def sentiment_features(X_texts, model, sentiment_axis):
    features = []
    for text in X_texts:
        tokens = simple_tokenize(text)
        projections = []
        positions = []

        for i, token in enumerate(tokens):
            if token in model:
                vec = model[token]
                sentiment_score = np.dot(vec, sentiment_axis) / np.linalg.norm(vec)
                projections.append(sentiment_score)
                positions.append(i)

        if not projections:
            features.append([0.0] * 10)
            continue

        projections = np.array(projections)
        positions = np.array(positions)
        norm_positions = positions / max(len(tokens) - 1, 1)
        position_weights = norm_positions / norm_positions.sum()
        weighted_sentiment = np.sum(projections * position_weights)
        last3_sentiment = np.mean(projections[-3:]) if len(projections) >= 3 else np.mean(projections)

        num_pos = np.sum(projections > 0.3)
        num_neg = np.sum(projections < -0.3)

        features.append([
            np.mean(projections),
            np.sum(projections),
            weighted_sentiment,
            last3_sentiment,
            len(projections),
            num_pos,
            num_neg,
            num_pos - num_neg,
            num_pos / len(projections),
            int(np.any(projections > 0.7)),
        ])
    return np.array(features)

def text_to_sentiment_matrix(text, model, sentiment_axis, max_sentences=3, max_words=25):
    sentences = re.split(r'[.!?]', text)
    matrix = np.zeros((max_sentences, max_words))
    for i, sentence in enumerate(sentences[:max_sentences]):
        tokens = simple_tokenize(sentence)
        for j, token in enumerate(tokens[:max_words]):
            if token in model:
                vec = model[token]
                polarity = np.dot(vec, sentiment_axis) / np.linalg.norm(vec)
                matrix[i, j] = polarity
            else:
                matrix[i, j] = 0.0
    return matrix.flatten()

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(min_df=2, lowercase=False,stop_words='english')
X_tfidf = vectorizer.fit_transform(X)
print(f"TF-IDF shape: {X_tfidf.shape}")

# === Sentiment Features Extraction ===
sent_feat = sentiment_features(X, w2v_model, sentiment_axis)
scaler = StandardScaler()
sent_feat_scaled = scaler.fit_transform(sent_feat)
print(f"Sentiment feature shape: {sent_feat_scaled.shape}")

# === Sentiment Matrix Extraction ===
matrices = np.array([text_to_sentiment_matrix(text, w2v_model, sentiment_axis) for text in X])
print(f"Sentiment matrix shape: {matrices.shape}")

# === Combine all Features ===
X_combined = hstack([
    X_tfidf, 
    csr_matrix(sent_feat_scaled),
    csr_matrix(matrices)
])

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

# === Train Classifier ===
falr = FoldAveragedLogisticRegression(n_folds=5, random_state=42)
falr.fit(X_train, y_train)

# === Evaluate ===
print(f"Average Model Test Accuracy: {falr.score(X_test, y_test):.1%}")
individual_scores = falr.get_individual_scores(X_test, y_test)
print(f"Individual Model Accuracies: {[f'{s:.1%}' for s in individual_scores]}")
print(f"Mean Individual Accuracy: {np.mean(individual_scores):.1%}")

ensemble_pred = falr.get_ensemble_predict(X_test, method='average')
print(f"Ensemble (avg probs) Accuracy: {accuracy_score(y_test, ensemble_pred):.1%}")


