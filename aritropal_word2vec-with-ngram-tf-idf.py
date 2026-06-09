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


zip_path_labeled = '/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip'
zip_path_unlabeled = '/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip'
zip_path_test = '/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip'

labeled_file = 'labeledTrainData.tsv'
unlabeled_file = 'unlabeledTrainData.tsv'
test_file = 'testData.tsv'


with zipfile.ZipFile(zip_path_labeled, mode='r') as zf:
    with zf.open(labeled_file, 'r') as f:
        labeled_train = pd.read_csv(f, delimiter='\t', quoting=3)

labeled_train.head()


with zipfile.ZipFile(zip_path_unlabeled, mode='r') as zf:
    with zf.open(unlabeled_file, 'r') as f:
        unlabeled_train = pd.read_csv(f, delimiter='\t', quoting=3)

unlabeled_train.head()


with zipfile.ZipFile(zip_path_test, mode='r') as zf:
    with zf.open(test_file, 'r') as f:
        test = pd.read_csv(f, delimiter='\t', quoting=3)

test.head()


from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords


def number_preprocessing(review, strategy='context_aware'):
    if strategy == 'remove_all':
        return re.sub(r'\d+', '', review)

    if strategy == 'replace_token':
        # single digit
        review = re.sub(r'\b\d\b', '<DIGIT>', review)
        # multiple digits
        review = re.sub(r'\b\d{2,}\b', '<NUMBER>', review)
        return review

    if strategy == 'hash_encode':
        def replace_with_hash(match):
            return '#'*len(match.group())
        return re.sub(r'\d+', replace_with_hash, review)

    if strategy == 'context_aware':
        preserve_patterns = [
            r'\b\d+\.\d+\b',             # Decimal numbers
            r'\b\d+/\d+\b',              # Fractions
            r'\b\d+\s*stars?\b',         # Star ratings
            r'\b\d+\s*out\s+of\s+\d+\b', # X out of Y ratings
            r'\b\d+\s*(st|nd|rd|th)\b'   # Ordinals 1st, 2nd 
        ]

        protected_numbers = {}
        c = 0

        for pattern in preserve_patterns:
            matches = re.finditer(pattern, review, re.IGNORECASE)
            for match in matches:
                placeholder = f"__PROTECTED_NUM_{c}__"
                protected_numbers[placeholder] = match.group()
                review = review.replace(match.group(), placeholder, 1)
                c += 1

        review = re.sub(r'\b\d+\b', '', review)

        for placeholder, original in protected_numbers.items():
            review = review.replace(placeholder, original)

        return review

    return review


!pip install contractions


import contractions


def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text()
    review_text = review_text.lower()
    review_text = contractions.fix(review_text)

    review_text = number_preprocessing(review_text)

    review_text = re.sub(r'[^\w\s<>-]', ' ', review_text)

    review_text = ' '.join(review_text.split())

    words = review_text.split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)


import nltk.data
nltk.download('punkt')


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    sentences = tokenizer.tokenize(review.strip())
    sentences_wordlist = []
    for s in sentences:
        if len(s)>0:
            sentences_wordlist.append(review_to_wordlist(s, remove_stopwords))

    return sentences_wordlist


train_data = []

for review in labeled_train['review']:
    train_data += review_to_sentences(review, tokenizer)

for review in unlabeled_train['review']:
    train_data += review_to_sentences(review, tokenizer)


len(train_data)


num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

from gensim.models import word2vec
model = word2vec.Word2Vec(train_data, 
                          workers = num_workers,
                          vector_size = num_features,
                          min_count = min_word_count,
                          window = context,
                          sample = downsampling)


model.init_sims(replace=True)
model_name = "tutorial_model"
model.save(model_name)


model.wv.vectors.shape


from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfNgramEmbedder:
    def __init__(self, model, num_features, n=3, max_features=10000, token_pattern=r'(?u)\b\w+\b', lowercase=False):
        self.model = model
        self.num_features = num_features
        self.n = n
        self.max_features = max_features
        self.token_pattern = token_pattern
        self.lowercase = lowercase

    def fit(self, reviews):
        documents = [' '.join(review) for review in reviews]
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features = self.max_features,
            lowercase = self.lowercase,
            token_pattern = self.token_pattern,
            ngram_range = (1, self.n)
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()

        word_index = set(self.model.wv.index_to_key)
        feature_emb_list = []
        valid_idxs = []

        for i, feat in enumerate(self.feature_names):
            words = feat.split()
            vecs = [self.model.wv[w] for w in words if w in word_index]
            if len(vecs) > 0:
                feature_emb_list.append(np.mean(vecs, axis=0).astype('float32'))
                valid_idxs.append(i)
            else:
                feature_emb_list.append(None)

        self.valid_indices = np.array(valid_idxs, dtype = int)
        self.valid_feature_emb_list = np.vstack([feature_emb_list[i] for i in self.valid_indices]).astype('float32')
        return self

    def transform(self, reviews):
        documents = [' '.join(review) for review in reviews]
        tfidf = self.tfidf_vectorizer.transform(documents)

        n_reviews = len(reviews)
        X = np.zeros((n_reviews, self.num_features), dtype='float32')

        for i in range(n_reviews):
            weights = tfidf[i, self.valid_indices].toarray().ravel()
            wsum = weights.sum()
            if wsum>0:
                weighted = (self.valid_feature_emb_list * weights[:, None]).sum(axis=0) / wsum
                X[i] = weighted
            else:
                X[i] = np.zeros((self.num_features,), dtype='float32')

        return X

    def fit_transform(self, reviews):
        self.fit(reviews)
        return self.transform(reviews)


# def makeWeightedFeatureVec(index, model, num_features, tfidf_matrix, valid_word_embeddings, valid_indices):
#     doc_weights = tfidf_matrix[index, valid_indices].toarray().flatten()

#     if doc_weights.sum()>0:
#         weighted_embeddings = valid_word_embeddings * doc_weights[:, np.newaxis]
#         return weighted_embeddings.sum(axis=0) / doc_weights.sum()
#     else:
#         return np.zeros((num_features,), dtype="float32")


# def getWeightedAvgFeatureVecs(reviews, model, num_features):
#     documents = [' '.join(review) for review in reviews]

#     tfidf_vectorizer = TfidfVectorizer(
#         max_features = 10000,
#         lowercase = False,
#         token_pattern = r'(?u)\b\w+\b'
#     )
#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    
#     feature_names = tfidf_vectorizer.get_feature_names_out()
#     word_index = set(model.wv.index_to_key)
#     valid_words = [word for word in feature_names if word in word_index]
    
#     valid_indices = np.array([i for i, word in enumerate(feature_names) if word in word_index])
#     valid_word_embeddings = np.array([model.wv[word] for word in valid_words])
    
#     reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

#     for i in range(len(reviews)):
#         reviewFeatureVecs[i] = makeWeightedFeatureVec(
#             i, model, num_features, tfidf_matrix, valid_word_embeddings, valid_indices)

#     return reviewFeatureVecs


embedder = TfidfNgramEmbedder(model, num_features)


train_reviews = []
for review in labeled_train['review']:
    train_reviews.append(review_to_wordlist(review, True))

trainDataVecs = embedder.fit_transform(train_reviews)


trainDataVecs.shape


test_reviews = []
for review in test['review']:
    test_reviews.append(review_to_wordlist(review, True))

testDataVecs = embedder.transform(test_reviews)


testDataVecs.shape


from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score


# def objective(trial):
#     params = {
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',
#         'reg_lambda': trial.suggest_int('lambda', 0, 10),
#         'reg_alpha': trial.suggest_int('alpha', 0, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'n_jobs': -1
#     }

#     model = XGBClassifier(**params)
#     cv = StratifiedKFold(n_splits=5, shuffle=True)
#     scores = cross_val_score(model, trainDataVecs, labeled_train['sentiment'], 
#                              cv=cv, scoring='roc_auc', n_jobs=-1)
    
#     return np.mean(scores)


# study = optuna.create_study(direction='maximize', 
#                            sampler=optuna.samplers.TPESampler(seed=42),
#                            pruner=optuna.pruners.MedianPruner())

# study.optimize(objective, n_trials=50)


model = XGBClassifier(
    n_estimators = 754,
    subsample = 0.8063883618113541,
    max_depth = 9,
    learning_rate = 0.0810975323497118,
    reg_alpha = 4,
    reg_lambda = 5
)


model.fit(trainDataVecs, labeled_train['sentiment'])


results = model.predict(testDataVecs)


output = pd.DataFrame({
    'id': test['id'],
    'sentiment': results
})

output.to_csv("submission.csv", index=False, quoting=3)
output.head()

