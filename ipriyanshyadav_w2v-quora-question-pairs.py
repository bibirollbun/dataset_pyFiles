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
import seaborn as sns

import re
from bs4 import BeautifulSoup

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


a = pd.read_csv("../quora-question-pairs/train.csv")
a


a.shape


a = a[["question1", "question2", "is_duplicate"]].dropna()
a.shape


a["is_duplicate"].value_counts()


print("Ratios")
a["is_duplicate"].value_counts() / len(a) * 100


a["is_duplicate"].value_counts().plot(kind="pie")


a1 = a[a["is_duplicate"] == 0]
a2 = a[a["is_duplicate"] == 1]


print(a1["is_duplicate"].value_counts())
print()
print(a2["is_duplicate"].value_counts())


a1 = a1.sample(149263, random_state=42)
a1.shape


a2 = a2.sample(149263, random_state=42)
a2.shape


df = pd.concat([a1, a2]).reset_index(drop=True)
df.shape


df["is_duplicate"].value_counts().plot(kind="pie")


def preprocess(q):

    q = str(q).lower().strip()

    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    # Stemming
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    q = ' '.join([stemmer.stem(word) for word in q.split()])

    return q


df["question1"]= df["question1"].apply(preprocess)
df["question2"]= df["question2"].apply(preprocess)


df1 = df[["question1", "question2"]].copy()

df1["q1_len"] = df1["question1"].str.len()
df1["q2_len"] = df1["question2"].str.len()

df1["q1_num_words"] = df1["question1"].apply(lambda x: len(str(x).split()))
df1["q2_num_words"] = df1["question2"].apply(lambda x: len(str(x).split()))

df1.head()


def common_words(row):
    w1 = set(map(lambda x: x.lower().strip(), str(row["question1"]).split()))
    w2 = set(map(lambda x: x.lower().strip(), str(row["question2"]).split()))
    return len(w1 & w2)

def word_total(row):
    w1 = str(row['question1']).split()
    w2 = str(row['question2']).split()
    return len(w1) + len(w2)

df1["word_common"] = df1.apply(common_words, axis=1)
df1["word_total"] = df1.apply(word_total, axis=1)

df1["word_share"] = df1.apply(lambda x: round(x["word_common"] / x["word_total"], 2) if x["word_total"] > 0 else 0, axis=1)

df1.head()


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

def fetch_token_features(row):

    q1 = row['question1']
    q2 = row['question2']

    SAFE_DIV = 0.0001

    STOP_WORDS = stopwords.words("english")

    token_features = [0.0]*8

    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))

    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))

    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))


    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


token_features = df1.apply(fetch_token_features, axis=1)

df1["cwc_min"]       = list(map(lambda x: x[0], token_features))
df1["cwc_max"]       = list(map(lambda x: x[1], token_features))
df1["csc_min"]       = list(map(lambda x: x[2], token_features))
df1["csc_max"]       = list(map(lambda x: x[3], token_features))
df1["ctc_min"]       = list(map(lambda x: x[4], token_features))
df1["ctc_max"]       = list(map(lambda x: x[5], token_features))
df1["last_word_eq"]  = list(map(lambda x: x[6], token_features))
df1["first_word_eq"] = list(map(lambda x: x[7], token_features))

df1.head()


import distance

def fetch_length_features(row):

    q1 = row['question1']
    q2 = row['question2']

    length_features = [0.0]*3

    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))

    #Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens))/2

    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1) if strs else 0

    return length_features


length_features = df1.apply(fetch_length_features, axis=1)

df1['abs_len_diff'] = list(map(lambda x: x[0], length_features))
df1['mean_len'] = list(map(lambda x: x[1], length_features))
df1['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))

df1.head()


from fuzzywuzzy import fuzz

def fetch_fuzzy_features(row):

    q1 = row['question1']
    q2 = row['question2']

    fuzzy_features = [0.0]*4

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features


fuzzy_features = df1.apply(fetch_fuzzy_features, axis=1)

# Creating new feature columns for fuzzy features
df1['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
df1['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
df1['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
df1['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))

df1.head()


df1["is_duplicate"] = df["is_duplicate"]
df= df1.copy()
df.head()


sns.pairplot(df[['ctc_min', 'cwc_min', 'csc_min', 'is_duplicate']],hue='is_duplicate')


sns.pairplot(df[['ctc_max', 'cwc_max', 'csc_max', 'is_duplicate']],hue='is_duplicate')


sns.pairplot(df[['last_word_eq', 'first_word_eq', 'is_duplicate']],hue='is_duplicate')


sns.pairplot(df[['mean_len', 'abs_len_diff','longest_substr_ratio', 'is_duplicate']],hue='is_duplicate')


sns.pairplot(df[['fuzz_ratio', 'fuzz_partial_ratio','token_sort_ratio','token_set_ratio', 'is_duplicate']],hue='is_duplicate')


df1= df[["question1", "question2"]].copy()
df1.head()


df2= df.drop(columns= ["question1", "question2", "is_duplicate"])
df2.head()


import gensim


questions= list(df1["question1"]) + list(df["question2"])

ques_sent= []
for sentence in questions:
    ques_sent.append(gensim.utils.simple_preprocess(sentence))


model= gensim.models.Word2Vec(window= 5, min_count= 3, sg= 0, vector_size= 100)
model.build_vocab(ques_sent)

model.train(corpus_iterable= ques_sent, total_examples= model.corpus_count, epochs= model.epochs)


len(model.wv.index_to_key)


def document_vector(doc):
    # remove oov words
    doc = [word for word in doc.split() if word in model.wv.index_to_key]
    if len(doc) == 0:
        return np.zeros(model.vector_size)  # return zero vector for empty docs
    return np.mean(model.wv[doc], axis=0)


from tqdm import tqdm


X_list= []
for doc in tqdm(df1["question1"].values):
    X_list.append(document_vector(doc))

y_list= []
for doc in tqdm(df1["question2"].values):
    y_list.append(document_vector(doc))

X_array= np.array(X_list)
y_array= np.array(y_list)


w2v_features= np.concatenate([X_array, y_array], axis= 1)
manual_features = df2[["q1_len", "q2_len", "q1_num_words", "q2_num_words", "word_common", "word_total", "word_share", "cwc_min", "cwc_max", "csc_min", "csc_max", "ctc_min", "ctc_max", "last_word_eq", "first_word_eq", "abs_len_diff", "mean_len", "longest_substr_ratio", "fuzz_ratio", "fuzz_partial_ratio", "token_sort_ratio", "token_set_ratio"]].values

X = np.concatenate([w2v_features, manual_features], axis=1)
y = df["is_duplicate"].values

print(f"Final feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42, stratify= y)

rf= RandomForestClassifier(n_estimators= 100, random_state= 42)
rf.fit(X_train, y_train)

y_pred= rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy: {accuracy:.3f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


def get_w2v_features(text):
    """Convert text to W2V features by averaging word vectors"""
    words = text.split()
    word_vectors = []

    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])

    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)



def prediction(q1, q2):
    """Predict if two questions are duplicates using W2V features"""

    # Preprocess the questions
    q1_processed = preprocess(q1)
    q2_processed = preprocess(q2)

    # W2V features
    q1_w2v = get_w2v_features(q1_processed)
    q2_w2v = get_w2v_features(q2_processed)

    # Manual features
    q1_len = len(q1_processed)
    q2_len = len(q2_processed)
    q1_num_words = len(q1_processed.split())
    q2_num_words = len(q2_processed.split())

    # Common words
    w1 = set(map(lambda x: x.lower().strip(), q1_processed.split()))
    w2 = set(map(lambda x: x.lower().strip(), q2_processed.split()))
    word_common = len(w1 & w2)
    word_total = len(q1_processed.split()) + len(q2_processed.split())
    word_share = round(word_common / word_total, 2) if word_total > 0 else 0

    # Token features
    from nltk.corpus import stopwords
    STOP_WORDS = stopwords.words("english")
    SAFE_DIV = 0.0001

    q1_tokens = q1_processed.split()
    q2_tokens = q2_processed.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        token_features = [0.0] * 8
    else:
        q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
        q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

        common_word_count = len(q1_words.intersection(q2_words))
        common_stop_count = len(q1_stops.intersection(q2_stops))
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

        token_features = [
            common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV),
            common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV),
            common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV),
            common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV),
            common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV),
            common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV),
            int(q1_tokens[-1] == q2_tokens[-1]) if q1_tokens and q2_tokens else 0,
            int(q1_tokens[0] == q2_tokens[0]) if q1_tokens and q2_tokens else 0
        ]

    # Length features
    import distance
    abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
    mean_len = (len(q1_tokens) + len(q2_tokens)) / 2
    strs = list(distance.lcsubstrings(q1_processed, q2_processed))
    longest_substr_ratio = len(strs[0]) / (min(len(q1_processed), len(q2_processed)) + 1) if strs else 0

    # Fuzzy features
    from fuzzywuzzy import fuzz
    fuzz_ratio = fuzz.QRatio(q1_processed, q2_processed)
    fuzz_partial_ratio = fuzz.partial_ratio(q1_processed, q2_processed)
    token_sort_ratio = fuzz.token_sort_ratio(q1_processed, q2_processed)
    token_set_ratio = fuzz.token_set_ratio(q1_processed, q2_processed)

    # Combine features
    w2v_features = np.concatenate([q1_w2v, q2_w2v])
    manual_features = np.array([
        q1_len, q2_len, q1_num_words, q2_num_words, word_common, word_total, word_share,
        *token_features, abs_len_diff, mean_len, longest_substr_ratio,
        fuzz_ratio, fuzz_partial_ratio, token_sort_ratio, token_set_ratio
    ])

    features = np.concatenate([w2v_features, manual_features]).reshape(1, -1)

    # Make prediction
    pred = rf.predict(features)[0]
    prob = rf.predict_proba(features)[0]

    return pred, prob


test_cases = [
    ("What is the capital of India", "What is the capital of India"),
    ("What is the capital of India", "What is the capital of France"),
    ("How do I learn Python", "How can I learn Python programming"),
    ("What is machine learning", "How does a car engine work"),
    ("How to lose weight fast", "What are quick ways to reduce weight")
]

for q1, q2 in test_cases:
    pred, prob = prediction(q1, q2)
    print(f"Q1: {q1}")
    print(f"Q2: {q2}")
    confidence = prob[1]*100 if pred == 1 else (1 - prob[1])*100
    print(f"Prediction: {pred} (Confidence: {confidence:.1f}%)")
    print("-" * 80)

