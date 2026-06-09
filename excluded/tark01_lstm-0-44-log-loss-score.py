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


import seaborn as sns
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
import sklearn

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('/kaggle/input/quora-df/quora_feautures.csv')


df['is_duplicate'].value_counts()


df = df.dropna()


df.shape


test_data = pd.read_csv("/kaggle/input/test-data-adv/test_adv_feature.csv")





def preprocess(q):
    
    q = str(q).lower().strip()
    
    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('â‚¹', ' rupee ')
    q = q.replace('â‚¬', ' euro ')
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

    
    return q


df['question1'] = df['question1'].apply(preprocess)
df['question2'] = df['question2'].apply(preprocess)


df['q1_len'] = df['question1'].str.len() 
df['q2_len'] = df['question2'].str.len()


df['q1_num_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
df['q2_num_words'] = df['question2'].apply(lambda row: len(row.split(" ")))
df.head()


def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return len(w1 & w2)


df['word_common'] = df.apply(common_words, axis=1)
df.head()


def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return (len(w1) + len(w2))


df['word_total'] = df.apply(total_words, axis=1)
df.head()


df['word_share'] = round(df['word_common']/df['word_total'],2)
df.head()


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


token_features = df.apply(fetch_token_features, axis=1)

df["cwc_min"]       = list(map(lambda x: x[0], token_features))
df["cwc_max"]       = list(map(lambda x: x[1], token_features))
df["csc_min"]       = list(map(lambda x: x[2], token_features))
df["csc_max"]       = list(map(lambda x: x[3], token_features))
df["ctc_min"]       = list(map(lambda x: x[4], token_features))
df["ctc_max"]       = list(map(lambda x: x[5], token_features))
df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
df["first_word_eq"] = list(map(lambda x: x[7], token_features))


df.head(5)


pip install Distance


import distance

def fetch_length_features(row):
    q1 = row['question1']
    q2 = row['question2']
    
    length_features = [0.0] * 3  # Initialize features to default values
    
    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features  # Return default values if either question is empty
    
    # Absolute length difference
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    # Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    
    # Longest Common Substring Ratio
    strs = list(distance.lcsubstrings(q1, q2))
    if strs:  # Check if strs is non-empty
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    else:
        length_features[2] = 0.0  # Default value if no common substring
    
    return length_features



# Apply the function
length_features = df.apply(fetch_length_features, axis=1)

# Create new columns
df['abs_len_diff'] = list(map(lambda x: x[0], length_features))
df['mean_len'] = list(map(lambda x: x[1], length_features))
df['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))


df.shape


df.columns


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


fuzzy_features = df.apply(fetch_fuzzy_features, axis=1)

# Creating new feature columns for fuzzy features
df['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
df['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
df['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
df['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
q1_tfidf = vectorizer.fit_transform(df['question1'])
q2_tfidf = vectorizer.transform(df['question2'])

# Compute cosine similarity
#df['similarity'] = [cosine_similarity(q1_tfidf[i], q2_tfidf[i])[0][0] for i in range(len(df))]


df['similarity'] = [
    cosine_similarity(q1_tfidf[i].reshape(1, -1), q2_tfidf[i].reshape(1, -1))[0][0]
    for i in range(len(df))
]


df.shape


import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained GloVe word vectors (50D for faster computation)
glove_model = api.load("glove-wiki-gigaword-50")  # You can change to 100D, 200D, etc.

# Function to get sentence embedding by averaging word embeddings
def get_sentence_embedding(sentence):
    words = sentence.split()  # Tokenize sentence
    word_vectors = [glove_model[word] for word in words if word in glove_model]
    
    if len(word_vectors) == 0:  # If no word embeddings are found, return a zero vector
        return np.zeros(50)
    
    return np.mean(word_vectors, axis=0)  # Average word embeddings

# Compute sentence embeddings
q1_embeddings = np.array([get_sentence_embedding(q) for q in df['question1']])
q2_embeddings = np.array([get_sentence_embedding(q) for q in df['question2']])



df['embedding_similarity'] = [
    cosine_similarity(q1_embeddings[i].reshape(1, -1), q2_embeddings[i].reshape(1, -1))[0][0]
    for i in range(len(df))
]





df = pd.read_csv('/kaggle/input/quora-df/quora_feautures.csv')


df.head()


df.shape


df.shape


df.isnull().sum()


df = df.dropna()





question = list(df['question1']) + list(df['question2'])


from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=45000,  oov_token="<OOV>")



tokenizer.fit_on_texts(question)


tokenizer.word_index


q1_seq = tokenizer.texts_to_sequences(df['question1'])


q2_seq = tokenizer.texts_to_sequences(df['question2'])





from tensorflow.keras.preprocessing.sequence import pad_sequences


len_of_seq = [len(x) for x in  q2_seq]


max(len_of_seq)


sequence = tokenizer.texts_to_sequences(question)


sequence


q1_padded = pad_sequences(q1_seq, maxlen=248, padding='post', truncating='post')
q2_padded = pad_sequences(q2_seq, maxlen=248, padding='post', truncating='post')


q1_padded.shape


df_cor = df.drop(['id', 'qid1', 'qid2', 'question1', 'question2'], axis=1)


cor = df_cor.corr()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16,20))
sns.heatmap(cor, annot=True)











df = pd.read_csv('/kaggle/input/quora-df/quora_feautures.csv')


df.head()


df.isnull().sum()


df = df.dropna()


new_train_data = df.drop(['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate', 'Unnamed: 0'], axis=1)


new_train_data.columns


new_train_data_arr = np.array(new_train_data)


new_train_data_arr.shape


q1_padded.shape[1]


y = df['is_duplicate']


y = np.array(y, dtype=np.float32)


q1_padded = np.array(q1_padded, dtype=np.int32)
q2_padded = np.array(q2_padded, dtype=np.int32)

# Ensure `train_data_arr` is also a NumPy array
train_data_arr = np.array(new_train_data_arr, dtype=np.float32)


print(q1_padded.shape)
print(new_train_data_arr.shape)


train_data = np.hstack([q1_padded, q2_padded, train_data_arr])


train_data.shape


train_data


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.2, random_state=42, stratify=y)


y_train.shape


X_train.shape


from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate,  Dropout
from tensorflow.keras.models import Sequential


max_len = 248  # Since you padded both q1 & q2 to 248
vocab_size = 35000  # Adjust based on your tokenizer's vocab size
embedding_dim = 300  # Use pre-trained embeddings like GloVe for better performance


model = Sequential()

model.add(Embedding(input_dim=35000, output_dim=128, input_length=248))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.summary()


from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0005) 


history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=7, batch_size=64)


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


model.save("lstm_model.h5")


import pickle

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)




