!pip install langdetect
!pip install contractions
!pip install imblearn


import re
import os
import math
import string
from pathlib import Path
from collections import Counter
import emoji
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import nltk
from lightgbm import LGBMClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.naive_bayes import MultinomialNB, GaussianNB

#from imblearn.over_sampling import RandomOverSampler
from langdetect import detect, LangDetectException
import contractions
from scipy.sparse import hstack, csr_matrix

RANDOM_STATE = 42
import warnings
warnings.filterwarnings('ignore')

# PyTorch for LSTM............
import torch
from torch import optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, dataset,DataLoader, TensorDataset, SequentialSampler




is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron')
nltk.download('punkt_tab')

stopwords_set = set(stopwords.words('english'))


train_rules = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')

train_rules.head(5)


def create_article_path( text_id, article_id):
    path = f'/kaggle/input/fake-or-real-the-impostor-hunt/data/train/article_{article_id}/file_{text_id}.txt'
    return path

def read_file(file_path):
    # Function to read the content of the file at the given path
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    return text


# Create the fake_text_id column: if real is 1, fake is 2; otherwise fake is 1
train_rules['fake_text_id'] = train_rules['real_text_id'].apply(lambda x:2 if x ==1 else 1)

# Format article ID with leading zeros (e.g., '0001')
train_rules['article_id'] = train_rules['id'].apply(lambda x: str(x).zfill(4))

# Construct file paths for real and fake texts
train_rules['real_text_file'] = train_rules[['real_text_id', 'article_id']].apply(lambda x:create_article_path(x['real_text_id'], x['article_id']), axis=1)

train_rules['fake_text_file']= train_rules[['fake_text_id', 'article_id']].apply(lambda x:create_article_path(x['fake_text_id'], x['article_id']), axis=1)


# Load text content from each file
train_rules['real_text'] = train_rules['real_text_file'].apply(read_file)
train_rules['fake_text'] = train_rules['fake_text_file'].apply(read_file)



train_rules.head(5)


# real text is : 1 and fake text is : 0
df_real = train_rules[['article_id', 'real_text']].copy()
df_real.columns = ['article_id', 'text']
df_real['label'] = 1

df_fake = train_rules[['article_id', 'fake_text']].copy()
df_fake.columns = ['article_id', 'text']
df_fake['label'] = 0

# Concatenate real and fake data into one DataFrame
df_full = pd.concat([df_real, df_fake], ignore_index=True)

# Show result
df_full.head()



sample_article = '0090'
print(f"Label: {df_full[df_full['article_id'] == sample_article].label.iloc[0]}")
df_full[df_full['article_id'] == sample_article].text.iloc[0]


# Clean emojis from text
def strip_emoji(text):
    return emoji.get_emoji_regexp().sub("", text)

# Remove punctuations, stopwords, links, mentions and new line characters
def strip_all_entities(text):
    text = re.sub(r'\r|\n', ' ', text.lower())  # Replace newline and carriage return with space, and convert to lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)  # Remove links and mentions
    text = re.sub(r'[^\x00-\x7f]', '', text)  # Remove non-ASCII characters
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = ' '.join(word for word in text.split() if word not in stopwords_set)
    return text

# Clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    # Remove hashtags at the end of the sentence
    new_tweet = re.sub(r'(\s+#[\w-]+)+\s*$', '', tweet).strip()

    # Remove the # symbol from hashtags in the middle of the sentence
    new_tweet = re.sub(r'#([\w-]+)', r'\1', new_tweet).strip()

    return new_tweet

# Filter special characters such as & and $ present in some words
def filter_chars(text):
    return ' '.join('' if ('$' in word) or ('&' in word) else word for word in text.split())

# Remove multiple spaces
def remove_mult_spaces(text):
    return re.sub(r"\s\s+", " ", text)

# Function to check if the text is in English, and return an empty string if it's not
def filter_non_english(text):
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    return text if lang == "en" else ""

# Expand contractions
def expand_contractions(text):
    return contractions.fix(text)

# Remove numbers
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# Lemmatize words
def lemmatize(text):
    # Initialize lemmatizer for text cleaning
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Remove short words
def remove_short_words(text, min_len=2):
    words = text.split()
    long_words = [word for word in words if len(word) >= min_len]
    return ' '.join(long_words)

# Replace elongated words with their base form
def replace_elongated_words(text):
    regex_pattern = r'\b(\w+)((\w)\3{2,})(\w*)\b'
    return re.sub(regex_pattern, r'\1\3\4', text)

# Remove repeated punctuation
def remove_repeated_punctuation(text):
    return re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)

# Remove extra whitespace
def remove_extra_whitespace(text):
    return ' '.join(text.split())

def remove_url_shorteners(text):
    return re.sub(r'(?:http[s]?://)?(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|tr\.im|is\.gd|cli\.gs|u\.nu|url\.ie|tiny\.cc|alturl\.com|ow\.ly|bit\.do|adoro\.to)\S+', '', text)

# Remove spaces at the beginning and end of the tweet
def remove_spaces_tweets(tweet):
    return tweet.strip()

# Remove short tweets
def remove_short_tweets(tweet, min_words=3):
    words = tweet.split()
    return tweet if len(words) >= min_words else ""

# Function to call all the cleaning functions in the correct order
def clean_tweet(tweet):
    #tweet = strip_emoji(tweet)
    tweet = expand_contractions(tweet)
    tweet = filter_non_english(tweet)
    tweet = strip_all_entities(tweet)
    tweet = clean_hashtags(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_mult_spaces(tweet)
    tweet = remove_numbers(tweet)
    tweet = lemmatize(tweet)
    tweet = remove_short_words(tweet)
    tweet = replace_elongated_words(tweet)
    tweet = remove_repeated_punctuation(tweet)
    tweet = remove_extra_whitespace(tweet)
    tweet = remove_url_shorteners(tweet)
    tweet = remove_spaces_tweets(tweet)
    tweet = remove_short_tweets(tweet)
    tweet = ' '.join(tweet.split())  # Remove multiple spaces between words
    return tweet


df_full['text'] = df_full['text'].apply(clean_tweet)


def Stemming(text):
    ps = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_words = []
    for token in tokens:
        stemmed_token = ps.stem(token)
        stemmed_words.append(stemmed_token)
    return ' '.join(stemmed_words)


df_full['text'] = df_full['text'].apply(Stemming)


df_full.head()


# df = df_full.copy()

# # shuffle the dataset............
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# X = df['text']
# y = df['label'].values

# ct = CountVectorizer()
# X_ct = ct.fit_transform(X)



# # Combine Count and TF-IDF using FeatureUnion
# combined_features = FeatureUnion([
#     ("count_vec", CountVectorizer()),
#     ("tfidf_vec", TfidfVectorizer(use_idf=True))
# ])
#
# # Full pipeline
# pipeline = Pipeline([
#     ("features", combined_features),
#     ("classifier", RandomForestClassifier())
# ])
#
# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)
#
# # Train model
# pipeline.fit(X_train, y_train)
#
# # Predict
# predictions = pipeline.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, predictions))



# tfidf = TfidfVectorizer(use_idf=True).fit(X)
# X_tfidf = tfidf.transform(X)



# from gensim.models import Word2Vec, KeyedVectors
# # Download pretrained GoogleNews vectors (300D)
# # https://code.google.com/archive/p/word2vec/
# # !mkdir -p /kaggle/input/word2vec
# # !kaggle datasets download -d leadbest/googlenewsvectorsnegative300 -p /kaggle/input/word2vec --unzip

# # w2v_path = "/kaggle/input/word2vec-google/GoogleNews-vectors-negative300.bin"
# # w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
# # Tokenize your training text
# texts = df_full['text']

# # # Train Word2Vec (100D, window=5, min_count=2)

# # embedding_dim = 200
# # max_len = 200  # truncate/pad sequence length

# w2v = Word2Vec(sentences=texts, vector_size=300, window=5, min_count=1, workers=4)

# def text_to_vec(words, model, size=300):
#     vecs = [model.wv[w] for w in words if w in model.wv]
#     if len(vecs) == 0:
#         return np.zeros(size)
#     return np.mean(vecs, axis=0)

# X = np.array([text_to_vec(t, w2v) for t in texts])
# y = np.array(df_full['label'])


from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"  # you can also try "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name).to(device)
bert_model.eval()


def text_to_bert_embedding(text, tokenizer, model, max_len=128):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_len,
                       truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_state = outputs.last_hidden_state  # [batch, seq, hidden]
        
        # Option 1: CLS token
        cls_embedding = hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        # Option 2: Mean pooling (better sometimes)
        # mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_state.size())
        # mean_embedding = (hidden_state * mask).sum(1) / mask.sum(1)
        # cls_embedding = mean_embedding.squeeze().cpu().numpy()
    
    return cls_embedding


texts = df_full['text']

X = np.array([text_to_bert_embedding(t, tokenizer, bert_model) for t in texts])
y = np.array(df_full['label'])


# X_train, X_val, y_train, y_val = train_test_split(
#     X,y, test_size=0.25, random_state=42
# )


model ={
    'RandomForest' : RandomForestClassifier(
        random_state=22,
        n_estimators=1000,
        max_depth=15,
        n_jobs=-1,
        min_samples_split=2,
        ccp_alpha=0.01
    ),
    'LogisticRegression' : LogisticRegression(
        max_iter=600,
        random_state=20,
    ),
    'XGBoost' : XGBClassifier(
        n_estimators=600,
        n_jobs=-1,
        device='cuda',
        random_state=22,
        tree_method='gpu_hist',
    ),
    "SVM": SVC(
        probability=True,
        random_state=22,
        max_iter=500
    ),
    'decisionTree': DecisionTreeClassifier(
        random_state=22,
        splitter='best',
        criterion='gini',
        max_depth=15,
        ccp_alpha=0.02
    ),
    # 'naiveBayes': MultinomialNB(
    #     alpha=0.1,
    # ),
    # 'lightgbm': lgb.LGBMClassifier(
    #     random_state=22,
    #     learning_rate=0.01,
    #     n_jobs=-1,
    #     n_estimators=500,
    #     max_depth=15,
    #     force_col_wise=False
    # )
}

# 5 Fold stratified cross validation
st = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

for name, model in model.items():
    accuracy , precision, recall, f1s = [],[],[],[]

    for train_index, test_index in st.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        prec, recalls, f1 , _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

        accuracy.append(acc)
        precision.append(prec)
        recall.append(recalls)
        f1s.append(f1)

    print(f"\nğŸ“Œ Model: {name}")
    print(f"Accuracy:  {np.mean(accuracy):.4f}")
    print(f"Precision: {np.mean(precision):.4f}")
    print(f"Recall:    {np.mean(recall):.4f}")
    print(f"F1-score:  {np.mean(f1s):.4f}")


# 1. Model and pipeline definition
svm = SVC(probability=True)

pipeline = Pipeline([
    ('clf', svm)
])

# 2. Hyperparameters to search
param_grid = {
    'clf__kernel': ['linear', 'rbf'],
    'clf__C': [0.1, 1, 10],
    'clf__gamma': ['scale', 'auto']  # Only affects RBF kernel
}

# 3. Stratified 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,  # Use all CPU cores
    verbose=2
)

# 4. Run Grid Search
grid.fit(X, y)

# 5. Results
print("ğŸ”� Best parameter combination:")
print(grid.best_params_)
print("\nğŸ“ˆ Best mean F1-score (cross-validation):")
print(grid.best_score_)


# param_grid = {
#     'n_estimators': [300, 500],          # number of trees
#     'max_depth': [10, 20],           # depth of each tree
#     'min_samples_split': [2, 5],               # minimum samples to split a node
#     'min_samples_leaf': [ 2, 4],                 # minimum samples per leaf
#     'max_features': ['sqrt', 'log2'],        # number of features to consider for best split
#     'criterion': ['gini', 'entropy']               # splitting criterion
# }


# randomF = RandomForestClassifier(random_state=42)

# # pipeline = Pipeline([
# #     ('rf', RandomForestClassifier(random_state=42)),
# # ])

# # 3. Stratified 5-fold cross-validation
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=32)

# grid = GridSearchCV(
#     estimator=randomF,
#     param_grid=param_grid,
#     scoring='f1_macro',
#     cv=cv,
#     n_jobs=1,  # Use all CPU cores
#     verbose=2
# )

# # 4. Run Grid Search
# grid.fit(X, y)

# # 5. Results
# print("ğŸ”� Best parameter combination:")
# print(grid.best_params_)
# print("\nğŸ“ˆ Best mean F1-score (cross-validation):")
# print(grid.best_score_)


# class NewsDataset(Dataset):
#     def __init__(self, texts, labels, w2v, max_len):
#         self.texts = texts
#         self.labels = labels.values
#         self.w2v = w2v
#         self.max_len = max_len
#         self.embedding_dim = w2v.vector_size

#     def __len__(self):
#         return len(self.texts)

#     def text_to_vec(self, text):
#         words = text.split()
#         vecs = []
#         for w in words[:self.max_len]:
#             if w in self.w2v.wv:
#                 vecs.append(self.w2v.wv[w])
#             else:
#                 vecs.append(np.zeros(self.embedding_dim))
#         while len(vecs) < self.max_len:
#             vecs.append(np.zeros(self.embedding_dim))
#         return np.array(vecs)

#     def __getitem__(self, idx):
#         x = self.text_to_vec(self.texts.iloc[idx])
#         y = self.labels[idx]
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# train_dataset = NewsDataset(X_train, y_train, w2v, max_len)
# val_dataset = NewsDataset(X_val, y_val, w2v, max_len)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64)


# class FakeNewsClassifier(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, num_classes=2):
#         super(FakeNewsClassifier, self).__init__()
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim*2, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         _, (h, _) = self.lstm(x)
#         h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
#         out = self.dropout(h)
#         return self.fc(out)


# model = FakeNewsClassifier(embedding_dim=embedding_dim, hidden_dim=128).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)



# def train_model(model, train_loader, val_loader, epochs):
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for x_batch, y_batch in train_loader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             optimizer.zero_grad()
#             outputs = model(x_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         avg_loss = total_loss/len(train_loader)

#         # Validation
#         model.eval()
#         preds, labels = [], []
#         with torch.no_grad():
#             for x_batch, y_batch in val_loader:
#                 x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#                 outputs = model(x_batch)
#                 _, predicted = torch.max(outputs, 1)
#                 preds.extend(predicted.cpu().numpy())
#                 labels.extend(y_batch.cpu().numpy())
#         acc = accuracy_score(labels, preds)
#         print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {acc:.4f}")


# train_model(model, train_loader, val_loader, epochs=100)


# # Retrieve the best model from the grid search
# best_model = grid.best_estimator_

# # Retrain the best model using the full dataset
# best_model.fit(X_ct, y)


# import pandas as pd
# from pathlib import Path
# import torch

# # === 1. Load Test Data ===
# data_path = Path('/kaggle/input/fake-or-real-the-impostor-hunt/data/test')
# folders = sorted([f for f in data_path.iterdir() if f.is_dir()])

# submission_rows = []

# # Put model in evaluation mode
# model.eval()

# with torch.no_grad():
#     for idx, folder in enumerate(folders):
#         texts = []
#         for file in ['file_1.txt', 'file_2.txt']:
#             fp = folder / file
#             texts.append(read_file(fp))   # your custom read function

#         # === 2. Preprocess ===
#         feat_rows = [clean_tweet(t) for t in texts]   # clean text
        
#         def text_to_vec(text):
#             words = text.split()
#             vecs = []
#             for w in words[:max_len]:
#                 if w in w2v.wv:
#                     vecs.append(w2v.wv[w])
#                 else:
#                     vecs.append(np.zeros(embedding_dim))
#             while len(vecs) < max_len:
#                 vecs.append(np.zeros(embedding_dim))
#             return np.array(vecs)

#         vecs = [text_to_vec(t) for t in feat_rows]
#         #vecs = [w2v.wv(t) for t in feat_rows]  # convert to vectors (same as training)
#         vecs = torch.tensor(vecs, dtype=torch.float32).to(device)

#         # === 3. Model Prediction ===
#         outputs = model(vecs)
#         preds = torch.argmax(outputs, dim=1).cpu().numpy()

#         # preds will be [0,1] or [1,0] depending on fake/real order
#         # We want: 1 or 2 (which text is REAL)
#         real_text_id = preds.argmax() + 1  # add +1 because file_1=1, file_2=2

#         submission_rows.append([idx, real_text_id])

# # === 4. Save Submission File ===
# submission_df = pd.DataFrame(submission_rows, columns=["id", "real_text_id"])




# # 1. Read test folders in sorted order
# data_path = Path('/kaggle/input/fake-or-real-the-impostor-hunt/data/test')
# folders = sorted([f for f in data_path.iterdir() if f.is_dir()])

# submission_rows = []
# #best_model = model["SVM"]  # pick best one

# for idx, folder in enumerate(folders):
#     texts =[]
#     for file in ['file_1.txt', 'file_2.txt']:
#         fp = folder / file
#         texts.append(read_file(fp))

#     feat_rows = [clean_tweet(t) for t in texts] # Preprocessing call clean_tweet function

#     #X_test = ct.transform(feat_rows)  # CountVectorization
#     #X_test = np.array([text_to_vec(t, w2v) for t in feat_rows]) # Word2Vec
#     X_test = np.array([text_to_bert_embedding(t, tokenizer, bert_model) for t in feat_rows])
    
#     #scores = best_model.predict_proba(X_test)[:, 1]
#     scores = grid.predict_proba(X_test)[:, 1]
#     chosen = 1 if scores[0] > scores[1] else 2

#     submission_rows.append({"id": idx, "real_text_id": chosen})

# submission = pd.DataFrame(submission_rows)



# submission_df.to_csv("submission.csv", index=False)

# print(submission_df.head(100))


test_path = Path("/kaggle/input/fake-or-real-the-impostor-hunt/data/test")
folders = sorted([f for f in test_path.iterdir() if f.is_dir()])

submission_rows = []

for idx, folder in enumerate(folders):
    f1 = clean_tweet(open(folder/"file_1.txt").read())
    f2 = clean_tweet(open(folder/"file_2.txt").read())
    
    f1_vec = text_to_bert_embedding(f1, tokenizer, bert_model).reshape(1, -1)
    f2_vec = text_to_bert_embedding(f2, tokenizer, bert_model).reshape(1, -1)
    
    p1 = grid.predict_proba(f1_vec)[0][1]
    p2 = grid.predict_proba(f2_vec)[0][1]
    
    real_text_id = 1 if p1 > p2 else 2
    submission_rows.append([idx, real_text_id])

submission_df = pd.DataFrame(submission_rows, columns=["id", "real_text_id"])
submission_df.to_csv("submission.csv", index=False)
submission_df.head()

