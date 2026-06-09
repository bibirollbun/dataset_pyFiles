# (Cell 1) - run this first
!pip install -q transformers sentence-transformers   # optional for BERT/embeddings; Kaggle often already has them

import os, gc, re, math, random
import numpy as np, pandas as pd
from tqdm import tqdm

# sklearn / scipy / tf / keras
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from scipy.sparse import csr_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Lambda, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# display options
pd.set_option('display.max_colwidth', 200)



import os

# List all datasets mounted in the Kaggle input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    for filename in filenames[:5]:  # show first 5 files per folder
        print("   ", filename)



import pandas as pd

# Load train.csv from zip
train = pd.read_csv("/kaggle/input/quora-question-pairs/train.csv.zip")

# Load test.csv (already extracted)
test = pd.read_csv("/kaggle/input/quora-question-pairs/test.csv")

# Load sample_submission.csv from zip
sub = pd.read_csv("/kaggle/input/quora-question-pairs/sample_submission.csv.zip")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Submission shape:", sub.shape)



# Step 1: EDA
print(train.head())
print("\nMissing values in train:\n", train.isnull().sum())

print("\nTarget distribution:")
print(train['is_duplicate'].value_counts(normalize=True))

# Average question length
train['q1_len'] = train['question1'].astype(str).apply(len)
train['q2_len'] = train['question2'].astype(str).apply(len)

print("\nAverage length of Question 1:", train['q1_len'].mean())
print("Average length of Question 2:", train['q2_len'].mean())

# Class balance visualization
import matplotlib.pyplot as plt

train['is_duplicate'].value_counts().plot(kind='bar')
plt.title("Class Distribution (0 = not duplicate, 1 = duplicate)")
plt.show()



# text-preprocessing 
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords & wordnet if not already
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text):  # handle missing
        return ""
    # Lowercase
    text = text.lower()
    # Remove special chars & numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize & remove stopwords
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Apply to training set
train['q1_clean'] = train['question1'].apply(clean_text)
train['q2_clean'] = train['question2'].apply(clean_text)

print(train[['question1','q1_clean']].head(10))



# (Cell 3) - quick EDA
print("Missing in train:", train[['question1','question2']].isnull().sum().to_dict())
print("Missing in test:", test[['question1','question2']].isnull().sum().to_dict())
print("Label distribution:\n", train['is_duplicate'].value_counts(normalize=True))

# show example pairs
train.sample(5)



# (Cell 4)
import string
def clean_text(s):
    if pd.isnull(s): return ""
    s = str(s)
    s = s.strip().lower()
    # basic normalization: remove weird characters, keep alphanumerics and spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

for df in [train, test]:
    df['q1'] = df['question1'].apply(clean_text)
    df['q2'] = df['question2'].apply(clean_text)



# (Cell 5) - engineered features
def add_basic_feats(df):
    df['len_q1'] = df['q1'].apply(len)
    df['len_q2'] = df['q2'].apply(len)
    df['words_q1'] = df['q1'].apply(lambda x: len(x.split()))
    df['words_q2'] = df['q2'].apply(lambda x: len(x.split()))
    df['common_words'] = df.apply(lambda x: len(set(x['q1'].split()).intersection(set(x['q2'].split()))), axis=1)
    df['total_words'] = df['words_q1'] + df['words_q2']
    df['word_share'] = df['common_words'] / (df['total_words'].replace(0, np.nan))
    df['word_share'] = df['word_share'].fillna(0)
    df['char_diff'] = (df['len_q1'] - df['len_q2']).abs()
    return df

train = add_basic_feats(train)
test = add_basic_feats(test)

train[['len_q1','len_q2','words_q1','words_q2','common_words','word_share']].head()



# (Cell 6)
all_text = pd.concat([train['q1'], train['q2'], test['q1'], test['q2']]).unique()
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), analyzer='word')
tfidf.fit(all_text)

# transform
q1_tfidf_train = tfidf.transform(train['q1'])
q2_tfidf_train = tfidf.transform(train['q2'])

# cosine similarity per row: (x.multiply(y)).sum(axis=1) when rows align
cosine_sim_train = (q1_tfidf_train.multiply(q2_tfidf_train)).sum(axis=1)
train['tfidf_cosine'] = np.asarray(cosine_sim_train).reshape(-1)

# same for test
q1_tfidf_test = tfidf.transform(test['q1'])
q2_tfidf_test = tfidf.transform(test['q2'])
cosine_sim_test = (q1_tfidf_test.multiply(q2_tfidf_test)).sum(axis=1)
test['tfidf_cosine'] = np.asarray(cosine_sim_test).reshape(-1)

train[['tfidf_cosine']].describe()



# (Cell 7)
feat_cols = ['len_q1','len_q2','words_q1','words_q2','common_words','word_share','char_diff','tfidf_cosine']
X = train[feat_cols].fillna(0)
y = train['is_duplicate']
X_test = test[feat_cols].fillna(0)

# scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# quick stratified split for validation
X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42, stratify=y)

# train logistic regression
clf = LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced')
clf.fit(X_tr, y_tr)

# evaluate
val_preds = clf.predict_proba(X_val)[:,1]
print("Validation Log Loss:", log_loss(y_val, val_preds))
print("Validation AUC:", roc_auc_score(y_val, val_preds))

# predict test and save submission
test_preds = clf.predict_proba(X_test_scaled)[:,1]

# create submission
if 'test_id' in test.columns:
    sub_df = pd.DataFrame({'test_id': test['test_id'], 'is_duplicate': test_preds})
else:
    # fallback to sample_submission index
    sub_df = pd.DataFrame({'test_id': sub['test_id'], 'is_duplicate': test_preds})

sub_df.to_csv('/kaggle/working/submission_baseline.csv', index=False)
print("Saved submission_baseline.csv")



# (Cell 8) - prepare tokenizer and sequences
MAX_NUM_WORDS = 100000
MAX_LEN = 30   # adjust (30-40 usually good for Quora)
EMBEDDING_DIM = 300  # if using glove 300d; otherwise choose 100 or 200

texts = pd.concat([train['q1'], train['q2'], test['q1'], test['q2']]).astype(str).values
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(tqdm(texts, desc="Fitting tokenizer"))

# convert to sequences
train_q1_seq = tokenizer.texts_to_sequences(train['q1'].astype(str))
train_q2_seq = tokenizer.texts_to_sequences(train['q2'].astype(str))
test_q1_seq  = tokenizer.texts_to_sequences(test['q1'].astype(str))
test_q2_seq  = tokenizer.texts_to_sequences(test['q2'].astype(str))

X_tr_q1 = pad_sequences(train_q1_seq, maxlen=MAX_LEN, padding='post')
X_tr_q2 = pad_sequences(train_q2_seq, maxlen=MAX_LEN, padding='post')
X_test_q1 = pad_sequences(test_q1_seq, maxlen=MAX_LEN, padding='post')
X_test_q2 = pad_sequences(test_q2_seq, maxlen=MAX_LEN, padding='post')
y = train['is_duplicate'].values
word_index = tokenizer.word_index
print("Unique tokens:", len(word_index))



# (Cell 9) - try to load glove if available, else random
EMBEDDING_PATH = "/kaggle/input/glove-global/glove.6B.300d.txt"  # adjust this path if you uploaded glove
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

embedding_matrix = None
if os.path.exists(EMBEDDING_PATH):
    embeddings_index = {}
    with open(EMBEDDING_PATH, 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            values = line.rstrip().split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= num_words: continue
        vec = embeddings_index.get(word)
        if vec is not None and vec.shape[0] == EMBEDDING_DIM:
            embedding_matrix[i] = vec
    print("Loaded GloVe, embedding matrix shape:", embedding_matrix.shape)
else:
    print("GloVe file not found. Using random embeddings.")
    embedding_matrix = np.random.normal(size=(num_words, EMBEDDING_DIM)).astype(np.float32)



# (Cell 10) - define siamese model
def build_siamese(maxlen, num_words, embed_dim, embedding_matrix=None, trainable_emb=False):
    inp1 = Input(shape=(maxlen,), name='q1_input')
    inp2 = Input(shape=(maxlen,), name='q2_input')

    if embedding_matrix is not None:
        embedding_layer = Embedding(num_words, embed_dim, weights=[embedding_matrix], trainable=trainable_emb, name='emb')
    else:
        embedding_layer = Embedding(num_words, embed_dim, trainable=True, name='emb')

    e1 = embedding_layer(inp1)
    e2 = embedding_layer(inp2)

    shared_lstm = Bidirectional(LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2), name='shared_lstm')
    v1 = shared_lstm(e1)
    v2 = shared_lstm(e2)

    # merge using abs diff and multiply
    from tensorflow.keras import backend as K
    diff = Lambda(lambda x: K.abs(x[0] - x[1]))([v1, v2])
    mult = Lambda(lambda x: x[0] * x[1])([v1, v2])
    merged = tf.keras.layers.concatenate([diff, mult])

    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model([inp1, inp2], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_siamese(MAX_LEN, num_words, EMBEDDING_DIM, embedding_matrix, trainable_emb=False)
model.summary()



# (Cell 11) - training (use subset for quick runs)
SAMPLE = None   # set to e.g. 100000 to limit training rows while experimenting; else None for full
if SAMPLE:
    sample_idx = np.random.choice(len(y), SAMPLE, replace=False)
    X1 = X_tr_q1[sample_idx]; X2 = X_tr_q2[sample_idx]; y_sub = y[sample_idx]
else:
    X1 = X_tr_q1; X2 = X_tr_q2; y_sub = y

X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1, X2, y_sub, test_size=0.1, random_state=42, stratify=y_sub)

BATCH = 512
EPOCHS = 6

checkpoint_path = "/kaggle/working/siamese_lstm.h5"
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
]

model = build_siamese(MAX_LEN, num_words, EMBEDDING_DIM, embedding_matrix, trainable_emb=False)
history = model.fit([X1_train, X2_train], y_train, validation_data=([X1_val, X2_val], y_val),
                    epochs=EPOCHS, batch_size=BATCH, callbacks=callbacks, verbose=2)



# (Cell 11) - training (use subset for quick runs)
SAMPLE = None   # set to e.g. 100000 to limit training rows while experimenting; else None for full
if SAMPLE:
    sample_idx = np.random.choice(len(y), SAMPLE, replace=False)
    X1 = X_tr_q1[sample_idx]; X2 = X_tr_q2[sample_idx]; y_sub = y[sample_idx]
else:
    X1 = X_tr_q1; X2 = X_tr_q2; y_sub = y

# Train/validation split
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
    X1, X2, y_sub, test_size=0.1, random_state=42, stratify=y_sub
)

# Hyperparameters
BATCH = 512
EPOCHS = 6

# âœ… Use .keras format instead of .h5 (avoids pickle issue)
checkpoint_path = "/kaggle/working/siamese_lstm.keras"

callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
]

# Build Siamese LSTM model
model = build_siamese(
    MAX_LEN, num_words, EMBEDDING_DIM, embedding_matrix, trainable_emb=False
)

# Train the model
history = model.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_val, X2_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=2
)

# âœ… Reload the best saved model (optional, after training)
# from keras.models import load_model
# best_model = load_model("/kaggle/working/siamese_lstm.keras")



# (Cell 12) - validation logloss and test predictions
val_pred = model.predict([X1_val, X2_val], batch_size=1024, verbose=1).ravel()
print("Siamese LSTM val LogLoss:", log_loss(y_val, val_pred))
print("Siamese LSTM val AUC:", roc_auc_score(y_val, val_pred))

# predict test (may be slow)
test_pred = model.predict([X_test_q1, X_test_q2], batch_size=1024, verbose=1).ravel()

# prepare submission
if 'test_id' in test.columns:
    sub_out = pd.DataFrame({'test_id': test['test_id'], 'is_duplicate': test_pred})
else:
    sub_out = pd.DataFrame({'test_id': sub['test_id'], 'is_duplicate': test_pred})
sub_out.to_csv('/kaggle/working/submission_siamese_lstm.csv', index=False)
print("Saved submission_siamese_lstm.csv")



# Create submission
submission = pd.DataFrame({
    "test_id": test["test_id"],
    "is_duplicate": test_pred  # probabilities from model.predict()
})

submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved!")



# Evaluate on validation
val_pred = model.predict([X1_val, X2_val], batch_size=1024, verbose=1).ravel()
print("Validation LogLoss:", log_loss(y_val, val_pred))
print("Validation AUC:", roc_auc_score(y_val, val_pred))

# Predict test
test_pred = model.predict([X_test_q1, X_test_q2], batch_size=1024, verbose=1).ravel()

# Save submission
import pandas as pd
submission = pd.DataFrame({'test_id': test['test_id'], 'is_duplicate': test_pred})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("âœ… Submission saved!")



import os

# Check current files in working directory
print(os.listdir('/kaggle/working/'))

# Create zip of your model and tokenizer
!zip -r quora_model_files.zip /kaggle/working/siamese_lstm.keras /kaggle/working/tokenizer.pkl

# Move the zip to Kaggle output folder so you can download it
!mv quora_model_files.zip /kaggle/output/

print("âœ… Zip created! Go to the Output tab to download 'quora_model_files.zip'")



# === SAVE TOKENIZER AS .pkl FILE ===
import pickle
import pandas as pd

# 1. Find where your tokenizer is created in your notebook
# Look for lines like:
# tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
# tokenizer.fit_on_texts(list(questions))

# 2. Add this AFTER tokenizer creation:
print("Saving tokenizer as .pkl file...")
with open('tokenizer.pkl', 'wb') as f:  # 'wb' = write binary
    pickle.dump(tokenizer, f)
print("âœ… tokenizer.pkl saved!")

# 3. Verify it worked
import os
if os.path.exists('tokenizer.pkl'):
    file_size = os.path.getsize('tokenizer.pkl')
    print(f"ğŸ“� tokenizer.pkl created successfully! Size: {file_size} bytes")
else:
    print("â�Œ tokenizer.pkl not created!")

