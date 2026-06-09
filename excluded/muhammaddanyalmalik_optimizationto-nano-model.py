# Three deep-learning models ensemble for Jigsaw rule_violation classification
# Models: MLP, BiLSTM, CNN (text)
# Usage: place this script in Kaggle notebook; it reads train.csv and test.csv from input dir

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Conv1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GlobalAveragePooling1D
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# ---------- Config ----------
DATA_DIR = '/kaggle/input/jigsaw-agile-community-rules'
MAX_WORDS = 50000    # vocabulary size
MAX_LEN = 200        # sequence length
EMB_DIM = 128
BATCH_SIZE = 128
EPOCHS = 100
RANDOM_STATE = 42

# ---------- Load data ----------
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Combine fields into a single text column
for df in [train, test]:
    df['text'] = df['body'].fillna('') + ' ' + df['rule'].fillna('') + ' ' + df['subreddit'].fillna('')

# Target
y = train['rule_violation'].astype(int)

# ---------- Tokenize ----------
tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True, oov_token='<OOV>')
tokenizer.fit_on_texts(train['text'].astype(str))

X = tokenizer.texts_to_sequences(train['text'].astype(str))
X_test = tokenizer.texts_to_sequences(test['text'].astype(str))

X = pad_sequences(X, maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=MAX_LEN, padding='post', truncating='post')

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

# ---------- Callbacks ----------
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# ---------- Model definitions ----------

# 1) Simple MLP (uses averaged embeddings)

def build_gru_cnn(max_words=50000, emb_dim=128, max_len=200):
    inp = Input(shape=(max_len,), name='gru_cnn_input')
    x = Embedding(input_dim=max_words, output_dim=emb_dim)(inp)
    
    # GRU branch
    gru = Bidirectional(LSTM(64, return_sequences=True))(x)
    gru = GlobalMaxPooling1D()(gru)
    
    # CNN branch
    cnn = Conv1D(128, 3, activation='relu')(x)
    cnn = GlobalMaxPooling1D()(cnn)
    
    # Combine GRU + CNN
    combined = Concatenate()([gru, cnn])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(64, activation='relu')(combined)
    out = Dense(1, activation='sigmoid')(combined)

    model = Model(inp, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# def build_mlp(max_words=30000, emb_dim=128, max_len=200):
#     inp = Input(shape=(max_len,), name='mlp_input')
#     x = Embedding(input_dim=max_words, output_dim=emb_dim)(inp)
#     x = GlobalAveragePooling1D()(x)  # ✅ Fixed: use a Keras layer instead of tf.reduce_mean
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     x = Dense(64, activation='relu')(x)
#     out = Dense(1, activation='sigmoid')(x)
#     model = Model(inp, out)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# 2) BiLSTM
def build_bilstm(max_words=MAX_WORDS, emb_dim=EMB_DIM, max_len=MAX_LEN):
    inp = Input(shape=(max_len,), name='bilstm_input')
    x = Embedding(input_dim=max_words, output_dim=emb_dim, input_length=max_len)(inp)
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out, name='BiLSTM')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3) CNN
def build_cnn(max_words=MAX_WORDS, emb_dim=EMB_DIM, max_len=MAX_LEN):
    inp = Input(shape=(max_len,), name='cnn_input')
    x = Embedding(input_dim=max_words, output_dim=emb_dim, input_length=max_len)(inp)
    x = Conv1D(filters=128, kernel_size=5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out, name='CNN')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---------- Build models ----------
print('Building models...')
mlp_model = build_gru_cnn()
bilstm_model = build_bilstm()
cnn_model = build_cnn()

# Print summaries (optional)
mlp_model.summary()
bilstm_model.summary()
cnn_model.summary()

# ---------- Train models ----------
print('\nTraining MLP...')
mlp_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, rlr], verbose=1)

print('\nTraining BiLSTM...')
bilstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, rlr], verbose=1)

print('\nTraining CNN...')
cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, rlr], verbose=1)

# ---------- Ensemble on validation set (average probabilities) ----------
print('\nPredicting on validation set and ensembling...')
val_p1 = mlp_model.predict(X_val, batch_size=64)
val_p2 = bilstm_model.predict(X_val, batch_size=64)
val_p3 = cnn_model.predict(X_val, batch_size=64)

val_avg = (val_p1 + val_p2 + val_p3) / 3.0
val_pred = (val_avg > 0.5).astype(int).reshape(-1)

# Metrics
acc = accuracy_score(y_val, val_pred)
f1 = f1_score(y_val, val_pred)
prec = precision_score(y_val, val_pred)
rec = recall_score(y_val, val_pred)

print('\nEnsemble Validation Metrics:')
print(f'  Accuracy: {acc:.4f}')
print(f'  F1 score: {f1:.4f}')
print(f'  Precision: {prec:.4f}')
print(f'  Recall: {rec:.4f}')
print('\nClassification report:\n', classification_report(y_val, val_pred))
print('\nConfusion matrix:\n', confusion_matrix(y_val, val_pred))

# ---------- Retrain on full training data (optional) ----------
print('\nRetraining models on full training data (this may take time)...')
mlp_model.fit(X, y, epochs=5, batch_size=BATCH_SIZE, verbose=1)
bilstm_model.fit(X, y, epochs=5, batch_size=BATCH_SIZE, verbose=1)
cnn_model.fit(X, y, epochs=5, batch_size=BATCH_SIZE, verbose=1)

# ---------- Predict on test and create submission ----------
print('\nPredicting on test set and ensembling...')
test_p1 = mlp_model.predict(X_test, batch_size=64)
test_p2 = bilstm_model.predict(X_test, batch_size=64)
test_p3 = cnn_model.predict(X_test, batch_size=64)

test_avg = (test_p1 + test_p2 + test_p3) / 3.0
test_pred = (test_avg > 0.5).astype(int).reshape(-1)

submission = pd.DataFrame({'row_id': test['row_id'], 'rule_violation': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print('\n✅ submission.csv ready at /kaggle/working/submission.csv')




