import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# conifguration 
pd.set_option('display.max_columns', 200)

# Load Data
try:
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        print("Running on Kaggle, reading from Kaggle's input directory.")
        train_df = pd.read_csv('/kaggle/input/depi-r-3-competition-1/xy_train.csv')
        test_df = pd.read_csv('/kaggle/input/depi-r-3-competition-1/x_test.csv')
        sample_submission_df = pd.read_csv('/kaggle/input/depi-r-3-competition-1/sample_submission.csv')
    else:
        print("Running locally, reading from the current directory.")
        train_df = pd.read_csv('xy_train.csv')
        test_df = pd.read_csv('x_test.csv')
        sample_submission_df = pd.read_csv('sample_submission.csv')
except FileNotFoundError:
    print("File issue!!")

train_df.info()

#label distribution
print(train_df['label'].value_counts())

#let's see the distribution
plt.figure(figsize=(6,4))
sns.countplot(x='label', data = train_df)
plt.title("Fake vs Real")
plt.xlabel('label')
plt.ylabel('count')
plt.show()


train_df.head()


test_df.head()


sample_submission_df.head()


# since we found a label=2 we should remove it
train_df = train_df[train_df['label'] != 2]



print(train_df.info())
print(train_df['label'].value_counts())


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


X = train_df['text']
y = train_df['label']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")


from sklearn.linear_model import LogisticRegression

#create the pipeline for the Tfidf + Logistic Reg
logistic_pipeline = Pipeline(
    [
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('model', LogisticRegression(random_state=42, solver='liblinear'))
    ]
)

print("\nTraining the model....")
logistic_pipeline.fit(X_train, y_train)
print(".....Taining complete.....")

y_pred = logistic_pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Trial 1 | Logistic Regression | acc = {accuracy:.4f}")


from sklearn.naive_bayes import MultinomialNB

naive_bayes_pipeline = Pipeline(
    [
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('model', MultinomialNB())
    ]
)

print("\nTraining the model....")
naive_bayes_pipeline.fit(X_train, y_train)
print(".....Taining complete.....")

y_pred = naive_bayes_pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Trial 2 | Naive Bayes | acc = {accuracy:.4f}")



import xgboost as xgb

xgb_pipeline = Pipeline(
    [
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('model', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ]
)

print("\nTraining the model....")
xgb_pipeline.fit(X_train, y_train)
print(".....Taining complete.....")

y_pred = xgb_pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Trial 3 | XGBoost | acc = {accuracy:.4f}")


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping


# tokenizer
vocab_size = 10000
max_len = 150
embedding_dim = 128

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_val_padded = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')

print(f"Padded training data shape: {X_train_padded.shape}")
print(f"Padded validation data shape: {X_val_padded.shape}")


# model
lstm = Sequential(
    [
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        SpatialDropout1D(0.2),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ]
)

lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm.summary()


# training
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
print("\nTraining the model....")
history = lstm.fit(
    X_train_padded,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_val_padded, y_val),
    callbacks=[early_stopping],
    verbose=2
)

# evaluation
val_loss, val_accuracy = lstm.evaluate(X_val_padded, y_val)
print(f"\nTrial 4 | LSTM | val_acc = {val_accuracy:.4f}")

