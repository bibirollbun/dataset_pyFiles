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
import zipfile
import os
from collections import defaultdict
import numpy as np
import re

# Define paths to the ZIP files (replace with your actual paths)
train_zip_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip'
test_zip_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip'
test_labels_zip_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip'

# Extract each ZIP file into a directory
extract_dir = './extracted_data'
os.makedirs(extract_dir, exist_ok=True)  # Create directory if it doesn't exist

# Extract train.csv
with zipfile.ZipFile(train_zip_path, 'r') as z:
    z.extractall(extract_dir)

# Extract test.csv
with zipfile.ZipFile(test_zip_path, 'r') as z:
    z.extractall(extract_dir)

# Extract test_labels.csv
with zipfile.ZipFile(test_labels_zip_path, 'r') as z:
    z.extractall(extract_dir)

# Load datasets from extracted files
train_df = pd.read_csv(os.path.join(extract_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(extract_dir, 'test.csv'))
test_labels_df = pd.read_csv(os.path.join(extract_dir, 'test_labels.csv'))

# Verify data
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
print("Label distribution:\n", train_df[label_cols].sum())


train_texts = train_df['comment_text'].tolist()
train_labels = train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

test_texts = test_df['comment_text'].tolist()
test_labels = test_labels_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values



vocabulary = {"<OOV>": 1}
word_counts = defaultdict(int)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and punctuation (keeping apostrophes)
    text = re.sub(r"[^a-zA-Z0-9']", " ", text)
    
    # Split into tokens
    tokens = text.split()
    
    return tokens

# Process each text and build vocabulary
for text in train_texts:
    tokens = preprocess_text(text)
    for token in tokens:
        word_counts[token] += 1

# Sort words by frequency and add to vocabulary
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
for idx, (word, count) in enumerate(sorted_words, start=2):  # Start from 2 (since 0 is padding, 1 is OOV)
    vocabulary[word] = idx


def text_to_sequence(text, vocabulary, max_length=100):
    tokens = preprocess_text(text)
    sequence = []
    for token in tokens:
        if token in vocabulary:
            sequence.append(vocabulary[token])
        else:
            sequence.append(vocabulary["<OOV>"])
    
    # Pad or truncate sequence to max_length
    if len(sequence) < max_length:
        sequence += [0] * (max_length - len(sequence))  # Pad with zeros
    else:
        sequence = sequence[:max_length]  # Truncate
    
    return sequence

max_length = 30  # Fixed sequence length
train_sequences = []
for text in train_texts:
    sequence = text_to_sequence(text, vocabulary, max_length)
    train_sequences.append(sequence)

test_sequences = []
for text in test_texts:
    sequence = text_to_sequence(text, vocabulary, max_length)
    test_sequences.append(sequence)
    
print(len(train_sequences))


train_sequences = np.array(train_sequences)
test_sequences = np.array(test_sequences)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
print("Training data shape:", train_sequences.shape)
print("Training labels shape:", train_labels.shape)
print("Test data shape:", test_sequences.shape)
print("Test labels shape:", test_labels.shape)
print("Vocabulary size:", len(vocabulary))


import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import precision_score, recall_score, f1_score

# Define the model
def build_model(vocabulary_size, max_length):
    model = models.Sequential()
    
    # Embedding layer
    model.add(layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=100,  # Dimension of the embedding vectors
        input_length=max_length,
        mask_zero=True
    ))
    
    # LSTM layer
    model.add(layers.LSTM(units=256, return_sequences=False, use_cudnn=False))
    
    # Dense layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.45))  # To prevent overfitting
    
    # Output layer with sigmoid activation
    model.add(layers.Dense(6, activation='sigmoid'))  # 6 labels
    
    return model

# Compile the model
def compile_model(model, learning_rate=0.00075):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model

# Function to evaluate the model
def evaluate_model(model, test_sequences, test_labels):
    # Making predictions
    y_pred = model.predict(test_sequences)
    
    # Rounding the predictions
    y_pred = (y_pred > 0.75).astype(int)
    
    # Calculating metrics
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(test_labels, y_pred)
    print("Accuracy:", accuracy.result().numpy())
    
    precision = precision_score(test_labels, y_pred, average='macro')
    print("Precision:", precision)
    
    recall = recall_score(test_labels, y_pred, average='macro')
    print("Recall:", recall)
    
    f1 = f1_score(test_labels, y_pred, average='macro')
    print("F1-score:", f1)

# Build and compile the model
model = build_model(len(vocabulary), max_length=100)
model = compile_model(model)


# Training the model
def train_model(model, train_sequences, train_labels, validation_data, epochs=10, batch_size=32):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_sequences,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=[early_stopping]
    )
    return model, history

model, history = train_model(
    model,
    train_sequences,
    train_labels,
    validation_data=(test_sequences, test_labels),
    epochs=10,  # Adjust as needed
    batch_size=32  # Adjust as needed
)


# Make predictions
sample_predictions = model.predict(test_sequences)


sample_predictions = (sample_predictions > 0.75).astype(int)
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
submission_df = pd.DataFrame(sample_predictions, columns=label_columns)

# Combine with id
submission_df = pd.concat([test_df['id'], submission_df], axis=1)

submission_df


submission_df.to_csv('submission.csv', index=None)

