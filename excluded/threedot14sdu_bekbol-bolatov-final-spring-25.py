# Step 1: Install necessary libraries
# !pip install nltk

# Step 2: Import required libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 3: Load dataset
df = pd.read_csv('/kaggle/input/d/uciml/sms-spam-collection-dataset/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]  # Select relevant columns
df.columns = ['label', 'message']  # Rename columns

# Step 4: Data cleaning and preprocessing
ps = PorterStemmer()
corpus = []

for i in range(len(df)):
    # Remove non-alphabetic characters
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    # Convert to lowercase
    review = review.lower()
    # Split into words
    review = review.split()
    # Remove stopwords and stem
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    # Join words back to sentence
    review = ' '.join(review)
    corpus.append(review)

# Step 5: Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 6: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(corpus).toarray()
y = df['label'].values

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Step 9: Predictions and Evaluation
y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Step 10: Tokenization and Padding for Keras Models

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# Parameters for tokenization and padding
vocab_size = 10000
max_length = 100
embedding_dim = 100

# Tokenize the corpus
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(corpus) # Use the cleaned corpus from Stage I
sequences = tokenizer.texts_to_sequences(corpus)

# Pad the sequences
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Split the padded sequences for training and testing
X_train_padded, X_test_padded, y_train_dl, y_test_dl = train_test_split(padded_sequences, df['label'].values, test_size=0.2, random_state=42)

print("Stage II - Part 1: Keras Embedding + 1D CNN Model")
print("Padded sequences shape:", padded_sequences.shape)
print("X_train_padded shape:", X_train_padded.shape)
print("X_test_padded shape:", X_test_padded.shape)


# Step 11: Build the Keras Embedding + 1D CNN Model

model_cnn = Sequential()
model_cnn.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model_cnn.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model_cnn.add(GlobalMaxPooling1D()) # Use GlobalMaxPooling to reduce dimensions
model_cnn.add(Dense(64, activation='relu'))
model_cnn.add(Dropout(0.5)) # Add dropout for regularization
model_cnn.add(Dense(1, activation='sigmoid')) # Output layer for binary classification

# Compile the model
model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_cnn.summary()

# Step 12: Train the 1D CNN Model

epochs = 10 # You can experiment with the number of epochs
batch_size = 32

history_cnn = model_cnn.fit(X_train_padded, y_train_dl,
                            epochs=epochs,
                            validation_data=(X_test_padded, y_test_dl),
                            batch_size=batch_size,
                            verbose=1)


# Step 13: Evaluate the 1D CNN Model

loss_cnn, accuracy_cnn = model_cnn.evaluate(X_test_padded, y_test_dl, verbose=0)
print(f"\n1D CNN Model Accuracy: {accuracy_cnn:.4f}")

# Predictions for Confusion Matrix
y_pred_cnn = (model_cnn.predict(X_test_padded) > 0.5).astype("int32")

print("\n1D CNN Model Confusion Matrix:")
print(confusion_matrix(y_test_dl, y_pred_cnn))
print("\n1D CNN Model Classification Report:")
print(classification_report(y_test_dl, y_pred_cnn))


# Step 14: Train Word2Vec Model

import gensim
from gensim.models import Word2Vec

# Tokenize the corpus into lists of words for Word2Vec training
corpus_w2v = [sentence.split() for sentence in corpus]

# Train the Word2Vec model
# vector_size: Dimension of word vectors
# window: Maximum distance between the current and predicted word within a sentence
# min_count: Ignores all words with total frequency lower than this
# workers: Use these many worker threads to train the model
w2v_model = Word2Vec(sentences=corpus_w2v, vector_size=100, window=5, min_count=1, workers=4)

print("\nStage II - Part 2: Word2Vec Embedding + LSTM Model")
print(f"Word2Vec model trained with {len(w2v_model.wv)} unique words.")

# Step 15: Create Embedding Matrix for Keras

# Initialize embedding matrix with zeros
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Fill embedding matrix with vectors from Word2Vec model
for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
        # Words not in Word2Vec model will remain zeros

print("Embedding matrix shape:", embedding_matrix.shape)


# Step 16: Build the Word2Vec Embedding + LSTM Model

from tensorflow.keras.layers import LSTM, GRU # Or SimpleRNN

model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=vocab_size, # Use the same vocab_size as before
                          output_dim=embedding_dim, # Use the same embedding_dim as Word2Vec
                          weights=[embedding_matrix], # Initialize with Word2Vec weights
                          trainable=False)) # Set to False to keep embeddings fixed initially
                                           # You can set to True for fine-tuning
# Choose one of LSTM, GRU, or SimpleRNN
model_lstm.add(LSTM(128, return_sequences=False)) # Set return_sequences=True if stacking recurrent layers
# model_lstm.add(GRU(128, return_sequences=False))
# model_lstm.add(SimpleRNN(128, return_sequences=False))

model_lstm.add(Dense(64, activation='relu'))
model_lstm.add(Dropout(0.5))
model_lstm.add(Dense(1, activation='sigmoid'))

# Compile the model
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_lstm.summary()

# Step 17: Train the LSTM Model

epochs = 10 # You can experiment with the number of epochs
batch_size = 32

history_lstm = model_lstm.fit(X_train_padded, y_train_dl, # Use the same padded sequences
                              epochs=epochs,
                              validation_data=(X_test_padded, y_test_dl),
                              batch_size=batch_size,
                              verbose=1)


# Step 18: Evaluate the LSTM Model

loss_lstm, accuracy_lstm = model_lstm.evaluate(X_test_padded, y_test_dl, verbose=0)
print(f"\nLSTM Model Accuracy: {accuracy_lstm:.4f}")

# Predictions for Confusion Matrix
y_pred_lstm = (model_lstm.predict(X_test_padded) > 0.5).astype("int32")

print("\nLSTM Model Confusion Matrix:")
print(confusion_matrix(y_test_dl, y_pred_lstm))
print("\nLSTM Model Classification Report:")
print(classification_report(y_test_dl, y_pred_lstm))

