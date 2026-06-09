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


# Import necessary libraries
import pandas as pd
import numpy as np
import zipfile
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


# Step 1: Extract the zipped dataset files
zip_files = ['/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip', 
             '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip',
             '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip',
             '/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip']

for zip_file in zip_files:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('.')


# Step 2: Load the extracted CSV files
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
test_labels = pd.read_csv('test_labels.csv')  # Optional, for local validation
sample_submission = pd.read_csv('sample_submission.csv')


# Step 3: Explore the data
print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print("Sample of training data:\n", train_data.head())
print("Columns in training data:", train_data.columns.tolist())


print("Missing values in train data:\n", train_data.isnull().sum())


# Step 4: Preprocess the text data
text_column = 'comment_text'
max_words = 25000 
max_len = 200     


# Initialize and fit the tokenizer on training data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data[text_column].values)


# Convert text to sequences of integers
X_train = tokenizer.texts_to_sequences(train_data[text_column].values)
X_test = tokenizer.texts_to_sequences(test_data[text_column].values)


# Pad sequences to ensure uniform length
X_train_padded = pad_sequences(X_train, maxlen=max_len)
X_test_padded = pad_sequences(X_test, maxlen=max_len)


# Step 5: Prepare the labels
label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = train_data[label_columns].values


y_test = test_labels[label_columns].values if 'toxic' in test_labels.columns else None


# Step 6: Visualize data distribution (example: count of toxic labels)
toxic_counts = train_data[label_columns].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
toxic_counts.plot(kind='bar')
plt.title('Distribution of Toxic Labels in Training Data')
plt.xlabel('Label Categories')
plt.ylabel('Count')
plt.show()


# Step 7: Save preprocessed data 
np.save('X_train_padded.npy', X_train_padded)
np.save('X_test_padded.npy', X_test_padded)
np.save('y_train.npy', y_train)
if y_test is not None:
    np.save('y_test.npy', y_test)

print("Data preprocessing complete. Shapes:")
print("X_train_padded shape:", X_train_padded.shape)
print("X_test_padded shape:", X_test_padded.shape)
print("y_train shape:", y_train.shape)


# Import necessary libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Load preprocessed data (assuming saved from part one)
X_train_padded = np.load('X_train_padded.npy')
X_test_padded = np.load('X_test_padded.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy') if os.path.exists('y_test.npy') else None

# Define model parameters
max_words = 10000
max_len = 200
embedding_dim = 100
num_classes = y_train.shape[1]

# Build the CNN model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))  # Sigmoid for multi-label

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test_padded)
y_pred_binary = (y_pred > 0.5).astype(int)



#  Plot training history
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Import necessary libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Load preprocessed data (assuming saved from part one)
X_train_padded = np.load('X_train_padded.npy')
X_test_padded = np.load('X_test_padded.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy') if os.path.exists('y_test.npy') else None

# Define model parameters
max_words = 10000
max_len = 200
embedding_dim = 100
num_classes = y_train.shape[1]

# Build the LSTM model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(LSTM(128, return_sequences=False))  # Single LSTM layer
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))  # Sigmoid for multi-label

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_padded, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test_padded)
y_pred_binary = (y_pred > 0.5).astype(int)



#  Plot training history
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()




