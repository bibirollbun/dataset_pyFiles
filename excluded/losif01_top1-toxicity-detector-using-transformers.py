# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
test_labels = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
sample_submission = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')


print("Train shape:", train.shape);print("Test shape:", test.shape)
print(test_labels.head())
print(sample_submission.head())




train.head(100)


test.head()


sample_submission.head()


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train[label_cols].sum().sort_values(ascending=False).plot(kind='bar', figsize=(8,5), color='blue')
plt.title("Number of Comments per Toxic Class")
plt.ylabel("Count")
plt.xticks(rotation=60)
plt.show()


train['label_sum'] = train[label_cols].sum(axis=1)

train['label_sum'].value_counts().sort_index().plot(kind='bar', color='g')
plt.title("Multi-Label Distribution")
plt.xlabel("Number of Toxic Tags per Comment")
plt.ylabel("Number of Comments")
plt.show()


import random
for label in label_cols:
    print(f"\n\n Example of '{label}':\n")
    example = train[train[label] == 1]['comment_text'].iloc[random.randint(0,100)]
    print(example)


example = train[train[label] == 1]['comment_text']
example.head(100)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


train['clean_text'] = train['comment_text'].apply(clean_text)
test['clean_text'] = test['comment_text'].apply(clean_text)


train


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define label columns
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']



# Prepare test data (assuming 'test', 'test_labels', and 'train' DataFrames are available)
test_merged = test.merge(test_labels, on='id')
test_merged = test_merged[(test_merged[label_cols] != -1).all(axis=1)]
test_data = test_merged['clean_text']
test_labels = test_merged[label_cols].values

# Split training data
train_data, val_data, train_labels, val_labels = train_test_split(
    train['clean_text'], train[label_cols].values, test_size=0.2, random_state=42
)



# Tokenize the text
tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data)
train_sequences = tokenizer.texts_to_sequences(train_data)
val_sequences = tokenizer.texts_to_sequences(val_data)
test_sequences = tokenizer.texts_to_sequences(test_data)



# Pad sequences
max_length = 128
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')





# Define hyperparameters
vocab_size = 20000 + 1  # +1 for OOV token
embedding_dim = 128     # Size of the embedding vectors
d_model = embedding_dim
num_heads = 8           # Number of attention heads
ff_dim = 512            # Feed-forward network dimension
num_layers = 2          # Number of transformer layers
dropout_rate = 0.1      # Dropout rate

# Positional Encoding Layer
class PositionalEncoding(layers.Layer):
    def __init__(self, max_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_length, d_model)
    
    def positional_encoding(self, max_length, d_model):
        angle_rads = self.get_angles(np.arange(max_length)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        return inputs + pos_encoding

# Encoder Layer
def encoder_layer(d_model, num_heads, ff_dim, dropout_rate=0.1):
    inputs = layers.Input(shape=(None, d_model))
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(inputs, inputs)
    attention = layers.Dropout(dropout_rate)(attention)
    attention = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    ff = layers.Dense(ff_dim, activation='relu')(attention)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout_rate)(ff)
    outputs = layers.LayerNormalization(epsilon=1e-6)(attention + ff)
    return keras.Model(inputs=inputs, outputs=outputs)



# Build Transformer Model
inputs = layers.Input(shape=(max_length,), dtype='int32')
embedding = layers.Embedding(vocab_size, embedding_dim)(inputs)  # Randomly initialized embeddings
pos_encoding = PositionalEncoding(max_length, embedding_dim)(embedding)
x = pos_encoding
for _ in range(num_layers):
    x = encoder_layer(embedding_dim, num_heads, ff_dim, dropout_rate)(x)
pooled = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(6, activation='sigmoid')(pooled)  # 6 output classes for multi-label classification
model = keras.Model(inputs=inputs, outputs=outputs)




# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_padded, train_labels,
    validation_data=(val_padded, val_labels),
    epochs=10,
    batch_size=32,
    verbose=1
)



# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Evaluate on test set
test_loss, test_acc = model.evaluate(test_padded, test_labels, verbose=0)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

# Plot heatmap of label correlations
corr = train[label_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Toxicity Labels')
plt.show()


from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


test_sequences = tokenizer.texts_to_sequences(test['clean_text'])
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Predict probabilities
predictions = model.predict(test_padded)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'toxic': predictions[:,0],
    'severe_toxic': predictions[:,1],
    'obscene': predictions[:,2],
    'threat': predictions[:,3],
    'insult': predictions[:,4],
    'identity_hate': predictions[:,5]
})

# Save to CSV
submission.to_csv('submission.csv', index=False)


print("Submission shape:", submission.shape)
print(submission.head())

