#1
import pandas as pd
import numpy as np
import nltk
import re

#2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer

#3
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

#4
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Attention
from tensorflow.keras.models import Model




#1
# Load the dataset
df = pd.read_csv('/kaggle/input/asap-aes/training_set_rel3.tsv', sep='\t', encoding='ISO-8859-1')

# Select relevant columns
df = df[['essay', 'domain1_score']]

# Basic text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

df['clean_essay'] = df['essay'].apply(preprocess_text)

# Split into training and testing sets
X = df['clean_essay']
y = df['domain1_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#2
# Feature extraction
def extract_features(text_series):
    features = pd.DataFrame()
    features['essay_length'] = text_series.apply(len)
    features['word_count'] = text_series.apply(lambda x: len(x.split()))
    features['avg_word_length'] = features['essay_length'] / features['word_count']
    return features

X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_features, y_train)

# Predict and evaluate
y_pred_lr = lr_model.predict(X_test_features)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'Linear Regression MSE: {mse_lr}')


#3
# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
maxlen = 300
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=maxlen))
model.add(LSTM(64))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Predict and evaluate
y_pred_lstm = model.predict(X_test_pad)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
print(f'LSTM Model MSE: {mse_lstm}')



print(f'Linear Regression MSE: {mse_lr}')
print(f'LSTM Model MSE: {mse_lstm}')


embeddings_index = {}
with open('/kaggle/input/glove-100/glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs



embedding_dim = 100
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=maxlen,
                    trainable=False))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()



model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.1)



#saving the model

# from transformers import BertTokenizer

# model.save("/kaggle/input/bert_model/keras/default/1/BiLSTM.h5")
# model.save('/kaggle/working/bilstm_model.h5')
model.save_weights('/kaggle/working/bilstm.weights.h5')

# tokenizer.save_pretrained("bert_model")





#install library
!pip install transformers


from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



def tokenize_essays(essays, tokenizer, max_len):
    return tokenizer(
        essays.tolist(),
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

max_len = 512
train_encodings = tokenize_essays(X_train, tokenizer, max_len)
test_encodings = tokenize_essays(X_test, tokenizer, max_len)



from transformers import TFBertModel
from tf_keras.layers import Input, Dense
from tf_keras.models import Model
from tf_keras.optimizers import Adam
import tensorflow as tf
import os

# Set the environment variable before importing TensorFlow
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Define maximum sequence length
max_len = 512  # Adjust as needed

# Load the pre-trained BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Define input layers
input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

# Obtain BERT outputs
outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
cls_output = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token's output

# Add a regression head
out = Dense(1, activation='linear')(cls_output)

# Build and compile the model
model = Model(inputs=[input_ids, attention_mask], outputs=out)
model.compile(optimizer=Adam(learning_rate=2e-5), loss='mean_squared_error')
model.summary()



model.fit(
    x={'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']},
    y=y_train,
    validation_split=0.1,
    epochs=3,
    batch_size=8
)


from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertTokenizer
from tensorflow.keras.models import load_model



# Save the entire fine-tuned model
model.save("/kaggle/working/bert_regression_model")

# # Save the tokenizer
# tokenizer.save_pretrained("/kaggle/working/bert_regression_model")



import shutil

# Zips the entire folder into a .zip file
shutil.make_archive('/kaggle/working/bert_regression_model', 'zip', '/kaggle/working/bert_regression_model')



from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("/kaggle/working/bert_regression_model")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/kaggle/working/bert_regression_model")



y_pred = model.predict(X_test)


# Calculate Evaluation Metrics:

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")



from scipy.stats import pearsonr

corr, _ = pearsonr(y_test, y_pred.flatten())
print(f"Pearson Correlation: {corr}")



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred.flatten())
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs. Predicted Scores")
plt.show()



residuals = y_test - y_pred.flatten()
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residuals")
plt.title("Distribution of Residuals")
plt.show()


