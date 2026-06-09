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


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer


# To prevent the error in the training section
# it forces TensorFlow to run all functions in eager execution 
# mode, meaning operations execute immediately without graph compilation.
tf.config.run_functions_eagerly(True)



# Reading the datasets
df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

display(df_train)
display(df_test)


len(df_train)


# Check properties information
df_train.describe()


# Check for null values - train set
df_train.isna().sum()


# Check for null values - test set
df_test.isna().sum()


# Dropping null values
df_train.dropna(inplace=True)


# Check for empty strings - train set
blanks_train = []

for i, tid, t, tst, snt in df_train.itertuples():
    if type(t) == str and t.isspace():
        blanks_train.append(i)
        
if len(blanks_train) > 0:
    print(len(blanks_train))
    df_train.drop(blanks_train, inplace=True)


# Print the first row in the training dataset
print("First Row in the Training Dataset:")
print(df_train.iloc[0])

print('-' * 30)

# Count the number of rows where 'text' is equal to 'selected_text'
matching_rows = len(df_train[df_train['text'] == df_train['selected_text']])
print(f"Number of Matching Rows: {matching_rows}")


# Check for empty strings - train set
blanks_train = []

for i, tid, t, tst, snt in df_train.itertuples():
    if type(t) == str and t.isspace():
        blanks_train.append(i)
        
if len(blanks_train) > 0:
    print(len(blanks_train))
    df_train.drop(blanks_train, inplace=True)


# Check for empty strings - test set
blanks_test = []

for i, tid, t, snt in df_test.itertuples():
    if type(t) == str and t.isspace():
        blanks_test.append(i)
        
if len(blanks_test) > 0:
    print(len(blanks_test))
    df_train.drop(blanks_test, inplace=True)


# Check for blanacement
df_train['sentiment'].value_counts()


# Performing T-Test
from scipy import stats

df_train['text_length'] = df_train['text'].apply(lambda t: len(str(t)))

positive_text_lengths = df_train[df_train['sentiment'] == 'positive']['text_length']
negative_text_lengths = df_train[df_train['sentiment'] == 'negative']['text_length']
neutral_text_lengths = df_train[df_train['sentiment'] == 'neutral']['text_length']

# Perform ANOVA test
f_statistic, p_value = stats.f_oneway(positive_text_lengths, negative_text_lengths, neutral_text_lengths)

# Print the results
print("ANOVA Test Results:")
print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

# Interpret the results
alpha = 0.05  # Set your significance level
if p_value < alpha:
    print("The means of at least two groups are significantly different.")
else:
    print("There is no significant difference in the means of the groups.")


# Preparing the data
max_len = 32
num_words = 500


# Tokenizing
tok = Tokenizer(num_words=num_words)
tok.fit_on_texts(df_train['text'])


# Defining X and y
X_train = df_train['text']
y_train = df_train['sentiment']

X_test = df_test['text']
y_test = df_test['sentiment']


X_train_mat = tok.texts_to_sequences(X_train)
X_test_mat = tok.texts_to_sequences(X_test)


# Pad sequences to the same length
X_train_padded = pad_sequences(X_train_mat, maxlen=max_len)
X_test_padded = pad_sequences(X_test_mat, maxlen=max_len)


y_train = pd.get_dummies(y_train, drop_first=True, dtype=int).to_numpy()
y_test = pd.get_dummies(y_test, drop_first=True, dtype=int).to_numpy()


# Create NN Architectures

# RNN
def create_rnn_model(units, dropout_rate):
    model = Sequential()
    model.add(Embedding(num_words, 128, input_length=max_len))
    model.add(SimpleRNN(units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='sigmoid'))
    return model

# LSTM
def create_lstm_model(units, dropout_rate):
    model = Sequential()
    model.add(Embedding(num_words, 32, input_length=max_len))
    model.add(LSTM(units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='sigmoid'))
    return model

# GRU
def create_gru_model(units, dropout_rate):
    model = Sequential()
    model.add(Embedding(num_words, 32, input_length=max_len))
    model.add(GRU(units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation='sigmoid'))
    return model


# Define hyperparameters
units = 128
dropout_rate = 0.3


# Create and compile the models

# RNN
rnn_model = create_rnn_model(units, dropout_rate)
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# LSTM
lstm_model = create_lstm_model(units, dropout_rate)
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# GRU
gru_model = create_gru_model(units, dropout_rate)
gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the models with early stopping
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)


# Train the models

# RNN
rnn_history = rnn_model.fit(X_train_padded, y_train, epochs=20, batch_size=128, 
                            validation_split=0.2, callbacks=[early_stopping])

# LSTM
lstm_history = lstm_model.fit(X_train_padded, y_train, epochs=20, batch_size=128, 
                              validation_split=0.2, callbacks=[early_stopping])

# GRU
gru_history = gru_model.fit(X_train_padded, y_train, epochs=20, batch_size=128, 
                            validation_split=0.2, callbacks=[early_stopping])

