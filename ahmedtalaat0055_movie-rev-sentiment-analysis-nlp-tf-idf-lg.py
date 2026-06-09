import pandas as pd

train_df = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/test.csv')

print("Train DataFrame head:")
display(train_df.head())

print("\nTest DataFrame head:")
display(test_df.head())


display(train_df.head())
display(test_df.head())
train_df.info()
test_df.info()
display(train_df.isnull().sum())
display(test_df.isnull().sum())


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_words = 10000  
max_len = 100     

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['review'])

train_sequences = tokenizer.texts_to_sequences(train_df['review'])
test_sequences = tokenizer.texts_to_sequences(test_df['review'])

X_train_padded = pad_sequences(train_sequences, maxlen=max_len, truncating='post', padding='post')
X_test_padded = pad_sequences(test_sequences, maxlen=max_len, truncating='post', padding='post')

y_train = np.array(train_df['sentiment'])

print("Original training review example:", train_df['review'].iloc[0])
print("Padded training sequence example:", X_train_padded[0])
print("Original training sentiment label:", y_train[0])

print("\nOriginal test review example:", test_df['review'].iloc[0])
print("Padded test sequence example:", X_test_padded[0])


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

embedding_dim = 16  
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid'))
model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


epochs = 15 # Choosing a reasonable number of epochs
history = model.fit(X_train_padded, y_train, epochs=epochs, validation_split=0.2)


print("Final validation loss:", history.history['val_loss'][-1])
print("Final validation accuracy:", history.history['val_accuracy'][-1])


# Make predictions on the padded test data
predictions = model.predict(X_test_padded)

# Convert probabilities to binary predictions (0 or 1) using a threshold of 0.5
binary_predictions = (predictions > 0.5).astype(int)

print("Example raw prediction (probability):", predictions[0])
print("Example binary prediction:", binary_predictions[0])


submission_df = pd.DataFrame()
submission_df["id"] = test_df["id"]
submission_df["sentiment"] = binary_predictions
output_path = "/kaggle/working/submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission file saved as: {output_path}")
display(submission_df.head())

