import pandas as pd

train_df = pd.read_csv('/kaggle/input/kcvanguard-deep-learning-assignment/train-reviews-gmaps.csv')
print(train_df.head())


test_df = pd.read_csv('/kaggle/input/kcvanguard-deep-learning-assignment/test-review-gmaps-new.csv')
print(test_df.head())


import re
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

train_df['cleaned_reviews'] = train_df['reviews'].apply(preprocess_text)
test_df['cleaned_reviews'] = test_df['reviews'].apply(preprocess_text)

stop_words = set(stopwords.words('indonesian'))

tokenizer = Tokenizer(num_words=10000, lower=True, split=' ')
tokenizer.fit_on_texts(train_df['cleaned_reviews'])

X_train = tokenizer.texts_to_sequences(train_df['cleaned_reviews'])
X_train = pad_sequences(X_train, padding='post', maxlen=200) 

X_test = tokenizer.texts_to_sequences(test_df['cleaned_reviews'])
X_test = pad_sequences(X_test, padding='post', maxlen=200)

y_train = train_df['label'].map({'negative': 0, 'positive': 1}).values



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_shape=(2000,)))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train_split, y_train_split, epochs=4, batch_size=64, validation_data=(X_val_split, y_val_split))




y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  


submission_df = pd.DataFrame({
    'id': test_df['id'],
    'label': ['positive' if label == 1 else 'negative' for label in y_pred.flatten()]
})

submission_df.to_csv('CRP_AYUSH.csv', index=False)


