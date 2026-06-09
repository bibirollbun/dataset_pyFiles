import nltk
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')


data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
data.head()


# Drop unique identifier and useless column
data = data.drop(['textID', 'selected_text'], axis=1)
data.head()


data.isna().sum()


data.duplicated().sum()


# Drop NaN row
data = data.dropna()


# See Distribution of neutral, negative and positive
plt.figure(figsize=(8,6))
sns.countplot(
    data=data,
    x='sentiment'
)
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiments')
plt.ylabel('Count')
plt.show()


# Initialize pre-processing tool
eng_stopwords = stopwords.words('english')
punctuations = string.punctuation
wnl = WordNetLemmatizer()


def preprocess(text):
    words = word_tokenize(text.lower())
    words_tag = pos_tag(words)

    words = [wnl.lemmatize(word, get_tag(tag)) for word, tag in words_tag if word not in eng_stopwords and word not in punctuations and word.isalpha()]
    return ' '.join(words)

def get_tag(tag):
    if tag == 'JJ':
        return 'a'

    if tag in ['NN', 'VB', 'RB']:
        return tag[0].lower()

    return 'n'


text = [preprocess(text) for text in data['text']]


le = LabelEncoder()
data['sentiment'] = le.fit_transform(data['sentiment'])
num_label = len(le.classes_)
label_names = le.classes_.tolist()


tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(text)

sequences = tokenizer.texts_to_sequences(text)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=15)

y = data['sentiment']


X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)


model = models.Sequential([
    layers.Input(shape=(15,)),
    layers.Embedding(input_dim=1000, output_dim=16),
    Bidirectional(layers.LSTM(256)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_label, activation='softmax')
])

model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)


EPOCHS = 10
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
)


test_result = model.evaluate(X_test, y_test, return_dict=True)
accuracy = test_result['accuracy']
print(f'Model accuracy on test set: {accuracy*100:.2f}%')


metrics = history.history
plt.figure(figsize=(16, 4))
plt.plot(history.epoch, np.array(metrics['accuracy']) * 100, np.array(metrics['val_accuracy']) * 100)
plt.xlabel('Epochs')
plt.ylabel('Accuracy [%]')
plt.legend(['accuracy', 'val_accuracy'])
plt.show()


# Predictions
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
submission_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

sequences = tokenizer.texts_to_sequences(test_df['text'].tolist())
padded_sequences = pad_sequences(sequences, padding='post', maxlen=15)

predictions = model.predict(padded_sequences, batch_size=64, verbose=1)
pred_indices = np.argmax(predictions, axis=1)
pred_labels = [label_names[idx] for idx in pred_indices]

submission_df['selected_text'] = pred_labels


submission_df.to_csv('submission.csv', index=False)

