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


data=pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Train.csv')


data.head()


import nltk
from nltk.corpus import stopwords

text=data['Text'].apply(lambda x: x.split())
labels=data['Category']

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

text=text.apply(lambda x : ' '.join([word for word in x if word not in stop_words]))


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
text_train, text_test, labels_train, labels_test = train_test_split(text, labels, test_size=0.2, random_state=42)


import tensorflow as tf

vectorized_layer=tf.keras.layers.TextVectorization(ragged=True,max_tokens=20000)
vectorized_layer.adapt(text_train)


train_sequence=vectorized_layer(text_train)
test_sequence=vectorized_layer(text_test)

train_padded=tf.keras.utils.pad_sequences(
    train_sequence.numpy(),
    maxlen=350,
    padding='pre',
    truncating='pre',
)
test_padded=tf.keras.utils.pad_sequences(
    test_sequence.numpy(),
    maxlen=350,
    padding='pre',
    truncating='pre',
)


label_vectorizer=tf.keras.layers.StringLookup(num_oov_indices=0)
label_vectorizer.adapt(labels_train)


labels_train=label_vectorizer(labels_train)
labels_test=label_vectorizer(labels_test)


train_padded=tf.data.Dataset.from_tensor_slices(train_padded)
test_padded=tf.data.Dataset.from_tensor_slices(test_padded)

labels_train=tf.data.Dataset.from_tensor_slices(labels_train)
labels_test=tf.data.Dataset.from_tensor_slices(labels_test)

train_dataset=tf.data.Dataset.zip(train_padded,labels_train)
test_dataset=tf.data.Dataset.zip(test_padded,labels_test)


train_dataset=train_dataset.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE).batch(32)
test_dataset=test_dataset.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE).batch(32)


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(350,)),  # Input layer
    tf.keras.layers.Embedding(20000, 128),  # Embedding layer

    # Conv1D layer
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
    # First Bidirectional LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.4),
    # Second Bidirectional LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.6),
    # Dense layer
    tf.keras.layers.Dense(128, activation='relu'),
    # Output layer
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()


es=tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    verbose=1,
    restore_best_weights=True,
)


history = model.fit(train_dataset, epochs=30, validation_data=test_dataset,callbacks=[es,])


model.evaluate(test_dataset)


ranked=pd.read_csv('/kaggle/input/learn-ai-bbc/BBC News Test.csv')
ranked.head()


text=ranked['Text'].apply(lambda x: x.split())
text=text.apply(lambda x : ' '.join([word for word in x if word not in stop_words]))
ArticleId=ranked['ArticleId']


text_sequence=vectorized_layer(text)
train_padded=tf.keras.utils.pad_sequences(
    text_sequence.numpy(),
    maxlen=350,
    padding='pre',
    truncating='pre',
)


predictions=model.predict(train_padded)
predicted_categories = np.argmax(predictions, axis=1)
predicted_labels = [label_vectorizer.get_vocabulary()[i] for i in predicted_categories]

submission_df = pd.DataFrame({'ArticleId': ArticleId, 'Category': predicted_labels})
submission_df.to_csv('submission.csv', index=False)



submission_df.head(15)

