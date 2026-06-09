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


import zipfile
import os
import pandas as pd
import tensorflow as tf
import numpy as np

# Unzip and load each file
with zipfile.ZipFile('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/temp')
train = pd.read_csv('/kaggle/temp/train.csv')

with zipfile.ZipFile('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/temp')
test = pd.read_csv('/kaggle/temp/test.csv')

with zipfile.ZipFile('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/temp')
test_labels = pd.read_csv('/kaggle/temp/test_labels.csv')

with zipfile.ZipFile('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/temp')
sample_submission = pd.read_csv('/kaggle/temp/sample_submission.csv')


# Check shapes
print('Train:', train.shape)
print('Test:', test.shape)

# Preview
train.head()



df = pd.read_csv(os.path.join('/kaggle/temp/train.csv'))


from tensorflow.keras.layers import TextVectorization


X = df['comment_text']
y = df[df.columns[2:]].values


df.columns


df[df.columns[2:]]


MAX_FEATURES = 2000000   # Number of words in the vocab


vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')


vectorizer.adapt(X.values)


vectorized_text = vectorizer(X.values)


# Tensorflow dara pipeline:

#MCSHBAP - map, chache, shuffle, batch, prefetch  from_tensor_slices, list_file
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # helps bottlenecks


train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding, InputLayer


model = Sequential()
# Add an InputLayer to specify the input shape:
model.add(InputLayer(shape=(1800,)))
# Create the embedding layer
model.add(Embedding(MAX_FEATURES+1, 32))
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='tanh')))
# Feature extractor Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# Final layer
model.add(Dense(6, activation='sigmoid'))


model.compile(loss='BinaryCrossentropy', optimizer='Adam')


model.summary()


history = model.fit(train, epochs=1, validation_data=val)


from matplotlib import pyplot as plt
plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()


batch = test.as_numpy_iterator().next


input_text = vectorizer('You freaking suck! I am going to hit you.')


from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# For Keras TextVectorization layer
res = model.predict(np.array([input_text]))[0]


(res > 0.5).astype(int)


# model.predict(np.array([input_text]))
model.predict(np.expand_dims(input_text,0))


batch_X, batch_y = test.as_numpy_iterator().next()


(model.predict(batch_X) > 0.5).astype(int)


res.shape


from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy


pre = Precision()
re = Recall()
acc = CategoricalAccuracy()


# Convert to list if dataset isn't too large
test_batches = list(test.as_numpy_iterator())

for i, batch in enumerate(test_batches):
    X_true, y_true = batch
    yhat = model.predict(X_true, verbose=0)
    
    # Update metrics
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)
    
    # Optional: Print progress
    if i % 10 == 0:
        print(f"Processed {i+1}/{len(test_batches)} batches")


print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')


!pip install gradio jinja2


import tensorflow as tf
import gradio as gr


model.save('toxicity.h5')


model = tf.keras.models.load_model('toxicity.h5')


input_str = vectorizer('hey i freaken hate you!')


res = model.predict(np.expand_dims(input_str,0))


res


def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text


import gradio as gr
from tensorflow.keras.preprocessing.sequence import pad_sequences

def score_comment(comment):
    # Preprocess and predict
    sequences = tokenizer.texts_to_sequences([comment])
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    results = model.predict(padded, verbose=0)[0]
    
    # Format results
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    return {label: float(score) for label, score in zip(labels, results)}

# Create interface
interface = gr.Interface(
    fn=score_comment,
    inputs=gr.Textbox(label="Enter your comment"),
    outputs=gr.Label(label="Toxicity Scores"),
    examples=[
        ["Have a nice day!"],
        ["I hate you so much!"]
    ]
)

# Launch without specific port to avoid conflicts
interface.launch()


interface.launch(share=True)

