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


import nltk
import xgboost as xgb
import tensorflow as tf
import tensorflow_hub as hub
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, GRU, Bidirectional, Dense, SimpleRNN, Dropout
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', None)


train_df=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")
test_df=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")
train_df.head()


train_df.shape


class_counts = train_df["target"].value_counts()
plt.figure(figsize=(5, 3))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Distribution of Target Classes")
plt.xticks([0, 1])
plt.show()


sentence_lengths = [len(sentence.split()) for sentence in train_df.question_text]
length_counts = Counter(sentence_lengths)
lengths = sorted(length_counts.keys())
counts = [length_counts[length] for length in lengths]
plt.figure(figsize=(10, 2))
plt.boxplot(lengths, vert=False, patch_artist=True, widths=0.5)
plt.xlabel('Number of Words in Sentence')
plt.yticks([])
plt.title('Sentence Length Distribution')
plt.show()


train_inputs, val_inputs, y_train, y_val = train_test_split(train_df.question_text, train_df.target, 
                                                                        test_size=0.3, random_state=42)


y_train = y_train.to_numpy()
y_val = y_val.to_numpy()


train_inputs.shape


classes, counts = np.unique(y_train, return_counts=True)
for c, count in zip(classes, counts):
    print(f"Class {c}: {count}")



english_stopwords = stopwords.words('english')
stemmer = SnowballStemmer(language='english')


def preprocess(doc):
    doc_stp=[[word for word in text.split(" ") if word.lower() not in english_stopwords]for text in doc]
    doc_stem=[[stemmer.stem(word) for word in words]for words in doc_stp]
    doc_final=[' '.join(words) for words in doc_stem]
    return doc_final
    
text_pipeline = Pipeline([
    ('preprocess', FunctionTransformer(preprocess)),
    ('vectorize', TfidfVectorizer(max_features=50000))
])


%%time
x_train=text_pipeline.fit_transform(train_inputs)
x_val=text_pipeline.transform(val_inputs)
x_test=text_pipeline.transform(test_df.question_text)


counter = Counter(y_train)
scale_pos_weight = counter[0]/counter[1]
model = xgb.XGBClassifier(
    objective='binary:logistic',
    early_stopping_rounds=20,
    eval_metric='logloss',
    n_estimators=500,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    device = "cuda"
)


%%time
evals = [(x_train, y_train), (x_val, y_val)]
model.fit(x_train, y_train, eval_set=evals, verbose=None)


results = model.evals_result()
plt.plot(results['validation_0']['logloss'], label='Train Loss')
plt.plot(results['validation_1']['logloss'], label='Validation Loss')
plt.legend(); plt.title("Logloss over Iterations")


%%time
y_pred=model.predict(x_val)
f1_score(y_val, y_pred)


%%time
vocab = 50000
max_len = 50

tokenizer = Tokenizer(num_words=vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(train_inputs)
sequences_train = tokenizer.texts_to_sequences(train_inputs)

padded_train_inputs = pad_sequences(sequences_train, maxlen=max_len, padding="post")

sequences_val = tokenizer.texts_to_sequences(val_inputs)
padded_val_inputs = pad_sequences(sequences_val, maxlen=max_len, padding="post")


embedd_size = 50
rnn_units=64
dropout_rate=0.3
model = Sequential([
    Embedding(vocab, embedd_size),
    SimpleRNN(units=rnn_units),
    Dropout(dropout_rate),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


%%time
model.fit(
    padded_train_inputs, y_train,
    epochs=15, batch_size=256, verbose=1)


y_pred=model.predict(padded_val_inputs)
y_pred_binary = (y_pred >= 0.5).astype(int)
f1_score(y_val, y_pred_binary)


embedd_size = 50
gru_units = 64
dropout_rate=0.3
GRU_model = Sequential([
    Embedding(vocab, embedd_size), 
    Bidirectional(GRU(gru_units)),
    Dropout(dropout_rate),
    Dense(1, activation='sigmoid')
])
GRU_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


%%time
GRU_model.fit(padded_train_inputs, y_train,
               epochs=15, batch_size=256, verbose=1)


y_pred=GRU_model.predict(padded_val_inputs)
y_pred_binary = (y_pred >= 0.5).astype(int)
f1_score(y_val, y_pred_binary)


from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam as TFAdam
from transformers.optimization_tf import AdamWeightDecay


model_checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


%%time
MAX_LEN=50
encoded_input = tokenizer(
    train_inputs.tolist(),
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_tensors="tf"
)


train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': encoded_input['input_ids'],
        'attention_mask': encoded_input['attention_mask']
    },
    y_train
)).batch(256).prefetch(tf.data.AUTOTUNE)


model = TFAutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=1)


model.compile(
    optimizer=AdamWeightDecay(learning_rate=5e-5),
    loss=BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


model.fit(train_dataset, epochs=1)


encoded_val_input = tokenizer(val_inputs.to_list(),
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_tensors="tf"
                         )


val_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': encoded_val_input['input_ids'],
        'attention_mask': encoded_val_input['attention_mask']
    },
    y_val
)).batch(256).prefetch(tf.data.AUTOTUNE)


y_pred=model.predict(val_dataset)



logits = y_pred.logits  
y_pred_binary = (logits >= 0).astype(int).flatten() 
f1 = f1_score(y_val, y_pred_binary)
f1




