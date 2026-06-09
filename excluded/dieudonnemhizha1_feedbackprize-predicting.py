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


MODEL_PATH = '../input/huggingface-bert-variants/bert-base-cased/bert-base-cased' 


!pip install pyforest


from pyforest import *
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import time
import datetime
from scipy import sparse
import datasets, transformers
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForMaskedLM
os.environ['WANDB_DISABLED'] = 'true'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from transformers import TFBertModel


AUTO = tf.data.experimental.AUTOTUNE
EPOCHS = 20
BATCH_SIZE = 32
MAX_LEN = 128


train_df = pd.read_csv('/kaggle/input/feedback-prize-effectiveness/train.csv')


test_df = pd.read_csv('/kaggle/input/feedback-prize-effectiveness/test.csv')


sample = pd.read_csv('/kaggle/input/feedback-prize-effectiveness/sample_submission.csv')


train_df.head()


test_df.head()


sample.head()


train_df.shape


test_df.shape


sample.shape


def cleanup_text(text):
    words = re.sub(pattern = '[^a-zA-Z]', repl = ' ', string = text)
    words = words.lower()
    return words
cleanup_text('Every Mountain Speaks Different Ways!')


text_preprocessed = train_df['discourse_text'].apply(cleanup_text)
print(text_preprocessed)


train_df['text_preprocessed'] = text_preprocessed
display(train_df.head())


def bert_encode(texts, tokenizer, max_len=MAX_LEN):
    input_ids = []
    token_type_ids = []
    attention_mask = []
    
    for text in texts:
        token = tokenizer(text, max_length=max_len, truncation=True, padding='max_length', add_special_tokens=True)
        input_ids.append(token['input_ids'])
        token_type_ids.append(token['token_type_ids'])
        attention_mask.append(token['attention_mask'])
    return np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)


tokenizer = transformers.BertTokenizer.from_pretrained('../input/huggingface-bert-variants/distilbert-base-cased/distilbert-base-cased')
tokenizer.save_pretrained('.')


sep = tokenizer.sep_token
sep


train_df['inputs'] = train_df.discourse_type + sep + train_df.text_preprocessed
train_df.head()


bin_map = {'discourse_effectiveness': {'Ineffective': 0, 'Adequate': 1, 'Effective': 2}}
train_df = train_df.replace(bin_map)


train_df.head()


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_df['inputs'], train_df['discourse_effectiveness'], test_size=0.2, random_state=12)


X_train = bert_encode(X_train.astype(str), tokenizer)
X_valid = bert_encode(X_valid.astype(str), tokenizer)
y_train = y_train.values
y_valid = y_valid.values


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)


def build_model(bert_model, max_len=MAX_LEN):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    token_type_ids = Input(shape=(max_len,), dtype=tf.int32, name='token_type_ids')
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
    
    sequence_output = bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
    clf_output = sequence_output[:, 0, :]
    clf_output = Dropout(.1)(clf_output)
    out = Dense(3, activation='softmax')(clf_output)
    model = Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=out)
    model.compile(Adam(lr=1e-5), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


%%time
transformer_layer = (TFBertModel.from_pretrained('../input/huggingface-bert-variants/distilbert-base-cased/distilbert-base-cased'))
model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()


from tensorflow.keras.utils import plot_model
plot_model(model)


train_history = model.fit(
        train_dataset,
        steps_per_epoch = 200,
        validation_data = valid_dataset,
        epochs = EPOCHS
)


test_preprocessed = test_df['discourse_text'].apply(cleanup_text)
test_preprocessed


test_df['text_preprocessed'] = test_preprocessed
test_df.head()


test_df['processed'] = test_df.discourse_type + sep + test_df.text_preprocessed


test_processed = bert_encode(test_df.processed.astype(str), tokenizer)


preds = model.predict(test_processed, verbose = 1)
preds


sample['Ineffective'] = preds[:,0]
sample['Adequate'] = preds[:,1]
sample['Effective'] = preds[:,2]
sample.sample(2)


sample.to_csv("submission.csv", index=False)
print('Success')




