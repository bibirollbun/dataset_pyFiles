import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
snowball = SnowballStemmer(language='english')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as XGB
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import transformers
from transformers import DistilBertTokenizer, TFDistilBertModel
from transformers import TFDistilBertForSequenceClassification, DistilBertConfig
from numba import cuda
from nltk.tokenize import word_tokenize
import tensorflow_text as tftext
import tensorflow_hub as tfhub
import keras_nlp
import keras
from keras import layers


# define light‐augmentation & cleaning helpers

import random
import re

def random_deletion(sentence, p=0.05):
    """Randomly drop each word with probability p, keep at least one."""
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new = [w for w in words if random.random() > p]
    return " ".join(new) if new else random.choice(words)

def typo_noise(sentence, p=0.01):
    """Randomly swap adjacent characters with probability p."""
    s = list(sentence)
    for i in range(len(s)-1):
        if random.random() < p:
            s[i], s[i+1] = s[i+1], s[i]
    return "".join(s)

def clean_text(text):
    """Strip HTML tags and collapse whitespace."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


data = pd.read_csv('/kaggle/input/daigt-proper-train-dataset/train_drcat_01.csv')
data


data = data.drop(columns=['source','fold'])


data_classes = data.groupby('label').count()['text']
print(data_classes)
plt.title('class distribution in dataset')
plt.pie(data_classes , labels = ['0','1'] ,autopct='%0.0f%%')


data_0 = data[data['label']==0].iloc[0:7263,:]
data_1 = data[data['label']==1]


final_data = pd.concat([data_0,data_1])
final_data


plt.title('new class distribution of the dataset')
plt.pie(final_data.groupby('label').count()['text'],labels=['0','1'],autopct='%0.0f%%')


final_data['text'].index = np.arange(0,final_data.shape[0])
final_data['text']


x_train, x_test ,y_train , y_test = train_test_split(final_data.iloc[:,0:2], final_data['label'],test_size=0.2)


import random

AUG_N = 1000    
aug_texts, aug_labels = [], []

for txt, lbl in zip(
    x_train['text'].iloc[:AUG_N],
    y_train.iloc[:AUG_N]
):
    rd = random_deletion(txt, p=0.05)  # milder deletion
    ty = typo_noise(rd, p=0.01)        # milder typos
    aug_texts.append(ty)
    aug_labels.append(lbl)

aug_df = pd.DataFrame({'text': aug_texts, 'label': aug_labels})
orig_len = len(x_train)

# concatenate and shuffle so noise is mixed in
x_train = pd.concat([x_train, aug_df], ignore_index=True) \
           .sample(frac=1, random_state=42) \
           .reset_index(drop=True)
y_train = x_train['label']

print(f"Added {len(aug_df)} samples → train size {orig_len} → {len(x_train)}")


# : clean whitespace/HTML artifacts ──
x_train['text'] = x_train['text'].apply(clean_text)
x_test['text']  = x_test['text'].apply(clean_text)


len_train = []

for i in range(x_train.shape[0]):
    len_train.append(len(x_train.iloc[i,0]))
print('average characters per essay are ' , np.mean(len_train))


x_train.groupby('label').count()


# simplified LR schedule + full fine-tuning

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# 1) Preprocessor 
preprocessor1 = keras_nlp.models.DistilBertPreprocessor.from_preset(
    "distil_bert_base_en_uncased",
    sequence_length=512,
)

# 2) Classifier 
classifier = keras_nlp.models.DistilBertClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=1,
)

# 3) Unfreeze the entire backbone
classifier.backbone.trainable = True

# 4) LR schedule: Exponential decay from 3e-5 → 1e-7 over total_steps
BATCH_SIZE = 8
EPOCHS     = 6
steps_per_epoch = len(x_train) // BATCH_SIZE
total_steps     = steps_per_epoch * EPOCHS

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-5,
    decay_steps=total_steps,
    decay_rate=0.95,    # decay by 5% every total_steps
    staircase=False,
    name="exp_decay",
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# 5) Compile
classifier.compile(
    loss="binary_crossentropy",
    optimizer=optimizer,
    jit_compile=True,
    metrics=["accuracy", "AUC"],
)

# 6) EarlyStopping
early_stop = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=2,
    restore_best_weights=True,
)

# 7) Train
history = classifier.fit(
    x=x_train["text"].to_list(),
    y=x_train["label"],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test["text"].to_list(), y_test),
    callbacks=[early_stop],
)


final_test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
final_test


pred = classifier.predict(final_test['text'].to_list())
pred


test_result = pred[:,0]
test_result


final_submission = pd.DataFrame(final_test['id'])
final_submission['generated'] = test_result
final_submission


final_submission.to_csv('submission.csv', index=False)

