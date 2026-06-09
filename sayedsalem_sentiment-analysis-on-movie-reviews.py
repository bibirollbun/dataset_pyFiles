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


!pip install emoji


import matplotlib.pyplot as plt
import seaborn as sns

import re

from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import class_weight

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

import emoji

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from transformers import DistilBertTokenizer, TFDistilBertModel

import zipfile

from tqdm import tqdm


train_df = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip', sep='\t')
train_df


test_df = pd.read_csv(f'/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip', sep='\t')
test_df


train_df.isna().sum()


test_df.isna().sum()


test_df['Phrase'] = test_df['Phrase'].fillna('')


chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA?": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don't care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "BFF": "Best friends forever",
    "CSL": "Can't stop laughing"
  }

def chat_conversion(text: str):
    new_text = []
    for word in text.split():
        if word.upper() in chat_words:
            new_text.append(chat_words[word.upper()])
        else:
            new_text.append(word)
    return " ".join(new_text)


def remove_stopwords(text: str):

  words = word_tokenize(text)

  new_text = []

  stopword = stopwords.words('english')

  for word in words:

    if word not in stopword:
      new_text.append(word)

  return " ".join(new_text)


def remove_emojis(text: str):
    return emoji.replace_emoji(text, replace='')


def clean_text(raw_text: str):

  # Handling chatwords
  cleaned_text = chat_conversion(raw_text)

  #

  # Convert the sentence to lower letter
  cleaned_text = cleaned_text.lower()

  # Remove emojis
  cleaned_text = remove_emojis(cleaned_text)

  # Remove HTML Tags
  cleaned_text = re.sub('<.*?>', '', cleaned_text)

  # Remove URLs
  cleaned_text = re.sub(r'https?://\S+|www\.\S+', '', cleaned_text)

  # Punctuation removal
  cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

  # Remove stopwords
  cleaned_text = remove_stopwords(cleaned_text)

  # Remove multiple spaces
  cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

  return cleaned_text


# Let's test our text cleaning function

text = "The review mentioned a website,ðŸ”¥ chÃ©ck it out at https://some-movie-site.com for more info.<html><body><p> Movie 1</p><p> Actor - Aamir Khan</p><p> Click here to <a href='http://google.com'>download</a></p></body></html> Google    search here www.google.com LOL ðŸ˜˜  Ãœ"
new_text = clean_text(text)
new_text


train_df['cleaned_phrase'] = train_df['Phrase'].apply(clean_text)
train_df


train_df['cleaned_phrase'].duplicated().sum()


test_df['cleaned_phrase'] = test_df['Phrase'].apply(clean_text)
test_df


test_df['cleaned_phrase'].duplicated().sum()


X_train_text = train_df['cleaned_phrase'].tolist()
X_train_text


y_train = train_df['Sentiment'].values
y_train


X_test_text = test_df['cleaned_phrase'].tolist()
X_test_text


model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
bert_model = TFDistilBertModel.from_pretrained(model_name)


def calculate_max_length(tokenizer, texts):

    encodings = tokenizer(texts, truncation=False, padding=False)
    input_ids = encodings['input_ids']

    sequence_lengths = [len(ids) for ids in input_ids]

    return max(sequence_lengths)


calculate_max_length(tokenizer, X_train_text)


calculate_max_length(tokenizer, X_test_text)


def get_cls_embeddings(tokenizer, bert_model, X_train_text, max_length=50):
    encodings = tokenizer(X_train_text, truncation=True, padding=True, max_length=max_length, return_tensors='tf')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # We can embed our phrases by batchs to save our gpu

    batch_size = 1024
    num_samples = len(input_ids)
    all_cls_embeddings = []

    for i in tqdm(range(0, num_samples, batch_size)):
        batch_input_ids = input_ids[i:i + batch_size]
        batch_attention_mask = attention_mask[i:i + batch_size]

        bert_output = bert_model(batch_input_ids, attention_mask=batch_attention_mask)
        last_hidden_state = bert_output.last_hidden_state

        batch_cls_embeddings = last_hidden_state[:, 0, :].numpy() # embedding of [CLS] the executive summary

        all_cls_embeddings.append(batch_cls_embeddings)

    cls_embeddings = np.concatenate(all_cls_embeddings, axis=0)
    return cls_embeddings


train_cls_embedding = get_cls_embeddings(tokenizer, bert_model, X_train_text)
train_cls_embedding.shape


plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=train_df)
plt.title('Distribution of Sentiment Classes in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


model = Sequential([

    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.5),

    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.4),

    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.3),

    Dense(16, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),

    Dense(5, activation='softmax')
])


model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
  )


model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    train_cls_embedding,
    y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stopping]
)


# Plotting the training history
history_dict = history.history

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history_dict['accuracy'], label='Training Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


def model_prediction(model, tokenizer, bert_model, X_test_text):

    test_cls_embedding = get_cls_embeddings(tokenizer, bert_model, X_test_text)

    predictions = model.predict(test_cls_embedding)
    predicted_classes = np.argmax(predictions, axis=1)

    return predicted_classes


predicted_classes = model_prediction(model, tokenizer, bert_model, X_test_text)
predicted_classes


submission_df = pd.DataFrame({
        'PhraseId': test_df['PhraseId'],
        'Sentiment': predicted_classes
    })


submission_df.head()


submission_df.to_csv('submission_bert.csv', index=False)

