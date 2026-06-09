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


df = pd.read_csv("/kaggle/input/spooky-author-identification/train.zip", index_col='id')



df.head()


df = df.apply(lambda x: x.lower() if isinstance(x, str) else x)


df


import re

url_pattern = re.compile(r'https?://\S+')

def remove_urls(text):
    return url_pattern.sub('', text)

df['text'] = df['text'].apply(remove_urls)


df = df.replace(to_replace=r'[^\w\s]', value='', regex=True)


df = df.replace(to_replace=r'\d', value='', regex=True)


import nltk
from nltk.tokenize import word_tokenize

df['text'] = df['text'].apply(word_tokenize)


import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: [word for word in x if word not in stop_words])



df.head(5)


import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd

stemmer = PorterStemmer()

def stem_words(words):
    return [stemmer.stem(word) for word in words]

df['stemmed_messages'] = df['text'].apply(stem_words)


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    
    return lemmas

df['lemmatized_messages'] = df['text'].apply(lemmatize_tokens)


df.head()

