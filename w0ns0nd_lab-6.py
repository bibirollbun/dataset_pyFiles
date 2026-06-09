import zipfile, os

input_path = "/kaggle/input/spooky-author-identification/"
for fname in ["train.zip", "test.zip", "sample_submission.zip"]:
    with zipfile.ZipFile(os.path.join(input_path, fname), 'r') as z:
        z.extractall("/kaggle/working/")
import pandas as pd
import numpy as np
import re
import string
from collections import Counter

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/working/train.csv")
df.head()



print("Кількість текстів:", len(df))
print("\nРозподіл авторів:")
print(df['author'].value_counts())

df['char_len'] = df['text'].str.len()
df['word_len'] = df['text'].str.split().apply(len)

print("\nОпис довжин (символи):")
print(df['char_len'].describe())

print("\nОпис довжин (слова):")
print(df['word_len'].describe())

df['author'].value_counts().plot(kind='bar', title="Розподіл текстів за авторами")
plt.show()



stop_words = set(stopwords.words('english'))
punct_table = str.maketrans('', '', string.punctuation)

contractions_map = {
    "don't": "do not", "can't": "cannot", "i'm": "i am", "it's": "it is",
    "you're": "you are", "they're": "they are", "we're": "we are",
    "i've": "i have", "you've": "you have", "they've": "they have",
    "isn't": "is not", "wasn't": "was not", "weren't": "were not",
    "won't": "will not", "wouldn't": "would not", "shouldn't": "should not"
}



ps = PorterStemmer()
wnl = WordNetLemmatizer()

def expand_contractions(text, cmap):
    tokens = text.lower().split()
    tokens = [cmap.get(tok, tok) for tok in tokens]
    return " ".join(tokens)

def preprocess_text(text):
    # нижній регістр
    text = text.lower()
    # заміна скорочень
    text = expand_contractions(text, contractions_map)
    # видалення пунктуації
    text = text.translate(punct_table)
    # токенізація
    tokens = word_tokenize(text)
    # стоп-слова
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    # лематизація
    tokens = [wnl.lemmatize(t) for t in tokens]
    return tokens



df['tokens'] = df['text'].apply(preprocess_text)
df['text_clean'] = df['tokens'].apply(lambda toks: " ".join(toks))

df[['text', 'text_clean', 'author']].head(10)



def top_words(token_series, n=15):
    c = Counter()
    for toks in token_series:
        c.update(toks)
    return c.most_common(n)

print("ТОП-15 слів (усі автори):")
print(top_words(df['tokens']))

for author in df['author'].unique():
    print(f"\nАвтор: {author}")
    print(top_words(df[df['author']==author]['tokens']))



df['word_len_clean'] = df['text_clean'].str.split().apply(len)

print(df[['word_len','word_len_clean']].describe())

df['word_len_clean'].hist(bins=40)
plt.title("Розподіл довжин після обробки")
plt.xlabel("К-сть слів")
plt.ylabel("К-сть текстів")
plt.show()



df[['id','author','text_clean']].to_csv("/kaggle/working/spooky_preprocessed.csv", index=False)
print("Файл збережено.")



for i in range(5):
    print("Оригінал:", df.loc[i, "text"])
    print("Після обробки:", df.loc[i, "text_clean"])
    print("-"*80)


