!pip install -q textblob

import pandas as pd
import numpy as np
import os
import re
import nltk

# Завантажуємо необхідні ресурси для токенізації, лематизації та фільтрації стоп-слів
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Читаємо train.csv прямо з архіву
df = pd.read_csv("/kaggle/input/spooky-author-identification/train.zip")

# Переглядаємо перші кілька рядків таблиці
print(df.head())

# Перевіряємо унікальні класи авторів
print("Автори:", df['author'].unique())


# Приводимо текст до нижнього регістру
def set_lower(data_list):
    df = data_list[0].copy()
    target = data_list[1]
    df[target] = df[target].apply(lambda x: " ".join(x.lower() for x in x.split()))
    return df

# Видаляємо пунктуацію (усі символи, крім літер і пробілів)
def set_nonpunct(data_list):
    df = data_list[0].copy()
    target = data_list[1]
    df[target] = df[target].str.replace(r'[^\w\s]', '', regex=True)
    return df

# Видаляємо стоп-слова (частки, прийменники, займенники тощо)
def set_nonstop_words(data_list):
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    df = data_list[0].copy()
    target = data_list[1]
    df[target] = df[target].apply(lambda x: " ".join(word for word in x.split() if word not in stop))
    return df

# Лематизуємо слова (приводимо до початкової граматичної форми)
def set_lemmatize(data_list):
    from textblob import Word
    df = data_list[0].copy()
    target = data_list[1]
    df[target] = df[target].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))
    return df

# Стемінг (обрізка слова до кореня)
def set_stemming(data_list):
    from nltk.stem import PorterStemmer
    df = data_list[0].copy()
    target = data_list[1]
    st = PorterStemmer()
    df[target] = df[target].apply(lambda x: " ".join(st.stem(word) for word in x.split()))
    return df


# Застосовуємо кроки обробки до текстового стовпця 'text'
data = [df, 'text']

df = set_lower(data)
df = set_nonpunct([df, 'text'])
df = set_nonstop_words([df, 'text'])
df = set_lemmatize([df, 'text']) 


# Перевіряємо результат
df.head()


# Додаємо кількість слів у кожному тексті
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

# Візуалізація розподілу довжини текстів
import matplotlib.pyplot as plt
df['word_count'].hist(bins=20)
plt.title('Розподіл кількості слів у текстах')
plt.xlabel('Кількість слів')
plt.ylabel('Кількість текстів')
plt.grid(True)
plt.show()


# Аналіз найчастіших слів
from collections import Counter

all_words = " ".join(df['text']).split()
word_freq = Counter(all_words)

# Виведення 20 найчастіших слів
print("Найпоширеніші слова:")
print(word_freq.most_common(20))


