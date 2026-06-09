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


# --- Импорт библиотек ---
!pip install lxml
!pip install nltk
!pip install wordcloud
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os
import re
import lxml.html as lh
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')

# --- Загрузка данных ---
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_table('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip')
test = pd.read_table('/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip')
sub = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/sampleSubmission.csv')

print("Данные успешно загружены.")

# --- Предобработка данных ---
lemmatizer = WordNetLemmatizer()

def preprocess(data):
    data = data.str.lower()  # Приведение к нижнему регистру
    data = data.apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Удаление пунктуации
    stop_words = set(stopwords.words('english'))
    data = data.apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))  # Удаление стоп-слов
    return data

def remove_html_tags(data):
    clean_data = []
    for text in data:
        html_content = lh.fromstring(text)
        clean_data.append(html_content.text_content())
    return clean_data

train['clean_review'] = remove_html_tags(train['review'])
train['clean_review'] = preprocess(train['clean_review'])
test['clean_review'] = remove_html_tags(test['review'])
test['clean_review'] = preprocess(test['clean_review'])

print("Предобработка данных завершена.")

# --- Генерация облака слов ---
positive = train.loc[train['sentiment'] == 1]
negative = train.loc[train['sentiment'] == 0]

# Облако слов для положительных отзывов
text_positive = " ".join(review for review in positive.clean_review)
wordcloud_positive = WordCloud(collocations=False, background_color='white').generate(text_positive)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title("Облако слов для положительных отзывов")
plt.show()

# Облако слов для отрицательных отзывов
text_negative = " ".join(review for review in negative.clean_review)
wordcloud_negative = WordCloud(collocations=False, background_color='black').generate(text_negative)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title("Облако слов для отрицательных отзывов")
plt.show()

# --- Подготовка данных для модели ---
MAX_WORDS = 25000
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(train.clean_review)

# Преобразование текстов в последовательности
train_sequences = tokenizer.texts_to_sequences(train.clean_review)
test_sequences = tokenizer.texts_to_sequences(test.clean_review)

# Дополнение последовательностей
max_seq_length = max(len(seq) for seq in train_sequences)
X_train_padded = pad_sequences(train_sequences, maxlen=max_seq_length, padding='pre')
X_test_padded = pad_sequences(test_sequences, maxlen=max_seq_length, padding='pre')

# Метки
Y_train = np.array(train['sentiment'])

# --- Определение и обучение модели ---
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=max_seq_length),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

# Обучение модели
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train_padded, Y_train, batch_size=32, epochs=20, validation_split=0.2, callbacks=[early_stopping])

# --- Оценка производительности модели ---
plt.figure(figsize=(12, 6))

# График потерь
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# График точности
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# --- Предсказание и генерация файла для отправки ---
predictions = model.predict(X_test_padded)
test['sentiment'] = (predictions >= 0.5).astype(int)

# Создание файла для отправки
submission = test[['id', 'sentiment']]
submission.to_csv('submission.csv', index=False)

print("Файл submission.csv успешно сохранен.")


