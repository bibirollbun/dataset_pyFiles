import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from numpy import array

from tensorflow.keras.preprocessing.text import Tokenizer


!pip install transformers tensorflow


from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

# Загрузка предобученных модели и токенизатора
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = TFDistilBertForSequenceClassification.from_pretrained(model_name)


df = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', delimiter='\t')
tsv_file_path = '/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip'

df_un = pd.read_csv(tsv_file_path, delimiter='\t', quoting=3)
tsv_file_path = '/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip'

df_test = pd.read_csv(tsv_file_path, delimiter='\t', quoting=3)


# можно также для удаления тегов использовать библиотеку BeautifulSoup
#from bs4 import BeautifulSoup             
# используем регулярные выражения
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # альтернатива :    sentence = BeautifulSoup(sen).get_text() 

    # Удалить все символы, не являющиеся буквами
    sentence = re.sub('[^a-zA-Z]', # шаблон - что заменяем, ^a-zA-Z - все символы не являющиеся буквами
                        ' ',      # шаблон - на что заменяем
                      sentence) #  где заменяем

    # Удалить единственный символ (регулярное выражение: пробел, буква, пробел)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Удалим множественные пробелы (>=1 пробела заменяем на ровно 1 пробел)
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
# удаляем теги
TAG_RE = re.compile(r'<[^>]+>')

# функция для удаления тегов

def remove_tags(text):
    return TAG_RE.sub('', text)


X_train = []
sentences = list(df['review'])
for sen in sentences:
    X_train.append(preprocess_text(sen))


X_test = []
sentences = list(df_test['review'])
for sen in sentences:
    X_test.append(preprocess_text(sen))


y = df['sentiment']


y_train = np.array(list(map(lambda x: 1 if x==1 else 0, y)))


len(X_test)



# Токенизация текста (возвращает словарь input_ids и attention_mask)
le = 100
s = 0
probs = []
while s < len(X_test):
    inputs = tokenizer(X_test[s:s + le], return_tensors="tf", padding=True, truncation=True, max_length=512)
    s += le
    outputs = model(inputs)
    logits = outputs.logits

# Преобразование logits в вероятности через softma
    probs.extend(tf.nn.softmax(logits, axis=1))
    #if s == 200:
    #    break


def convert_array(x):
    if x >= 0.5:
    #arr[arr >= 0.5] = 1
        return 1
    return 0

y_final = [convert_array(probs[i].numpy()[1]) for i in range(len(probs))]
#y_final = y_final.astype(int)


result = pd.DataFrame({"id":df_test["id"], "sentiment":y_final})
result.head()


result.to_csv("output.csv", index=False, quoting=3, escapechar='\\')

