import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")


!unzip /kaggle/input/quora-question-pairs/train.csv.zip -d /kaggle/working
!unzip /kaggle/input/quora-question-pairs/test.csv.zip -d /kaggle/working


df_train = pd.read_csv("/kaggle/working/train.csv")
df_test = pd.read_csv("/kaggle/working/test.csv")


df_train.head()


df_train.isna().sum()


df_train.dropna(inplace=True)


df_train.drop(['id', 'qid1', 'qid2'], axis=1, inplace=True)


df_train.shape


df_train.duplicated().sum()


#text preprocessing
import re
import emoji

chat_map = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "pls": "please",
    "plz": "please",
    "idk": "i do not know",
    "btw": "by the way",
    "brb": "be right back",
    "dm": "direct message"
}

def text_preprocessing(text):
    text = text.lower()

    #url removal
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    #remove punctuation
    punctuation_pattern = r'["#,\-.@\[\\\]_{|}~]'
    text = re.sub(punctuation_pattern, '', text)

    #emoji removal
    text = emoji.demojize(text)
    text = re.sub(r':[a-z_&]+:', ' ', text)

    #chat word treatment
    text = ' '.join([chat_map.get(word, word) for word in text.split()])

    #whitespace removal
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


df_train['question1'] = df_train['question1'].apply(text_preprocessing)
df_train['qu']

