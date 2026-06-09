import numpy as np
import pandas as pd 
from tqdm import tqdm
tqdm.pandas()

train = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")
test = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)


def build_vocab(sentences, verbose =  True):
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

sentences = train["question_text"].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})


from gensim.models import KeyedVectors
news_path = '/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)


import operator 

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab))) # 词汇种数比例
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i))) # 单词个数比例
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

oov = check_coverage(vocab,embeddings_index)
oov[:10]


def clean_text(x):
    x = str(x)
    # 1. 对 "/-'" 这三个符号，用空格替换
    for punct in "/-'":
        x = x.replace(punct, ' ')
    # 2. 对 '&' 这个符号，在其前后加上空格，使其成为独立单词
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    # 3. 对一大串标点符号，直接删除（用空字符串替换）
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))
sentences = train["question_text"].apply(lambda x: x.split())
vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)
oov[:10]


import re

def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

train["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
vocab = build_vocab(sentences)
oov = check_coverage(vocab,embeddings_index)
oov[:20]


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re # 把key键 也就是需要被替换的单词拼起来 方便后续检索替换


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'

                }
mispellings, mispellings_re = _get_mispell(mispell_dict) # 字典；需要被替换的单词拼起来

def replace_typical_misspell(text): # 替换拼写
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x)) # 替换
sentences = train["question_text"].progress_apply(lambda x: x.split()) # 分开单词
to_remove = ['a','to','of','and']
sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)
oov = check_coverage(vocab,embeddings_index)
oov[:20]

