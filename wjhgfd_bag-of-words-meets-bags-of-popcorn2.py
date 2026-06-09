!nvidia-smi


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


import pandas as pd

train = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', header=0,
                   delimiter='\t', quoting=3)
test = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip', header=0,
                  delimiter='\t', quoting=3)
unlabeled_train = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip',
                             header=0, delimiter='\t', quoting=3)
print('Read {} labeled train reviewsd, {} labeled test resviews and {} unlabeled reviews'.format(train['review'].size, test['review'].size, unlabeled_train['review'].size))


# 导入各种字符串清理模块
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
    
def review_to_wordlist(review, remove_stopwords=False):
    # 函数将文档转换为单词序列
    # 可选的移除停用词, 返回一个单词列表

    # 移除HTML
    review_text = BeautifulSoup(review).get_text()

    # 移除非字母字符
    review_text = re.sub('[^a-zA-Z]', " ", review_text)

    # 将单词转换为小写并分割
    words = review_text.lower().split()

    # 可选：移除停用词(默认为false)
    if remove_stopwords:
        stops = set(stopwords.words('english'))
        words = [w for w in words if not w in stops]

    return (words)


import nltk.data

# 加载punkt分词器
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# tokenizer = nltk.data.load('tokenizers/punkt/chinese.pickle')

# 定义函数, 将评论拆分为解析后的句子
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentences) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


sentences = [] # 初始化一个空的句子列表

for review in train["review"]: 
    sentences += review_to_sentences(review, tokenizer) 

for review in unlabeled_train["review"]: 
    sentences += review_to_sentences(review, tokenizer)


len(sentences)


sentences[0]


sentences[1]


!pip install gensim



import logging
from gensim.models import Word2Vec

# 日志设置（方便查看训练进度）
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

# 参数设置
num_features = 300      # 词向量维度
min_word_count = 40     # 最小词数
num_workers = 4         # 并行线程数
context = 10            # 上下文窗口大小
downsampling = 1e-3     # 高频词下采样

# 训练模型
model = Word2Vec(
    sentences=sentences,
    vector_size=num_features,
    workers=num_workers,
    min_count=min_word_count,
    window=context,
    sample=downsampling
)

# 让模型节省内存——如果不再训练可以执行此行
model.init_sims(replace=True)

# 保存模型
model_name = '300features_40minwords_10context.model'
model.save(model_name)


model.wv.doesnt_match("man woman child kitchen".split())


model.wv.doesnt_match("france england germany berlin".split()) 
'berlin'


model.wv.doesnt_match("paris berlin london austria".split())


model.wv.most_similar('man')


model.wv.most_similar('queen')


model.wv.most_similar('awful')


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype='float32')
    nwords = 0
    index2word_set = set(model.wv.key_to_index)
    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model.wv[word])
    featureVec = np.divide(featureVec, nwords)

    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype='float32')
    for review in reviews:
        if counter % 1000 == 0:
            print('第{}条评论, 共{}条'.format(counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVecs


# 计算训练集和测试集的平均特征向量，开始使用停用词
clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

clean_test_reviews = []
for review in test['review']:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)

forest = forest.fit(trainDataVecs, train['sentiment'])

result = forest.predict(testDataVecs)

output = pd.DataFrame(data={'id': test['id'], 'sentiment': result})
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )




