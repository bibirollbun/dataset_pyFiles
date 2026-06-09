import numpy as np
import pandas as pd
import zipfile
import re
import gc
from bs4 import BeautifulSoup
import nltk
# nltk.download()
from nltk.corpus import stopwords


train_path='/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip'
test_path='/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip'
valid_path='/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip'


def get_data():
    with zipfile.ZipFile(train_path, 'r') as f:
        with f.open(re.split("/",train_path)[-1][:-4]) as f:
            tr=pd.read_csv(f, delimiter="\t")
    
    with zipfile.ZipFile(test_path, 'r') as f:
        with f.open(re.split("/",test_path)[-1][:-4]) as f:
            te=pd.read_csv(f, delimiter="\t")

    with zipfile.ZipFile(valid_path, 'r') as f:
        with f.open(re.split("/",valid_path)[-1][:-4]) as f:
            va=pd.read_csv(f, delimiter="\t",quoting=3)
    return tr,te,va
train_df,test_df,val_df=get_data()


print(sorted(train_df.id)[:10],sorted(val_df.id)[:10],sorted(test_df.id)[:10],sep='\n')


train_df.head()


train_df.loc[0].review


train_df.sentiment.value_counts()


def preprocess(x):
    x=BeautifulSoup(x).get_text()
    x=x.lower()
    x=re.sub('[^A-Za-z]'," ",x)
    x=x.split()
    stop_words=set(stopwords.words("english"))
    return " ".join([i for i in x if i not in stop_words])

# Score of 0.85448 with tese features

# train_df['cleaned']=train_df.review.apply(preprocess)
# test_df['cleaned']=test_df.review.apply(preprocess)

def preprocess(x):
    x=BeautifulSoup(x).get_text()
    x=x.lower()
    x=x.replace("!","exclam ")
    x=re.sub('[^A-Za-z]'," ",x)
    x=x.split()
    stop_words=set(stopwords.words("english"))
    return " ".join([i for i in x if i not in stop_words])

# All else held constant, with exclam tag, model perf improves to 0.8560

train_df['cleaned']=train_df.review.apply(preprocess)
test_df['cleaned']=test_df.review.apply(preprocess)


from sklearn.feature_extraction.text import CountVectorizer


vec=CountVectorizer(lowercase=False,max_features=5000)
fitter=vec.fit(train_df.cleaned.values)

train_tokenized=fitter.transform(train_df.cleaned.values).toarray()
test_tokenized=fitter.transform(test_df.cleaned.values).toarray()


import xgboost as xgb


# model=xgb.XGBClassifier(random_state=42)
# model.fit(train_tokenized,train_df.sentiment)
# pd.DataFrame({'id':test_df.id,'sentiment':model.predict(test_tokenized)}).to_csv("submission.csv",index=False)


from sklearn.model_selection import StratifiedKFold

def better_training(X,y,unlabel):
    oofs=np.zeros(len(X))
    outs=np.zeros(len(unlabel))
    skf=StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
    for train,test in skf.split(X,y):
        X_train,y_train=X[train],y.iloc[train]
        X_test,y_test=X[test],y.iloc[test]
        model=xgb.XGBClassifier(random_state=42).fit(X_train,y_train)
        oofs[test]=model.predict(X_test)
        outs+=model.predict(unlabel)
    print((oofs==y).sum()/len(X))
    return int(np.round(outs/5))

# preds=better_training(train_tokenized,train_df.sentiment,test_tokenized)
# pd.DataFrame({'id':test_df.id,'sentiment':preds}).to_csv("submission.csv",index=False)


train_df,test_df,val_df=get_data()


import nltk.data
punkt_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
regex_pattern=re.compile("[^A-Za-z]")
stop_word_filter=set(stopwords.words('english'))

def word_embed_clean(x,stop_words=None):
    x=BeautifulSoup(x).get_text()
    x=x.lower()
    x=regex_pattern.sub(" ",x)
    x=x.split()
    if stop_words:
        return [i for i in x if i not in stop_word_filter]
    return x

def word_embed(x,tokenizer):
    raw_sentences=tokenizer.tokenize(x)
    outs=[]
    return [word_embed_clean(sentence) for sentence in raw_sentences if len(sentence)>0]

def sentence_gen(corpus):
    return [sentence for text in corpus for sentence in word_embed(text,punkt_tokenizer)]
# all_sents=sentence_gen(train_df.review.to_list()+val_df.review.to_list())


import pickle

# with open('token_data','wb') as f:
#     pickle.dump(all_sents,f)

with open('/kaggle/input/token-array/token_data','rb') as f:
    all_sents=pickle.load(f)


print(all_sents[0])


import gensim
from gensim.models import Word2Vec

vec_model=Word2Vec(sentences=all_sents,vector_size=600,min_count=40,window=10,workers=4,sample=1e-3)
vec_model.init_sims(replace=True)
# vec_model.save('word_embed.model')

# vec_model=Word2Vec.load('/kaggle/input/token-array/word_embed.model')


vec_model.wv.most_similar("person",topn=5)


def average_tokens(words):
    outs=np.zeros((feature_dim,),dtype=np.float32)
    for word in words:
        if word in word_vocab:
            outs+=vec_model.wv[word]
    return outs/feature_dim

# Averaged because making BOW model
def run_average(reviews):
    outs=np.zeros((len(reviews),feature_dim),dtype=np.float32)
    for idx,review in enumerate(reviews):
        words=word_embed_clean(review,punkt_tokenizer)
        outs[idx]=average_tokens(words)
    return outs

word_vocab=set(vec_model.wv.index_to_key)
feature_dim=len(vec_model.wv['person'])

train_embeddings=run_average(train_df.review.values)
test_embeddings=run_average(test_df.review.values)


model=xgb.XGBClassifier(random_state=42)
model.fit(train_embeddings,train_df.sentiment)
pd.DataFrame({'id':test_df.id,'sentiment':model.predict(test_embeddings)}).to_csv("submission.csv",index=False)




