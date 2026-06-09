# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def getBatch(Data,batch_size=32,is_shuffle=True):
    if is_shuffle:
        random.shuffle(Data)
    sindex=0
    eindex=batch_size
    while sindex<len(Data):
        if eindex <= len(Data)-1:
            yield Data[sindex:eindex]
            sindex+=batch_size
            eindex+=batch_size
        else:
            yield Data[sindex:]
            sindex += batch_size
            eindex += batch_size
# 可以选择处理停用词，没有处理后缀，没有处理标点
def getData2(file_path,input_size,is_stop=False):

        with open(file_path, 'r') as f:
            data  =f.readlines()
        data =data[1:]# 第一行是'PhraseId\tSentenceId\tPhrase\tSentiment'
        seq_id =[ da.split('\t')[1] for da in data ]
        data=[t.strip() for t in data]
        Data =[]
        for i in range(len(data)):
        #     if i== 0 or seq_id[i] != seq_id[i - 1]:
                 try:
        #             #print(i)
                     Data.append([data[i].split('\t')[0], data[i].split('\t')[2]])
                 except:
        #             #157451	8588是空
                     Data.append([data[i].split('\t')[0] ,'<PAD> <PAD>'])
        #             print(data[i])

        if is_stop:
            Stop = getStopWords()
        ID = []
        XX = []
        for i in range(len(Data)):
            id, x = Data[i]
            X = x.split(' ')
            if is_stop:
                X = [x for x in X if x not in Stop]
            if len(X) <= input_size:
                while len(X) < input_size:
                    X.append('<PAD>')
            else:
                X = X[:input_size]
            ID.append(int(id))
            XX.append(X)

        return  list(zip(ID, XX))
def getData(file_path ,input_size):

    try:
        with open(file_path, 'r') as f:
            data  =f.readlines()
        data =data[1:]# 第一行是'PhraseId\tSentenceId\tPhrase\tSentiment'
        seq_id =[ da.split('\t')[1] for da in data ]
        data=[t.strip() for t in data]
        Data =[]
        for i in range(len(data)):
           # if i== 0 or seq_id[i] != seq_id[i - 1]:
                Data.append([data[i].split('\t')[1], data[i].split('\t')[2], data[i].split('\t')[3]])


        ID = []
        XX = []
        YY = []
        for i in range(len(Data)):
            id, x, y = Data[i]
            X = x.split(' ')
            if len(X) <= input_size:
                while len(X) < input_size:
                    X.append('<PAD>')
            else:
                X=X[:input_size]
            ID.append(int(id))
            XX.append(X)
            YY.append(int(y))

        data= list(zip(ID, XX, YY))
        return data
    except:
        print("your data pile is error")
        return None

def prepare_sequence(data,Index,n=1,is_torch=False):
    if n==1:
       idxs = list(map(lambda w: Index[w] if w in Index else  Index['<PAD>'], data))
    else:
        idxs = []
        for i in range(len(data)):
            if data[i]=='<PAD>' or i+n>=len(data):
                idxs.append(Index['<PAD>'])
            else:
                str = ""
                for j in range(n):
                    str += data[j + i] + ' '
                if str in Index:
                    idxs.append(Index[str])
                else:
                    idxs.append(Index['<PAD>'])

    if is_torch:
        idxs =Variable(torch.LongTensor(idxs))
    return idxs

def precossing(file_path, te_path=None, n_gram= 1, input_size=50, is_torch=False,Word2index=None):

    Data =getData(file_path,input_size)
    if Data is None:
        print("your train data pile is error")
        return None,None,None

    ID,Words,Target=zip(*Data)
    ID =[id for id in ID]
    Words = [word for word in Words]
    Target = [target for target in Target]
    if Word2index is None:
        Word2index = {'<PAD>': 0, '<UNK>': 1}
        if n_gram == 1:
            for w in set(flatten(Words)):
                Word2index[w] = len(Word2index)
        else:
            for word in Words:
                for i in range(len(word))[:-n_gram + 1]:
                    str = ""
                    if word[i] == '<PAD>':
                        continue
                    for j in range(n_gram):
                        str += word[j + i] + ' '
                    if str not in Word2index:
                        Word2index[str] = len(Word2index)

    Train=[]
    for i in range(len(ID)):
        x=prepare_sequence(Words[i],Word2index,n_gram,is_torch)
        ta=torch.LongTensor([Target[i]]) if is_torch else Target[i]
        Train.append((ID[i],x,ta))


    if te_path is not None:
        Test =getData2(te_path,input_size)
        if Test is None:
            print("your test data pile is error")
            return Word2index,None,None
        ID,Words=zip(*Test)

        Test=[]
        for i in range(len(ID)):
            x = prepare_sequence(Words[i], Word2index,n_gram, is_torch)
   #         ta = torch.LongTensor([Target[i]]) if is_torch else Target[i]
            Test.append((ID[i], x))


        return Word2index,Train,Test

    else:
        return Word2index,Train,None

#pre_embedding(50)
def make_embedding(config):
   # str1 = "word2index" + str(config.embedding_size) + ".npy"
    #str2 = "index2emb" + str(config.embedding_size) + ".npy"
    try:
        word2emb=np.load("/kaggle/input/glovedata/word2emb.npy",allow_pickle=True).item()#使用np.
        #word2index=np.load(str1)
        #index2emb=np.load(str2)
    except:
        print("error to extract data")

    pad_vec=np.zeros(config.embedding_size)
    tmp={'<PAD>':pad_vec}
    word2emb={**tmp,**word2emb}
    embedding=np.array(list(word2emb.values()))
    embedding=torch.tensor(embedding,dtype=torch.float32)
    Word2index={key:idx for idx,key in enumerate(word2emb)}
    _,Train,Test=precossing(config.train_data_path,config.test_data_path,1,
                            config.input_size,is_torch=True,Word2index=Word2index)

    return embedding,Word2index,Train,Test


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd


class Config():
    def __init__(self,embedding_size):
        self.num_class=5
        self.conv_size= (3,4,5)
        self.input_size=50
        self.batch_size=32
        self.filter_num=256
        self.embedding_size=embedding_size
        self.hidden_size=128
        self.epoch=25

        self.embedding_pretrained=None
        self.model_type='CNN'
        self.RNN_type='LSTM'
        self.dropout=0.5
        self.learning_rate=0.002
        self.train_data_path="/kaggle/input/moviedata/train.tsv"
        self.test_data_path="/kaggle/input/moviedata/test.tsv"
        self.num_layers=1




class TextCNN(nn.Module):
    def __init__(self,config):
        super(TextCNN,self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=True)
        else:
            self.embedding = nn.Embedding(config.vocal_size,config.embedding_size,padding_idx=0)
        self.convs=nn.ModuleList(
            [nn.Conv2d(1,config.filter_num,kernel_size=(k,config.embedding_size))
             for k in config.conv_size ]
        )
        self.dropout=nn.Dropout(config.dropout)
        self.fc=nn.Linear(config.filter_num*len(config.conv_size),config.num_class)

    def conv_and_pool(self,x,conv):#N map的数量  L:句子长度
        x =F.relu(conv(x)).squeeze(3)#(B,N,L-k+1,1)->(B,N,L-k+1)
        x=F.max_pool1d(x,x.size(2)).squeeze(2)
        #(B,N,L-k+1)->(B,N,1)->(B,N)
        return x
    def forward(self,x):
        out = self.embedding(x)
        out=out.unsqueeze(1)
        out =torch.cat([self.conv_and_pool(out,conv) for conv in self.convs],dim=1)#(B,N*3)
        out = self.dropout(out)
        out = self.fc(out)
        return out



class TextRNN(nn.Module):
    def __init__(self,config):
        super(TextRNN,self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=True)
        else :
            self.embedding = nn.Embedding(config.vocal_size,config.embedding_size)
        if config.RNN_type=='LSTM':
            self.rnn = nn.LSTM(config.input_size,config.hidden_size,config.num_layers,batch_first=True,bidirectional=False)
        elif config.RNN_type=='GRU':
            self.rnn = nn.GRU(config.input_size,config.hidden_size,config.num_layers,batch_first=True)
        self.fc=nn.Linear(config.hidden_size,config.num_class)

    def forward(self,x):
        embed=self.embedding(x)
        out,_=self.rnn(embed)
        pre=self.fc(out[:,-1,:])#(B,I,H)
        return pre



def train(data,config):
    if  config.model_type=='CNN':
        model=TextCNN(config)
    else:
        model=TextRNN(config)
    optimizer=optim.Adam(model.parameters(),lr=config.learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(config.epoch):
        for batch in getBatch(data, config.batch_size):
            optimizer.zero_grad()
            id, x, y = zip(*batch)
            x = torch.cat(x).view(-1, config.input_size)
            y = torch.cat(y)
            predict = model.forward(x)
            loss = loss_func(predict, y)
            loss.backward()
            optimizer.step()
    return model


config=Config(50)
embedding, Word2index, Train, Test = make_embedding(config)
#config.embedding_pretrained = embedding
config.vocal_size=len(Word2index)

if config.model_type =='CNN':
            Model=TextCNN(config)
else:
            Model=TextRNN(config)
Model.load_state_dict(torch.load("/kaggle/input/addmodel3/TextCNN1q.bin"))
#Model=train(Train,config)
Predict=[]
ID,_=zip(*Test)
for batch in getBatch(Test, config.batch_size,is_shuffle=False):
            _,xx=zip(*batch)
            xx=torch.cat(xx).view(-1, config.input_size)
            predic=Model.forward(xx)
            predic=np.argmax(predic.detach().numpy(), axis=1)
            Predict=np.r_[Predict,predic]
Predict=[int(t) for t in Predict]
answer={'Phraseid':ID,'Sentiment':Predict}
#print(Predict)
answer=pd.DataFrame(answer)
answer.to_csv('submission.csv', index=False)

