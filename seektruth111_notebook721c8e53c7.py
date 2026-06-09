import torch
import random
import argparse
from torch.autograd import Variable


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
            with open('/kaggle/input/stopword2/en_stopword.txt', 'r', encoding='utf-8') as f:
               data =f.read()
            Stop =data.split(" ")
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
def getData(file_path ,input_size ,is_stop=False):

    try:
        with open(file_path, 'r') as f:
            data  =f.readlines()
        data =data[1:]# 第一行是'PhraseId\tSentenceId\tPhrase\tSentiment'
        seq_id =[ da.split('\t')[1] for da in data ]
        data=[t.strip() for t in data]
        Data =[]
        for i in range(len(data)):
           # if i== 0 or seq_id[i] != seq_id[i - 1]:
                Data.append([data[i].split('\t')[0], data[i].split('\t')[2], data[i].split('\t')[3]])

        if is_stop:
            with open('/kaggle/input/stopword2/en_stopword.txt', 'r', encoding='utf-8') as f:
               data =f.read()
            Stop =data.split(" ") 
        ID = []
        XX = []
        YY = []
        for i in range(len(Data)):
            id, x, y = Data[i]
            X = x.split(' ')
            if is_stop:
                X = [x for x in X if x not in Stop]
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

    Data =getData(file_path,input_size,is_stop=False)
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
        Test =getData2(te_path,input_size,is_stop=False)
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
def getData(file_path ,input_size ,is_stop=False):

    try:
        with open(file_path, 'r') as f:
            data  =f.readlines()
        data =data[1:]# 第一行是'PhraseId\tSentenceId\tPhrase\tSentiment'
        seq_id =[ da.split('\t')[1] for da in data ]
        data=[t.strip() for t in data]
        Data =[]
        for i in range(len(data)):
           # if i== 0 or seq_id[i] != seq_id[i - 1]:
                Data.append([data[i].split('\t')[0], data[i].split('\t')[2], data[i].split('\t')[3]])

        if is_stop:
            Stop = getStopWords()
        ID = []
        XX = []
        YY = []
        for i in range(len(Data)):
            id, x, y = Data[i]
            X = x.split(' ')
            if is_stop:
                X = [x for x in X if x not in Stop]
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

    Data =getData(file_path,input_size,is_stop=False)
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
        Test =getData2(te_path,input_size,is_stop=False)
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




flatten =lambda l: [item for sublist in l for item in sublist]
def getBatch(X,Y,batch_size,is_shuffle=True):
    if is_shuffle:
        indices=np.random.permutation(len(X))
        try:
           X=X[indices]
           Y=Y[indices]
        except:
           X=np.array(X)
           Y=np.array(Y)
           X = X[indices]
           Y = Y[indices]
    sindex=0
    eindex=batch_size
    while sindex<len(X):
        if eindex <= len(Y)-1:
            yield X[sindex:eindex],Y[sindex:eindex]
            sindex+=batch_size
            eindex+=batch_size
        else:
            yield X[sindex:],Y[sindex:]
            sindex += batch_size
            eindex += batch_size

#由于onehot编码会十分的缓慢，这里只是把词配对上了
#我们使用更改模型处理的方式来模拟BOW

def dot(X,W,type):
    if type =='BOW':
        X1 = np.zeros((len(X), W.shape[1]))
        for i in range(len(X)):
            for j in range(W.shape[1]):
                X1[i,j]=np.sum([ W[x,j] for x in X[i] ])
        return X1
    elif type =='N_gram':
        return np.dot(X,W)
    else:
        print("your text_feature error")
        return None

def cross_loss(output,target):
    loss=-np.mean(np.log(output[np.arange(len(target)),target]))
    return loss
def cal_grad(X, P1, Y, type, input_size,output_size):
    grad = np.zeros((input_size, output_size))  # 初始化梯度矩阵

    for i in range(len(Y)):
        if Y[i] >= P1.shape[1]:  # 确保 Y[i] 的值是有效类别
            continue

        for j in range(len(X[i])):
            if type == 'BOW':  # Bag of Words 模式
                for k in range(P1.shape[1]):
                    if k == Y[i]:
                        grad[X[i][j], Y[i]] -= (P1[i, Y[i]] - 1) * X[i][j] / len(Y)
                    else:
                        grad[X[i][j], k] -= P1[i, k] * X[i][j] / len(Y)
            else:  # 非 BOW 模式（假设 X[i] 是 feature vector）
                for k in range(output_size):
                    if k == Y[i]:
                        grad[j, Y[i]] -= (P1[i, Y[i]] - 1) * X[i][j] / len(Y)
                    else:
                        grad[j, k] -= P1[i, k] * X[i][j] / len(Y)
    return grad
#BFGS有使用运行会出现内存不足
class LogisticRegression:
    def __init__(self,input_size,output_size,batch_size):
        self.W=np.random.uniform(-0.1,0.1,(input_size,output_size-1))#W1：I,O
        self.input_size=input_size
        self.output_size=output_size-1
        self.batch_size=batch_size

    def predict(self,X,text_type):
        X = dot(X, self.W, text_type)
        Y=np.exp(X)/(1+np.sum(np.exp(X),axis=1,keepdims=True))
        Y=np.c_[Y,1-np.sum(Y,axis=1,keepdims=True)]
        return Y

    def fit_BFGS(self,X,Y,config):
        if config.text_feature=='BOW':
            print("BOW词维度太高，无法使用拟牛顿法")
            return
        Bk=np.eye(self.input_size*self.output_size)
        Wk=self.W#(I,O-1)

        for epoch in range(config.epoch):
            W1 = Wk
            X1=dot(X,W1,config.text_feature)
            X1 = np.c_[X1, np.zeros((len(X), 1))]
            P1 = np.exp(X1 - np.max(X1, axis=1, keepdims=True)) / (
                np.sum(np.exp(X1 - np.max(X1, axis=1, keepdims=True)), axis=1, keepdims=True))


            grad = -cal_grad(X,P1,Y,config.text_feature,self.input_size,self.output_size)

            D = np.dot(Bk, flatten(grad))
            D=D.reshape(self.input_size,-1)
            alpha =0.5
            m=1#线性搜索
            while(m<15):
                W2=W1 - alpha**m*D*config.learning_rate
                X2=dot(X,W2,config.text_feature)
                X2=np.c_[X2,np.zeros((len(X), 1))]
                P2 = np.exp(X2 - np.max(X2, axis=1, keepdims=True)) / (
                    np.sum(np.exp(X2 - np.max(X2, axis=1, keepdims=True)), axis=1, keepdims=True))  # (B,O-1)
                if(cross_loss(P2,Y)<cross_loss(P1,Y)):
                    break
                m+=1

            W2 = W1 - alpha ** m * D*config.learning_rate
            X2 = dot(X, W2, config.text_feature)
            X2 = np.c_[X2, np.zeros((len(X), 1))]
            P2 = np.exp(X2 - np.max(X2, axis=1, keepdims=True)) / (
                np.sum(np.exp(X2 - np.max(X2, axis=1, keepdims=True)), axis=1, keepdims=True))  # (B,O)
            grad2 = cal_grad(X,P2,Y,config.text_feature,self.input_size,self.output_size)

            s = np.array(flatten(W2 - W1))
            yy = np.array(flatten(grad2 - grad))
            rho = 1.0 / (np.dot(yy.T, s)) if np.dot(yy.T, s) > 1e-8 else 0
            if rho > 0:
                I = np.eye(self.input_size * self.output_size)
                V = I - rho * np.outer(s, yy)
                Bk = V @ Bk @ V.T + rho * np.outer(s, s)
                # 直接用 V.T 避免重复计算

            self.W=W2

            if np.linalg.norm(flatten(grad))< config.grad:
                break

    def fit_SGD(self,X,Y,config):
        #loss =  log p   *1/N
        # p= exp()/(sum(exp)( )  +1)
        # X = X @ W
        # dloss / dw1 = dloss/ d p  *  d p / d X'  * dX' /d w
        #             =1/N* 1/p        *  p*(1-p)  * x
        #             =1/N  *(1-p)*x1
        #对于非y 的      =1/N*p*x1
        maxepoch = config.epoch
        lr = config.learning_rate
        for epoch in range(maxepoch):
            grad =np.zeros((self.input_size,config.output_size-1))
            for x,y in getBatch(X,Y,config.batch_size,is_shuffle=False if config.text_feature=='BOW' else True):
                x1=dot(x,self.W,config.text_feature)#(B,I)
                x1=np.c_[x1,np.zeros((len(x),1))]
                p1=np.exp(x1-np.max(x1,axis=1,keepdims=True))/(np.sum(np.exp(x1-np.max(x1,axis=1,keepdims=True)),axis=1,keepdims=True))                                            #(B,O-1)
                p1=p1[:,:-1]

                D=cal_grad(x,p1,y,config.text_feature,self.input_size,self.output_size)
                self.W+= lr* D
                grad+=D

            grad=[x for x in flatten(grad)if x!=0]
            grad =np.array(grad)/(len(Y)/config.batch_size)
            print(f"epoch{epoch+1} the grad:{np.linalg.norm(grad)}")
            #print(self.W[0])
            if np.linalg.norm(grad) <config.grad:
                break

class SoftmaxRegression:
    def __init__(self,input_size,output_size,batch_size):
        self.input_size=input_size
        self.output_size=output_size
        self.batch_size=batch_size
        self.W = np.random.uniform(-0.1,0.1,(self.input_size,output_size))

    def predict(self,X,text_type):
        X= dot(X,self.W,text_type)
        X1=np.exp(X-np.max(X,axis=1,keepdims=True))
        Y=np.exp(X1)/np.sum(X1,axis=1,keepdims=True)
        return Y

    def fit_SGD(self,X,Y,config):
        maxepoch = config.epoch
        lr = config.learning_rate
        for epoch in range(maxepoch):
            Grad = np.zeros((self.input_size,config.output_size))
            for x,y in getBatch(X,Y,config.batch_size):
                x1= dot(x,self.W,config.text_feature)
                X1 = np.exp(x1 - np.max(x1, axis=1, keepdims=True))
                Y1 = np.exp(X1) / np.sum(X1, axis=1, keepdims=True)
                grad = cal_grad(x,Y1,y,config.text_feature,self.input_size,self.output_size)
                Grad += grad
                self.W +=lr*grad

               #print(self.W)
            Grad =Grad/ ( len(Y)//config.batch_size)
            if (epoch+1)%10==0:
                print(f"epoch{epoch+1} the grad:{np.linalg.norm(flatten(Grad))}")
            if np.linalg.norm(flatten(Grad)) < config.grad:
                break

    def fit_BFGS(self,X,Y,config):
        if config.text_feature=='BOW':
            print("BOW词维度过大，不适合拟牛顿法")
            return
        Bk=np.eye(self.input_size*self.output_size)
        for epoch in range(config.epoch):#一般来说，BFGS是使用全批数据更新的，运行时间会很长
                W1 = self.W
                X1 = dot(X, W1, config.text_feature)
                X1 = np.exp(X1 - np.max(X1, axis=1, keepdims=True))
                P1 = np.exp(X1) / np.sum(X1, axis=1, keepdims=True)
                grad = -cal_grad(X, P1, Y, config.text_feature, self.input_size,self.output_size)

                #print(f"epoch{epoch + 1} the grad:")
                #print(np.linalg.norm(flatten(grad)))
                #print(f"loss:{cross_loss(P1,Y)}")
                #print(W1[0])

                if np.linalg.norm(flatten(grad)) < config.grad:
                    break

                D = np.dot(Bk, flatten(grad))#(O*I,O*I)  (O*I,1)
                D=D.reshape(self.input_size,-1)
                alpha = 0.5
                m = 1
                while (m < 15):
                    W2 = W1 - alpha ** m * D
                    X2 = dot(X, W2, config.text_feature)
                    X2=np.exp(X2-np.max(X2,axis=1,keepdims=True))
                    P2 = np.exp(X2) / (np.sum(np.exp(X2), axis=1, keepdims=True))
                    if cross_loss(P2, Y) < cross_loss(P1, Y):
                        break
                    m += 1
               # print(f"alpha:{alpha**m}")
                W2 = W1 - alpha ** m * D*config.learning_rate
                X2 = dot(X, W2, config.text_feature)
                X2 = np.exp(X2 - np.max(X2, axis=1, keepdims=True))
                P2 = np.exp(X2) / (np.sum(np.exp(X2), axis=1, keepdims=True))
                grad2 = -cal_grad(X, P2, Y, config.text_feature, self.input_size,self.output_size)

                s = np.array(flatten(W2 - W1))
                yy = np.array(flatten(grad2 - grad))
                rho = 1.0 / (np.dot(yy.T, s)) if np.dot(yy.T, s) > 1e-8 else 0
                if rho > 0:
                    I = np.eye(self.input_size*self.output_size)
                    V = I - rho * np.outer(s, yy)
                    Bk = V @ Bk @ V.T + rho * np.outer(s, s)

                self.W = W2




class Config:
    def __init__(self):
        # Model parameters
        self.input_size = 50
        self.output_size = 5
        self.batch_size = 32
        self.epoch = 10
        self.learning_rate = 0.001
        self.grad = 1e-3
        
        # Text feature settings
        self.text_feature = 'BOW'
        self.n_gram = 2
        self.frequence = 0
        self.is_sort = False
        
        # Model selection
        self.model_type = 'logistic_regression'
        self.optimization = 'SGD'
        
        # Data paths
        self.train_data_path = '/kaggle/input/uesddata/train.tsv'
        self.test_data_path = '/kaggle/input/uesddata/test.tsv'




def train(config,Word2index,Train_data):
    id, xx, yy = zip(*Train_data)
    xx = list(xx)
    yy = list(yy)
    if config.model_type == 'logistic_regression':
        Logist = LogisticRegression(config.input_size if config.text_feature == 'N_gram' else len(Word2index),
                                    config.output_size, config.batch_size)

        if config.optimization == 'SGD':
            Logist.fit_SGD(xx, yy, config)
        elif config.optimization == 'BFGS':
            Logist.fit_BFGS(xx, yy, config)
        return Logist

    if config.model_type == 'softmax_regression':
        Softmax = SoftmaxRegression(config.input_size if config.text_feature == 'N_gram' else len(Word2index),
                                    config.output_size, config.batch_size)

        if config.optimization == 'SGD':
            Softmax.fit_SGD(xx, yy, config)
        elif config.optimization == 'BFGS':
            Softmax.fit_BFGS(xx, yy, config)
        return Softmax




config=Config()
Train,Test,Word2index =None,None,None
if config.text_feature=='BOW':
        Word2index,Train, Test = precossing(config.train_data_path, config.test_data_path)
        

elif config.text_feature=='N_gram':
        Word2index,Train,Test=precossing(config.train_data_path,config.test_data_path,config.n_gram)
        
if config.model_type == 'logistic_regression':
            Model = LogisticRegression(config.input_size if config.text_feature == 'N_gram' else len(Word2index),
                                        config.output_size-1, config.batch_size)
else :
            Model = SoftmaxRegression(config.input_size if config.text_feature == 'N_gram' else len(Word2index),
                                    config.output_size, config.batch_size)
Model=train(config,Word2index,Train)
id2, te_X = zip(*Test)
te_X = list(te_X)
Predict = Model.predict(te_X, config.text_feature)


#Predict=[int(t) for t in Predict]
Predict = np.argmax(Predict, axis=1)
answer={'Phraseid':id2,'Sentiment':Predict}
#print(Predict)
answer=pd.DataFrame(answer)
answer.to_csv('submission.csv',index=False)


Predict[5754]

