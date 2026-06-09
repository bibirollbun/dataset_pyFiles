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


import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,in_features=4,h1=8,h2=8,out_features=3):
        super().__init__()
        self.fc1=nn.Linear(in_features,h1)
        self.fc2=nn.Linear(h1,h2)
        self.out=nn.Linear(h2,out_features)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
        


torch.manual_seed(32)
model = Model()


df = pd.read_csv("/kaggle/input/ece657aw21-assignment1-iris-dataset/iris_train.csv")
df


X = df.drop("species",axis = 1)
y = df['species']
y = y.replace('Iris-versicolor',2)
y = y.replace('Iris-setosa',0)
y = y.replace('Iris-virginica',1)


X = X.fillna(X.mean())
X.isnull().sum()


y=y.values
X=X.values


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=32)


X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


cri = nn.CrossEntropyLoss()
optimz = torch.optim.Adam(model.parameters(),lr=0.01)


for i in range(100):
    pred = model.forward(X_train)
    loss = cri(pred,y_train)
    optimz.zero_grad()
    loss.backward()
    print(loss)
    optimz.step()


t = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
e =[]
with torch.no_grad():
    for i,data in enumerate(X_test):
        pred = model.forward(data)
        loss = cri(pred,y_test[i])
        e.append(t[pred.argmax().item()])


loss


e=np.array(e)
e


ts = pd.read_csv('/kaggle/input/ece657aw21-assignment1-iris-dataset/iris_test.csv')
u=torch.FloatTensor(ts[['sepal_length',	'sepal_width',	'petal_length',	'petal_width']].values)



t = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
e =[]
with torch.no_grad():
    for i,data in enumerate(u):
        pred = model.forward(data)
        e.append(pred.argmax().item())



sub = pd.DataFrame(np.array(ts['id']),columns=['id'])
sub['species']=np.array(e)
sub





sub.to_csv('submission.csv', index=False)




