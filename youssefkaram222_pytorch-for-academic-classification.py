import numpy as np
import pandas as pd
import matplotlib.pyplot
import seaborn as sns

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


import torch
import torch.nn as nn
import torch.functional as f
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader




df=pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')


df


x=df.iloc[:,1:-1]
y=df.iloc[:,-1]




encoder=LabelEncoder()

y=encoder.fit_transform(y)


scaler=StandardScaler()

x=scaler.fit_transform(x)


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)


X_train=torch.tensor(X_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.long)
y_test=torch.tensor(y_test,dtype=torch.long)


X_test


train_data=TensorDataset(X_train,y_train)
test_data=TensorDataset(X_test,y_test)



train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=True)


class ANNmodel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(ANNmodel,self).__init__()
        self.fc1=nn.Linear(input_dim,64)
        self.dropout=nn.Dropout(0.2)
        self.fc2=nn.Linear(64,32)
        self.dropout2=nn.Dropout(0.15)
        self.out=nn.Linear(32,output_dim)
    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=self.dropout(x)
        x=torch.relu(self.fc2(x))
        x=self.dropout2(x)
        x = self.out(x)
        return torch.softmax(x, dim=1)
        


out=len(np.unique(y))


model=ANNmodel(X_train.shape[1],out)
print(model)


criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)


for epoch in range(70):
    model.train()
    total_loss=0
    for inputs,y_train in train_loader:
        optimizer.zero_grad()
        y_pred=model(inputs)
        loss_value = criterion(y_pred, y_train)
        loss_value.backward()
        optimizer.step()
        total_loss += loss_value.item()

    print(f"Epoch: {epoch+1}, Loss: {total_loss}")



from sklearn.metrics import classification_report

model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test)
    y_preds = torch.argmax(y_pred_probs, dim=1)

print(classification_report(y_test, y_preds))







