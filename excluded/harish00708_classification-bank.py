import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df.head()


df['y'].value_counts()


X = df.drop(columns=['id','y'])
y = df['y'].astype(float).values.reshape(-1,1)


def preprocess(df , train=True , encoder_dict={}):
    label_col = df.select_dtypes('object').columns
    if train:
        for i in label_col:
            encoder_dict[i] = df[i].unique().tolist()

    for col in label_col:
        df[col] = df[col].map(lambda x: encoder_dict[col].index(x) if x in encoder_dict[col] else -1)
    
    return df , encoder_dict
    


df , encoder_dict = preprocess(X)
df.head()


X_train, X_test , y_train, y_test = train_test_split(df,y , test_size=0.3 , random_state = 42)

Scaler = StandardScaler()

X_train = Scaler.fit_transform(X_train)
X_test = Scaler.transform(X_test)

y_test = np.array(y_test)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


X_train = torch.tensor(X_train , dtype = torch.float).to(device)
X_test = torch.tensor(X_test , dtype = torch.float).to(device)
y_train = torch.tensor(y_train , dtype = torch.float).to(device)
y_test = torch.tensor(y_test , dtype = torch.float).to(device)


class LogisticRegression(nn.Module):
    def __init__(self , in_feature):
        super().__init__()
        self.layer1 = nn.Linear(in_feature , 128)
        self.dropout = nn.Dropout(0.15)
        self.layer2 = nn.Linear(128,64)
        self.layer3 = nn.Linear(64,32)
        self.layer4 = nn.Linear(32,1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        
        x = self.layer4(x)

        return x
        
        


in_feature = X_train.shape[1]

model = LogisticRegression(in_feature).to(device)

# Loss and Optimzer
criterian = nn.BCEWithLogitsLoss()
optimzer = optim.Adam(model.parameters(), lr=1e-3)



epochs = 150

for epoch in range(epochs):

    model.train()
    output = model(X_train)
    loss = criterian(output , y_train)
    
    preds = (torch.sigmoid(output) > 0.5).float()
    correct_pred = (preds == y_train).sum().item()
    accuracy = correct_pred / len(y_train)

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    if (epoch+1) % 10 == 1: 
        print(f"epoch : {epoch+1}  Loss {loss.item():.3f}  Accuracy {accuracy:.3f}")


model.eval()

with torch.inference_mode():
    output = model(X_test)
    preds = (torch.sigmoid(output) > 0.5).float()
    accuracy = accuracy_score(preds.cpu().numpy(), y_test.cpu().numpy())

    print(f'Accuracy Score: {accuracy:.2f}')
    


# For Submission

X_train = torch.cat((X_train, X_test), dim=0)
y_train = torch.cat((y_train, y_test), dim=0)

epochs = 128

in_feature = X_train.shape[1]

model = LogisticRegression(in_feature).to(device)

# Loss and Optimzer
criterian = nn.BCEWithLogitsLoss()
optimzer = optim.Adam(model.parameters(), lr=1e-3)



for epoch in range(epochs):

    model.train()
    output = model(X_train)
    loss = criterian(output , y_train)
    
    preds = (torch.sigmoid(output) > 0.5).float()
    correct_pred = (preds == y_train).sum().item()
    accuracy = correct_pred / len(y_train)

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    if (epoch+1) % 10 == 1: 
        print(f"epoch : {epoch+1}  Loss {loss.item():.3f}  Accuracy {accuracy:.3f}")




Test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

ids = Test['id']

Test.drop(columns=['id'], inplace = True)

Test.head()

Test ,_= preprocess(Test, train = False , encoder_dict = encoder_dict)

Test_scaled = Scaler.transform(Test)

Test_scaled_Tensor = torch.tensor(Test_scaled , dtype = torch.float).to(device)

model.eval()

with torch.inference_mode():
    output = model(Test_scaled_Tensor)
    preds = (torch.sigmoid(output) > 0.5).float()

    print(torch.unique(preds, return_counts=True))




preds_cpu = preds.cpu().numpy().flatten()  # move to CPU and convert to numpy
Submission = pd.DataFrame({'id':ids,
                        'predictions': preds_cpu})

Submission.sample(10)
Submission['predictions'].value_counts()


Submission.to_csv('Submission.csv' , index=False)




