import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch 
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split


train = pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv",index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv',index_col='id')


train


train.info()


train.describe()


label_encoder = LabelEncoder()
train["Target"] = label_encoder.fit_transform(train["Target"])


corr_train = train.corr()
corr_train.head(5)


corr_train = train.corr()
fig, axi = plt.subplots(figsize=(25, 20))
sns.heatmap(corr_train , annot=True , center=0.2, fmt='.2f', linewidths=0.5)
plt.show()


corr_tar = corr_train["Target"].sort_values(ascending=False)
corr_tar


plt.figure(figsize=(10, 8))
corr_tar.plot(kind="bar" , color = "blue")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()


x = train.drop('Target',axis=1)
x = x.drop('International',axis=1)
x = x.drop('Unemployment rate',axis=1)
x = x.drop('Educational special needs',axis=1)
x = x.drop("Father's qualification",axis=1)
x = x.drop('Nacionality',axis=1)

y = train.Target


x


scaler = StandardScaler()
x = scaler.fit_transform(x)
x


x_,x_test,y_,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train,x_val,y_train,y_val=train_test_split(x_,y_,test_size=0.3,random_state=42)


x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)


class ANN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(ANN,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256,128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128,64)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64,output_size)
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x


input_size = x_train.shape[1]
hidden_size = 64
output_size = len(np.unique(y))


model = ANN(input_size,hidden_size,output_size)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")



model.eval()
with torch.no_grad():
    test_outputs = model(x_test_tensor)
    test_predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (test_predictions == y_test_tensor).sum().item() / y_test_tensor.size(0)
print(f"Test Accuracy: {accuracy:.4f}")


import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class_names = ["Dropout", "Enrolled", "Graduate"]

model.eval()
with torch.no_grad():
    pred = model(torch.tensor(x_, dtype=torch.float32))  

pred = torch.argmax(pred, dim=1).numpy()  

conf_matrix = confusion_matrix(y_, pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()





