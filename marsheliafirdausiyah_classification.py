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


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('/kaggle/input/latihan-kuis-sml/train.csv')


data


data.info()


#Cek duplikat data
duplikat=data.duplicated()
print(f'Data Dupliat:{duplikat}')

jmlh_duplikat=data.duplicated().sum()
print(f'Jumlah Data Duplikat:{jmlh_duplikat}')


#Cek Missing Value
data.isnull().sum()


#Cek Outlier Numerik
numerical_col=data.select_dtypes(include=['float64','int64'])

def count_outliers_iqr(data, col_out):
    jumlah_outlier ={}
    for col in col_out:
        q1 = np.percentile(data[col],25)
        q3 = np.percentile(data[col],75)
        IQR = q3-q1
        lwr_bound = q1 - (1.5*IQR)
        upr_bound = q3 + (1.5*IQR)

    outliers = data[(data[col]<lwr_bound)|(data[col])>upr_bound]
    jumlah_outlier[col]=len(outliers)
    return jumlah_outlier

jumlah_outlier = count_outliers_iqr(data,numerical_col)
print("Outlier untuk setiap kolom:")
print(jumlah_outlier)


#Class NN
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1=nn.Linear(input_size, hidden_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.relu2=nn.ReLU()
        self.fc3=nn.Linear(hidden_size,hidden_size)
        self.relu3=nn.ReLU()
        self.fc4=nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


print(data['Air Quality'].unique())
print("Jumlah kategori unik:", len(data['Air Quality'].unique()))


#DEFINISI PARAMETER
input_size = 10
hidden_size = 40
output_size = 4
batch_size = 80
num_epochs = 200


#MODEL NN
model = SimpleNN(input_size,hidden_size,output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)


#DEFINISI X DAN Y
y = LabelEncoder().fit_transform(data['Air Quality'])
x = data.drop('Air Quality',axis=1).values


#SPLIT DATA
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


#KONSERVSI KE TENSOR PYTORCH
x_train_tensor = torch.tensor(x_train,dtype=torch.float32)
y_train_tensor = torch.tensor(y_train,dtype=torch.long)
x_test_tensor = torch.tensor(x_test,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test,dtype=torch.long)


#DATASET DAN DATA LOADER
train_dataset=TensorDataset(x_train_tensor, y_train_tensor)
train_loader=DataLoader(train_dataset, batch_size, shuffle=True)


#ITERASI DAN EPOCH
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%10==0:
        print (f'Epoch[{epoch+1}/{num_epochs}],Loss:{loss.item():.4f}')


#EVALUASI TRAINING
with torch.no_grad():
    y_pred = model(x_train_tensor)
    predicted = torch.argmax(y_pred, dim=1)
    acc = (predicted == y_train_tensor).float().mean()
    print(f'Akurasi:{acc:.4f}')


#EVALUASI TESTING
with torch.no_grad():
    y_pred = model(x_test_tensor)
    predicted = torch.argmax(y_pred, dim=1)
    acc = (predicted==y_test_tensor).float().mean()
    print(f'Akurasi:{acc:.4f}')


data_test=pd.read_csv('/kaggle/input/latihan-kuis-sml/test.csv')


data_test


id_test = data_test['index'] if 'index' in data_test.columns else np.arange(len(data_test))


data_test = data_test.values
data_test_tensor = torch.tensor(data_test, dtype=torch.float32)


#Prediksi Label dengan Model
with torch.no_grad():
    y_test_pred = model(data_test_tensor)
    y_test_pred_class = torch.argmax(y_test_pred, dim=1).numpy()


le = LabelEncoder()
le.fit(data['Air Quality'])  
y_labels = le.inverse_transform(y_test_pred_class)


file_submission = pd.DataFrame({
    'index': id_test, 
    'Air Quality': y_labels
})
submission.to_csv('file_submission.csv', index=False)


print("Contoh hasil submission:")
print(submission.head())

