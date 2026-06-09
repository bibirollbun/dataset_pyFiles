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


import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, accuracy_score
import torch.optim as optim
import torch.utils.data as data
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier


train_dataset = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_dataset = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_dataset = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train_dataset.head(5)


test_dataset.head(5)


sample_dataset.head(5)


print(train_dataset.isna().sum().sum())
print(train_dataset.isna().sum().sum())


df = train_dataset.drop(columns=['id'])


def correlate():
    mat = df.corr()
    plt.figure(figsize=(10,6))
    sns.heatmap(mat,annot=True,cmap='coolwarm',fmt=".2f")
    plt.title("Feature Correlations")
    plt.show()

correlate()


# df1 = df


def outlier_plot(col1,col2='day'):
    plt.figure(figsize=(8,4))
    sns.set_style('darkgrid')
    sns.scatterplot(x=df[col2],y=df[col1],color='red')
    plt.title(f'{col1} before outlier is fixed')
    plt.xlabel('day')
    plt.ylabel(col1)
    plt.show()
    


outlier_plot('humidity')


def handle_outlier(col):
    
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    median_value = df[col].median()

    df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound),median_value,df[col])

    return df
    


for col in df.columns:
    if col != 'day' and col != 'rainfall':
        df = handle_outlier(col)


def fixed_outlier_plot(col1,col2='day'):
    plt.figure(figsize=(8,4))
    sns.set_style('darkgrid')
    sns.scatterplot(x=df[col2],y=df[col1])
    plt.title(f'{col1} after fixing outliers')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()


fixed_outlier_plot('humidity')


def feature_eng(df):
    # cyclic encoding
    df['day_sin'] = np.sin(2 * np.pi * df['day']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/365)
    
    #feels like temperature
    df['heat_index'] = (df['temparature'] * df['humidity'])/100

    # wind chill factor
    df['wind_chill'] = df['temparature'] - (df['windspeed'] * 0.7)

    return df

df = feature_eng(df)
    


def new_correlate(df):
    mat = df.corr()
    plt.figure(figsize=(10,5))
    sns.heatmap(mat,annot=True,cmap='coolwarm',fmt='.2f')
    plt.title('New Feature Correlations')
    plt.show()

new_correlate(df)


df.head(5)


x = df.drop(columns=['rainfall'])
y = df['rainfall']


X_train,X_test,Y_train,Y_test = train_test_split(
    x.drop(columns=['day_sin','day_cos','heat_index','wind_chill']),y,test_size=0.25,random_state=42)

# scaling down the values between 0 and 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Convert to tensor
X_train_tensor = torch.tensor(X_train,dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train,dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test,dtype=torch.float32)
Y_test_tensor = torch.tensor(np.array(Y_test),dtype=torch.float32).unsqueeze(1)

batch_size = 32

train_dataset = data.TensorDataset(X_train_tensor,Y_train_tensor)
train_dataloader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)


class BinaryClassifier(nn.Module):
    def __init__(self,input_size=11):
        super().__init__()
        self.fc1 = nn.Linear(input_size,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,1)

        self.sigmoid_layer = nn.Sigmoid()

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid_layer(x)

        return x


model = BinaryClassifier()


model.parameters


learning_rate = 0.005
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.BCELoss()


num_epochs = 50
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()

    for batch_x,batch_y in train_dataloader:
        
        optimizer.zero_grad()
        outputs = model(batch_x)
    
        loss = criterion(outputs,batch_y)
    
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(),'best_model.pth')
        print(f'epoch: {epoch} loss: {loss.item():.3f}')


print(best_loss)


best_model = BinaryClassifier()
best_model.load_state_dict(torch.load('best_model.pth'))


best_model.eval()
with torch.no_grad():
    y_probs = best_model(X_test_tensor).numpy()
    y_pred = (y_probs > 0.5).astype(int)
    print("Accuracy: ",accuracy_score(Y_test,y_pred))


test_dataset.head(5)


def feature_eng(df):
    # cyclic encoding
    df['day_sin'] = np.sin(2 * np.pi * df['day']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/365)
    
    #feels like temperature
    df['heat_index'] = (df['temparature'] * df['humidity'])/100

    # wind chill factor
    df['wind_chill'] = df['temparature'] - (df['windspeed'] * 0.7)

    return df

test_df = feature_eng(test_dataset.drop(columns=['id']))


test_df.head(5)


test_df['winddirection'].median()


test_df['winddirection'].fillna(test_df['winddirection'].median(),inplace = True)


test_df.isna().sum()


best_model.eval()
with torch.no_grad():
    test_probs = best_model(torch.tensor(scaler.fit_transform(test_df.drop(columns=['day_sin','day_cos','heat_index','wind_chill'])),dtype=torch.float32)).numpy()
    test_preds = (test_probs > 0.5).astype(int)
    


test_df.isna().sum()


final_output = pd.concat([pd.DataFrame(test_dataset['id']),pd.DataFrame(test_preds)],
                         axis=1,ignore_index=True)


final_output.head(5)


final_output.rename(columns={0:'id',1:'rainfall'},inplace=True)


final_output.head(5)


final_output.to_csv('submission.csv',index=False)




