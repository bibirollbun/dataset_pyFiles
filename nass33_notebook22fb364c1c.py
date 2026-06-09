import kagglehub
from kagglehub import KaggleDatasetAdapter

import pandas as pd

df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")



import pandas as pd


df = df.drop(columns=["id"])
print(df.head())

numeric_data = df.select_dtypes(include=['number'])
print(numeric_data.head())

categorical_data = df.select_dtypes(include=['object','category'])
print(categorical_data.head())

boolean_data = df.select_dtypes(include=['boolean'])
print(boolean_data.head())

boolean_data = boolean_data.apply(lambda x: x.astype(int) if x.dtype=='bool' else x)
print(boolean_data.head())

numeric_data = pd.concat([numeric_data,boolean_data],axis=1)
print(numeric_data.head())

road_type = categorical_data['road_type'].unique()
print(road_type)

lighting = categorical_data['lighting'].unique()
print(lighting)

time_of_day = categorical_data['time_of_day'].unique()
print(time_of_day)

weather = categorical_data['weather'].unique()
print(weather)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(categorical_data)
encoder_df = pd.DataFrame(
    encoded_data.toarray(),
    columns=encoder.get_feature_names_out(categorical_data.columns)
)
print(encoder_df.head())

final_df = pd.concat([numeric_data , encoder_df],axis=1)
print(final_df.head())

X = final_df.drop(columns=['accident_risk']).values
y = final_df["accident_risk"].values

from sklearn.model_selection import train_test_split
X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=64)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , TensorDataset


X_train_tensor = torch.tensor(X_train , dtype =torch.float32)
X_test_tensor = torch.tensor(X_test,dtype=torch.float32)

y_train_tensor = torch.tensor(y_train,dtype=torch.float32).view(-1,1)
y_test_tensor = torch.tensor(y_test,dtype=torch.float32).view(-1,1)

train_df = TensorDataset(X_train_tensor ,y_train_tensor )
test_df = TensorDataset(X_test_tensor , y_test_tensor )

train_loader = DataLoader(train_df,batch_size=64,shuffle=True)
test_loader = DataLoader(test_df,batch_size=64,shuffle=False)

input_feature = X.shape[1]
hidden_layer = 64
output_feature = 1

class Accident(nn.Module):
    def __init__(self,input_feature,hidden_layer,output_feature):
        super().__init__()

        self.accident = nn.Sequential(
            nn.Linear(input_feature,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,output_feature)
        )

    def forward(self,x):
        return self.accident(x)

model = Accident(input_feature,hidden_layer,output_feature)
optimizer = optim.Adam(model.parameters(),lr=0.01)
loss_function = nn.MSELoss()

epochs = 5

for epoch in range(epochs):
    for x , y in train_loader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred,y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

with torch.no_grad():
    total_loss = 0
    for xt, yt in test_loader:
        y_pred = model(xt)
        loss = loss_function(y_pred,yt)
        total_loss += loss
    print(total_loss)



