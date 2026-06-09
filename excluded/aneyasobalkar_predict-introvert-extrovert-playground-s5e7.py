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


!pip install imbalanced-learn==0.11.0


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


print(train_data.columns,train_data.shape)


train_data.isnull().sum()


train_data["Personality"].value_counts()


#Training Data
train_id = train_data["id"]
train_target = train_data["Personality"].map({"Introvert":0, "Extrovert":1})
train_data = train_data.drop(["id", "Personality"], axis = 1)
#Test Data
test_id = test_data["id"]
test_data = test_data.drop("id", axis = 1)


def preprocess_fn(data):
    numerical_columns = data.select_dtypes(include='float64').columns
    object_columns = data.select_dtypes(include = "object").columns
    for column in numerical_columns:
        data[column] = data[column].fillna(data[column].median())
    for column in object_columns:
        data[column] = data[column].map({"No":0, "Yes":1})
        data[column]= data[column].fillna(0.0)


#numerical_columns = train_data.select_dtypes(include='float64').columns
#object_columns = train_data.select_dtypes(include = "object").columns[0:2]
#for column in numerical_columns:
#    train_data[column] = train_data[column].fillna(train_data[column].median())
#for column in object_columns:
#    train_data[column] = train_data[column].map({"No":0, "Yes":1})
#train_data["Personality"] = train_data["Personality"].map({"Introvert":0, "Extrovert":1})
#train_data["Stage_fear"]= train_data["Stage_fear"].fillna(0.0)
#train_data["Drained_after_socializing"]= train_data["Drained_after_socializing"].fillna(0.0)


preprocess_fn(train_data)
preprocess_fn(test_data)


print(f"Training Data Shape: {train_data.shape}")
print(train_data.isnull().sum())
print(f"\nTesting Data Shape: {test_data.shape}")
print(test_data.isnull().sum())


fig, axes = plt.subplots(train_data.shape[1],1,figsize=(22,30))
for num,column in enumerate(train_data.columns.values):
    sns.histplot(data = train_data, x= column, hue = train_target, ax = axes[num], kde= True)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size = 0.2, random_state = 42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


sm = SMOTE(random_state = 42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
print(X_train_resampled.shape, y_train_resampled.shape)
print(y_train_resampled.value_counts())


X_train_resampled = torch.Tensor(X_train_resampled.to_numpy()).to(device)
X_test = torch.Tensor(X_test.to_numpy()).to(device)
y_train_resampled = torch.Tensor(y_train_resampled.to_numpy()).to(device)
y_test = torch.Tensor(y_test.to_numpy()).to(device)


print(X_train_resampled.shape, X_test.shape,y_train_resampled.shape, y_test.shape)
print(X_train_resampled.device, X_test.device,y_train_resampled.device, y_test.device)


class PersonalityClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features = 7, out_features = 16),
            nn.ReLU(),
            nn.Linear(in_features = 16, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 16),
            nn.ReLU(),
            nn.Linear(in_features = 16, out_features = 8),
            nn.ReLU(),
            nn.Linear(in_features = 8, out_features = 1),
        )
    def forward(self, x:torch.Tensor):
        return self.layers(x)
torch.manual_seed(42)
model_0 = PersonalityClassifier().to(device)


loss_fn = nn.BCEWithLogitsLoss()
accuracy_fn = BinaryAccuracy().to(device)
optimizer = torch.optim.Adam(params = model_0.parameters(), lr = 0.05)


epochs = 200
losses = []
accuracies = []
for epoch in range(epochs):
    #Set training mode
    model_0.train()
    #Initial Prediction from forward pass
    y_logits_train = model_0(X_train_resampled).squeeze()
    y_pred_train = torch.sigmoid(y_logits_train).round()
    #Compute loss and accuracy
    loss = loss_fn(y_logits_train, y_train_resampled)
    acc = accuracy_fn(y_pred_train, y_train_resampled) 
    #Add to list
    losses += [loss.item()]
    accuracies += [acc.item()]
    #Zero the gradients
    optimizer.zero_grad()
    #Compute gradient via back propagation
    loss.backward()
    #Step in calculated direction from gradient
    optimizer.step()
    if epoch%10 == 0:
        print(f"Epoch: {epoch} Loss: {loss} Accuracy: {acc*100}%")


fig, axes = plt.subplots(2, 1, figsize = (20,10))
axes[0].plot(losses)
axes[0].set_title("Losses")
axes[1].plot(accuracies)
axes[1].set_title("Accuracies")
plt.show()


model_0.eval()
with torch.inference_mode():
    y_logits_test = model_0(X_test).squeeze()
y_pred_test = torch.sigmoid(y_logits_test).round()


print(f"The accuracy is {accuracy_fn(y_pred_test, y_test).item()*100}%")


#numerical_columns = test_data.select_dtypes(include='float64').columns
#object_columns = test_data.select_dtypes(include = "object").columns[0:2]
#for column in numerical_columns:
#    test_data[column] = test_data[column].fillna(test_data[column].median())
#for column in object_columns:
#    test_data[column] = test_data[column].map({"No":0, "Yes":1})


test_data = torch.Tensor(test_data.to_numpy()).to(device)


with torch.inference_mode():
    y_pred_test = torch.sigmoid(model_0(test_data)).round().squeeze()
y_pred_test = y_pred_test.to(int)


y_pred_test.cpu().numpy().shape, test_id.shape


submission = pd.DataFrame({"id": test_id, "Personality": y_pred_test.cpu().numpy()})
submission["Personality"] = submission["Personality"].map({0:"Introvert", 1:"Extrovert"})


submission.to_csv("submission_1.csv", index = False)

