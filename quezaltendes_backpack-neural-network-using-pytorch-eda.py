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
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
training_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


training_extra


train = pd.concat([train, training_extra], axis=0)


train


y_train = train['Price']
X_train = train.drop(columns=['id', 'Price', 'Color', 'Compartments', 'Waterproof'])


X_train


X_train.isna().sum()


X_test = test.drop(columns=['id', 'Color', 'Compartments', 'Waterproof']) 


'''
random_array = []
import random
for i in range(X_train.shape[0]):
    random_array.append(random.randint(0, 1))
random = pd.DataFrame({'random': random_array})


importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)
# after using Linear Regression model'''


X_train[['Brand', 'Material', 'Size', 'Style', 'Laptop Compartment']] = \
X_train[['Brand', 'Material', 'Size', 'Style', 'Laptop Compartment']].fillna('Unknown')


X_test[['Brand', 'Material', 'Size', 'Style', 'Laptop Compartment']] = \
X_test[['Brand', 'Material', 'Size', 'Style', 'Laptop Compartment']].fillna('Unknown')


X_train[['Weight Capacity (kg)']] = X_train[['Weight Capacity (kg)']].fillna(X_train[['Weight Capacity (kg)']].sum() / X_train.shape[0])
X_test[['Weight Capacity (kg)']] = X_test[['Weight Capacity (kg)']].fillna(X_test[['Weight Capacity (kg)']].sum() / X_test.shape[0])


cat_column = X_train.columns[X_train.dtypes == 'object']
cat_columns = [x for x in cat_column]


cat_columns


num_column = X_train.columns[X_train.dtypes != 'object']
num_columns = [x for x in num_column]


num_columns[0]




plt.hist(X_train['Weight Capacity (kg)'], bins=500)
plt.show()


sns.boxplot(x=X_train['Weight Capacity (kg)'])

plt.show()


plt.hist(y_train, bins=500)
plt.show()


cat_train = X_train[cat_columns]
cat_test = X_test[cat_columns]

cat = pd.concat([cat_train, cat_test], axis=0)
cat = pd.get_dummies(cat)

cat_train = cat[:X_train.shape[0]]
cat_test = cat[X_train.shape[0]:]

num_train = X_train[num_columns]
cat_train = pd.get_dummies(cat_train)

X_train = pd.concat([num_train, cat_train], axis=1)

num_test = X_test[num_columns]
cat_test = pd.get_dummies(cat_test)

X_test = pd.concat([num_test, cat_test], axis=1)


X_test


X_train


plt.figure(figsize=(10, 8))
sns.heatmap(X_train.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Корреляция признаков")
plt.show()


import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim



X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
X_test_tensor = torch.tensor(X_test.values.astype(np.float32))


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, torch.zeros(len(X_test_tensor)))

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model_nn = nn.Sequential(
    nn.Linear(23, 32),
    nn.Dropout(0.1),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Dropout(0.1),
    nn.ReLU(),
    nn.Linear(16, 1)
    
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model_nn.parameters(), lr=1e-3)


def run_train(model_nn, dataloader, loss_fn, optimizer):
    model_nn.train()
    total_loss = 0
    for X, y in dataloader:
        pred = model_nn(X)
        loss = loss_fn(y, pred.squeeze(1))
        total_loss += loss.item()

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()
    return total_loss
        


def eval(model_nn, dataloader):
    model_nn.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0]
            pred = model_nn(X)
            predictions.append(pred.cpu())

    return torch.cat(predictions, dim=0)



def show_loss(totalloss):
    plt.plot(totalloss)
    plt.xlabel('EPOCHS')
    plt.ylabel('Loss (MSE)')
    plt.show()


from tqdm import tqdm


from tqdm import tqdm

total_loss = []
NUM_EPOCHS = 50

for i in tqdm(range(NUM_EPOCHS)):
    loss = run_train(model_nn, train_dataloader, loss_fn, optimizer)
    total_loss.append(loss)



show_loss(total_loss)


y_pred = np.array(eval(model_nn, test_dataloader))


loss


y_pred = y_pred.flatten()


output = pd.DataFrame({'id': test['id'], 'Price': y_pred})
output.to_csv('submission.csv', index=False)

