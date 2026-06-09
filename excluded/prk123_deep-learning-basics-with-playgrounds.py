import pandas as pd

training = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')


training.head()


X = training.iloc[:, 1:-1]


y = training.iloc[:, -1]



 X.columns



categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for cols in categorical_columns:
    X[cols] = encoder.fit_transform(X[cols])


X.head()


X = X.to_numpy()
y = y.to_numpy()



import torch
from torch.utils.data import TensorDataset

dataset = TensorDataset(torch.tensor(X), torch.tensor(y))


from torch.utils.data import DataLoader

batch_size = 10

shuffle = True

trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


for i, (batch, target) in enumerate(trainloader):
    if i >= 2:
        break

    print("Batch: ", batch)
    print("Inputs: ", target)
    


import torch.nn as nn
criterion = nn.BCELoss()


model = nn.Sequential(
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Linear(2, 1),
    nn.Sigmoid()
)


import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr = 0.008)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


from tqdm import tqdm
num_epochs = 10
trainingloss = []

model = model.to(device)
for epoch in range(num_epochs):
    
    training_loss = 0.0
    for data in tqdm(trainloader, desc=f"Epoch: {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()

        feature, target = data

        feature, target = feature.float().to(device), target.unsqueeze(1).float().to(device)
        
        output = model(feature)
        
        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        training_loss += loss.item()

    epoch_loss = training_loss / len(trainloader)
    print(f"Epoch {epoch + 1}: {epoch_loss}")
    trainingloss.append(epoch_loss)


testing = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


testing.head()


test_X = testing.iloc[:, 1:]


categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for cols in categorical_columns:
    test_X[cols] = encoder.fit_transform(test_X[cols])


test_X.head()


testdataset = torch.tensor(test_X.to_numpy())

testdataset = testdataset.float().to(device)


model.eval()

with torch.no_grad():
    predictions = model(testdataset)

predictions = predictions.cpu().numpy().flatten()



#predictions = (predictions >= 0.5).astype(int)


submission = pd.DataFrame(
    {
        'id': testing['id'],
        'y': predictions
    }
)


submission.to_csv('submission.csv', index=False)


submission.head()

