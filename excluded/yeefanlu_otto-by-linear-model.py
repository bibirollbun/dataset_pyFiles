import pandas
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

device = torch.device("cpu")#better use cuda

#1:prepare data

scaler = StandardScaler()
Le = LabelEncoder()

train_data = pandas.read_csv("/kaggle/input/otto-group-product-classification-challenge/train.csv")#read
x_train = train_data.drop(['id', 'target'], axis=1)#drop
y_train = train_data['target']#select

x_train = scaler.fit_transform(x_train)#scale
y_train = Le.fit_transform(y_train)#encode

x_data = torch.tensor(x_train, dtype=torch.float32)#tensor
y_data = torch.tensor(y_train, dtype=torch.long)#tensor

test_data = pandas.read_csv("/kaggle/input/otto-group-product-classification-challenge/test.csv")
x_test = test_data.drop(['id'], axis=1)
x_test = scaler.transform(x_test)
x_test_data = torch.tensor(x_test, dtype=torch.float32)
print(x_data.shape,y_data.shape,x_test_data.shape)

train_dataset = torch.utils.data.TensorDataset(x_data, y_data)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

#2:build model
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(93, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.drop1 = torch.nn.Dropout(0.3)

        self.fc2 = torch.nn.Linear(256, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.drop2 = torch.nn.Dropout(0.3)

        self.fc3 = torch.nn.Linear(128, 9)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        x = self.fc3(x)
        return x

model = Net().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

#3:train model
for epoch in range(100):
    for i,(features,labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        y_hat = model(features)
        loss = criterion(y_hat, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 300 == 0:
            print(f'epoch:{epoch},loss:{loss.item()}')

#4:predict
text_dataset = torch.utils.data.TensorDataset(x_test_data)
test_loader = torch.utils.data.DataLoader(dataset=text_dataset, batch_size=256)

model.eval()
all_probs = []
with torch.no_grad():
    for (x_test,) in test_loader:
        x_test = x_test.to(device)

        outputs = model(x_test)
        probs = torch.softmax(outputs.data, 1)
        all_probs.append(probs)

all_probs_numpy = torch.cat(all_probs, dim=0).cpu().numpy()
print(all_probs_numpy.shape)

submission = pd.DataFrame(all_probs_numpy, columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])

submission['id'] = test_data['id'].values
submission = submission[['id'] + ['Class_' + str(i) for i in range(1, 10)]]

submission.to_csv('submission.csv', index=False)












