import random
import numpy as np
import torch
import tensorflow as tf
from sklearn import config_context
import PIL
import torchvision.transforms as transforms

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tf.random.set_seed(seed)

fix_seed()


from IPython.display import Image, display

image_path = '/kaggle/input/glasses-or-no-glasses/faces-spring-2020/faces-spring-2020/face-1.png'
display(Image(image_path, width=250))


image_path = '/kaggle/input/glasses-or-no-glasses/faces-spring-2020/faces-spring-2020/face-10.png'
display(Image(image_path, width=250))


import pandas as pd

train_df = pd.read_csv('/kaggle/input/glasses-or-no-glasses/train.csv')
test_df = pd.read_csv('/kaggle/input/glasses-or-no-glasses/test.csv')
X, y = train_df.drop(['glasses', 'id'], axis=1), train_df['glasses']


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.DataFrame(X_test['id'])
df['glasses'] = 0

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    rf_classifier.fit(X_train, y_train)
    
    val_accuracy = accuracy_score(y_val, rf_classifier.predict(X_val))
    print(f"Validation Accuracy: {val_accuracy}")

    df['glasses'] += rf_classifier.predict_proba(X_test.drop('id', axis=1))[:, 1] 

df['glasses'] /= n_splits
# df['glasses'] = (df['glasses'] > 0.5).astype(int)

df.to_csv('kaggle_submission.csv', index=False)

print("Kaggle submission file saved as 'kaggle_submission.csv'")


import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

df = pd.DataFrame(X_test['id'])
df['glasses'] = 0

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

logreg = LogisticRegression(random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    logreg.fit(X_train, y_train)
    
    val_preds = logreg.predict(X_val)
    val_pred_proba = logreg.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, val_preds)
    val_log_loss = log_loss(y_val, val_pred_proba)
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Validation Log Loss: {val_log_loss}")

    df['glasses'] += logreg.predict_proba(X_test.drop('id', axis=1))[:, 1]

df['glasses'] /= n_splits

df.to_csv('kaggle_submission.csv', index=False)
print("Kaggle submission file saved as 'kaggle_submission.csv'")


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.DataFrame(X_test['id'])
df['glasses'] = 0

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

logreg = LogisticRegression(random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test.drop('id', axis=1))

    logreg.fit(X_train, y_train)
    
    val_preds = logreg.predict(X_val)
    val_pred_proba = logreg.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, val_preds)
    val_log_loss = log_loss(y_val, val_pred_proba)
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Validation Log Loss: {val_log_loss}")

    df['glasses'] += logreg.predict_proba(X_test_scaled)[:, 1]

df['glasses'] /= n_splits

df.to_csv('kaggle_submission.csv', index=False)
print("Kaggle submission file saved as 'kaggle_submission.csv'")


X_train = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test.drop('id', axis=1))

logreg.fit(X_train, y)

df['glasses'] = logreg.predict_proba(X_test_scaled)[:, 1]
df.to_csv('kaggle_submission.csv', index=False)
print("Kaggle submission file saved as 'kaggle_submission.csv'")


import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file).set_index('id')['glasses']
        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.ToTensor()
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = f'{self.img_dir}/face-{idx+1}.png' 
        image = PIL.Image.open(img_path)
        label = self.img_labels.loc[idx+1]
        return self.transform(image), label


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((512, 512)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = CustomImageDataset(
    annotations_file='/kaggle/input/glasses-or-no-glasses/train.csv',
    img_dir='/kaggle/input/glasses-or-no-glasses/faces-spring-2020/faces-spring-2020',
    transform=transform
)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                          shuffle=True, num_workers=2)


class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(7, 7), stride=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=2)
        
        self.fc1 = nn.Linear(in_features=1024, out_features=500)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(in_features=500, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)
        
    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool1(self.relu1(self.conv2(x)))
        x = self.maxpool1(self.relu1(self.conv3(x)))
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net(in_channels=3).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(net.parameters())


losses = []
for epoch in range(10):  
    print(f'{epoch}-th epoch...')
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        outputs = net(inputs)[:,0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    losses.append(running_loss)
    print(losses[-1])

print('Finished Training')


correct = 0
total = 0
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        inputs = images.to(device)
        labels = labels.float().to(device)

        outputs = net(inputs)[:,0]
        predicted = (outputs > 0).int()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 train images: %d %%' % (
    100 * correct / total))


idxs = []
labels = []
img_dir = '/kaggle/input/glasses-or-no-glasses/faces-spring-2020/faces-spring-2020'
with torch.no_grad():
    for idx in X_test['id']:
        img_path = f'{img_dir}/face-{idx}.png'
        image = transform(PIL.Image.open(img_path)).to(device)
        image = image[None, :]
        outputs = net(image)[:,0]
        predicted = nn.Sigmoid()(outputs).item()
        idxs.append(idx)
        labels.append(predicted)

result = pd.DataFrame({'id': idxs, 'glasses': labels})
result.to_csv('submission3.csv', index=False)


class CustomImageGanDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file).set_index('id')['glasses']
        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.vectors = pd.read_csv(annotations_file).loc[:, 'v1':'v512'].values
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = f'{self.img_dir}/face-{idx+1}.png' 
        image = PIL.Image.open(img_path)
        label = self.img_labels.loc[idx+1]
        vector = self.vectors[idx]
        new_chan = torch.Tensor(vector).unsqueeze(1).repeat(1, 512).T[None, :] # 1x512x512
        return torch.concat((self.transform(image), new_chan)), label


train_set = CustomImageGanDataset(
    annotations_file='/kaggle/input/glasses-or-no-glasses/train.csv',
    img_dir='/kaggle/input/glasses-or-no-glasses/faces-spring-2020/faces-spring-2020',
    transform=transform
)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                          shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net(in_channels=4).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(net.parameters())

losses = []
for epoch in range(5):  
    print(f'{epoch}-th epoch...')
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        outputs = net(inputs)[:,0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    losses.append(running_loss)
    print(losses[-1])

print('Finished Training')


correct = 0
total = 0
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        inputs = images.to(device)
        labels = labels.float().to(device)

        outputs = net(inputs)[:,0]
        predicted = (outputs > 0).int()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 train images: %d %%' % (
    100 * correct / total))


idxs = []
labels = []
img_dir = '/kaggle/input/glasses-or-no-glasses/faces-spring-2020/faces-spring-2020'
vectors = X_test.loc[:, 'v1':'v512'].values

with torch.no_grad():
    for idx in X_test['id']:
        img_path = f'{img_dir}/face-{idx}.png'
        image = transform(PIL.Image.open(img_path))
        image = image
        vector = vectors[idx - 4501]
        new_chan = torch.Tensor(vector).unsqueeze(1).repeat(1, 512).T[None, :] # 1x512x512
        input = torch.concat((image, new_chan))[None, :].to(device)
        outputs = net(input)[:,0]
        predicted = nn.Sigmoid()(outputs).item()
        idxs.append(idx)
        labels.append(predicted)

result = pd.DataFrame({'id': idxs, 'glasses': labels})
result.to_csv('submission.csv', index=False)

