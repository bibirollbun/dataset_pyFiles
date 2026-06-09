import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# /kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv
# /kaggle/input/grand-xray-slam-division-a/train1.csv


class CFG:
    test  = "/kaggle/input/grand-xray-slam-division-a/test1/"
    train = "/kaggle/input/grand-xray-slam-division-a/train1/"
    modv1 = "/kaggle/input/chexnet14/pytorch/v-01/1/model.pth.tar"
    modv2 = "/kaggle/input/chexnet14/pytorch/v-01/2/model.pth"
    EPOCHS = 500
    n_classes = 14
    diseases = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity','No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]


train = pd.read_csv("/kaggle/input/grand-xray-slam-division-a/train1.csv")


train.head(2)


plt.figure(figsize = (10, 5))
sns.distplot(train['Age']);
plt.title("Age Distribution", loc = 'center', pad = 10, fontdict = {"size" : 15, 'weight': 'bold', 'color': "#c9a02c"})
plt.show();


j = iter([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
          [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])

fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(30, 7));
fig.suptitle('Age Distribution on each Disease', fontsize=20)

for i in CFG.diseases:
    index = next(j)
    sns.boxplot(data = train, y = 'Age', x = i, ax = axes[*index]);
    axes[*index].set_title(i)
    axes[*index].set_xlabel(None)
    axes[*index].legend().remove()
    
plt.subplots_adjust(top=0.9)
plt.show();


sns.histplot(train['Sex'], color = '#526958');
plt.title("Sex Distribution")
plt.show()


j = iter([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6],
          [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])

fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(30, 7));
fig.suptitle('Normalized Stacked Barplot for Sex distribution', fontsize=20)

def normalize(i):
    temp = pd.DataFrame(train[['Sex', i]].groupby(by = ['Sex', i]).size(), columns = ['values']).reset_index()
    temp_tot = (temp['values'][0] +  temp['values'][1])
    temp['values'][0] = temp['values'][0]/temp_tot
    temp['values'][1] = temp['values'][1]/temp_tot
    temp_tot = (temp['values'][2] +  temp['values'][3])
    temp['values'][2] = temp['values'][2]/temp_tot
    temp['values'][3] = temp['values'][3]/temp_tot
    return temp

for i in CFG.diseases:
    index = next(j)
    temp = normalize(i)
    sns.histplot(data = temp, x = 'Sex', hue = i, weights = 'values', ax = axes[*index], multiple = 'stack');
    axes[*index].set_title(i)
    axes[*index].set_xlabel(None)
    axes[*index].legend().remove()
    
plt.subplots_adjust(top=0.9)
plt.show();

# There is no influence of Gender in getting the Disease


# How many patients are suffering from each Disease?
train.drop_duplicates(subset = ['Patient_ID'], keep = 'last')[CFG.diseases].sum(axis = 0)

# Note: `Pnuemothorax` and `Pleural Other` are rare disease and hard to classify


# How many patients are there with more than on Disease?
train.drop_duplicates(subset = ['Patient_ID'], keep = 'last')[CFG.diseases].sum(axis = 1).value_counts().sort_index()

# Note: Multi-label classification is needed because there are patients suffering with more than one disease


def print_img(i):
    img = Image.open(f'{CFG.train}{i}');
    plt.imshow(img);


print_img(train['Image_name'][0])


import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from sklearn.model_selection import train_test_split


from torchinfo import summary


img = Image.open(f"{CFG.train}{train['Image_name'][0]}").convert("RGB")


plt.imshow(img);


class cdataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)        
    def __getitem__(self, index):
        img = Image.open(f"{CFG.train}{self.X.iloc[index]}").convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img) #.unsqueeze(0)
        return img_tensor, self.y[index]


X_train, X_test, y_train, y_test = train_test_split(train['Image_name'], train[CFG.diseases], test_size = 0.2)


y_train = torch.from_numpy(y_train.to_numpy().astype(np.float32))
y_test  = torch.from_numpy(y_test.to_numpy().astype(np.float32))


train_dataset = cdataset(X_train, y_train)
test_dataset = cdataset(X_test, y_test)


dl_train = DataLoader(train_dataset, batch_size=8, shuffle=True,)
dl_test  = DataLoader( test_dataset, batch_size=8, shuffle=False)


class DenseNet121(nn.Module):
    """
    Model modified.
    ``````````````
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=False)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cudnn.benchmark = True

# initialize and load the model
model = DenseNet121(CFG.n_classes).cuda()
model = torch.nn.DataParallel(model).cuda()

if os.path.isfile(CFG.modv1):
    print("=> loading checkpoint")
    checkpoint = torch.load(CFG.modv1)
    model.load_state_dict(checkpoint['state_dict'], strict = False)
    print("=> loaded checkpoint")
else:
    print("=> no checkpoint found")


summary(model, input_size = (1, 3, 224, 224))


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# for epoch in range(CFG.EPOCHS):
#     total_epoch_loss = 0
#     for batch_features, batch_labels in dl_train:
#         batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
#         output = model(batch_features)
#         loss = criterion(output, batch_labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_epoch_loss = total_epoch_loss + loss.item()
#     avg_loss = total_epoch_loss/len(dl_train)
#     if (epoch+1)%5 == 0:
#         print(f'Epoch: {epoch + 1} , Loss: {avg_loss}')


# for epoch in range(CFG.EPOCHS):
#     total_epoch_loss = 0
#     for batch_features, batch_labels in dl_train:
#         batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
#         output = model(batch_features)
#         loss = criterion(output, batch_labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_epoch_loss = total_epoch_loss + loss.item()
#     avg_loss = total_epoch_loss/len(dl_train)
#     if (epoch+1)%10 == 0:
#         print(f'Epoch: {epoch + 1} , Loss: {avg_loss}')

