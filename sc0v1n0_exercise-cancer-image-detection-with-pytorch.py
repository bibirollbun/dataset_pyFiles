import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import accumulate
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

%matplotlib inline

import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset, random_split
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

sns.set_context('notebook')
sns.set_style('white')


path_data = '/kaggle/input/histopathologic-cancer-detection'
print(os.listdir(path_data))

print(os.listdir(os.path.join(path_data, 'train'))[:5])
print(os.listdir(os.path.join(path_data, 'test'))[:5])


from torch.cuda import is_available, get_device_name

if is_available():
    print(f"The environment has a compatible GPU ({get_device_name()}) available.")
else:
    print(f"The environment does NOT have a compatible GPU model available.")


from numpy import clip , array
from matplotlib import pyplot as plt
from torch import Tensor

def imshow(inp: Tensor) -> None:
    """Imshow for Tensor."""
    inp = inp.cpu().numpy()
    inp = inp.transpose((1, 2, 0))
    mean = array([0.485, 0.456, 0.406])
    std = array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()


## Load the label of data
labels_df = pd.read_csv("/kaggle/input/histopathologic-cancer-detection/train_labels.csv")
labels_df.head()


fig = plt.figure(figsize=(25, 8))

train_imgs = os.listdir(os.path.join(path_data, 'train'))

for idx, img in enumerate(np.random.choice(train_imgs, 40)):

    ax = fig.add_subplot(4, 40//4, idx+1)

    im = Image.open(os.path.join(path_data, 'train', img))

    plt.imshow(im)

    lab = labels_df.loc[labels_df["id"] == img.split('.')[0], 'label'].values[0]

    ax.set_title(f"Label: {lab}")



labels_df.shape


labels_df = labels_df.set_index('id')
labels_df


import random

random.seed(666)
idx = list(labels_df.index)
idx[:5]


idx_p = list(range(len(idx)))
print(idx_p[:5])
random.shuffle(idx_p)
print(idx_p[:5])


idx_random = [idx[x] for x in idx_p]
idx_random[:5]


total = len(idx_random)
idx_70 = int(total * 0.70)
idx_90 = int(total * 0.90)

idx_frac_70 = idx_random[:idx_70] 
idx_frac_20 = idx_random[idx_70:idx_90]
idx_frac_10 = idx_random[idx_90:] 

print(f"(70%): {len(idx_frac_70)} itens")
print(f"(20%): {len(idx_frac_20)} itens")
print(f"(10%): {len(idx_frac_10)} itens")
print(f"Total: {len(idx_frac_70) + len(idx_frac_20) + len(idx_frac_10)} itens")


from pathlib import Path

class CancerDataset(Dataset):
    def __init__(
        self,
        path_to_dataset: str,
        transform,
        dataset_type=None):

        path_dataset = Path(path_to_dataset)
        if not path_dataset.is_dir():
            raise OSError('This is not directory')

        check_data_split =  ['train', 'test']
        sub_dirs = [os.path.basename(str(x)) for x in path_dataset.iterdir()]
        
        if not check_data_split[0] in sub_dirs and not check_data_split[1] in sub_dirs:
            raise Exception('Does not exists train dir or test dir')

        self.path_dataset_train = path_dataset.joinpath(check_data_split[0])
        self.path_dataset_test = path_dataset.joinpath(check_data_split[1])

        if not 'train_labels.csv' in sub_dirs:
            raise Exception('File labels does not found')
        
        self.labels_file = path_dataset / "train_labels.csv"
        self.df_labels = pd.read_csv(self.labels_file)
        self.df_labels.set_index("id", inplace=True)

        random.seed(666)
        idx = list(labels_df.index)
        idx_p = list(range(len(idx)))
        random.shuffle(idx_p)
        idx_random = [idx[x] for x in idx_p]

        total = len(idx_random)
        idx_70 = int(total * 0.70)
        idx_90 = int(total * 0.90)
        
        idx_frac_70 = idx_random[:idx_70] 
        idx_frac_20 = idx_random[idx_70:idx_90]
        idx_frac_10 = idx_random[idx_90:] 
        
        #print(f"Total: {len(idx_frac_70) + len(idx_frac_20) + len(idx_frac_10)} itens")
    
        if dataset_type == "train":
            self.labels = list(self.df_labels.loc[idx_frac_70, 'label'])
            self.full_filenames = [self.path_dataset_train / f"{f}.tif" for f in idx_frac_70]
            print(f"(70%): {len(idx_frac_70)} itens")
            print("training dataset")
            
        elif dataset_type == "val":
            self.labels = list(self.df_labels.loc[idx_frac_20, 'label'])
            self.full_filenames = [self.path_dataset_train / f"{f}.tif" for f in idx_frac_20]
            print(f"(20%): {len(idx_frac_20)} itens")
            print("validation dataset")
            
        elif dataset_type == "test":
            self.labels = list(self.df_labels.loc[idx_frac_10, 'label'])
            self.full_filenames = [self.path_dataset_train / f"{f}.tif" for f in idx_frac_10]
            print("testing dataset")
            print(f"(10%): {len(idx_frac_10)} itens")
            
        else:
            raise Exception('Fail')

        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        img = Image.open(self.full_filenames[idx]) # PIL image
        img = self.transform(img)
        
        return img, self.labels[idx]


from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform_train = transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(degrees=5),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)
                               ])

transform_pos_processing = transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)
                               ])


path_data


str(next(Path(path_data).iterdir()))


training_set = CancerDataset(
    path_to_dataset=path_data,
    transform=transform_train,
    dataset_type="train"
)
 
validation_set = CancerDataset(
    path_to_dataset=path_data,
    transform=transform_pos_processing,
    dataset_type="val"
)

test_set = CancerDataset(
    path_to_dataset=path_data,
    transform=transform_pos_processing,
    dataset_type="test"
)

print(f'training dataset length: {len(training_set)}')
print(f'validation dataset length: {len(validation_set)}')
print(f'test dataset length: {len(test_set)}')



train_loader = torch.utils.data.DataLoader(
    training_set,
    batch_size=10,
    shuffle=True,
    num_workers=2)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=10,
    shuffle=False,
    num_workers=2)

N_CLASSES: int = 2
BATCH_SIZE: int = 30
LEARNING_RATE: float = 3e-4
N_EPOCHS: int = 2
print("done")


from torchvision import models

model = models.resnet34(
    pretrained=True
)


from torch.nn import CrossEntropyLoss

criterion = CrossEntropyLoss()


from torch.optim import Adam

optimizer: Adam = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, N_CLASSES)


for epoch in range(N_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader)):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 100 == 99:
            tqdm.write('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0



try:
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
except Exception as e:
    print(e, flush=True)


