import sys
import os
import sys
import json
import math
import random
import cv2
timm_path = "../input/timm-pytorch-image-models/pytorch-image-models-master"
sys.path.append(timm_path)
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm
import random
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch import optim
from torchvision import transforms
from transformers import  get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
set_seed(42)


df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')


df.isnull().sum()


df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['sex'] = df['sex'].fillna(-1)


df['age_approx'] /= 90
df['age_approx'] = df['age_approx'].fillna(0)


df['n_images'] = df.patient_id.map(df.groupby(['patient_id']).image_name.count())
df.loc[df['patient_id'] == -1, 'n_images'] = 1


df['path'] = [f"/kaggle/input/siic-isic-224x224-images/train/{x}.png" for x in df["image_name"].values]
dense_features = [
    'sex', 'age_approx', 'n_images'
]


strat_kfold = model_selection.StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)

# Create an empty 'fold' column in the DataFrame
df['fold'] = -1  # Initialize with a default value

for i, (_, train_index) in enumerate(strat_kfold.split(df.index, y=df['target'], groups=df['patient_id'])):
    df.loc[df.index[train_index], 'fold'] = i

df['fold'] = df['fold'].astype('int')


df.head()


df.sex.value_counts()


df.n_images = df.n_images/df.n_images.max()


image_size = 224
train_aug = A.Compose(
    [   A.RandomResizedCrop(image_size,image_size,p= 0.8),
        A.Resize(image_size,image_size,p=1.0),
        A.HorizontalFlip(p=0.5),   
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
          
   A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ]
)
val_aug = A.Compose(
    [ 
     A.Resize(image_size,image_size,p=1.0),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ]
)


class Op4bio_data(Dataset):
    def __init__(self,df, augs):
        self.df = df
        self.augs = augs
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        img_src = self.df.loc[idx,'path']
        image = cv2.imread(img_src)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = self.augs(image=image)
        image = transformed['image']
        
        meta = self.df[dense_features].iloc[idx, :].values
        
        label  = self.df['target'][idx]
        
        return image , torch.FloatTensor(meta) , label


t_data = Op4bio_data(df, augs = train_aug)


t_data[11]


class Model(nn.Module):
    def __init__(self,pretrained = True):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnet_b2_ns', pretrained=pretrained, num_classes=0, drop_rate=0.1,global_pool='',in_chans=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lin = nn.Linear(1408,64)
        self.norm = nn.BatchNorm1d(1408)
        self.do = nn.Dropout(p=0.4)
    
    def forward(self,image ):
        image = self.backbone(image)     
        image = self.pool(image)
        image = image.view(image.shape[0], -1)
        image = self.norm(image)
        x = self.lin(image)
        return x


embedding_size = 64

class Lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(3, 6, kernel_size=5)
        self.cnn2 = nn.Conv2d(6, 16, kernel_size=5)
        
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        # Calculate the input size for the fully connected layers
        self.fc_input_size = 16 * 53 * 53  # This is for 224x224 input images
        
        self.fc1 = nn.Linear(self.fc_input_size, 120)
        self.fc2 = nn.Linear(120, embedding_size)
    
    def forward(self, x):
        x = self.relu1(self.cnn1(x))
        x = self.pool1(x)
        x = self.relu2(self.cnn2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

    
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, embedding_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(embedding_size),
            nn.Linear( embedding_size,embedding_size),
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.squeeze(1)
        x = self.layer(x)
        #x = self.sigmoid(x)
        return x
    
    
class CLIPModel(nn.Module):
    def __init__(self, model1, model2):
        super(CLIPModel, self).__init__()
        self.model1 = model1  # Lenet
        self.model2 = model2  # Logistic
        self.t = nn.Parameter(torch.Tensor([1]))  # Learnable parameter

    def forward(self, x, y):
        out1 = self.model1(x)
        out2 = self.model2(y)
        
        #print(out1)
        #print(out2)

        # Apply L2 normalization
        out1_norm = F.normalize(out1, p=2, dim=1)
        out2_norm = F.normalize(out2, p=2, dim=1)
        
        out1_norm = out1_norm.to(out2_norm.dtype)  # Cast out1_norm to the same dtype as out2_norm


        # Compute dot product
        dot_product = torch.matmul(out1_norm, out2_norm.t())

        # Apply exponential weighting
        dot_product_weighted = torch.exp(self.t) * dot_product

        return dot_product_weighted


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(train_loader,model,optimizer,criterion,e,epochs,device):
    '''Trains the model for a single epoch and returns Loss,Accuracy, AUC for that epoch'''

    losses = AverageMeter()
    model.train()
    global_step = 0
    loop = tqdm(enumerate(train_loader),total = len(train_loader))
    
    for step,(image,tabular,l) in loop:
        image = image.to(device)
        tabular = tabular.to(device)
        
        batch_size = l.size(0)

        output = model(image,tabular)
        
        #print(output)
        
        labels = torch.arange(batch_size) 
        labels = labels.to(device)
        
        loss1 = criterion(output, labels)
        loss2 = criterion(output.T, labels)
        loss = (loss1 + loss2) / 2.0
        
        losses.update(loss.item(), batch_size)

        loss.backward()
         # Clip gradients to avoid gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm value as needed
        
        
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1

        loop.set_description(f"Epoch {e+1}/{epochs}")
        loop.set_postfix(model_loss = loss.item() ,stage = 'train')        

    return losses.avg


def val_one_epoch(loader,model,criterion,device):
    '''Validates the model for a single epoch and returns Loss,Accuracy, AUC for that epoch'''
    losses = AverageMeter()
    model.eval()
    global_step = 0
    loop = tqdm(enumerate(loader),total = len(loader))
    
    for step,(image,tabular,l) in loop:
        image = image.to(device)
        tabular = tabular.to(device)
        batch_size = l.size(0)
        
        labels = torch.arange(batch_size) 
        labels = labels.to(device)
        
        with torch.no_grad():
            output = model(image,tabular)

        loss1 = criterion(output, labels)
        loss2 = criterion(output.T, labels)
        loss = (loss1 + loss2) / 2.0        
        losses.update(loss.item(), batch_size)
        loop.set_postfix(model_loss = loss.item() ,stage = 'val')
        global_step += 1

    return losses.avg


def fit(t_loader ,v_loader, model, OUTPUT_DIR,device,optimizer):
    
    T_LOSS1 = []
    V_LOSS1 = []
    model.to(device)
    #model.to(device)
    
    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optimizer
    
    epochs = 25

    loop = range(epochs)
    for e in loop:
      
        loss = train_one_epoch(t_loader,model,optimizer,criterion,e,epochs,device)
        
        print(f'For epoch {e+1}/{epochs}')
        print(f'average model train_loss {loss}')
        
        T_LOSS1.append(loss)

        val_loss = val_one_epoch(v_loader,model,criterion,device)
        
        print(f'average model val_loss {val_loss}')
        
        V_LOSS1.append(val_loss)

        torch.save(model.state_dict(),OUTPUT_DIR+ f'model_val_loss {val_loss}.pth')

    return T_LOSS1,V_LOSS1


train_data= df[df.fold != 0]
val_data  = df[df.fold == 0]
    
t_data = Op4bio_data(train_data.reset_index(drop=True) , augs = train_aug)
v_data = Op4bio_data(val_data.reset_index(drop=True) , augs = val_aug)


t_loader = DataLoader(t_data, shuffle=True,
                        num_workers=2,
                        batch_size=16*6,drop_last =True)

v_loader = DataLoader(v_data, shuffle=False,
                        num_workers=2,
                        batch_size=16*8,drop_last =False)


OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


model1 =  Model()
model2 =  LogisticRegression()


combined_model = CLIPModel(model1, model2)
optimizer = optim.AdamW(combined_model.parameters(), lr=1e-6 , weight_decay = 1e-5 ) 
T_LOSS1, V_LOSS1= fit(t_loader ,v_loader, combined_model, OUTPUT_DIR,device,optimizer )


import matplotlib.pyplot as plt

# Generate x-axis values
epochs = len(T_LOSS1)
x = list(range(1, epochs + 1))

# Plot training loss
plt.plot(x, T_LOSS1, label='Training Loss')

# Plot validation loss
plt.plot(x, V_LOSS1, label='Validation Loss')

# Set plot labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Show the plot
plt.show()

