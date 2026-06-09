!pip install -U albumentations


import pandas as pd
import numpy as np
import cv2
import os
import hashlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import (
    Compose, Resize, HorizontalFlip, ShiftScaleRotate, RandomResizedCrop, 
    GaussianBlur, MedianBlur, RandomBrightnessContrast, Normalize
)

import warnings  
warnings.filterwarnings('ignore')


DIR_INPUT = '/kaggle/input/plant-pathology-2020-fgvc7'

SEED = 42
NUM_EPOCHS = 50
BATCH_SIZE = 64
HEIGHT = 360
WIDTH = 512
NUM_WORKERS = 4


train=pd.read_csv(DIR_INPUT + "/train.csv")
test=pd.read_csv(DIR_INPUT + "/test.csv")

base_path=DIR_INPUT+"/images/"
def generate_image_path(image_id):
    return f"{base_path}{image_id}.jpg"

train['img_path'] = train['image_id'].apply(generate_image_path)
test['img_path'] = test['image_id'].apply(generate_image_path)


def calculate_hash(im):
    md5 = hashlib.md5()
    md5.update(np.array(im).tostring())
    
    return md5.hexdigest()
    
def get_image_meta(image_id, image_src, dataset='train'):
    im = Image.open(image_src)
    extrema = im.getextrema()

    meta = {
        'image_id': image_id,
        'dataset': dataset,
        'hash': calculate_hash(im),
        'r_min': extrema[0][0],
        'r_max': extrema[0][1],
        'g_min': extrema[1][0],
        'g_max': extrema[1][1],
        'b_min': extrema[2][0],
        'b_max': extrema[2][1],
        'height': im.size[0],
        'width': im.size[1],
        'format': im.format,
        'mode': im.mode
    }
    return meta


data = []

for i, image_id in enumerate(tqdm(train['image_id'].values, total=train.shape[0])):
    data.append(get_image_meta(image_id, DIR_INPUT + '/images/{}.jpg'.format(image_id)))


for i, image_id in enumerate(tqdm(test['image_id'].values, total=test.shape[0])):
    data.append(get_image_meta(image_id, DIR_INPUT + '/images/{}.jpg'.format(image_id), 'test'))


meta = pd.DataFrame(data)
meta.head()


meta.groupby(by='dataset')[['width', 'height']].aggregate(['min', 'max'])


duplicates = meta.groupby(by='hash')[['image_id']].count().reset_index()
duplicates = duplicates[duplicates['image_id'] > 1]
duplicates.reset_index(drop=True, inplace=True)

duplicates = duplicates.merge(meta[['image_id', 'hash']], on='hash')

duplicates.head(20)


fig, ax = plt.subplots(5, 2, figsize=(8, 16))
ax = ax.flatten()

for i in range(0, min(duplicates.shape[0], 10), 2):
    image_i = cv2.imread(DIR_INPUT + '/images/{}.jpg'.format(duplicates.iloc[i, 2]), cv2.IMREAD_COLOR)
    image_i = cv2.cvtColor(image_i, cv2.COLOR_BGR2RGB)
    ax[i].set_axis_off()
    ax[i].imshow(image_i)
    ax[i].set_title(duplicates.iloc[i, 2])
    
    image_i_1 = cv2.imread(DIR_INPUT + '/images/{}.jpg'.format(duplicates.iloc[i + 1, 2]), cv2.IMREAD_COLOR)
    image_i_1 = cv2.cvtColor(image_i_1, cv2.COLOR_BGR2RGB)
    ax[i + 1].set_axis_off()
    ax[i + 1].imshow(image_i_1)
    ax[i + 1].set_title(duplicates.iloc[i + 1, 2])


labels = ['healthy', 'multiple_diseases', 'rust', 'scab']
diseases = dict()
for column in labels:
    counts = pd.DataFrame(train[column].value_counts())
    diseases[column] = counts.iloc[1,0]

diseases


plt.bar(diseases.keys(),diseases.values(), color=["#6666ff"])
plt.title('Bar Chart', fontsize=18)
plt.show()


explode = (0.02,0.02,0.02,0.02)
plt.pie(diseases.values(),labels = diseases.keys(), colors=["#6666ff","#4da6ff","#1bc7ff","#c44dff"], autopct='%1.1f%%', explode=explode)
plt.title('Pie Chart', fontsize=18)
plt.axis('equal') 
plt.show()


torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Custom Dataset class
class PlantDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.labels = ['healthy', 'multiple_diseases', 'rust', 'scab']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, self.df.iloc[idx]['image_id']
        else:
            labels = torch.tensor(
                self.df.iloc[idx][self.labels].values.astype(np.float32)
            )
            return image, labels



train_transform = transforms.Compose([
        transforms.RandomResizedCrop((HEIGHT, WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=30,translate=(0.1,0.1),scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=0.2,contrast=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_idx, val_idx = train_test_split(
    range(len(train)),
    test_size=0.2,
    random_state=SEED,
    stratify=train[labels].values.argmax(axis=1)
)

train_dataset = PlantDataset(
    train.iloc[train_idx].reset_index(drop=True),
    transform=train_transform
)

val_dataset = PlantDataset(
    train.iloc[val_idx].reset_index(drop=True),
    transform=val_transform
)

test_dataset = PlantDataset(
    test,
    transform=val_transform,
    is_test=True
)


train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)


class PlantModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(PlantModel, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        for param in self.backbone.fc.parameters():
            param:requires_grad = True

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
        
model = PlantModel(num_classes=len(labels))
model = model.to(device)


#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({'loss': loss.item()})
        
        optimizer.step()
        optimizer.zero_grad()
        
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = None
    all_targets = None
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            preds = torch.softmax(outputs, dim=1).data.cpu()
            if all_preds is None:
                all_preds = preds
            else:
                all_preds = torch.cat((all_preds, preds), dim=0)
            if all_targets is None:
                all_targets = labels.clone().squeeze(-1).cpu()
            else:
                all_targets = torch.cat((all_targets, labels.squeeze(-1).cpu()), dim=0)
    
    val_loss = running_loss / len(dataloader.dataset)
    return val_loss, all_preds, all_targets


train_losses = []
val_losses = []
val_scores = []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)

    val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)
    val_losses.append(val_loss)

    print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    validation_score = roc_auc_score(val_targets, val_preds, average='macro')
    val_scores.append(validation_score)
    print(f"Validation score: {validation_score:.4f}")
    
    scheduler.step(val_loss)


plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS+1), train_losses, label='Train Loss')
plt.plot(range(1, NUM_EPOCHS+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_history.png')
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS+1), val_scores, label='Validation Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation Score')
plt.legend()
plt.grid(True)
plt.savefig('score_history.png')
plt.show()


model.eval()
test_predictions = []
test_ids = []

with torch.no_grad():
    for inputs, image_ids in tqdm(test_loader, desc='Testing'):
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs).cpu().numpy()
        
        test_predictions.append(preds)
        test_ids.extend(image_ids)
test_predictions = np.concatenate(test_predictions, axis=0)

submission_df = pd.DataFrame({'image_id': test_ids})
for i, class_name in enumerate(labels):
    submission_df[class_name] = test_predictions[:, i]

submission_df.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


submission_df

