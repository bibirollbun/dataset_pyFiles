import os
import random

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm.notebook import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For scheduler
from transformers import get_cosine_schedule_with_warmup

# Make plots inline
%matplotlib inline



# Fix Seed
seed = 10

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   # keep False for determinism
torch.backends.cudnn.enabled = True      # <-- don't turn this off

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))



# PREPROCESS IMAGES ONE TIME (resize to 600x600)

input_img_dir  = '/kaggle/input/plant-pathology-2020-fgvc7/images'
resized_img_dir = '/kaggle/working/resized_600'
IMG_SIZE = (600, 600)

os.makedirs(resized_img_dir, exist_ok=True)

from tqdm import tqdm

print("Resizing images to 600x600 (only for files not yet processed)...")
for fname in tqdm(os.listdir(input_img_dir)):
    src_path = os.path.join(input_img_dir, fname)
    dst_path = os.path.join(resized_img_dir, fname)

    if os.path.exists(dst_path):
        continue  # skip already processed

    img = cv2.imread(src_path)
    if img is None:
        continue
    img = cv2.resize(img, IMG_SIZE)
    cv2.imwrite(dst_path, img)

print("Done preprocessing images.")



data_path = '/kaggle/input/plant-pathology-2020-fgvc7/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')

print(train.shape, test.shape)
display(train.head())
display(test.head())
display(submission.head())



healthy = train.loc[train['healthy'] == 1]
multiple_diseases = train.loc[train['multiple_diseases'] == 1]
rust = train.loc[train['rust'] == 1]
scab = train.loc[train['scab'] == 1]

mpl.rc('font', size=15)
plt.figure(figsize=(7, 7))

labels = ['healthy', 'multiple diseases', 'rust', 'scab']
plt.pie(
    [len(healthy), len(multiple_diseases), len(rust), len(scab)],
    labels=labels,
    autopct='%1.1f%%'
);



def show_image(img_ids, rows=4, cols=3):
    assert len(img_ids) <= rows * cols

    plt.figure(figsize=(15, 15))
    grid = gridspec.GridSpec(rows, cols)

    for idx, img_id in enumerate(img_ids):
        img_path = f'{input_img_dir}/{img_id}.jpg'
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(grid[idx])
        ax.imshow(image)
        ax.axis('off')

last_healthy_img_ids = healthy['image_id'][-12:]
last_multiple_diseases_img_ids = multiple_diseases['image_id'][-12:]
last_rust_img_ids = rust['image_id'][-12:]
last_scab_img_ids = scab['image_id'][-12:]

show_image(last_healthy_img_ids)
show_image(last_multiple_diseases_img_ids)
show_image(last_rust_img_ids)
show_image(last_scab_img_ids)



# Split train data and valid data
_, valid = train_test_split(
    train,
    test_size=0.1,
    stratify=train[['healthy', 'multiple_diseases', 'rust', 'scab']],
    random_state=10
)

print("Train size:", len(train))
print("Valid size:", len(valid))



class ImageDataset(Dataset):
    def __init__(self, df, img_dir='./', transform=None, is_test=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        if self.is_test:
            return image
        else:
            label = np.argmax(self.df.iloc[idx, 1:5].values)
            return image, label



# Transformers for train data
transform_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.1,
                               contrast_limit=0.1, p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=25, p=0.7),
    A.OneOf([A.Emboss(p=1),
             A.Sharpen(p=1),
             A.Blur(p=1)], p=0.5),
    A.PiecewiseAffine(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

# Transformers for valid and test data
transform_test = A.Compose([
    A.Normalize(),
    ToTensorV2()
])



resized_dir = '/kaggle/working/resized_600/'

dataset_train = ImageDataset(train, img_dir=resized_dir, transform=transform_train)
dataset_valid = ImageDataset(valid, img_dir=resized_dir, transform=transform_test)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

batch_size = 4  # keep as in your original; you can try 8 later

loader_train = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,      
    pin_memory=False,   
)

loader_valid = DataLoader(
    dataset_valid,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,      
    pin_memory=False,  
)



!pip install -q efficientnet-pytorch==0.7.1

from efficientnet_pytorch import EfficientNet

# Load pre-trained efficientnet-b7 model
model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=4)
model = model.to(device)

print("Model on device:", next(model.parameters()).device)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00007, weight_decay=0.0001)

epochs = 38

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(loader_train) * 5,
    num_training_steps=len(loader_train) * epochs
)



for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0

    for images, labels in tqdm(loader_train):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_train_loss += loss.item()
        
        # ⚠️ free GPU memory
        del images, labels, outputs, loss
        torch.cuda.empty_cache()
    
    print(f'Epoch [{epoch+1}/{epochs}] - Train data loss : {epoch_train_loss/len(loader_train):.4f}')
    
    model.eval()
    epoch_valid_loss = 0
    preds_list = []
    true_onehot_list = []
    
    with torch.no_grad():
        for images, labels in loader_valid:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_valid_loss += loss.item()
            
            preds = torch.softmax(outputs.cpu(), dim=1).numpy()
            true_onehot = torch.eye(4)[labels].cpu().numpy()
            preds_list.extend(preds)
            true_onehot_list.extend(true_onehot)

            # free GPU memory
            del images, labels, outputs, loss
            torch.cuda.empty_cache()
        
        print(f'Epochs [{epoch+1}/{epochs}] - Valid data loss : {epoch_valid_loss/len(loader_valid):.4f} / Valid data ROC AUC : {roc_auc_score(true_onehot_list, preds_list):.4f}')



dataset_test = ImageDataset(test, img_dir=resized_dir, 
                            transform=transform_test, is_test=True)
loader_test = DataLoader(dataset_test, batch_size=batch_size, 
                         shuffle=False, worker_init_fn=seed_worker,
                         generator=g, num_workers=2, pin_memory=True)

# TTA dataset uses train transform (augmentations)
dataset_TTA = ImageDataset(test, img_dir=resized_dir, 
                           transform=transform_train, is_test=True)
loader_TTA = DataLoader(dataset_TTA, batch_size=batch_size, 
                        shuffle=False, worker_init_fn=seed_worker,
                        generator=g, num_workers=2, pin_memory=True)



model.eval()

preds_test = np.zeros((len(test), 4))

with torch.no_grad():
    for i, images in enumerate(loader_test):
        images = images.to(device)
        outputs = model(images)
        preds_part = torch.softmax(outputs.cpu(), dim=1).squeeze().numpy()
        preds_test[i * batch_size:(i + 1) * batch_size] += preds_part

submission_test = submission.copy()
submission_test[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds_test



num_TTA = 5

preds_tta = np.zeros((len(test), 4))

for t in range(num_TTA):
    with torch.no_grad():
        for i, images in enumerate(loader_TTA):
            images = images.to(device)
            outputs = model(images)
            preds_part = torch.softmax(outputs.cpu(), dim=1).squeeze().numpy()
            preds_tta[i * batch_size:(i + 1) * batch_size] += preds_part

preds_tta /= num_TTA

submission_tta = submission.copy()
submission_tta[['healthy', 'multiple_diseases', 'rust', 'scab']] = preds_tta

submission_test.to_csv('submission_test.csv', index=False)
submission_tta.to_csv('submission_tta.csv', index=False)



def apply_label_smoothing(df, target, alpha, threshold):
    df_target = df[target].copy()
    k = len(target)
    
    for idx, row in df_target.iterrows():
        if (row > threshold).any():
            row = (1 - alpha) * row + alpha / k
            df_target.iloc[idx] = row
    return df_target

alpha = 0.01
threshold = 0.99

submission_test_ls = submission_test.copy()
submission_tta_ls = submission_tta.copy()
target_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']

submission_test_ls[target_cols] = apply_label_smoothing(submission_test_ls, target_cols, alpha, threshold)
submission_tta_ls[target_cols] = apply_label_smoothing(submission_tta_ls, target_cols, alpha, threshold)

submission_test_ls.to_csv('submission_test_ls.csv', index=False)
submission_tta_ls.to_csv('submission_tta_ls.csv', index=False)



path = './'

torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
    }, path + 'EfficientNet-B7.tar')


