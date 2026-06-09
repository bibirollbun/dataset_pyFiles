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


import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchinfo import summary

from torchvision.models import resnet152, ResNet152_Weights

import albumentations 
from albumentations.pytorch.transforms import ToTensorV2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold

import os
import copy
import glob
import json
import random
import pathlib
from PIL import Image
import pickle 


BASE_PATH = '/kaggle/input/cassava-leaf-disease-classification/'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {DEVICE}')


df = pd.read_csv(BASE_PATH + 'train.csv')
df.head(10)


labels = json.load(open(BASE_PATH + "label_num_to_disease_map.json"))
labels = {int(key):value for key, value in labels.items()}
labels


df.info()


df['label'].value_counts()


def display_images(label, rows, cols):
    
    number = rows * cols
    
    new_df = df[df['label'] == label]
    
    img_list = random.sample(new_df['image_id'].tolist(), number)
    
    plt.figure(figsize=(12, 9))
    for index, img_id in enumerate(img_list):
        plt.subplot(rows, cols, index+1)
        image = Image.open(BASE_PATH + "/train_images/" + img_id)
        plt.imshow(image, aspect='auto')
        plt.axis('off')
        
    plt.suptitle(f'\n\n Class {label}: ' + labels[label], fontsize=20)
    plt.tight_layout()


random.seed(42)
for i in range(5):
    display_images(label=i, rows=3, cols=4)


example_id = df['image_id'][19]
example_image = Image.open(BASE_PATH + "/train_images/" + example_id)
plt.figure()
plt.imshow(example_image)
plt.axis('off')
plt.show()


vertical_transform = albumentations.VerticalFlip(p=1)
augmented_image = vertical_transform(image=np.array(example_image))['image']
plt.figure()
plt.imshow(augmented_image)
plt.axis('off')
plt.show()


horizontal_transform = albumentations.HorizontalFlip(p=1)
augmented_image = horizontal_transform(image=np.array(example_image))['image']
plt.figure()
plt.imshow(augmented_image)
plt.axis('off')
plt.show()


random_crop_transform = albumentations.RandomResizedCrop(size=(512, 512))
augmented_image = random_crop_transform(image=np.array(example_image))['image']
plt.figure()
plt.imshow(augmented_image)
plt.axis('off')
plt.show()


transpose_transform = albumentations.Transpose(p=1)
augmented_image = transpose_transform(image=np.array(example_image))['image']
plt.figure()
plt.imshow(augmented_image)
plt.axis('off')
plt.show()


shift_scale_rotate_transform = albumentations.ShiftScaleRotate(p=1)
augmented_image = shift_scale_rotate_transform(image=np.array(example_image))['image']
plt.figure()
plt.imshow(augmented_image)
plt.axis('off')
plt.show()


hue_saturation_value_transform = albumentations.HueSaturationValue(
    hue_shift_limit=20,
    sat_shift_limit=50,
    val_shift_limit=20,
    p=1)
augmented_image = hue_saturation_value_transform(image=np.array(example_image))['image']
plt.figure()
plt.imshow(augmented_image)
plt.axis('off')
plt.show()


random_brightness_contrast_transform = albumentations.RandomBrightnessContrast(
    brightness_limit=(-0.1, 0.1), 
    contrast_limit=(-0.1, 0.1), 
    p=1)
augmented_image = hue_saturation_value_transform(image=np.array(example_image))['image']
plt.figure()
plt.imshow(augmented_image)
plt.axis('off')
plt.show()


torch.manual_seed(1)

num_classes = 5
num_folds = 5    # for stratified K-fold cross-validation

width = 512     # for image augmentation, we will resize image width and height
height = 512


skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)

df_train_folds = []
df_valid_folds = []

for i, (train_index, valid_index) in enumerate(skf.split(X=df['image_id'], y=df['label'])):
    
    df_train_folds.append(df.loc[list(train_index)])
    df_valid_folds.append(df.loc[list(valid_index)])
    
fold_number = 0

df_train = df_train_folds[fold_number]
df_valid = df_valid_folds[fold_number]

df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)


train_transforms = albumentations.Compose([
    
    albumentations.RandomResizedCrop(size=(height, width)),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.ShiftScaleRotate(p=0.5),
    albumentations.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
    albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    
])

valid_transforms = albumentations.Compose([
    albumentations.CenterCrop(width, height, p=1.0),
    albumentations.Resize(width, height),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class CassavaDataset(Dataset):
    
    def __init__(self, image_ids, labels, target_directory, transform=None):
        
        self.transform = transform
        self.image_ids = image_ids
        self.labels = labels
        self.target_directory = target_directory
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        
        img = Image.open(os.path.join(BASE_PATH, self.target_directory, self.image_ids[index]))
        img = np.array(img)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        
        if self.transform:
            return self.transform(image=img)['image'], label 
        else:
            return img, label 


train_directory = pathlib.Path("train_images")
batch_size = 16

train_dataset = CassavaDataset(image_ids=df_train.image_id, labels=df_train.label, target_directory=train_directory, transform=train_transforms)
valid_dataset = CassavaDataset(image_ids=df_valid.image_id, labels=df_valid.label, target_directory=train_directory, transform=valid_transforms)

train_dl = DataLoader(train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(valid_dataset, batch_size, shuffle=True)


def get_resnet_model():
    
    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    
    for params in model.parameters():
        params.requires_grad = False
        
    in_feat = model.fc.in_features
        
    model.fc = nn.Sequential(
          nn.Linear(in_feat, 256),
          nn.ReLU(),
          nn.Dropout(p=0.3),
          nn.Linear(256, num_classes))
    
    model = model.to(DEVICE)
    
    return model


model = get_resnet_model()

summary(model, input_size=(batch_size, 3, width, height))


def train(model, num_epochs, train_dl, valid_dl):
    
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_valid_loss = np.inf
    
    for epoch in range(num_epochs):
        
        model.train()
        
        batch_num = 0
        
        for x_batch, y_batch in train_dl:
            
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            batch_num += 1
            if (batch_num % 100 == 0):
                print(f'Batch number: {batch_num}')
            
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().item()
        
        
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        
        scheduler.step()
        
        model.eval()
        
        with torch.no_grad():
            
            for x_batch, y_batch in valid_dl:
                
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().item()
                
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        
        if accuracy_hist_valid[epoch] > best_acc:
            best_acc = accuracy_hist_valid[epoch]
            best_model_wts = copy.deepcopy(model.state_dict())
        
        print(f'Epoch {epoch+1}:   Train accuracy: {accuracy_hist_train[epoch]:.4f}    Validation accuracy: {accuracy_hist_valid[epoch]:.4f} ')
    
    
        if loss_hist_valid[epoch] < min_valid_loss:
            counter = 0
        else:
            counter += 1
    
        if counter >= patience:
            break
    
    
    model.load_state_dict(best_model_wts)
    
    history = {}
    history['loss_hist_train'] = loss_hist_train
    history['loss_hist_valid'] = loss_hist_valid
    history['accuracy_hist_train'] = accuracy_hist_train
    history['accuracy_hist_valid'] = accuracy_hist_valid
    
    return model, history


num_epochs = 10
patience = 3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)


best_model, hist = train(model, num_epochs, train_dl, valid_dl)


# Saving loss and accuracy history during initial training phase

with open('hist_fold_' + str(fold_number) + '.pkl', 'wb') as f:
    pickle.dump(hist, f)


for i, block in enumerate(model.children()):
    print('\n\n')
    print(f'Block {i}: \n\n')
    print(block)
    print('\n\n')


for params in model.parameters():
    params.requires_grad = False
    
unfreeze_layers = [7]
i = 0
for layer in model.children():
    if i in unfreeze_layers:
        for param in layer.parameters():
            param.requires_grad = True
    i += 1    


# Print model summary again

summary(model, input_size=(batch_size, 3, width, height))


best_model_tuned, hist_tuned = train(best_model, num_epochs, train_dl, valid_dl)


# Saving loss and accuracy history during finetuning phase

with open('hist_tuned_fold_' + str(fold_number) + '.pkl', 'wb') as f:
    pickle.dump(hist_tuned, f)


torch.save(best_model_tuned, '/kaggle/working/best_model_tuned_resnet152_10_epochs_fold_' + str(fold_number) + '.pt')
torch.save(best_model_tuned.state_dict(), '/kaggle/working/best_model_tuned_weights_resnet152_10_epochs_fold_' + str(fold_number) + '.pt')


fig, axs = plt.subplots(ncols=2, figsize=(6.5, 3))

axs[0].plot(range(1, (2*num_epochs)+1), hist['accuracy_hist_valid']+hist_tuned['accuracy_hist_valid'], '-o', color='tab:blue', label='Validation set')
axs[0].plot(range(1, (2*num_epochs)+1), hist['accuracy_hist_train']+hist_tuned['accuracy_hist_train'], '-o', color='tab:orange', label='Training set')
axs[1].plot(range(1, (2*num_epochs)+1), hist['loss_hist_valid']+hist_tuned['loss_hist_valid'], '-o', color='tab:red', label='Validation set')
axs[1].plot(range(1, (2*num_epochs)+1), hist['loss_hist_train']+hist_tuned['loss_hist_train'], '-o', color='tab:green', label='Training set')

axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_xticks([1, 5, 10, 15, 20])
axs[0].grid(alpha=0.1)
axs[0].legend(frameon=True, edgecolor='black', fontsize=8)
axs[0].axvline(x=10, color='black', linestyle='dashed', linewidth=1)
axs[0].text(x=1.6, y=0.875, s='Start Finetuning ⟶', fontsize=7)

axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_xticks([1, 5, 10, 15, 20])
axs[1].grid(alpha=0.1)
axs[1].legend(frameon=True, edgecolor='black', fontsize=8)
axs[1].axvline(x=10, color='black', linestyle='dashed', linewidth=1)


plt.tight_layout()
plt.show()


# Get lists of true labels and our predictions (from our model trained in this Notebook only) for images 
# in the validation set:

eval_dl = DataLoader(valid_dataset, batch_size=1, shuffle=False)

label_list = []
prediction_list = []

with torch.no_grad():
    for image, label in eval_dl:
        
        image = image.to(DEVICE)
        logits = best_model_tuned(image)
        probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        prediction = np.argmax(probs)
        label_list.append(label.numpy())
        prediction_list.append(prediction)
    


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print(classification_report(label_list, prediction_list))


cm = confusion_matrix(label_list, prediction_list)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, linewidth=1, linecolor='white')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

