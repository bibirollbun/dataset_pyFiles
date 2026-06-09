import torch
import torch.nn as nn

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tqdm import tqdm
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


data_root_path = '/kaggle/input/ai-vs-human-generated-dataset'

train_df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/train.csv')
train_df.head(10)


len(train_df)


test_df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/test.csv')
test_df.head()


len(test_df)


image_0_url = 'train_data/041be3153810433ab146bc97d5af505c.jpg'
read_image_0 = cv2.imread(os.path.join(data_root_path,image_0_url))
read_image_0 = cv2.cvtColor(read_image_0,cv2.COLOR_BGR2RGB)
plt.imshow(read_image_0)
plt.title('Sample 0')
plt.axis('off')
plt.show()


image_1_url = 'train_data/a6dcb93f596a43249135678dfcfc17ea.jpg'
read_image_1 = cv2.imread(os.path.join(data_root_path,image_1_url))
read_image_1 = cv2.cvtColor(read_image_1,cv2.COLOR_BGR2RGB)
plt.imshow(read_image_1)
plt.title('Sample 1')
plt.axis('off')
plt.show()


train_df = train_df.drop(columns = ['Unnamed: 0'])


train_df['file_name'] = train_df['file_name'].apply(lambda x: os.path.join(data_root_path, x))


train_df.head()


train_df, val_df = train_test_split(train_df, test_size=0.2, random_state = 42)
len(train_df), len(val_df)


read_data = cv2.imread(train_df['file_name'][0])
plt.imshow(read_data)
plt.axis('off')
plt.show()


class ImageDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = pd.DataFrame(df)  
        self.transform = transform
        self.is_test = is_test  

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["file_name"]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img  
        else:
            label = self.df.iloc[idx]["label"]
            return img, label


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = ImageDataset(df=train_df, transform=train_transform, is_test=False)
val_dataset = ImageDataset(df=val_df, transform=val_transform, is_test=False)


batch_size = 64


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


batch_images, batch_labels = next(iter(train_loader))
print(f"Batch shape: {batch_images.shape}, Labels shape: {batch_labels.shape}")


test_img = batch_images[1].permute(1,2,0)
plt.imshow(test_img)
plt.axis('off')
plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
num_classes = 2
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
model = model.to(device)


for param in model.features.parameters():
    param.requires_grad = False


lr = 1e-3
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
num_epochs=10


train_losses = []
train_accuracy = []

val_losses = []
val_accuracy = []

for epoch in range(num_epochs):
# Training 
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).long()  

        outputs = model(images) 
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    train_losses.append(train_avg_loss)
    train_accuracy.append(train_acc)
    
#Valiadation
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    val_pred_classes, val_labels_list = [], []

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device).long()  
    
            outputs = model(images) 
            loss = criterion(outputs, labels)
    
            val_running_loss += loss.item()
    
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

            val_pred_classes.extend(predicted.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())
    
        val_avg_loss = val_running_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

    val_losses.append(val_avg_loss)
    val_accuracy.append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] | Train_Loss: {train_avg_loss:.4f} | Train_Accuracy: {train_acc:.4f} | Val_Loss: {val_avg_loss:.4f} | Val_Accuracy: {val_acc:.4f}')


plt.plot(train_accuracy,marker = 'o')
plt.plot(val_accuracy,marker = 'v')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train','Validation'])
plt.show()


plt.plot(train_losses,marker = 'o')
plt.plot(val_losses,marker = 'v')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train','Validation'])
plt.show()


conf_matrix = confusion_matrix(val_pred_classes, val_labels_list)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


torch.save(model.state_dict(), 'best_model.pth')

