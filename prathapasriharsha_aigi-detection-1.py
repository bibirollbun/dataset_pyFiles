import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/detect-ai-vs-human-generated-images/train.csv")
train_df.drop(columns=["Unnamed: 0"],inplace=True)
train_df.head()


def preprocess_image(img_path, img_size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    return img
    
class AIGIDataset(Dataset):
     def __init__(self, df, img_dir, transform=None):
         self.df = df
         self.img_dir = img_dir
         self.transform = transform
     
     def __len__(self):
         return len(self.df)

     def __getitem__(self, idx):
         img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
         img = preprocess_image(img_path,(224,224))
         label = self.df.iloc[idx, 1]
         if self.transform:
             img = self.transform(image=img)["image"]
         return img, torch.tensor(label, dtype=torch.long)

transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


train_ds = AIGIDataset(train_df,"/kaggle/input/ai-vs-human-generated-dataset/",transform = transform)
train_loader = DataLoader(train_ds,batch_size=64, shuffle=True, num_workers=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


model = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=2)
# model = torch.compile(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


num_epochs = 10
clip_grad = 1.0

for epoch in range(num_epochs):
    
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    # train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)  # Apply gradient clipping
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # train_loader_tqdm.set_postfix(loss=loss.item(), acc=100 * correct / total)

    avg_train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Train_acc: {train_acc:.2f}%")


torch.save(model.state_dict(), "model.pth")




