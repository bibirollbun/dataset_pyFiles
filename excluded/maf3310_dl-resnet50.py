import os
os.makedirs('/root/.kaggle', exist_ok=True)
with open('/root/.kaggle/kaggle.json', 'w') as f:
    f.write('{"username":"maf3310","key":"362ab0114dc18171a899d60b98355870"}')  # æ›¿æ�›ç‚ºæ–°é‡‘é‘°
!chmod 600 /root/.kaggle/kaggle.json


!ls -l /root/.kaggle/


import os
dataset_path = '/kaggle/input/plant-pathology-2020-fgvc7'
print("æ•¸æ“šé›†å…§å®¹ï¼š", os.listdir(dataset_path))


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ä½¿ç”¨è¨­å‚™ï¼š{device}")


# ğŸ“¦ åŒ¯å…¥åŸºæœ¬å¥—ä»¶
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image

# âœ… ç¢ºèª�ä½¿ç”¨ GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ä½¿ç”¨è¨­å‚™ï¼š{device}")

# âœ… Kaggle è³‡æ–™é›†è·¯å¾‘
dataset_path = "/kaggle/input/plant-pathology-2020-fgvc7"
print("æ•¸æ“šé›†å…§å®¹ï¼š", os.listdir(dataset_path))

# âœ… è¼‰å…¥è¨“ç·´è³‡æ–™
train_df = pd.read_csv(os.path.join(dataset_path, "train.csv"))
labels = ['healthy', 'multiple_diseases', 'rust', 'scab']
print(train_df.head())

# âœ… é¡�åˆ¥åˆ†ä½ˆè¦–è¦ºåŒ–
class_counts = train_df[labels].sum().reset_index()
class_counts.columns = ['é¡�åˆ¥', 'æ•¸é‡�']
sns.barplot(data=class_counts, x='é¡�åˆ¥', y='æ•¸é‡�')
plt.title('é¡�åˆ¥åˆ†ä½ˆ')
plt.show()

#  å®šç¾© Dataset é¡�åˆ¥
class PlantDataset(Dataset):
    def __init__(self, df, img_dir, labels, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, row['image_id']
        label = torch.tensor(row[self.labels].values.astype(np.float32))
        return image, label

# âœ… è³‡æ–™å¢�å¼·
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# âœ… å»ºç«‹ Dataset èˆ‡ DataLoader
image_dir = os.path.join(dataset_path, "images")
train_dataset = PlantDataset(train_df, image_dir, labels, train_transform)
test_df = pd.read_csv(os.path.join(dataset_path, "test.csv"))
test_dataset = PlantDataset(test_df, image_dir, labels, test_transform, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# âœ… æ¨¡å�‹å»ºç«‹ï¼ˆæ‰‹å‹•è¼‰å…¥ ResNet50 æ¬Šé‡�ï¼‰
weights_path = "/kaggle/input/resnet50/resnet50-0676ba61.pth"
model = models.resnet50(weights=None)
state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict)
model.fc = nn.Linear(model.fc.in_features, len(labels))
model = model.to(device)

# âœ… æ��å¤±èˆ‡å„ªåŒ–å™¨
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

#  è¨“ç·´å‡½å¼�ï¼ˆåŠ å…¥ precision/recall/f1-scoreï¼‰
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=5):
    best_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        scheduler.step()
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"ğŸ“˜ Epoch {epoch+1}/{num_epochs} | Loss: {running_loss / len(train_loader):.4f} | "
              f"ğŸ�¯ Precision: {precision:.4f}, ğŸ§  Recall: {recall:.4f}, â­� F1-score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_model.pth")
            print("âœ… æ¨¡å�‹å·²æ›´æ–°èˆ‡å„²å­˜")

# âœ… é–‹å§‹è¨“ç·´
train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=5)




# ğŸ§  æ�¨è«–æ¨¡å�‹
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

all_preds = []
image_ids = []

with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.extend(preds)
        image_ids.extend(ids)

# âœ… ç”¢ç”Ÿæ��äº¤æª” submission.csv
submission = pd.DataFrame([[img_id] + list(pred) for img_id, pred in zip(image_ids, all_preds)],
                          columns=["image_id"] + labels)
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv å·²æˆ�åŠŸç”¢å‡ºï¼�")

# ğŸ”— å»ºç«‹ä¸‹è¼‰é€£çµ�
from IPython.display import FileLink
FileLink("submission.csv")


!ls -lh submission.csv

