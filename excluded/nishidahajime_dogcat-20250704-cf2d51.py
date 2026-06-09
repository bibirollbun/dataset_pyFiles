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


import pandas as pd
import os, glob, time, copy, zipfile
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt


size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 32
num_epoch = 25
learning_rate = 0.0001
patience = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, img, phase):
        return self.data_transform[phase](img)


class DogCatDataset(data.Dataset):
    def __init__(self, file_list, labels, transform=None, phase='train'):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img, self.phase)
        label = self.labels[index]
        return img, label


!rm -rf ./data
!rm -f ./best_model.pth


# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿ã�¨ãƒ©ãƒ™ãƒ«ä»˜ã�‘
base_dir = '../input/dogs-vs-cats-redux-kernels-edition'
train_dir = './data/train'
test_dir = './data/test'

os.makedirs('./data', exist_ok=True)

with zipfile.ZipFile(os.path.join(base_dir, 'train.zip')) as train_zip:
    train_zip.extractall('./data')
with zipfile.ZipFile(os.path.join(base_dir, 'test.zip')) as test_zip:
    test_zip.extractall('./data')

# ã�™ã�¹ã�¦ã�®ç”»åƒ�ãƒ‘ã‚¹ã‚’å�–å¾—
all_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

# ãƒ©ãƒ™ãƒ«ï¼ˆç”»åƒ�ãƒ•ã‚¡ã‚¤ãƒ«å��ã�« cat or dog ã‚’å�«ã‚€ï¼‰
def get_label(file_list):
    labels = []
    for path in file_list:
        filename = os.path.basename(path)
        if 'dog' in filename:
            labels.append(1)
        else:
            labels.append(0)
    return labels

all_labels = get_label(all_list)


train_list, val_list, train_labels, val_labels = train_test_split(
    all_list, all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)


print(f"Train: {len(train_list)} æ�š")
print(f"Val:   {len(val_list)} æ�š")
print(f"Test:  {len(test_list)} æ�š")


transform = ImageTransform(size, mean, std)

train_dataset = DogCatDataset(
    train_list,
    train_labels,
    transform,
    phase='train'
)

val_dataset = DogCatDataset(
    val_list,
    val_labels,
    transform,
    phase='val'
)

train_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=6,
    pin_memory=True
)

val_loader = data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=6,
    pin_memory=True
)


# ResNet18ã�®èª­ã�¿è¾¼ã�¿ï¼ˆImageNetã�§äº‹å‰�å­¦ç¿’æ¸ˆã�¿ï¼‰
model = models.resnet34(pretrained=True)

# å…¨ã�¦ã�®ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ã‚’ä¸€æ—¦å›ºå®šï¼ˆrequires_grad=Falseï¼‰
for param in model.parameters():
    param.requires_grad = False

# âœ… layer3, layer4 ã�®ã�¿å¾®èª¿æ•´
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# fcå±¤ï¼ˆå…¨çµ�å�ˆï¼‰ã‚’2ã‚¯ãƒ©ã‚¹åˆ†é¡�ç”¨ã�«ç½®ã��æ�›ã�ˆ
model.fc = nn.Linear(model.fc.in_features, 2)

# ãƒ¢ãƒ‡ãƒ«ã‚’ãƒ‡ãƒ�ã‚¤ã‚¹ã�«é€�ã‚‹
model = model.to(device)

# æ��å¤±é–¢æ•°
criterion = nn.CrossEntropyLoss()

# âœ… æœ€é�©åŒ–å¯¾è±¡ï¼šlayer3, layer4, fc
params_to_update = (
    list(model.layer3.parameters()) +
    list(model.layer4.parameters()) +
    list(model.fc.parameters())
)
optimizer = optim.Adam(params_to_update, lr=0.0002)

# å­¦ç¿’ç�‡ã‚¹ã‚±ã‚¸ãƒ¥ãƒ¼ãƒ©ãƒ¼ï¼ˆ5ã‚¨ãƒ�ãƒƒã‚¯ã�”ã�¨ã�«å�Šåˆ†ã�«ï¼‰
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


best_acc = 0.0
early_stop_counter = 0

# æ��å¤±ã�¨ç²¾åº¦ã‚’è¨˜éŒ²ã�™ã‚‹ãƒªã‚¹ãƒˆ
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epoch):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    # === æ¤œè¨¼ãƒ•ã‚§ãƒ¼ã‚º ===
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    print(f'Epoch [{epoch+1}/{num_epoch}] '
          f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | '
          f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # æ��å¤±ã�¨ç²¾åº¦ã‚’è¨˜éŒ²
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        early_stop_counter = 0  # æ”¹å–„ã�—ã�Ÿã�®ã�§ãƒªã‚»ãƒƒãƒˆ
        torch.save(model.state_dict(), './best_model.pth')
        print('âœ… Best model updated and saved!')
    else:
        early_stop_counter += 1
        print(f'â�³ No improvement... [{early_stop_counter}/{patience}]')
        if early_stop_counter >= patience:
            print('ğŸ›‘ Early stopping triggered.')
            break
    #å­¦ç¿’ç�‡ã‚’èª¿æ•´
    scheduler.step()


import matplotlib.pyplot as plt

# === æ��å¤±ã�®æ�¨ç§»ã‚°ãƒ©ãƒ• ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid()

# === ç²¾åº¦ã�®æ�¨ç§»ã‚°ãƒ©ãƒ• ===
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ãƒ¢ãƒ‡ãƒ«ã�®é‡�ã�¿ã‚’èª­ã�¿è¾¼ã‚€ï¼ˆå¿µã�®ã�Ÿã‚�ï¼‰
model.load_state_dict(torch.load('./best_model.pth'))
model.eval()

# äºˆæ¸¬çµ�æ�œã‚’ä¿�å­˜ã�™ã‚‹ãƒªã‚¹ãƒˆ
results = []

# ç”»åƒ�ã�®å‰�å‡¦ç�†ï¼ˆãƒ�ãƒªãƒ‡ãƒ¼ã‚·ãƒ§ãƒ³ã�¨å�Œã�˜ï¼‰
transform = ImageTransform(size, mean, std)

for path in tqdm(test_list):
    img = Image.open(path).convert("RGB")
    img = transform(img, phase='val')
    img = img.unsqueeze(0)  # ãƒ�ãƒƒãƒ�æ¬¡å…ƒã‚’è¿½åŠ  (1, 3, H, W)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)  # shape: [1, 2]
        probs = torch.softmax(outputs, dim=1)  # shape: [1, 2]
        dog_prob = probs[0][1].item()  # çŠ¬ï¼ˆã‚¯ãƒ©ã‚¹1ï¼‰ã�®ç¢ºç�‡ã� ã�‘å�–å¾—

    # ãƒ•ã‚¡ã‚¤ãƒ«å��ï¼ˆä¾‹: 1234.jpgï¼‰ã�¨äºˆæ¸¬ç¢ºç�‡ï¼ˆ0ã€œ1ï¼‰
    id = os.path.basename(path).split('.')[0]
    results.append([int(id), dog_prob])

# DataFrameã�¨ã�—ã�¦CSVã�«ä¿�å­˜
df = pd.DataFrame(results, columns=['id', 'label'])
df.sort_values('id', inplace=True)
df.to_csv('submission.csv', index=False)

print("âœ… æ�¨è«–å®Œäº†ï¼ˆç¢ºç�‡ã�§å‡ºåŠ›ï¼‰submission.csv ã‚’ä½œæˆ�ã�—ã�¾ã�—ã�Ÿã€‚")




