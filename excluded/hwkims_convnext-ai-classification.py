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


import warnings
warnings.simplefilter(action='ignore')

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.backends.cudnn as cudnn

# 시드 고정
seed = 42
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True
cudnn.benchmark = False

# GPU 설정
ngpu = torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Number of GPUs Available: ", ngpu)
print("Device being used: ", device)

# 하이퍼파라미터 (필요에 따라 조정)
batch_size = 8 * ngpu  # 더 큰 배치 사이즈
num_epochs = 25  # 에폭 증가
learning_rate = 5e-4
image_size = 224
accumulation_steps = 4  # Gradient Accumulation steps
weight_decay = 1e-5 # weight decay 추가
clip_grad_norm = 1 # Gradient Clipping 추가

# 데이터셋 경로 (Kaggle Dataset 기준)
train_dir = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Train'
test_dir = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/Test'
train_csv_path = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/train.csv'
sample_submission_path = '/kaggle/input/cidaut-ai-fake-scene-classification-2024/sample_submission.csv'

# 데이터셋 transform
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터셋 로드
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.data_info = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_name = self.data_info.iloc[idx, 0]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label_str = self.data_info.iloc[idx, 1]
        if label_str == 'editada':
            label = 1  # Assuming 'editada' means Fake
        elif label_str == 'real':
            label = 0  # Assuming 'real' means Real
        else:
            raise ValueError(f"Unknown label: {label_str}")

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32) # Ensure label is a float tensor

train_dataset = CustomDataset(train_dir, train_csv_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# 테스트 데이터셋 로드 (라벨 없음)
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.img_names = os.listdir(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return img_name, image

test_dataset = TestDataset(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ConvNeXt 모델 정의 (이번에는 ConvNeXt-Large 전체 모델을 사용하고 마지막 레이어만 수정)
class ConvNextModel(nn.Module):
    def __init__(self, num_classes=1):  # 이진 분류이므로 num_classes=1
        super().__init__()
        self.convnext = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
        # Replace the classifier
        self.convnext.classifier[-1] = nn.Linear(self.convnext.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.convnext(x) # Forward pass

# 모델 초기화 및 GPU 설정
model = ConvNextModel().float()
if ngpu > 1:
    model = nn.DataParallel(model, device_ids=list(range(ngpu))).to(device)
else:
    model.to(device)

# 손실 함수 및 옵티마이저
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning Rate Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

# 학습
print("Starting Training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Gradient Accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

        if i % 50 == 0 and i > 0:
            print(f"  Batch {i+1}/{len(train_loader)}, Loss: {running_loss / (i+1):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {epoch_loss:.4f}")

    # 에폭마다 스케줄러 업데이트
    scheduler.step()

    # 에폭마다 모델 저장 (Kaggle Notebook의 /kaggle/working/ 디렉토리에 저장)
    torch.save(model.state_dict(), f"/kaggle/working/convnext_large_epoch_{epoch+1}.pth")

# 제출 파일 로드
fsc_submission = pd.read_csv(sample_submission_path, index_col="image")

# 추론
model.eval()
predictions = {}
with torch.no_grad():
    for img_name, img in tqdm(test_loader, desc="Inference"):
        img = img.to(device)
        outputs = model(img)
        outputs = outputs.sigmoid().cpu()
        for i, img_name in enumerate(img_name):
            predictions[img_name] = outputs[i].item()  # 원본 레이블 사용

# 결과 저장
submission = pd.DataFrame(columns=['image', 'label'])
submission['image'] = list(predictions.keys())
submission['label'] = list(predictions.values())
submission['label'] = 1 - submission['label'] # Label Inversion for submission
submission.to_csv(f"/kaggle/working/submission.csv", index=False)
print(submission)

