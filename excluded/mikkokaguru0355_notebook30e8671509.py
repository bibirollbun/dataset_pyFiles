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


import os
import zipfile

os.makedirs('/kaggle/working/train', exist_ok=True)
os.makedirs('/kaggle/working/test', exist_ok=True)

train_zip_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
train_extract_path = '/kaggle/working/train'

with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall(train_extract_path)

test_zip_path = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'
test_extract_path = '/kaggle/working/test'

with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall(test_extract_path)

print("ファイルの解凍が完了しました。")
print("学習用画像（一部）:", os.listdir('/kaggle/working/train/train')[:5])
print("テスト用画像（一部）:", os.listdir('/kaggle/working/test/test')[:5])


!pip install ultralytics timm


import os
import torch
from ultralytics import YOLO
from PIL import Image
from tqdm.notebook import tqdm

model = YOLO('yolov8n.pt')

coco_names = model.names
target_class_ids = [k for k, v in coco_names.items() if v in ['dog', 'cat']]

def crop_objects(input_dir, output_dir, image_limit=None):
    os.makedirs(output_dir, exist_ok=True)
    
    filenames = os.listdir(input_dir)
    if image_limit:
        filenames = filenames[:image_limit]
    
    cropped_count = 0
    for filename in tqdm(filenames, desc=f"Processing {os.path.basename(input_dir)}"):
        filepath = os.path.join(input_dir, filename)
        
        try:
            results = model(filepath, verbose=False)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        img = Image.open(filepath)
        
        for i, res in enumerate(results):
            boxes = res.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                
                if class_id in target_class_ids:
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    cropped_img = img.crop(xyxy)
                    
                    new_filename = f"{os.path.splitext(filename)[0]}_crop{i}{os.path.splitext(filename)[1]}"
                    save_path = os.path.join(output_dir, new_filename)
                    
                    cropped_img.save(save_path)
                    cropped_count += 1

    print(f"Finished processing {input_dir}. Created {cropped_count} cropped images in {output_dir}")

train_input_dir = '/kaggle/working/train/train'
train_output_dir = '/kaggle/working/cropped_train'
crop_objects(train_input_dir, train_output_dir)

test_input_dir = '/kaggle/working/test/test'
test_output_dir = '/kaggle/working/cropped_test'
crop_objects(test_input_dir, test_output_dir)

print("\n--- Cropped Train Images (sample) ---")
print(os.listdir(train_output_dir)[:5])
print("\n--- Cropped Test Images (sample) ---")
print(os.listdir(test_output_dir)[:5])


import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

train_dir = '/kaggle/working/cropped_train'

filepaths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
df = pd.DataFrame({'filepath': filepaths})
df['label'] = df['filepath'].apply(lambda x: 1 if 'dog' in os.path.basename(x) else 0)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

IMG_SIZE = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class CroppedDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['filepath']
        label = self.df.iloc[idx]['label']
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except (IOError, SyntaxError) as e:
            print(f"Skipping corrupted image: {image_path}")
            return self.__getitem__((idx + 1) % len(self))


BATCH_SIZE = 32
train_dataset = CroppedDataset(train_df, transform=data_transforms['train'])
val_dataset = CroppedDataset(val_df, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print("データ準備完了")
print(f"学習データ数: {len(train_dataset)}, 検証データ数: {len(val_dataset)}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model('convnext_small', pretrained=True, num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5) # ファインチューニングのため学習率は小さめに設定

print(f"モデル準備完了。デバイス: {device}")



NUM_EPOCHS = 5 # エポック数（必要に応じて調整）

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        progress_bar.set_postfix(loss=loss.item())
    
    epoch_train_loss = running_loss / len(train_dataset)

    model.eval()
    val_loss = 0.0
    corrects = 0
    with torch.no_grad():
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
        for inputs, labels in progress_bar_val:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            val_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = corrects.double() / len(val_dataset)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> "
          f"Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | "
          f"Val Acc: {epoch_val_acc:.4f}")

print("\n学習が完了しました。")


import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.notebook import tqdm
import torch.nn.functional as F
import re

test_dir = '/kaggle/working/cropped_test'

class TestDataset(Dataset):
    def __init__(self, filepaths, transform=None):
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        original_id = re.search(r'(\d+)_crop', os.path.basename(filepath))
        if original_id:
            image_id = int(original_id.group(1))
        else: # もしcropがない場合（例：123.jpg）
            image_id = int(os.path.splitext(os.path.basename(filepath))[0])
            
        try:
            image = Image.open(filepath).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, image_id
        except (IOError, SyntaxError) as e:
            print(f"Skipping corrupted image: {filepath}")
            return self.__getitem__((idx + 1) % len(self))

test_filepaths = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]

test_dataset = TestDataset(test_filepaths, transform=data_transforms['val'])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print("テストデータの準備完了")

model.eval()  # モデルを評価モードに設定
results = []

with torch.no_grad():
    progress_bar = tqdm(test_loader, desc="Inference")
    for inputs, image_ids in progress_bar:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        probabilities = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        
        for i, image_id in enumerate(image_ids):
            results.append({'id': image_id.item(), 'label': probabilities[i]})

print("推論完了")

inference_df = pd.DataFrame(results)
submission_df = inference_df.groupby('id')['label'].mean().reset_index()

sample_submission_df = pd.read_csv('/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv')
submission_df = sample_submission_df[['id']].merge(submission_df, on='id', how='left')

submission_df['label'] = submission_df['label'].fillna(0.5)

submission_df['label'] = submission_df['label'].clip(0.005, 0.995)

print("結果の集計完了")


submission_df.to_csv('submission.csv', index=False)

print("\nsubmission.csvが作成されました。")
print(submission_df.head())

