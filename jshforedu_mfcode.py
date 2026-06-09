import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights

import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

base_dir = '/kaggle/input/ai-vs-human-generated-dataset'
train_csv_path = os.path.join(base_dir, 'train.csv')
test_csv_path = os.path.join(base_dir, 'test.csv')

print("\nLoading datasets...")

df_train = pd.read_csv(train_csv_path)
df_train['file_name'] = df_train['file_name'].apply(lambda x: os.path.join(base_dir, x))

df_test = pd.read_csv(test_csv_path)
df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))

n_samples_train = min(25000, len(df_train))
n_samples_val = min(1000, len(df_train) - n_samples_train)

train_data = df_train.iloc[:n_samples_train].copy()
val_data = df_train.iloc[-n_samples_val:].copy()

train_paths = train_data['file_name'].values
train_labels = train_data['label'].values
val_paths = val_data['file_name'].values
val_labels = val_data['label'].values

print(f"Train Data: {len(train_paths)}")
print(f"Validation Data: {len(val_paths)}")


def extract_rgb_histogram(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
        hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()
        hist /= hist.sum()  # normalize
        return hist
    except:
        return None

from functools import partial

def extract_features_mp(paths, n_jobs=4):
    with Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(extract_rgb_histogram, paths), total=len(paths), desc="Extracting features"))
    features = []
    valid_indices = []
    for i, feat in enumerate(results):
        if feat is not None:
            features.append(feat)
            valid_indices.append(i)
    return np.array(features), np.array(valid_indices)



X_train, train_valid_idx = extract_features_mp(train_paths)
y_train = train_labels[train_valid_idx]

X_val, val_valid_idx = extract_features_mp(val_paths)
y_val = val_labels[val_valid_idx]

val_paths_valid = val_paths[val_valid_idx]

print(f"Train valid samples: {len(X_train)}")
print(f"Val valid samples: {len(X_val)}")


import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import accuracy_score

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'seed': 42
}

lgb_model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=100,
    callbacks=[
        early_stopping(stopping_rounds=10),
        log_evaluation(period=10)
    ]
)

val_preds = lgb_model.predict(X_val)
val_preds_binary = (val_preds > 0.5).astype(int)

val_acc = accuracy_score(y_val, val_preds_binary)
print(f"LightGBM Validation Accuracy: {val_acc:.4f}")


uncertain_idx = np.where((val_preds >= 0.1) & (val_preds <= 0.9))[0]
confident_idx = np.where((val_preds < 0.1) | (val_preds > 0.9))[0]

print(f"Confident validation samples: {len(confident_idx)}")
print(f"Uncertain validation samples: {len(uncertain_idx)}")

class AiHumanDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        else:
            return img


from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights = ResNet18_Weights.IMAGENET1K_V1
cnn_model = resnet18(weights=weights)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
cnn_model = cnn_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


uncertain_train_dataset = AiHumanDataset(
    image_paths=val_paths_valid[uncertain_idx],
    labels=y_val[uncertain_idx],
    transform=train_transform
)
uncertain_train_loader = DataLoader(uncertain_train_dataset, batch_size=32, shuffle=True, num_workers=2)


def train_cnn(model, dataloader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}")


train_cnn(cnn_model, uncertain_train_loader, criterion, optimizer, device, epochs=10)


def cnn_predict(model, dataloader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="CNN Predict"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)
            outputs = model(images)
            probs = nn.functional.softmax(outputs, dim=1)[:, 1]
            preds.extend(probs.cpu().numpy())
    return np.array(preds)


uncertain_val_dataset = AiHumanDataset(
    image_paths=val_paths_valid[uncertain_idx],
    labels=y_val[uncertain_idx],
    transform=val_transform
)
uncertain_val_loader = DataLoader(uncertain_val_dataset, batch_size=16, shuffle=False, num_workers=2)


cnn_probs_uncertain = cnn_predict(cnn_model, uncertain_val_loader)
cnn_preds_uncertain = (cnn_probs_uncertain > 0.5).astype(int)

acc_uncertain = accuracy_score(y_val[uncertain_idx], cnn_preds_uncertain)
print(f"2nd Stage CNN Accuracy on uncertain samples: {acc_uncertain:.4f}")


df_test['file_path'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))

X_test, test_valid_idx = extract_features_mp(df_test['file_path'].values)
print(f"Test samples with valid features: {len(X_test)}")

lgb_test_preds = lgb_model.predict(X_test)
print(f"LightGBM test prediction sample: {lgb_test_preds[:5]}")

uncertain_test_idx = np.where((lgb_test_preds >= 0.02) & (lgb_test_preds <= 0.99))[0]
confident_test_idx = np.where((lgb_test_preds < 0.02) | (lgb_test_preds > 0.99))[0]

print(f"Confident test samples: {len(confident_test_idx)}")
print(f"Uncertain test samples: {len(uncertain_test_idx)}")

uncertain_test_paths = df_test['file_path'].values[test_valid_idx][uncertain_test_idx]

cnn_test_dataset = AiHumanDataset(
    image_paths=uncertain_test_paths,
    labels=None,
    transform=val_transform
)
cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=16, shuffle=False, num_workers=2)

cnn_test_probs = cnn_predict(cnn_model, cnn_test_loader)
cnn_test_preds = (cnn_test_probs > 0.5).astype(int)


import os

final_test_preds = np.zeros(len(df_test), dtype=int)
final_test_preds_valid = np.zeros(len(test_valid_idx), dtype=int)

final_test_preds_valid[confident_test_idx] = (lgb_test_preds[confident_test_idx] > 0.5).astype(int)
final_test_preds_valid[uncertain_test_idx] = cnn_test_preds

for idx, valid_i in enumerate(test_valid_idx):
    final_test_preds[valid_i] = final_test_preds_valid[idx]

submission = pd.DataFrame({
    'id': 'test_data_v2/' + df_test['id'].apply(lambda x: os.path.basename(x)),
    'label': final_test_preds
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("/kaggle/working/submission.csv.")

