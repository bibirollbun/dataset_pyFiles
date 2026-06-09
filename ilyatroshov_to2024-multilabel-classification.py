import sys
sys.path.insert(0, "/kaggle/input/rasterio-2")  # Путь к rasterio 
sys.path.insert(0, "/kaggle/input/affine")  # Путь к affine 

!pip install --no-index --find-links "/kaggle/input/affine" affine
!pip install --no-index --find-links "/kaggle/input/rasterio-2" rasterio


import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df_train = pd.read_csv('/kaggle/input/tech-olympiad-2024-bahrain-nssa-challenge/train.csv')
df_val = pd.read_csv('/kaggle/input/tech-olympiad-2024-bahrain-nssa-challenge/validation.csv')
labels = pd.concat([df_train, df_val], axis=0)

def calculate_label_percentages(counts):
    total_labels = counts.sum() 
    return (counts / total_labels) * 100  

class_counts = labels.iloc[:, 1:].sum(axis=0)
class_percentages = calculate_label_percentages(class_counts)

plt.figure(figsize=(12, 6))
bars = plt.bar(class_percentages.index, class_percentages.values, color='skyblue')
plt.title('Распределение меток в исходных данных (% от общего числа меток)')
plt.xlabel('Классы')
plt.ylabel('Процент меток')
plt.xticks(rotation=45)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


import rasterio
import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from torchvision.models import efficientnet_b0
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim
import gc

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = csv_file
        self.img_dir     = img_dir
        self.transform   = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        with rasterio.open(img_path) as src:
            image = src.read()
        image = torch.tensor(image[:, :128, :128]).float()
        image = (image - image.min()) / (image.max() - image.min())
        label    = torch.tensor(self.annotations.iloc[idx, 1:].values.astype(int))

        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_dataloaders(csv_file, img_dir, batch_size=32, n_fold=0):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip() 
    ])

    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=2024)
    for i, (train_index, val_index) in enumerate(kf.split(dataset)):
        if i == n_fold:
            break
            
    train_dataset = Subset(dataset, train_index)
    
    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=None)
    val_dataset = Subset(dataset, val_index)
    print(len(train_dataset))
    print(len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    all_outputs = torch.sigmoid(torch.tensor(all_outputs)).numpy() 
    
    return epoch_loss, all_labels, all_outputs

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=11):
        super(EfficientNetB0, self).__init__()
        self.model = efficientnet_b0(pretrained=False)
        self.model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1] = nn.Linear(1280, num_classes)
        
    def forward(self, x):
        return self.model(x)

def train_model(csv_file, img_dir, model, model_name, num_epochs=10, batch_size=32, lr=1e-4, n_fold=0, device='cuda', patience=3, warmup_epochs=0):
    train_loader, val_loader = create_dataloaders(csv_file, img_dir, batch_size=batch_size, n_fold=n_fold)
    train_sets = len(train_loader)

    model     = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, train_sets * warmup_epochs, train_sets * num_epochs)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_losses = []
    val_losses   = []
    
    path = model_name + str(n_fold)
    os.makedirs(path, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_labels, val_outputs = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        val_preds = (val_outputs > 0.5).astype(int)
        f1 = f1_score(val_labels, val_preds, average='micro')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1 Score: {f1:.4f}')
        early_stopping(-f1, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

df_train = pd.read_csv('/kaggle/input/tech-olympiad-2024-bahrain-nssa-challenge/train.csv')
df_val = pd.read_csv('/kaggle/input/tech-olympiad-2024-bahrain-nssa-challenge/validation.csv')
labels = pd.concat([df_train, df_val], axis=0)
img_dir = '/kaggle/input/tech-olympiad-2024-bahrain-nssa-challenge/Images/Images'
dataset = CustomImageDataset(labels, img_dir)

model = EfficientNetB0(num_classes=11)

batch_size = 128
lr = 2e-3
n_fold = 0
train_model(labels, img_dir, model, 'EfficientNet', num_epochs=30, batch_size=batch_size, lr=lr, n_fold=n_fold, device='cuda', patience=5, warmup_epochs=0)
torch.cuda.empty_cache()
gc.collect()

def predict(csv_file, img_dir, model, model_name, batch_size=32, n_fold=0, device='cuda'):
    model     = model.to(device) # load the model into the GPU
    model.load_state_dict(torch.load(os.path.join(model_name + str(n_fold), 'checkpoint.pth')))
    criterion = nn.BCEWithLogitsLoss()
    
    test_dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    _, _, outputs = validate(model, test_loader, criterion, device)
    return outputs

labels = pd.read_csv('/kaggle/input/tech-olympiad-2024-bahrain-nssa-challenge/sample_submission.csv')
img_dir = '/kaggle/input/tech-olympiad-2024-bahrain-nssa-challenge/Images/Images'
preds = predict(labels, img_dir, model, 'EfficientNet', batch_size=batch_size, device='cuda', n_fold=n_fold)
preds = (preds > 0.5).astype(int)
df_preds = pd.DataFrame(preds, columns=labels.columns[1:])
submission = pd.concat([labels[['ID']], df_preds], axis=1)
submission.head()

submission.to_csv('submission.csv', index=False)



preds = predict(labels, img_dir, model, 'EfficientNet', batch_size=batch_size, device='cuda', n_fold=n_fold)
preds_binary = (preds > 0.5).astype(int)

def calculate_percentages(counts):
    return counts / counts.sum() * 100 

class_percentages = calculate_percentages(class_counts)

pred_counts = pd.DataFrame(preds_binary, columns=labels.columns[1:]).sum(axis=0)
pred_percentages = calculate_percentages(pred_counts)

comparison = pd.DataFrame({
    'Исходные (train+val), %': class_percentages,
    'Предсказанные (test), %': pred_percentages
})

plt.figure(figsize=(14, 6))
comparison.plot(kind='bar', alpha=0.7)
plt.title('Распределение классов (% от общего числа)')
plt.xlabel('Классы')
plt.ylabel('Процент примеров')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.legend()
plt.show()


import torch
import os

model_path = '/kaggle/working/trained_model_complete.pth'

checkpoint = {
    'model_state_dict': model.state_dict(),
    
    'model_config': {
        'architecture': 'EfficientNet-B0',
        'input_channels': 4,
        'num_classes': 11,
        'image_size': 224,
        'modified_layers': {
            'features.0.0': 'Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2))',
            'classifier.1': 'Linear(1280, 11)'
        }
    },
    
    'class_names': ['clear', 'agriculture', 'bare_ground', 'forest', 
                   'unshaded', 'partly_cloudy', 'cloudy', 'shaded',
                   'partly_shaded', 'habitation', 'water'],
    
    'training_metadata': {
        'batch_size': 16,
        'learning_rate': 3e-5,
        'epochs_trained': 150,
        'best_val_f1': 0.957  
    }
}

torch.save(checkpoint, model_path)
print(f"Полная модель сохранена в: {model_path}")
print(f"Размер файла: {os.path.getsize(model_path)/1024/1024:.2f} MB")


import torch
from torchvision import models
import torch.nn as nn

def load_custom_model(model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    
    model = models.efficientnet_b0(pretrained=False)
    
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        checkpoint['model_config']['input_channels'],
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )
    
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        checkpoint['model_config']['num_classes']
    )
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model, checkpoint

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_custom_model("trained_model_complete.pth", device)
    
    print("Модель успешно загружена!")
    print(f"Архитектура: {config['model_config']['architecture']}")
    print(f"Классы: {config['class_names']}")

