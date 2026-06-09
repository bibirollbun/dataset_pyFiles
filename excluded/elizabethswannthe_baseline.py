import os
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import pandas as pd

# Конфигурация
DATA_DIR_TRAIN = '/kaggle/input/offzone-deepfakes/train/app/hakaton/Hackaton/train/'
DATA_DIR_TEST = '/kaggle/input/offzone-deepfakes/test/app/hakaton/Hackaton/test'
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Датасет для обучения и валидации
class TrainDataset(Dataset):
    def __init__(self, root_dir, label_dir, split):
        """
        Args:
            root_dir (str): Путь к папке с изображениями.
            label_dir (str): Путь к папке с лейблами.
            split(str): Разделение на обучающую или валидационную выборку
        """
        image_paths = glob(root_dir+'*/*.png', recursive=True)
        self.image_paths_slit = image_paths[:int(0.8*len(image_paths))] if split=="train" else image_paths[int(0.8*len(image_paths)):]
        self.labels = pd.read_csv(label_dir)
        
        # Если нет трансформаций, задаём стандартные (ресайз + тензор)
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Пример размера для CNN
                transforms.ToTensor(),           # Конвертируем в тензор [0, 1]
            ])

    def __len__(self):
        return len(self.image_paths_slit)

    def __getitem__(self, idx):
        img_path = self.image_paths_slit[idx]
        label = self.labels[self.labels["folder_id"]==int(img_path.split("/")[-2])]["df_category"].values[0]
        image = self.transform(Image.open(img_path).convert("RGB")) 
        return image, label


# Датасет для тестирования 
class TestDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        """
        Args:
            root_dir (str): Путь к папке с изображениями.
            label_dir (str): Путь к папке с индексами папок.
        """
        self.root =  root_dir
        self.labels = pd.read_csv(label_dir)
        # Если нет трансформаций, задаём стандартные (ресайз + тензор)
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Пример размера для CNN
                transforms.ToTensor(),           # Конвертируем в тензор [0, 1]
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        folder_id = self.labels["folder_id"][idx]
        batch_path  = os.path.join(self.root, str(folder_id))
        image = [self.transform(Image.open(os.path.join(batch_path,i)).convert("RGB")) for i in os.listdir(batch_path)] 
        return torch.stack(image), folder_id


    # Подготовка данных
def prepare_data():
        train_dataset = TrainDataset(DATA_DIR_TRAIN, "/kaggle/input/offzone-deepfakes/train_metadata.csv", "train")
        val_dataset = TrainDataset(DATA_DIR_TRAIN, "/kaggle/input/offzone-deepfakes/train_metadata.csv", "val")
        test_dataset = TestDataset(DATA_DIR_TEST, "/kaggle/input/offzone-deepfakes/submission.csv")
    
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
        return train_loader, val_loader, test_loader



# Модель
def create_model(num_classes=1):
    model = models.resnet18()
    
    # Заменяем последний слой
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    return model.to(DEVICE)


# Обучение
def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.float().to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Считаем accuracy
        predicted = (outputs > 0.5).float()  
        batch_correct = (predicted == labels).sum().item()
        
        running_loss += loss.item()
        correct += batch_correct
        total += labels.size(0)
        
        # Вывод статистики по батчу
        print(f'Train Loss: {loss.item():.4f}, Accuracy: {batch_correct/labels.size(0):.2%}')
    
    # Вывод статистики по эпохе
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    print(f'Epoch {epoch+1} Summary:')
    print(f'Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2%}\n')


# Валидация
def val_model(model, val_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.float().to(DEVICE)
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            # Считаем accuracy
            predicted = (outputs > 0.5).float()  # Порог 0.5 для бинарной классификации
            batch_correct = (predicted == labels).sum().item()
            
            running_loss += loss.item()
            correct += batch_correct
            total += labels.size(0)
            
            # Вывод статистики по батчу
            print(f'Val Loss: {loss.item():.4f}, Accuracy: {batch_correct/labels.size(0):.2%}')
    
    # Вывод статистики по эпохе
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total


# Тестирование
def test_model(model, test_loader):
    model.eval()
    test_labels = pd.read_csv("/kaggle/input/offzone-deepfakes/submission.csv")
    test_labels["df_category"] = 0
    with torch.no_grad():
        for images, index in test_loader:
            images = images.squeeze().to(DEVICE)
            outputs = model(images).squeeze()
            predicted = (outputs > 0.5).float()
            mode,_ = torch.mode(predicted)
            test_labels.loc[test_labels["folder_id"]==int(index), "df_category"] = mode.item()
            
    test_labels.to_csv("/kaggle/working/submission.csv",index=False, encoding='utf-8')


# Подготовка данных
train_loader, val_loader, test_loader = prepare_data()

# Создание модели
model = create_model(num_classes=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
for epoch in range(EPOCHS):
    train_model(model, train_loader, criterion, optimizer, epoch=epoch)
    if epoch%2==0:
       val_model(model, val_loader, criterion, epoch=epoch)

# Тестирование
print('\nTesting model...')
test_accuracy = test_model(model, test_loader)










