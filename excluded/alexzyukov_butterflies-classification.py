# Установим библиотеку для работы с датасетами на kaggle
!pip install opendatasets --quiet


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import opendatasets as od


# Подгружаем датасет
url = 'https://www.kaggle.com/competitions/butterflies-classification'
od.download(url)


import os
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms, io, utils, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

from tqdm import tqdm


HOME_DIR = './butterflies-classification/'
TRAIN_DIR = HOME_DIR + 'train_butterflies/train_split'
TEST_DIR = HOME_DIR + 'test_butterflies/valid/'

# Подгружаем размеченные данные
base_ds = ImageFolder(root=TRAIN_DIR)

classes = base_ds.classes
classidx = base_ds.class_to_idx
counts = Counter(base_ds.targets)

# 50 видов бабочек
print(classes)

counts_df = pd.Series(counts).sort_index().rename({
    i: classes[i] for i in range(len(classes))
})
counts_vals = np.array(list(counts.values()))

# картинки распределены неравномерно по категориям,
# в каждой mean=99, std=15
#print(counts_df)
print(np.mean(counts_vals), np.std(counts_vals))

widths, heights = set(), set()
num_images = 0
for root, _, files in os.walk(TRAIN_DIR):
    for fname in files:
        path = os.path.join(root, fname)
        with Image.open(path) as img:
            w, h = img.size
            widths.add(w)
            heights.add(h)
        num_images += 1

# все картинки в разрешении 224x224
print(widths, heights)

# всего 4955 картинок
print(num_images)

"""
Особенности датасета:

1. У бабочки могут быть сложены или раскрыты
крылья — задача классификации усложняется,
нейросеть может ошибчно учитывать количество
крыльев, которые видны: одно или два.

2. Окрас и рисунок на разных сторонах крыльев
может быть разный. Поэтому изображения могут
значительно различаться даже внутри классов.
Например, в class_10 это серый и бурый цвета.

3. Бабочки расположены под разными углами
и ракурсами, яркость картинок также различается.
Значит, уместна соответствующая аугментация

4. У картинок есть фон, в основном это зелень.
Стоит его исключать, чтобы он не мешал
классификации бабочек. То есть это задача
сегментации, можно использовать простые
пороговые методы

5. Датасет несбалансирован, как выяснили в
коде выше. Картинок одних классов ~70,
а других — ~130

6. Отдельного упоминания требует class_39.
Он содержит картинки, на которых бабочки
напоминают небольших жуков. Причём бабочки
эти много меньше цветков, на которых
они запечатлены. Нейросеть может ошибочно
классифицировать фотографии с цветами как
экземпляры class_39

7. Цвет важен при классификации бабочек,
поэтому лучше не аугментировать по нему

Без аугментации и сегментации нейросеть
показывает accuracy на тесте, близкое 
к 100%
"""


class MyDataset(Dataset):
    def __init__(self, path, labels, transform=None):
        self.paths = np.array(path)
        self.labels = np.array(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return (img, label)


samples = base_ds.samples
paths, labels = zip(*samples)

# Разделили размеченные данные на тренировочную и тестовую выборки
# stratify=labels для равномерности количества экземпляров по классам
train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    paths, labels, test_size=0.2, stratify=labels
)

# Трансформация для использования нейросетей на основе ImageNet
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Такая же трансформация, что и для теста.
# Две переменные заведены, для более
# удобной аугментации, но в итоге она
# значительно не повышала точность
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

train_data = MyDataset(train_paths, train_labels, transform=train_transform)
valid_data = MyDataset(valid_paths, valid_labels, transform=train_transform)

# перемешивание shuffle=True, чтобы нейросеть не запоминала
# порядок индексов в тренировочной выборке
train_loader = DataLoader(
    train_data, batch_size=32, shuffle=True, num_workers=2
)
valid_loader = DataLoader(
    valid_data, batch_size=32, shuffle=False, num_workers=2
)


# Используем предобученную свёрточную нейронную сеть типа ImageNet
class RSNAModel(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet18'):
        super().__init__()

        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f'Unknown nn: {backbone_name}')

    def forward(self, batch):
        return self.backbone(batch)


def train_one_epoch(
    model,
    train_dataloader,
    optimizer,
    loss_fn,
    epoch,
    device,
    log_wandb=False,
    verbose=False
):
    model.train()

    # Инициализируем счётчики потерь, правильных предсказаний
    # и числа образцов
    running_loss = 0
    running_corrects = 0
    total_samples = 0

    # Свой прогресс-бар через tqdm для батчей
    pbar = tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        desc=f'train epoch: {epoch}'
    )

    for batch_idx, (inputs, labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Обнуляем градиенты, обратное распр. ошибки и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Считаем метрики
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        total_samples += inputs.size(0)

        # Выводим результаты
        if verbose:
            pbar.set_postfix({
                'train_loss': f'{running_loss/total_samples:.4f}',
                'train_acc': f'{running_corrects/total_samples:.4f}'
            })

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_corrects / total_samples

    return epoch_loss, epoch_accuracy


# Аналогично
@torch.no_grad()
def valid_one_epoch(
    model,
    valid_dataloader,
    loss_fn,
    epoch,
    device,
    log_wandb=False,
    verbose=False
):
    model.eval()

    running_loss = 0
    running_corrects = 0
    total_samples = 0

    pbar = tqdm(
        enumerate(valid_dataloader),
        total=len(valid_dataloader),
        desc=f'valid epoch: {epoch}'
    )

    for batch_idx, (inputs, labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        total_samples += inputs.size(0)

        if verbose:
            pbar.set_postfix({
                'val_loss': f'{running_loss/total_samples:.4f}',
                'val_acc': f'{running_corrects/total_samples:.4f}'
            })

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_corrects / total_samples

    return epoch_loss, epoch_accuracy


# Гиперпараметры
NUM_CLASSES = 50
LR = 1e-4
EPOCHS = 5
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Модель
model = RSNAModel(num_classes=NUM_CLASSES).to(DEVICE)
loss_fn = nn.CrossEntropyLoss() # задача классификации
optimizer = Adam(model.parameters(), lr=LR)

# Цикл обучения
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, loss_fn, epoch, DEVICE
    )
    valid_loss, valid_acc = valid_one_epoch(
        model, valid_loader, loss_fn, epoch, DEVICE
    )
    print(
        f'epoch: {epoch:02d}, train_loss={train_loss:.4f}, '
        f'train_acc={train_acc:.4f}, valid_loss={valid_loss:.4f}, '
        f'valid_acc={valid_acc:.4f}'
    )

"""
ResNet18 достаточно быстро переобучается: train_acc=0.9990,
loss на тренировочной выборке в 10 раз меньше, чем на
валидации. С валидацией справляется достаточно хорошо,
точность 0.9435
"""


model_names = ['resnet18', 'resnet50', 'efficientnet_b0']
results = []

for name in model_names:
    print(f'--- Training model: {name} ---')
    model = RSNAModel(num_classes=NUM_CLASSES, backbone_name=name).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)


    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, epoch, DEVICE
        )
        valid_loss, valid_acc = valid_one_epoch(
            model, valid_loader, loss_fn, epoch, DEVICE
        )
        print(
            f'{name} | epoch {epoch:02d} | '
            f'train_acc: {train_acc:.4f} | val_acc: {valid_acc:.4f}'
        )

    results.append((name, train_acc, valid_acc))


df_results = pd.DataFrame(results, columns=['model', 'train_acc', 'valid_acc'])
df_sorted = df_results.sort_values('valid_acc', ascending=False).reset_index(drop=True)
df_sorted


"""
Сравним нейросети:

ResNet50 самая медленная, зато с высокой валидационной точностью.
Имеет много параметров. valid_loss = 0.02, train_loss=0.17,
то есть хорошо сходится

EfficientNet-B0 более простая, но по валидации как ResNet50.
train_loss=0.05, valid_loss как у ResNet50

ResNet18 самая быстрая, но слишком быстро запоминает train:
точность на тренировочной выборке 0.9995, низкий train_loss,
то есть явное переобучение. Валидационная точность чуть хуже,
чем у других моделей

*значения для случая, если в train_test_split задали seed=42.
Иногда EfficientNet-B0 показывает наиболее стабильный результат:
не переобучается и показывает наибольшую точность на valid
"""


class TestDataset(Dataset):
    def __init__(self, path, transform=None):
        self.paths = sorted(
            [
                os.path.join(path, fname)
                for fname in os.listdir(path)
            ],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img


test_data = TestDataset(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    preds = []

    for batch in tqdm(dataloader):
        batch = batch.to(device)
        outputs = model(batch)
        batch_preds = outputs.argmax(dim=1).cpu().numpy()
        preds.extend(batch_preds)

    return preds


# Используем ResNet50 как лучшую на valid среди протестированных нейросетей.
# Обучаем 10 эпох, что долго, зато добиваемся наилучших весов и сохраняем их
model = RSNAModel(num_classes=NUM_CLASSES, backbone_name='resnet50').to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

best_acc = 0.0

for epoch in range(1, 10 + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, loss_fn, epoch, DEVICE
    )
    valid_loss, valid_acc = valid_one_epoch(
        model, valid_loader, loss_fn, epoch, DEVICE
    )
    print(
        f'epoch: {epoch:02d}, train_loss={train_loss:.4f}, '
        f'train_acc={train_acc:.4f}, valid_loss={valid_loss:.4f}, '
        f'valid_acc={valid_acc:.4f}'
    )

    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), 'best_model.pth')

# Наилучшие веса сохранены в файл best_model.pth


# Предсказываем test, то есть 250 неразмеченных картинок
model.load_state_dict(torch.load('best_model.pth'))
model.to(DEVICE)
test_preds = predict(model, test_loader, DEVICE)


# "Индексы" файлов в тесте. По сути список 0, ..., 249,
# но реализация более обобщённая
test_file_ids = [
    int(os.path.splitext(os.path.basename(p))[0])
    for p in test_data.paths
]

# Сопоставляем class_XX и XX
class_names = base_ds.classes
test_labels = [
    int(class_names[idx].split('_')[1])
    for idx in test_preds
]

results = pd.DataFrame({
    'index': test_file_ids,
    'label': test_labels
})

# Сохраняем предсказания в файл для Kaggle
results.to_csv('submission.csv', index=False)
print(results.head())


!kaggle competitions submit butterflies-classification -f submission.csv -m "My submission"

