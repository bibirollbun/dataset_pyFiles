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
        break

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torchvision import models
from torch import nn

from tqdm import tqdm
from IPython.display import clear_output

from collections import defaultdict
import time


BATCH_SIZE = 128


# Путь к папке с данными
data_dir_train = '/kaggle/input/classification-of-butterflies/train_butterflies/train_split'
data_dir_test = '/kaggle/input/classification-of-butterflies/test_butterflies'
# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера изображений
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

# Создание датасета с помощью ImageFolder

train_dataset = datasets.ImageFolder(root=data_dir_train, transform=transform)
test_dataset = datasets.ImageFolder(root=data_dir_test, transform=transform)

# Разделение на обучающую и тестовую выборки
train_size = int(0.8 * len(train_dataset))  # 80% данных для обучения
valid_size = len(train_dataset) - train_size  # 20% данных для тестирования
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

# Создание DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# Пример использования
for images, labels in train_loader:
    print(images.shape)  # Размер батча: (batch_size, channels, height, width)
    print(labels)  # Метки классов
    break


device = "cuda" if torch.cuda.is_available() else "cpu"
device


model_resnet18 = models.resnet18(pretrained=False)
model_resnet18.fc = nn.Linear(512, 50)
model_resnet18 = model_resnet18.to(device)


for i in train_loader:
    print(model_resnet18(i[0].to(device)).shape)
    break


model_resnet18


def plot_learning_curves(history):
    '''
    Функция для вывода графиков лосса и метрики во время обучения.
    '''
    fig = plt.figure(figsize=(20, 7))

    plt.subplot(1,2,1)
    plt.title('Лосс', fontsize=15)
    plt.plot(history['loss']['train'], label='train')
    plt.plot(history['loss']['val'], label='val')
    plt.ylabel('лосс', fontsize=15)
    plt.xlabel('эпоха', fontsize=15)
    plt.legend()

    plt.subplot(1,2,2)
    plt.title('Точность', fontsize=15)
    plt.plot(history['acc']['train'], label='train')
    plt.plot(history['acc']['val'], label='val')
    plt.ylabel('лосс', fontsize=15)
    plt.xlabel('эпоха', fontsize=15)
    plt.legend()
    plt.show()


def train(
    model,
    criterion,
    optimizer,
    scheduler,
    train_batch_gen,
    val_batch_gen,
    num_epochs=10
):
    '''
    Функция для обучения модели и вывода лосса и метрики во время обучения.
    '''

    history = defaultdict(lambda: defaultdict(list))

    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        start_time = time.time()

        # устанавливаем поведение dropout / batch_norm  в обучение
        model.train(True)
        model.to(device)

        # на каждой "эпохе" делаем полный проход по данным
        for X_batch, y_batch in tqdm(train_batch_gen):
            # обучаемся на текущем батче
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            loss = criterion(logits, y_batch.long().to(device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += np.sum(loss.detach().cpu().numpy())
            y_pred = logits.max(1)[1].detach().cpu().numpy()
            train_acc += np.mean(y_batch.cpu().numpy() == y_pred)
        scheduler.step()

        # подсчитываем лоссы и сохраням в "историю"
        train_loss /= len(train_batch_gen)
        train_acc /= len(train_batch_gen)
        history['loss']['train'].append(train_loss)
        history['acc']['train'].append(train_acc)

        # устанавливаем поведение dropout / batch_norm в режим тестирования
        model.train(False)

        # полностью проходим по валидационному датасету
        for X_batch, y_batch in tqdm(val_batch_gen):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch.long().to(device))
            val_loss += np.sum(loss.detach().cpu().numpy())
            y_pred = logits.max(1)[1].detach().cpu().numpy()
            val_acc += np.mean(y_batch.cpu().numpy() == y_pred)

        # подсчитываем лоссы и сохраням в "историю"
        val_loss /= len(val_batch_gen)
        val_acc /= len(val_batch_gen)
        history['loss']['val'].append(val_loss)
        history['acc']['val'].append(val_acc)

        clear_output()

        # печатаем результаты после каждой эпохи
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss (in-iteration): \t{:.6f}".format(train_loss))
        print("  validation loss (in-iteration): \t{:.6f}".format(val_loss))
        print("  training accuracy: \t\t\t{:.2f} %".format(train_acc * 100))
        print("  validation accuracy: \t\t\t{:.2f} %".format(val_acc * 100))

        plot_learning_curves(history)

    return model, history


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_resnet18.parameters(), lr=3e-4)
scheduler = StepLR(optimizer, step_size=3, gamma=0.2)

clf_model, history = train(
    model_resnet18, criterion, optimizer,
    scheduler, train_loader, valid_loader,
    num_epochs=12
)


################################
# thx to Сергей Панин (pan1ns) #
################################


test_dataset = datasets.ImageFolder(root=data_dir_test, transform=transform)


sample_name = [path.split('/')[-1].split('.')[0] for path, _ in test_dataset.samples]

data_pred = []
for data, _ in test_dataset:
    data_pred.append(data.clone().detach())

data_pred = torch.stack(data_pred, dim=0)
data_pred = data_pred.to(device)
pred = clf_model(data_pred)

str_nums = sorted([str(i) for i in range(50)])

to_real_index = {i : int(str_nums[i]) for i in range(50)}

res = [to_real_index[torch.argmax(item).item()] for item in pred]

cnt_output = len(res)

output = {
    "index": sample_name,
    "label": res,
}

output = pd.DataFrame(output)



output.to_csv('submission_resnet18_reindexed.csv', index=False)

