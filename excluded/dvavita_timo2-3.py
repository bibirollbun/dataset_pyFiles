import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import copy
import datetime
import random
import traceback
from IPython.display import display, clear_output
import os
import shutil
from torchvision.datasets import ImageFolder

seed = 0
random.seed(seed)
np.random.seed(seed)

torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



print(device)


train_transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),  
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomCrop(32, padding=4),  
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])  
])

test_transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),    
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])  
])

train_datasets = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_datasets = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=100, shuffle=True)


classes = train_datasets.classes
train_datasets.classes


class fc(torch.nn.Module):
    # Конструктор для класса
    # Инициализация нейронной сети
    def __init__(self):

        super(fc, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (3,3), padding=1)
        self.mp1 = torch.nn.MaxPool2d(kernel_size = (2,2))
        #32*16*16
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding=1)
        self.mp2 = torch.nn.MaxPool2d(kernel_size = (2,2))
        #64*8*8
        self.conv3 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding=1)
        #128*8*8
        self.fc1 = torch.nn.Linear(128*8*8, 512)
        self.activ1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 256)
        self.activ2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(256, 64)
        self.activ3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(64, 32)
        self.activ4 = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(32, 16)
        self.activ5 = torch.nn.ReLU()
        self.fc6 = torch.nn.Linear(16, 10)
        self.reg = torch.nn.Dropout(0.1)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, data_input):
        x = self.conv1(data_input)
        x = self.relu(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.reshape(-1, 128*8*8)
        x = self.reg(x)
        x = self.fc1(x)
        x = self.activ1(x)
        x = self.fc2(x)
        x = self.activ2(x)
        x = self.fc3(x)
        x = self.activ3(x)
        x = self.fc4(x)
        x = self.activ4(x)
        x = self.fc5(x)
        x = self.activ5(x)
        x = self.fc6(x)
        return x

    def inference(self, data_input):
        with torch.no_grad():
            x = self.forward(data_input)
            x = self.sm(x)
        return x


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))


def train_eval_loop(model, train_dataset, val_dataset, criterion,
                    lr=1e-4, epoch_n=10, batch_size=32,
                    device=None, early_stopping_patience=100, l2_reg_alpha=0,
                    max_batches_per_epoch_train=10000,
                    max_batches_per_epoch_val=1000,
                    data_loader_ctor=torch.utils.data.DataLoader,
                    optimizer_ctor=None,
                    lr_scheduler_ctor=None,
                    shuffle_train=True,
                    dataloader_workers_n=0, 
                    plot=False):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    if optimizer_ctor is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    else:
        optimizer = optimizer_ctor(model.parameters(), lr=lr)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = None

    train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                        num_workers=dataloader_workers_n)
    val_dataloader = data_loader_ctor(val_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=dataloader_workers_n)

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

# Dynamic plot
    if plot:
        plot_epoch_data = []
        plot_train_loss = []
        plot_val_loss = []
        plot_acc_train = []
        plot_acc_val = []
        
        fig, ax = plt.subplots(2,1, figsize=(10,6))
        line_train, = ax[0].plot([], [], 'r-')
        line_val, = ax[0].plot([], [], 'b-')
        line_acc_train, = ax[1].plot([], [], 'r-')
        line_acc_val, = ax[1].plot([], [], 'b-')
        ax[0].legend(['train', 'val'])
        ax[0].set_xlim(0, epoch_n)
        ax[1].legend(['train', 'val'])
        ax[1].set_xlim(0, epoch_n)
        ax[0].title.set_text('Функции потерь')
        ax[1].title.set_text('Accuracy')

        def add_point(epoch_i, train_loss, val_loss, acc_train, acc_val):
            max_loss = max(ax[0].viewLim.y1 / 1.1, train_loss, val_loss)
            ax[0].set_ylim(0, max_loss * 1.1)
            ax[1].set_ylim(0, 110)
            plot_epoch_data.append(epoch_i)
            plot_train_loss.append(train_loss)
            plot_val_loss.append(val_loss)
            plot_acc_train.append(100 * acc_train)
            plot_acc_val.append(100 * acc_val)
            line_train.set_data(plot_epoch_data, plot_train_loss)
            line_val.set_data(plot_epoch_data, plot_val_loss)
            line_acc_train.set_data(plot_epoch_data, plot_acc_train)
            line_acc_val.set_data(plot_epoch_data, plot_acc_val)
            clear_output(wait=True)
            display(fig)


    for epoch_i in range(epoch_n):
        try:
            epoch_start = datetime.datetime.now()
            

            print('Эпоха {}'.format(epoch_i))

            model.train()
            mean_train_loss = 0
            mean_train_acc = 0
            train_batches_n = 0
            for batch_i, (batch_x, batch_y) in enumerate(train_dataloader):
                if batch_i > max_batches_per_epoch_train:
                    break
                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)

                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                model.zero_grad()
                loss.backward()

                optimizer.step()
                temp = []
                for i in range(len(pred)):
                    temp.append(torch.argmax(pred[i]).cpu().numpy())
                acc = accuracy_score(temp, batch_y.cpu().numpy())
                mean_train_loss += float(loss)
                mean_train_acc += float(acc)
                train_batches_n += 1
                
            
            mean_train_loss /= train_batches_n
            mean_train_acc /= train_batches_n

            print('Эпоха: {} итераций, {:0.2f} сек'.format(train_batches_n,
                                                           (datetime.datetime.now() - epoch_start).total_seconds()))
            print('Среднее значение функции потерь на обучении', mean_train_loss)
            print('Средняя accuracy на обучении', mean_train_acc)



            model.eval()
            mean_val_loss = 0
            mean_val_acc = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in enumerate(val_dataloader):
                    if batch_i > max_batches_per_epoch_val:
                        break

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)

                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)
                    temp = []
                    for i in range(len(pred)):
                        temp.append(torch.argmax(pred[i]).cpu().numpy())
                    acc = accuracy_score(temp, batch_y.cpu().numpy())
                    mean_val_loss += float(loss)
                    mean_val_acc += float(acc)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            mean_val_acc /= val_batches_n

            if plot:
                add_point(epoch_i, mean_train_loss, mean_val_loss, mean_train_acc, mean_val_acc)
            else:
                pass
            
            print('Среднее значение функции потерь на валидации', mean_val_loss)
            print('Среднее значение accuracy на валидации', mean_val_acc)

            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)
                print('Новая лучшая модель! На эпохе {}'.format(epoch_i))
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('Модель не улучшилась за последние {} эпох, прекращаем обучение'.format(
                    early_stopping_patience))
                break

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

            print()
        except KeyboardInterrupt:
            print('Досрочно остановлено пользователем')
            break
        except Exception as ex:
            print('Ошибка при обучении: {}\n{}'.format(ex, traceback.format_exc()))
            break
        finally:
            if plot:
                plt.close(fig)

    return best_val_loss, best_model


model = fc()


best_loss, best_model = train_eval_loop(model=model, 
                train_dataset=train_datasets, 
                val_dataset=test_datasets, 
                criterion=torch.nn.CrossEntropyLoss(),
                lr=1e-3, 
                epoch_n=20, 
                batch_size=16,
                device=device, 
                l2_reg_alpha=0.5,
                max_batches_per_epoch_train=10000,
                max_batches_per_epoch_val=1000,
                optimizer_ctor=torch.optim.Adam,
                data_loader_ctor=torch.utils.data.DataLoader,
                lr_scheduler_ctor=torch.optim.lr_scheduler.ReduceLROnPlateau,
                shuffle_train=True,
                dataloader_workers_n=2,
                plot=True)


predicted_labels = []
actual_labels = []

model.eval()
with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predicted_labels.extend(predicted.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())


accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(actual_labels, predicted_labels, average='weighted', zero_division=0)

# Принт метрик
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Model Precision: {precision * 100:.2f}%")
print(f"Model Recall: {recall * 100:.2f}%")
print(f"Model F1 Score: {f1 * 100:.2f}%")

cm = confusion_matrix(actual_labels, predicted_labels)
class_names = test_loader.dataset.classes

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', annot_kws={"size": 15})
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()


df = pd.DataFrame({'preds':predicted_labels, 'actual':actual_labels})
df['err'] = abs(df.preds - df.actual)
df.query('err > 0')


i = 0
for el in iter(test_loader):
    for j in range(16):
        if i + j in df.query('err > 0').index and i + j < 1000:
            image = el[0][j]
            if image.shape[0] == 3:
                image = image[0, ...]
            plt.imshow(image)
            plt.title(f'Предсказали: {df.loc[i+j,"preds"]}; Настоящее: {df.loc[i+j,"actual"]}')
    i+=16
    plt.show()


train_transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),  
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.Resize((224,224)),  
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])  
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224,224)), 
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])  
])


class cnn(torch.nn.Module):
    # Конструктор для класса
    # Инициализация нейронной сети
    def __init__(self):

        super(cnn, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = (3,3), padding=1)
        self.mp1 = torch.nn.MaxPool2d(kernel_size = (2,2), stride=2)
        #32*112*112
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding=1)
        self.mp2 = torch.nn.MaxPool2d(kernel_size = (2,2), stride=2)
        #64*56*56
        self.conv3 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding=1)
        self.mp3 = torch.nn.MaxPool2d(kernel_size = (2,2), stride=2)
        #128*28*28
        self.conv4 = torch.nn.Conv2d(in_channels = 128, out_channels = 140, kernel_size = (3,3), padding=1)
        self.mp4 = torch.nn.MaxPool2d(kernel_size = (2,2), stride=2)
        #140*14*14
        self.conv5 = torch.nn.Conv2d(in_channels = 140, out_channels = 180, kernel_size = (3,3), padding=1)
        self.mp5 = torch.nn.MaxPool2d(kernel_size = (2,2), stride=2)
        #180*7*7
        self.fc1 = torch.nn.Linear(180*7*7, 512)
        self.activ1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(512, 256)
        self.activ2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(256, 200)
        self.activ3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(200, 150)
        self.activ4 = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(150, 100)
        self.activ5 = torch.nn.ReLU()
        self.fc6 = torch.nn.Linear(100, 90)
        self.reg = torch.nn.Dropout(0.1)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, data_input):
        x = self.conv1(data_input)
        x = self.relu(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.mp3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.mp4(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.mp5(x)
        x = x.reshape(-1, 180*7*7)
        x = self.reg(x)
        x = self.fc1(x)
        x = self.activ1(x)
        x = self.fc2(x)
        x = self.activ2(x)
        x = self.fc3(x)
        x = self.activ3(x)
        x = self.fc4(x)
        x = self.activ4(x)
        x = self.fc5(x)
        x = self.activ5(x)
        x = self.fc6(x)
        return x

    def inference(self, data_input):
        with torch.no_grad():
            x = self.forward(data_input)
            x = self.sm(x)
        return x


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_class_names(self):
        return self.classes


root_dir = '/kaggle/input/animal-categories-90-masters-of-survival/animals/animals/animals'


full_dataset = MyDataset(root_dir=root_dir, transform=train_transform)


train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])


val_dataset.dataset.transform = test_transform


batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


class_names = full_dataset.get_class_names()
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")


model = cnn()


best_loss, best_model = train_eval_loop(model=model, 
                train_dataset=train_dataset, 
                val_dataset=val_dataset, 
                criterion=torch.nn.CrossEntropyLoss(),
                lr=1e-3, 
                epoch_n=20, 
                batch_size=batch_size,
                device=device, 
                l2_reg_alpha=0.5,
                max_batches_per_epoch_train=10000,
                max_batches_per_epoch_val=1000,
                optimizer_ctor=torch.optim.Adam,
                data_loader_ctor=torch.utils.data.DataLoader,
                lr_scheduler_ctor=torch.optim.lr_scheduler.ReduceLROnPlateau,
                shuffle_train=True,
                dataloader_workers_n=2,
                plot=True)


predicted_labels = []
actual_labels = []

model.eval()
with torch.inference_mode():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predicted_labels.extend(predicted.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())


accuracy = accuracy_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(actual_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(actual_labels, predicted_labels, average='weighted', zero_division=0)

# Принт метрик
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Model Precision: {precision * 100:.2f}%")
print(f"Model Recall: {recall * 100:.2f}%")
print(f"Model F1 Score: {f1 * 100:.2f}%")

cm = confusion_matrix(actual_labels, predicted_labels)
class_names = val_loader.dataset.classes

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', annot_kws={"size": 15})
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()


df = pd.DataFrame({'preds':predicted_labels, 'actual':actual_labels})
df['err'] = abs(df.preds - df.actual)
df.query('err > 0')


i = 0
for el in iter(val_loader):
    for j in range(16):
        if i + j in df.query('err > 0').index and i + j < 10:
            image = el[0][j]
            if image.shape[0] == 3:
                image = image[0, ...]
            plt.imshow(image)
            plt.title(f'Предсказали: {df.loc[i+j,"preds"]}; Настоящее: {df.loc[i+j,"actual"]}')
    i+=16
    plt.show()

