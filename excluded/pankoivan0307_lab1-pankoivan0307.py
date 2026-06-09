import os
import cv2

import numpy as np
import pandas as pd

import librosa
import librosa.display
import IPython.display as ipd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Доступное устройство: {}'.format(device))


train = pd.read_csv("../input/freesound-audio-tagging/train.csv")
train.head()


labels = np.unique(train.label.values)
print('Всего классов: {}'.format(len(labels)))
labels_encoder = {label:i for i, label in enumerate(labels)}
print(labels_encoder)


trainPath = '../input/freesound-audio-tagging/audio_train/'
testPath = '../input/freesound-audio-tagging/audio_test/'

class Lab1Dataset(Dataset):
    def __init__(self, dataframe, test=False):
        self.dataframe = dataframe
        self.test = test
        
    def __getitem__(self, index):
        fileName = self.dataframe.fname.values[index]
        label = self.dataframe.label.values[index]
        
        path = (testPath if self.test else trainPath) + fileName
        signal, _ = librosa.load(path)
        signal = librosa.feature.melspectrogram(y=signal)    
        signal = librosa.power_to_db(signal, ref=np.max) 
        
        try:
            resized = cv2.resize(signal, (128, 128))
        except Exception as e:
            print(path)
            print(str(e))
            resized = np.zeros(shape=(128, 128))
        
        x = np.stack([resized] * 3)
        x = torch.tensor(x, dtype=torch.float32)

        if self.test == False:
            y = labels_encoder[label]
            return x, y
        else:
             return x
        
    def __len__(self):
        return self.dataframe.shape[0]


batch_size = 64
epochs = 10


train, validation = train_test_split(train, test_size=0.2, shuffle=True, random_state=5)

train_set = Lab1Dataset(train)
val_set = Lab1Dataset(validation)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set , batch_size=batch_size, shuffle=True)

print('Тренировочная выборка: {}'.format(train.shape[0]))
print('Валидационная выборка: {}'.format(validation.shape[0]))


model = efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
model.classifier[1] = torch.nn.Linear(1280, 41) # изменение классификатора под задачу
model = model.to(device)
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
cost = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    train_correct = 0
    val_correct = 0
    
    # перевод модели в режим обучения
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        x,y = x.to(device),y.to(device)
        
        # вычисление предсказания и потерь
        pred = model(x)
        loss = cost(pred, y)
        train_loss += cost(pred, y).item()
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # обратное распространение ошибки
        loss.backward()
        optimizer.step()
    
    # перевод модели в режим оценивания
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x,y = x.to(device),y.to(device)
            
            pred = model(x)
            loss = cost(pred, y)
            val_loss += cost(pred, y).item()
            val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    train_loss = train_loss/len(train_loader)
    val_loss = val_loss/len(val_loader)
    train_accuracy = train_correct / len(train)
    val_accuracy = val_correct / len(validation)
    print("Epoch = %d, train_loss = %.5f, val_loss = %.5f, train_accuracy = %.5f, val_accuracy = %.5f" % (epoch, train_loss, val_loss, train_accuracy, val_accuracy))


test = pd.read_csv('../input/freesound-audio-tagging/sample_submission.csv')

test_dataset = Lab1Dataset(test, test=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
predictions = torch.tensor([])
model.eval()

for x in test_loader:
    x = x.to(device)
    with torch.no_grad():
        y_hat = model(x)
    predictions = torch.cat([predictions, y_hat.cpu()])

predictions = torch.nn.functional.softmax(predictions, dim=1).detach().numpy()


submission_result = test.copy()

N = len(test)
for i in range(N):
    p = predictions[i, :]
    idx = np.argmax(p)
    submission_result.label[i] = labels[idx]

submission_result.to_csv('submission_final.csv', index=False)

submission_result.head()

