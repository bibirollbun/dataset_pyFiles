import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional
import librosa
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train = pd.read_csv("../input/freesound-audio-tagging/train.csv")
train.head()


unique_labels = train.label.unique()
print("Labels:", unique_labels)

labels = np.unique(train.label.values)
label_encoder = {label:i for i, label in enumerate(labels)}


model = torchvision.models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


model.classifier[1] = torch.nn.Linear(1280, 41)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


TRAIN_PATH = '../input/freesound-audio-tagging/audio_train/'
TEST_PATH = '../input/freesound-audio-tagging/audio_test/'
batch_size = 64
epochs = 10

class SoundDataset(Dataset):
    def __init__(self, dataframe, path, test=False):
        super(SoundDataset, self).__init__()
        self.dataframe = dataframe
        self.path = path
        self.test = test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        file_path = self.dataframe.fname.values[idx]
        label = self.dataframe.label.values[idx]
        path = (TEST_PATH if self.test else TRAIN_PATH) + file_path
        signal, _ = librosa.load(path)
        signal = librosa.feature.melspectrogram(y=signal)
        signal = librosa.power_to_db(signal, ref=np.max)

        try:
            resized = cv2.resize(signal, (128, 128))
        except Exception as e:
            print(path)
            print(str(e))
            resized = np.zeros(shape=(128, 128))

        X = np.stack([resized] * 3)  # Дублирование каналов
        X = torch.tensor(X, dtype=torch.float32)

        if not self.test:
            y = label_encoder[label]
            return X, y
        else:
            return X


x_train, x_validation, y_train, y_validation = train_test_split(train, train, test_size=0.2, shuffle=True, random_state=5)

train_dataset = SoundDataset(x_train, TRAIN_PATH)
val_dataset = SoundDataset(x_validation, TRAIN_PATH)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X.size(0)
        train_correct += torch.sum(pred.argmax(1) == y).item()

    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = train_correct / len(train_loader.dataset)

    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            val_loss += loss.item() * X.size(0)
            val_correct += torch.sum(pred.argmax(1) == y).item()

    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = val_correct / len(val_loader.dataset)

    print("Epoch {}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}".format(epoch+1, train_loss, train_accuracy, val_loss, val_accuracy))


test = pd.read_csv('../input/freesound-audio-tagging/sample_submission.csv')
test_dataset = SoundDataset(test, TEST_PATH, test=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
predictions = torch.tensor([])

model.eval()
with torch.no_grad():
    for X in test_loader:
        X = X.to(device)
        y_hat = model(X)
        predictions = torch.cat([predictions, y_hat.cpu()])

predictions = torch.nn.functional.softmax(predictions, dim=1).detach().numpy()

submission_top1 = test.copy()


for i in range(len(test)):
    p = predictions[i, :]
    idx = np.argmax(p)
    submission_top1.label[i] = labels[idx]

submission_top1.to_csv('submission_final.csv', index=False, header=True)

submission_top1.head()

