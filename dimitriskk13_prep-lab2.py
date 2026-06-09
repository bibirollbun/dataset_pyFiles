# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:1]:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# clone the respective repo
# make sure you have the Internet option enabled under Session options
!git clone https://github.com/slp-ntua/patrec-labs.git


# add path to system
import sys
sys.path.append('/kaggle/working/patrec-labs/lab3')


!ls


# install requirements
!pip install -r patrec-labs/lab3/requirements.txt


!pip install --force-reinstall --no-cache-dir numpy==1.20.3


file_path ='/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train_labels.txt'
df = pd.read_csv(file_path, sep='\t', header=None, names=['filename', 'label'])

row1 = df.sample(1,random_state=13)
label1 = row1['label'].values[0]

row2 = df[df['label'] != label1].sample(1,random_state=13)

print(f"Selection 1: {row1['filename'].values[0]} (Label: {label1})")
print(f"Selection 2: {row2['filename'].values[0]} (Label: {row2['label'].values[0]})")


from collections import namedtuple


def split(dir,selections):
    recording_dir = dir
    Sample = namedtuple('Sample', ['filename', 'label', 'mel', 'chroma'])
    selections = selections
    dataset =[] 
    for row in selections:
        raw_name = row['filename'].values[0]
        label = row['label'].values[0]
    
        real_name = raw_name.replace('.gz', '')
        full_path = os.path.join(recordings_dir, real_name)
        spec = np.load(full_path)
    
        mel_part = spec[:128]
        chroma_part = spec[128:]
    
        sample_obj = Sample(
            filename=real_name, 
            label=label, 
            mel=mel_part, 
            chroma=chroma_part
        )
        dataset.append(sample_obj)
        print(f"{real_name:<30} | {str(mel_part.shape):<15} | {str(chroma_part.shape):<15}")

    return dataset

selections = [row1,row2]
recordings_dir = '/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train'
dataset = split(recordings_dir,selections)



first_sample = dataset[0]
second_sample = dataset[1]

# Plot the spectrogram 

import librosa.display
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
img = librosa.display.specshow(first_sample.mel, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Spectrogram')
fig.colorbar(img, ax=ax, format="%+2.f dB")

fig, ax = plt.subplots()
img = librosa.display.specshow(second_sample.mel, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Spectrogram')
fig.colorbar(img, ax=ax, format="%+2.f dB")




first_sample.mel.shape


from collections import namedtuple

recordings_dir = '/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/'
selections = [row1,row2]

dataset_beat = split(recordings_dir,selections)


first_sample = dataset_beat[0]
second_sample = dataset_beat[1]

# Plot the spectrogram 

import librosa.display
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
img = librosa.display.specshow(first_sample.mel, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Spectrogram')
fig.colorbar(img, ax=ax, format="%+2.f dB")

fig, ax = plt.subplots()
img = librosa.display.specshow(second_sample.mel, x_axis='time', y_axis='linear', ax=ax)
ax.set(title='Spectrogram')
fig.colorbar(img, ax=ax, format="%+2.f dB")





first_sample = dataset[0]
second_sample = dataset[1]

# Plot the chromagram

import librosa.display
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
img = librosa.display.specshow(first_sample.chroma, y_axis='chroma', x_axis='time', ax=ax)
ax.set(title='Chromagram')
fig.colorbar(img, ax=ax)


fig, ax = plt.subplots()
img = librosa.display.specshow(second_sample.chroma, y_axis='chroma', x_axis='time', ax=ax)
ax.set(title='Chromagram')
fig.colorbar(img, ax=ax)



first_sample = dataset_beat[0]
second_sample = dataset_beat[1]

# Plot the chromagram

import librosa.display
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
img = librosa.display.specshow(first_sample.chroma, y_axis='chroma', x_axis='time', ax=ax)
ax.set(title='Chromagram')
fig.colorbar(img, ax=ax)


fig, ax = plt.subplots()
img = librosa.display.specshow(second_sample.chroma, y_axis='chroma', x_axis='time', ax=ax)
ax.set(title='Chromagram')
fig.colorbar(img, ax=ax)




from dataset import SpectrogramDataset, CLASS_MAPPING, torch_train_val_split


beat_mel_specs = SpectrogramDataset(
     '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/', class_mapping=CLASS_MAPPING, 
    train=True, feat_type='mel', max_length=-1
)
    
train_loader_beat_mel, val_loader_beat_mel = torch_train_val_split(beat_mel_specs, 32 ,32, val_size=.33)


datum = next(iter(train_loader_beat_mel)) #returns batch #0
print('Data shape')
print(datum[0].shape)  # shape of data
print('Labels')
print(datum[1])  # labels in batch
print('Lengths')
print(datum[2])  # length of each element in batch


beat_chroma = SpectrogramDataset(
     '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/', class_mapping=CLASS_MAPPING, 
    train=True, feat_type='chroma', max_length=-1
)
    
train_loader_beat_chroma, val_loader_beat_chroma = torch_train_val_split(beat_chroma, 32 ,32, val_size=.33)


datum = next(iter(train_loader_beat_chroma))
print('Data shape')
print(datum[0].shape)  # shape of data
print('Labels')
print(datum[1])  # labels in batch
print('Lengths')
print(datum[2])  # length of each element in batch


specs_fused = SpectrogramDataset(
     '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/', class_mapping=CLASS_MAPPING, 
    train=True, feat_type='fused', max_length=-1
)
train_loader, val_loader = torch_train_val_split(specs_fused, 32 ,32, val_size=.33)


datum = next(iter(train_loader))
print('Data shape')
print(datum[0].shape)  # shape of data
print('Labels')
print(datum[1])  # labels in batch
print('Lengths')
print(datum[2])  # length of each element in batch


import os
os.listdir("../input/patreco3-multitask-affective-music/data/")


# Step 4: Data loading and analysis
from dataset import CLASS_MAPPING, SpectrogramDataset

PARENT_DATA_DIR = '../input/patreco3-multitask-affective-music/data/'

train_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms_beat'), class_mapping=CLASS_MAPPING, train=True
)

train_dataset_all = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms_beat'), class_mapping=None, train=True
)



import matplotlib.pyplot as plt

def plot_class_histogram(dataset, title):
    plt.figure()
    plt.hist(dataset.labels, rwidth=0.5, align='mid')
    plt.xticks(dataset.labels, dataset.labels.astype(str))
    plt.title(title)
    plt.xlabel('Class Labels')
    plt.ylabel('Frequencies')


plot_class_histogram(train_dataset, title='Histogram of classes with merged classes')
plot_class_histogram(train_dataset_all, title='Histogram of classes with all classes')


print(f"files before class mapping: {len(train_dataset_all)}")
print(f"files after class mapping: {len(train_dataset)}")


import torch
import torch.nn as nn

def overfit_with_a_couple_of_batches(model, train_loader, optimizer, device):
    print('Training in overfitting mode...')
    epochs = 400
    
    # get only the 1st batch
    x_b1, y_b1, lengths_b1 = next(iter(train_loader))    
    model.train()
    for epoch in range(epochs):        
        loss, logits = model(x_b1.float().to(device), y_b1.to(device), lengths_b1.to(device))
        # prepare
        optimizer.zero_grad()
        # backward
        loss.backward()
        # optimizer step
        optimizer.step()

        if epoch == 0 or (epoch+1)%20 == 0:
            print(f'Epoch {epoch+1}, Loss at training set: {loss.item()}')    

# we need EarlyStopping that we'll adopt from: https://stackoverflow.com/a/73704579/19306080 and add some customizations
class EarlyStopper:
    def __init__(self, model, save_path, patience=1, min_delta=0):
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(self.model.state_dict(), self.save_path)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



class Classifier(nn.Module):
    def __init__(self, num_classes, backbone, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        num_classes (int): The number of classes
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(Classifier, self).__init__()
        self.backbone = backbone  # An LSTMBackbone or CNNBackbone
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()  # Loss function for classification

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        logits = self.output_layer(feats)
        loss = self.criterion(logits, targets)
        return loss, logits


# implemetation of the training process
DEVICE = 'cuda' 

def train_one_epoch(model, train_loader, optimizer, device=DEVICE):
    model.train()
    total_loss = 0
    for x, y, lengths in train_loader:        
        loss, logits = model(x.float().to(device), y.to(device), lengths.to(device))
        # prepare
        optimizer.zero_grad()
        # backward
        loss.backward()
        # optimizer step
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)    
    return avg_loss


def validate_one_epoch(model, val_loader, device=DEVICE):    
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y, lengths in val_loader:
            loss, logits = model(x.float().to(device), y.to(device), lengths.to(device))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)    
    return avg_loss


def train(model, train_loader, val_loader, optimizer, epochs, save_path='checkpoint.pth', device="cuda", overfit_batch=False):
    if overfit_batch:
        overfit_with_a_couple_of_batches(model, train_loader, optimizer, device)
    else:
        print(f'Training started for model {save_path.replace(".pth", "")}...')
        early_stopper = EarlyStopper(model, save_path, patience=5)
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer)
            validation_loss = validate_one_epoch(model, val_loader)
            if epoch== 0 or (epoch+1)%5==0:
                print(f'Epoch {epoch+1}/{epochs}, Loss at training set: {train_loss}\n\tLoss at validation set: {validation_loss}')          
            
            if early_stopper.early_stop(validation_loss):
                print('Early Stopping was activated.')
                print(f'Epoch {epoch+1}/{epochs}, Loss at training set: {train_loss}\n\tLoss at validation set: {validation_loss}')
                print('Training has been completed.\n')
                break


PARENT_DATA_DIR = '../input/patreco3-multitask-affective-music/data/'
BATCH_SIZE = 8
MAX_LENGTH = 150
DEVICE = 'cuda'

train_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms'), class_mapping=CLASS_MAPPING, 
    train=True, feat_type='mel', max_length=MAX_LENGTH
)

train_loader, val_loader = torch_train_val_split(
    train_dataset, BATCH_SIZE, BATCH_SIZE
)

# get the 1st batch values of the data loader
x_b1, y_b1, lengths_b1 = next(iter(train_loader))

# print the shape of the 1st item of the 1st batch of the data loader
input_shape = x_b1[0].shape
print(input_shape)


# run Training in overfitting mode
from lstm import LSTMBackbone

DEVICE = 'cuda' 
RNN_HIDDEN_SIZE = 64
NUM_CATEGORIES = 10

LR = 1e-4
epochs = 10

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs, device=DEVICE, overfit_batch=True)


# dataloaders for specific features type'
# mel

PARENT_DATA_DIR = '../input/patreco3-multitask-affective-music/data/'
BATCH_SIZE = 8
MAX_LENGTH = 150
DEVICE = 'cuda'

train_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms'), class_mapping=CLASS_MAPPING, 
    train=True, feat_type='mel', max_length=MAX_LENGTH
)

train_loader, val_loader = torch_train_val_split(
    train_dataset, BATCH_SIZE, BATCH_SIZE
)

# get the 1st batch values of the data loader
x_b1, y_b1, lengths_b1 = next(iter(train_loader))

# print the shape of the 1st item of the 1st batch of the data loader
input_shape = x_b1[0].shape
print(input_shape)


from lstm import LSTMBackbone


# run Training
DEVICE = 'cuda'
RNN_HIDDEN_SIZE = 128
NUM_CATEGORIES = 10

LR = 1e-4
epochs = 40
save_path='lstm_genre_mel.pth'

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs, save_path=save_path, device=DEVICE, overfit_batch=False)


# dataloaders for specific features type
# beat_synced mel
BATCH_SIZE = 8
MAX_LENGTH = 150
DEVICE = 'cuda'
PARENT_DATA_DIR = '../input/patreco3-multitask-affective-music/data/'

train_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms_beat'), class_mapping=CLASS_MAPPING, 
    train=True, feat_type='mel', max_length=MAX_LENGTH
)

train_loader, val_loader = torch_train_val_split(
    train_dataset, BATCH_SIZE, BATCH_SIZE
)

# get the 1st batch values of the data loader
x_b1, y_b1, lengths_b1 = next(iter(train_loader))

# print the shape of the 1st item of the 1st batch of the data loader
input_shape = x_b1[0].shape
print(input_shape)


# run Training
DEVICE = 'cuda'
RNN_HIDDEN_SIZE = 128
NUM_CATEGORIES = 10

LR = 1e-4
epochs = 40
save_path='lstm_genre_mel_beats.pth'

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs, save_path=save_path, device=DEVICE, overfit_batch=False)


# dataloaders for specific features type
# chroma 
BATCH_SIZE = 8
MAX_LENGTH = 150
DEVICE = 'cuda'
PARENT_DATA_DIR = '../input/patreco3-multitask-affective-music/data/'

train_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms'), class_mapping=CLASS_MAPPING, 
    train=True, feat_type='chroma', max_length=MAX_LENGTH
)

train_loader, val_loader = torch_train_val_split(
    train_dataset, BATCH_SIZE, BATCH_SIZE
)

# get the 1st batch values of the data loader
x_b1, y_b1, lengths_b1 = next(iter(train_loader))

# print the shape of the 1st item of the 1st batch of the data loader
input_shape = x_b1[0].shape
print(input_shape)


# run Training
DEVICE = 'cuda'
RNN_HIDDEN_SIZE = 128
NUM_CATEGORIES = 10

LR = 1e-4
epochs = 40
save_path='lstm_genre_chroma.pth'

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs, save_path=save_path, device=DEVICE, overfit_batch=False)


# dataloaders for specific features type, e.g. 'mel'
BATCH_SIZE = 8
MAX_LENGTH = 150
DEVICE = 'cuda'
PARENT_DATA_DIR = '../input/patreco3-multitask-affective-music/data/'

train_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, 'fma_genre_spectrograms'), class_mapping=CLASS_MAPPING, 
    train=True, feat_type='fused', max_length=MAX_LENGTH
)

train_loader, val_loader = torch_train_val_split(
    train_dataset, BATCH_SIZE, BATCH_SIZE
)

# get the 1st batch values of the data loader
x_b1, y_b1, lengths_b1 = next(iter(train_loader))

# print the shape of the 1st item of the 1st batch of the data loader
input_shape = x_b1[0].shape
print(input_shape)


# run Training
DEVICE = 'cuda'
RNN_HIDDEN_SIZE = 128
NUM_CATEGORIES = 10

LR = 1e-4
epochs = 40
save_path='lstm_genre_fused.pth'

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs, save_path=save_path, device=DEVICE, overfit_batch=False)


from sklearn.metrics import classification_report

def get_labels_predictions(model, data_loader, device):
    model.eval()
    y = []
    y_ = []
    with torch.no_grad():
        for x, labels, lengths in data_loader:
            loss, logits = model(x.float().to(device), labels.to(device), lengths.to(device))
            y.append(labels)
            y_.append(logits.argmax(dim=-1).detach().cpu().numpy())

    return y, y_ 


data_dir = 'fma_genre_spectrograms'
saved_model_path = 'lstm_genre_mel.pth'
PARENT_DATA_DIR = '../input/patreco3-multitask-affective-music/data/'
MAX_LENGTH = 150
print(f'Evaluation of model {saved_model_path.replace(".pth", "")} on the test set...')

test_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, data_dir), class_mapping=CLASS_MAPPING, 
    train=False, feat_type='mel',  max_length=MAX_LENGTH
)

test_loader, _ = torch_train_val_split(
    test_dataset, BATCH_SIZE, BATCH_SIZE, val_size=0.0
)

# get the input shape
x_b1, y_b1, lengths_b1 = next(iter(test_loader))
input_shape = x_b1[0].shape

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)
model.load_state_dict(torch.load(saved_model_path))

y, y_ = get_labels_predictions(model, test_loader, DEVICE)
# Add zero_division=0 to silence the warning
print(classification_report(np.hstack(y), np.hstack(y_), zero_division=0))


data_dir = 'fma_genre_spectrograms_beat'
saved_model_path = 'lstm_genre_mel_beats.pth'

print(f'Evaluation of model {saved_model_path.replace(".pth", "")} on the test set...')

test_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, data_dir), class_mapping=CLASS_MAPPING, 
    train=False, feat_type='mel',  max_length=MAX_LENGTH
)

test_loader, _ = torch_train_val_split(
    test_dataset, BATCH_SIZE, BATCH_SIZE, val_size=0.0
)

# get the input shape
x_b1, y_b1, lengths_b1 = next(iter(test_loader))
input_shape = x_b1[0].shape

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)
model.load_state_dict(torch.load(saved_model_path))

y, y_ = get_labels_predictions(model, test_loader, DEVICE)
print(classification_report(np.hstack(y), np.hstack(y_), zero_division=0))


data_dir = 'fma_genre_spectrograms'
saved_model_path = 'lstm_genre_chroma.pth'

print(f'Evaluation of model {saved_model_path.replace(".pth", "")} on the test set...')

test_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, data_dir), class_mapping=CLASS_MAPPING, 
    train=False, feat_type='chroma',  max_length=MAX_LENGTH
)

test_loader, _ = torch_train_val_split(
    test_dataset, BATCH_SIZE, BATCH_SIZE, val_size=0.0
)

# get the input shape
x_b1, y_b1, lengths_b1 = next(iter(test_loader))
input_shape = x_b1[0].shape

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)
model.load_state_dict(torch.load(saved_model_path))

y, y_ = get_labels_predictions(model, test_loader, DEVICE)
print(classification_report(np.hstack(y), np.hstack(y_), zero_division=0))


data_dir = 'fma_genre_spectrograms'
saved_model_path = 'lstm_genre_fused.pth'

print(f'Evaluation of model {saved_model_path.replace(".pth", "")} on the test set...')

test_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, data_dir), class_mapping=CLASS_MAPPING, 
    train=False, feat_type='fused',  max_length=MAX_LENGTH
)

test_loader, _ = torch_train_val_split(
    test_dataset, BATCH_SIZE, BATCH_SIZE, val_size=0.0
)

# get the input shape
x_b1, y_b1, lengths_b1 = next(iter(test_loader))
input_shape = x_b1[0].shape

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)
model.load_state_dict(torch.load(saved_model_path))

y, y_ = get_labels_predictions(model, test_loader, DEVICE)
print(classification_report(np.hstack(y), np.hstack(y_), zero_division=0))


!ls


# 2D CNN

# copy paste from convolution.py and add the implementation
import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self, input_dims, in_channels, filters, feature_size):
        super(CNNBackbone, self).__init__()
        self.input_dims = input_dims
        self.in_channels = in_channels
        self.filters = filters
        self.feature_size = feature_size
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, filters[0], kernel_size=(5,5), stride=1, padding=2),
            nn.BatchNorm2d((self.in_channels**1) * filters[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
                nn.Conv2d(filters[0], filters[1], kernel_size=(5,5), stride=1, padding=2),
                nn.BatchNorm2d((self.in_channels**2) * filters[1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv3 = nn.Sequential(
                nn.Conv2d(filters[1], filters[2], kernel_size=(3,3), stride=1, padding=1),
                nn.BatchNorm2d((self.in_channels**3) * filters[2]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv4 = nn.Sequential(
                nn.Conv2d(filters[2], filters[3], kernel_size=(3,3), stride=1, padding=1),
                nn.BatchNorm2d((self.in_channels**4) * filters[3]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        
        shape_after_convs = [input_dims[0]//2**(len(filters)), input_dims[1]//2**(len(filters))]
        self.fc1 = nn.Linear(filters[3] * shape_after_convs[0] * shape_after_convs[1], self.feature_size)
        
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, x.shape[1], x.shape[2])
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out


# run Training in overfitting mode for the CNN 
DEVICE = 'cuda'
NUM_CATEGORIES = 10
cnn_in_channels = 1
cnn_filters = [32, 64, 128, 256]
cnn_out_feature_size = 1000

LR = 1e-4
epochs = 10

# get the input shape
x_b1, y_b1, lengths_b1 = next(iter(train_loader))
input_shape = x_b1[0].shape

backbone = CNNBackbone(input_shape, cnn_in_channels, cnn_filters, cnn_out_feature_size)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs, device=DEVICE, overfit_batch=True)


data_dir = 'fma_genre_spectrograms'
saved_model_path = 'cnn_genre_mel.pth'

print(f'Evaluation of model {saved_model_path.replace(".pth", "")} on the test set...')

test_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, data_dir), class_mapping=CLASS_MAPPING, 
    train=False, feat_type='mel', max_length=MAX_LENGTH
)

test_loader, _ = torch_train_val_split(
    test_dataset, BATCH_SIZE, BATCH_SIZE, val_size=0.0
)

# get the input shape
x_b1, y_b1, lengths_b1 = next(iter(test_loader))
input_shape = x_b1[0].shape

backbone = CNNBackbone(input_shape, cnn_in_channels, cnn_filters, cnn_out_feature_size)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)
model.load_state_dict(torch.load(saved_model_path))

y, y_ = get_labels_predictions(model, test_loader, DEVICE)
print(classification_report(np.hstack(y), np.hstack(y_)))


# run Training
DEVICE = 'cuda'
NNUM_CATEGORIES = 10
cnn_in_channels = 1
cnn_filters = [32, 64, 128, 256]
cnn_out_feature_size = 1000

LR = 1e-4
epochs = 40
save_path='cnn_genre_mel.pth'

backbone = CNNBackbone(input_shape, cnn_in_channels, cnn_filters, cnn_out_feature_size)
model = Classifier(NUM_CATEGORIES, backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs, save_path=save_path, device=DEVICE, overfit_batch=False)


from ast_model import ASTBackbone


# add implementation of Regressor

class Regressor(nn.Module):
    def __init__(self, backbone, load_from_checkpoint=None):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        load_from_checkpoint (Optional[str]): Use a pretrained checkpoint to initialize the model
        """
        super(Regressor, self).__init__()
        self.backbone = backbone  # An LSTMBackbone or CNNBackbone
        if load_from_checkpoint is not None:
            self.backbone = load_backbone_from_checkpoint(
                self.backbone, load_from_checkpoint
            )
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, 1)
        self.criterion = nn.MSELoss()  # Loss function for regression

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        out = self.output_layer(feats)
        loss = self.criterion(out.float(), targets.float())
        return loss, out


# dataloaders for specific features type, e.g. 'mel'
BATCH_SIZE = 8
MAX_LENGTH = 150
DEVICE = 'cuda'


tasks = ['valence', 'energy', 'danceability']
task = 'valence'
label_index = tasks.index(task) 

train_dataset = SpectrogramDataset(
    os.path.join(PARENT_DATA_DIR, data_dir), 
    train=True, feat_type='mel', max_length=MAX_LENGTH, regression_label_index=label_index
)

train_loader, val_loader = torch_train_val_split(
    train_dataset, BATCH_SIZE, BATCH_SIZE
)


# run Training in overfitting mode
from lstm import LSTMBackbone

DEVICE = 'cuda' 
RNN_HIDDEN_SIZE = 64
NUM_CATEGORIES = 10

LR = 1e-4
epochs = 10

backbone = LSTMBackbone(input_shape[1], rnn_size=RNN_HIDDEN_SIZE, num_layers=2, bidirectional=True)
model = Regressor(backbone)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs, device=DEVICE, overfit_batch=True)


source_model_path = 'cnn_genre_fused_beat.pth'

# target model
backbone = CNNBackbone(input_shape, CNN_IN_CHANNELS, CNN_FILTERS, CNN_OUT_FEATURE_SIZE)
model = Regressor(backbone)
model.to(DEVICE)

# load source model
source_model = torch.load(source_model_path)

# remove weights of final layer
del source_model['output_layer.weight']
del source_model['output_layer.bias']

# initialize target model with the state of source model
model.load_state_dict(source_model, strict=False)

# train the model, initialized with genre's model weights, for valence (regression)
save_path = 'cnn_valence_transfer_fused_beat.pth'
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train(model, train_loader, val_loader, optimizer, epochs, save_path=save_path, device=DEVICE, overfit_batch=False)

