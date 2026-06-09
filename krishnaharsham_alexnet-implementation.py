import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.io import read_image
from torch.optim import SGD
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import os
import glob
import pandas as pd


# Model code
def init_weights(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.AlexNet = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.Linear(256*5*5, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, X):
        return self.AlexNet(X)


# Conversion hashmaps
synsetToClass = {}
classToSynset = {}
classToDescription = {}
i = 0
with open("/kaggle/input/imagenet-object-localization-challenge/LOC_synset_mapping.txt","r") as file:
    for idx, line in enumerate(file):
        parts = line.split(" ",1)
        synset = parts[0]
        synsetToClass[synset] = i
        classToSynset[i] = synset
        classToDescription[i] = parts[1].strip() if len(parts) > 1 else ''
        i += 1



os.makedirs("labels/train", exist_ok = True)
os.makedirs("labels/validation", exist_ok = True)


# Creating a CSV file for the training data
train_dir = glob.glob("/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/*")
i = 0
for folder_name in tqdm(train_dir):
    i += 1
    synset = os.path.basename(folder_name)
    class_no = synsetToClass[synset]

    img_files = glob.glob(os.path.join(f"/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/{synset}/", "*.JPEG"))
    img_files = [os.path.basename(img) for img in img_files]

    with open(f'labels/train/train_labels.csv', 'a') as f:
        if i == 1: 
            f.write("image, class \n")
        for img in img_files:
            f.write(f'{img}, {class_no} \n')


df = pd.read_csv('/kaggle/input/imagenet-object-localization-challenge/LOC_val_solution.csv')
df.sort_values(by='ImageId', inplace=True)

j = 0
for i, row in tqdm(df.iterrows()):
    j += 1
    parts = row['PredictionString'].split(' ', 1)
    img = row['ImageId'] + '.JPEG'
    class_no = synsetToClass[parts[0]]
    with open(f'labels/validation/val_labels.csv', 'a') as f:
        f.write(f'{img}, {class_no} \n')

print(f'Total number of valuation images: {j}')


class ImageNetDataSet(Dataset):
    def __init__(self,directory,annotations_file,transform = None, target_transform = None, train = True):
        self.directory = directory
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,idx):
        image_file = self.annotations.iloc[idx,0]
        class_no = int(self.annotations.iloc[idx,1])

        if self.train:
            folder_path = os.path.join(self.directory,classToSynset[class_no])
            image_path = os.path.join(folder_path,image_file)
        else:
            image_path = os.path.join(self.directory, image_file)

        image = read_image(image_path)
        if image.shape[0] == 1:
            image = image.expand(3,-1,-1)

        if image.shape[0] == 4:
            image = image[:3,:,:]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            class_no = self.target_transform(class_no)

        return image, class_no
            


transform = v2.Compose([
    v2.Resize(256),
    v2.RandomCrop((224,224)),
    v2.RandomHorizontalFlip(p = 0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.Normalize(mean = [0.485, 0.456, 0.406],  std = [0.229, 0.224, 0.225])
])

train_directory = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"
train_annotations = "/kaggle/working/labels/train/train_labels.csv"

train_set = ImageNetDataSet(
    directory = train_directory,
    annotations_file = train_annotations,
    transform = transform,
    train = True
)

validation_directory = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"
validation_annotations = "/kaggle/working/labels/validation/val_labels.csv"

val_set = ImageNetDataSet(
    directory = validation_directory,
    annotations_file = validation_annotations,
    transform = transform,
    train = False
)


train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=64, shuffle=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
device


model = AlexNet().to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model.to(device)
model.apply(init_weights)
optimizer = SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = 0.0005)
loss_fn = nn.CrossEntropyLoss()
scheduler = StepLR(
    optimizer,
    step_size=25,
    gamma=0.1
)


def train(model,train_dataloader,optimizer,loss_fn):
    size = len(train_dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (X, y) in tqdm(enumerate(train_dataloader),total = len(train_dataloader)):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    return avg_train_loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    avg_test_loss = test_loss / num_batches
    accuracy = correct / size
    return accuracy, avg_test_loss


epochs = 1
for t in range(epochs):
    avg_train_loss = train(model, train_dataloader,optimizer,loss_fn)
    if t == 0 or (t + 1) % 5 == 0 or t == epochs - 1:  # Print every 5 epochs or the last epoch
        accuracy, avg_test_loss = test(val_dataloader, model, loss_fn)
        print(f"Epoch {t+1}\n-------------------------------")
        print(f"Train Avg Loss: {avg_train_loss:>8f}")
        print(f"Test Accuracy: {(100 * accuracy):>0.1f}%, Test Avg Loss: {avg_test_loss:>8f} \n")
print("Done!")




