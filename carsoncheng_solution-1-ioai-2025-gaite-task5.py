import random
import numpy as np
import torch

seed = 42

random.seed(seed)                  # Python built-in random
np.random.seed(seed)               # NumPy
torch.manual_seed(seed)            # PyTorch (CPU)
torch.cuda.manual_seed(seed)       # PyTorch (single GPU)
torch.cuda.manual_seed_all(seed)   # PyTorch (all GPUs)

# Ensures deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import os
import zipfile
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import mobilenet
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
import os
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):   # Class used for data loading, DO NOT modify
    """
    Load spectrogram data from preprocessed .pt files.

    For training_set/, assumes:
        dataset/training_set/
            bonafide/
            spoof/

    For validation_set/ and testing_set/, assumes:
        dataset/validation_set/    (all .pt files in this folder, no subfolders)
        dataset/testing_set/       (all .pt files in this folder, no subfolders)

    No label will be provided for val/test sets to prevent label leakage.
    """

    def __init__(self, directory):
        self.samples = []

        if "training" in directory:
            label_map = {"bonafide": 0, "spoof": 1}
            for label_name, label in label_map.items():
                label_dir = os.path.join(directory, label_name)
                if not os.path.isdir(label_dir):
                    continue
                for fname in os.listdir(label_dir):
                    if fname.endswith(".pt"):
                        self.samples.append(
                            {"path": os.path.join(label_dir, fname), "label": label}
                        )
        else:
            for fname in sorted(os.listdir(directory)):
                if fname.endswith(".pt"):
                    self.samples.append({"path": os.path.join(directory, fname)})
            self.samples = [{"path": os.path.join(directory, f"{i}.pt")} for i in range(len(self.samples))]
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        spec = torch.load(item["path"])
        out = {"spectrogram": spec}
        if "label" in item:
            out["label"] = torch.tensor(item["label"], dtype=torch.long)
        return out



class MyModel(nn.Module): # use pretrained resnet18
    def __init__(self):
        super().__init__()
        model = resnet18(pretrained = True) # switch to pretrained model
        # model = resnet18(pretrained = ResNet18_Weights)  #use an offline pretrained model of resnet18
        # Setting pretrained = False means not importing the pre-trained model parameters of the current version of ResNet18.
        # Please do not set pretrained = True since the testing machine cannot connect to the internet.
        # You can change resnet18 to resnet 34 or 50 to achieve a high score
        # Other pretrained weights are not deployed ahead in advance
        # By reasonably designing the model, a score of 0.99 can be achieved
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        model.fc = nn.Linear(model.fc.in_features, 2)
        self.model = model

    def forward(self, x):
        return self.model(x)


def train_one_epoch(model, train_loader, val_loader, criterion, optimizer, device): # train only 1 epoch. You can train more epochs if needed
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc="Train"):
        x = batch["spectrogram"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Train Loss: {train_loss:.4f}")

    model.eval()
    val_loss = 0.0
    predictions = []
    gt = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val Split"):
            x = batch["spectrogram"].to(device)
            y = batch["label"].to(device)
            output = model(x)
            loss = criterion(output, y)
            predictions.append(torch.argmax(output, 1))
            gt.append(y)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Val Split Loss: {val_loss:.4f}")


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test"):
            x = batch["spectrogram"].to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            preds.extend(pred.cpu().numpy())
    return preds


def save_submission_csv(preds, save_name):
    #df = pd.DataFrame(preds)
    df = pd.read_csv("/kaggle/input/ioai-2025-gaite-task-5/sample_submission.csv")
    df['label'] = preds
    #print(df)
    df.to_csv(save_name, index=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


full_train_set = SpectrogramDataset("/kaggle/input/ioai-2025-gaite-task-5/train/training_set") # load training data and select 50% for training

val_size = int(0.5 * len(full_train_set)) # increase ratio to increase amount of training data
train_size = len(full_train_set) - val_size
train_set, val_split_set = random_split(full_train_set, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32)
val_split_loader = DataLoader(val_split_set, batch_size=32)
record = 0
train_one_epoch(model, train_loader, val_split_loader, criterion, optimizer, device)


#DATA_PATH is the secret environment variable to point the address of the validation set and test set on the testing machine. 
#You cannot access this address locally.
if os.environ.get('DATA_PATH'):
    DATA_PATH = os.environ.get("DATA_PATH")+"/" 
else:
    DATA_PATH = ""
    print("When the baseline is running, this error message will appear because the test set cannot be read, which is a normal phenomenon.") #When the baseline is running, this error message will appear because the test set cannot be read, which is a normal phenomenon.
    
#val_set = SpectrogramDataset(DATA_PATH + "/validation_set")
test_set = SpectrogramDataset("/kaggle/input/ioai-2025-gaite-task-5/test/testing_set")
#print(test_set.samples)
#val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

#val_preds = predict(model, val_loader, device)
test_preds = predict(model, test_loader, device)


#save_submission_csv(val_preds, "submissionA.csv")
save_submission_csv(test_preds, "submission.csv")


pd.read_csv("submission.csv")




