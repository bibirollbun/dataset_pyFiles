import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision import transforms, models
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score


root_dir = '/kaggle/input/state-farm-distracted-driver-detection/imgs/train'


transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


dataset = ImageFolder(root = root_dir, transform = transform)


train_dataset, test_dataset = random_split(dataset, [round(0.8*(len(dataset))), round(0.2*(len(dataset)))])


train_loader = DataLoader(train_dataset, batch_size=8, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle = False)


class EfficientNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.aboba = models.efficientnet_b3(pretrained=True)
        self.MLP = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.aboba(x)
        x = self.MLP(x)
        return x


model = EfficientNetClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


device = torch.device('cuda')
model = model.to(device)


losses = []
fs = []
valid = []

for epoch in range(10):
    model.train()
    
    running_loss = 0.0
    running_f1 = 0.0

    num_batches = 0
    for inputs, labels in (bar := tqdm(train_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
        
        
        score = f1_score(labels.detach().cpu().numpy(), np.argmax(outputs.detach().cpu().numpy(), axis=1), average='macro')
        running_f1 += score
        bar.set_description(f"Epoch: {epoch + 1}, Loss by batch: {loss.item():.4f}, F1 by epoch: {score:.4f}")


    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            outputs = model(inputs)
            preds = np.argmax(outputs.cpu(), axis=1)
    
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    f1_valid = f1_score(all_labels, all_preds, average='macro')
    print(f"Epoch: {epoch + 1}, mean loss on epoch: {running_loss / num_batches}, mean F1: {running_f1 / num_batches}, F1 valid: {f1_valid}")
    valid.append(f1_valid)
    losses.append(running_loss / num_batches)
    fs.append(running_f1 / num_batches)

