# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


trainData = Path('/kaggle/input/heads-or-tails-image-classification/train').rglob('*.jpg')

headCount, tailCount = 0, 0
headFilesTrain = []
tailFilesTrain = []
for file in trainData:
    if file.parent.name == "heads":
        headCount += 1
        headFilesTrain.append(file) 
    else:
        tailCount += 1
        tailFilesTrain.append(file) 
        
    
print(f"Number of train data, heads : {headCount}, tails: {tailCount}")





from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio = (1.0,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])  
])

trainDataPath = '/kaggle/input/heads-or-tails-image-classification/train'
trainDataSet = datasets.ImageFolder(root = trainDataPath, transform = transform)

# Dataloader
trainLoader = DataLoader(trainDataSet, batch_size=32, shuffle=True, num_workers=4)


import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

model = models.efficientnet_b0(weights='IMAGENET1K_V1')  # Pretrained
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)  # Adjust output
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


def Train(model, dataLoader, numEpochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(numEpochs):
        model.train()

        for inputs, labels in dataLoader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)  # shape: [batch,1]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()


        # -------- Compute ROC AUC on training data --------
        model.eval()
        all_labels = []
        all_scores = []
        with torch.no_grad():
            for inputs, labels in trainLoader:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())
    
        train_auc = roc_auc_score(all_labels, all_scores)
        print(f"Epoch {epoch+1}/{numEpochs}, Training ROC AUC: {train_auc:.4f}")
            
Train(model, trainLoader, 200)
torch.save(model.state_dict(), 'coinToss_model_v1p3p0.pt')


# Create a custom loader for test images
from PIL import Image
from torchvision import datasets, transforms

class TestImageFolder(datasets.ImageFolder):
    def __init__(self, rootDir, transform = None):
        self.transform = transform
        self.rootDir = rootDir
        self.imageNames = [file for file in Path(rootDir).glob('*.jpg')]

    def __len__(self):
        return len(self.imageNames)

    def __getitem__(self, idx):
        image = Image.open(self.imageNames[idx]).convert('RGB')
        if(self.transform):
            image = self.transform(image)
        return image, str(self.imageNames[idx])

testDataPath = '/kaggle/input/heads-or-tails-image-classification/test'
testDataSet = TestImageFolder(testDataPath, transform)

testDataLoader = DataLoader(testDataSet, batch_size=32, shuffle=False, num_workers=0)


import re

def findNumber(x):
    matches = re.findall(r'\d+', x)
    return int(matches[0]) if matches else -1

def Infer(model, testDataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    allProbs = []
    allPredictionIDs = []
    with torch.no_grad():
        for inputs, fileFullPaths in testDataLoader:
            fileNames = [findNumber(file) for file in fileFullPaths]
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze(1)
            probs = probs.cpu().numpy()
            allProbs.extend(probs)
            allPredictionIDs.extend(fileNames)
    return allProbs, allPredictionIDs

probsResult, predictionIds = Infer(model, testDataLoader)
    


# Sort keys and reorder values to match
sorted_pairs = sorted(zip(predictionIds, probsResult))  # sorts by keys
predictionIds, probsResult = zip(*sorted_pairs)

results = df = pd.DataFrame({
    'prediction_id': predictionIds,
    'probability_of_heads': [1 - probs for probs in probsResult]
})

results.to_csv("submission.csv")


