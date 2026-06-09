import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

from PIL import Image
import torchvision.models as models
from torchvision.transforms import v2
from torchvision import datasets


from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from transformers import pipeline 

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score


seed = 34

# Set seed for NumPy (for any NumPy-based operations)
np.random.seed(seed)

# Set seed for PyTorch CPU operations
torch.manual_seed(seed)


train = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/train.csv')
test = pd.read_csv('/kaggle/input/detect-ai-vs-human-generated-images/test.csv')

root = '/kaggle/input/ai-vs-human-generated-dataset/'


train.head()


train = train.drop(columns=['Unnamed: 0'])

train.head()


test.head()


train_df, val_df = train_test_split(train, test_size=0.2, stratify=train['label'])



# Set device
if torch.cuda.is_available():
    device='cuda'
elif torch.backends.xpu.is_available():
    device = 'xpu'
else:
    device = 'cpu'

device = torch.device(device)

print(f'Using device: {device}')


class ImageDataset(Dataset):
    def __init__(self, root, df, transform=None, is_train=True):
        self.root = root
        self.df = df
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img_path = os.path.join(self.root, img_path)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_train:
            label = self.df.iloc[idx, 1]            
            return image, label
        else:
            return image



import torchvision.transforms as transforms

# Training augmentations
train_transforms = transforms.Compose([
    transforms.Resize(232),  # Resize to match ConvNeXt preprocessing
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation and Test transforms
val_test_transforms = transforms.Compose([
    transforms.Resize(232),  # Resize to 232 as per ConvNeXt documentation
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load pretrained ConvNeXt Base model
model = models.convnext_base(weights="DEFAULT")

# Freeze all layers initially
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze the last two stages 
for param in model.features[-2:].parameters(): 
    param.requires_grad = True

# Replace the classifier head with a custom one
model.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
    nn.Flatten(),                  # Flatten the tensor
    nn.BatchNorm1d(1024),          # Add BatchNorm here
    nn.Linear(1024, 512),          # First fully connected layer
    nn.ReLU(),                     # Activation function
    nn.Dropout(0.4),               # Dropout for regularization
    nn.Linear(512, 2)              # Output layer (binary classification)
)

model = model.to(device)

# Define loss function, optimizer, and learning rate scheduler
optimizer = torch.optim.AdamW([
    {'params': model.features[-2:].parameters(), 'lr': 1e-5},  # Lower LR for backbone
    {'params': model.classifier.parameters(), 'lr': 1e-4}      # Higher LR for classifier
])

criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=5, gamma=0.7)


# !pip install torchviz -q
# from torchviz import make_dot

# # Get a mini-batch
# dummy_image, _ = next(iter(train_loader))
# dummy_image = dummy_image.to(device)

# # Forward pass
# output = model(dummy_image)

# # Generate visualization
# make_dot(output, params=dict(model.named_parameters())).render("model_architecture", format="png")



BATCH_SIZE = 64
NUM_WORKERS = 4

train_dataset = ImageDataset(root, train_df, transform=train_transforms)
val_dataset = ImageDataset(root, val_df, transform=train_transforms)
test_dataset = ImageDataset(root, test, transform=val_test_transforms, is_train=False)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True) 
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)



def train(model, optimizer, loss_fn, dataloader, scheduler=None):
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    model.train()

    epoch_loss = 0.0
    epoch_accuracy = 0.0

    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = outputs.argmax(dim=1)
        acc = (preds == y).float().mean().item()
        epoch_accuracy += acc

    epoch_loss /= num_batches
    epoch_accuracy /= num_batches

    if scheduler:
        scheduler.step()

    return epoch_loss, epoch_accuracy

def validate(model, loss_fn, dataloader):
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    model.eval()

    val_loss = 0.0
    val_acc = 0.0
    val_pred_classes = []
    val_y = []
    
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            
            loss = loss_fn(outputs, y)
            val_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            acc = (preds == y).float().mean().item()
            val_acc += acc
            
            val_pred_classes.extend(preds.cpu().numpy())
            val_y.extend(y.cpu().numpy())

    val_loss /= num_batches
    val_acc /= num_batches
    val_f1 = f1_score(np.array(val_y), np.array(val_pred_classes), average='binary')

    return val_loss, val_acc, val_f1


n_epochs = 5
train_loss, train_acc = [], []
val_loss, val_acc, val_f1 = [], [], []

for epoch in range(n_epochs):
    loss, acc = train(model, optimizer, criterion, train_loader)
    train_loss.append(loss)
    train_acc.append(acc)
    
    loss, acc, f1 = validate(model, criterion, val_loader)
    val_loss.append(loss)
    val_acc.append(acc)
    val_f1.append(f1)
    
    print(f'Epoch {epoch+1}:')
    print(f'\t train_loss: {train_loss[epoch]}, train_acc: {train_acc[epoch]}')
    print(f'\t val_loss: {val_loss[epoch]}, val_acc: {val_acc[epoch]}, val_f1: {val_f1[epoch]}\n')


predicted_classes = []
with torch.no_grad():
    model.eval()
    for X in test_loader:
        X = X.to(device)
        outputs = model(X)
        preds = outputs.argmax(dim=1)
        predicted_classes.extend(preds.cpu().numpy())

ss = pd.DataFrame({
    'id': test['id'],
    'label': predicted_classes
})

ss.to_csv('submission.csv')

