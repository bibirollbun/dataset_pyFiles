import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
%matplotlib inline


train_labels = pd.read_csv('histo_data/train_labels.csv')
print(train_labels.head())


print('Train shape:')
print(train_labels.shape)


len(train_labels['id'].unique())


# Now let's explore the test folder
from pathlib import Path

folder_path = Path("histo_data/test")

test_ids = []

for f in folder_path.iterdir():
    if f.is_file():
        test_ids.append(f.name)


print('Test shape:')
print(len(test_ids))
print('Test unique ids (Check for duplicates)')
print(len(np.unique(test_ids)))


sample_submission = pd.read_csv('histo_data/sample_submission.csv')
sample_submission.head()


print('Sample submission shape:')
print(sample_submission.shape)


test_ids[0:10]


image_path = f'histo_data/test/{test_ids[0]}'
img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()


print('Image shape:')
np.array(img).shape


train_labels['label'].value_counts() / len(train_labels)


import seaborn as sns


plt.figure(figsize=(5,3))
sns.countplot(data=train_labels, x='label', palette="Set2")

plt.xlabel("Cancer")
plt.ylabel("Frequency")
plt.title("Distriuction of the target variable (Cancer)")
plt.show()


cancer_sample_images = train_labels[train_labels['label'] == 1].sample(9)['id'].values
non_cancer_sample_images = train_labels[train_labels['label'] == 0].sample(9)['id'].values


plt.figure(figsize=(8, 8))

for i, img_name in enumerate(cancer_sample_images):
    img_path = f'histo_data/train/{img_name}.tif'
    img = Image.open(img_path)
    
    plt.subplot(3, 3, i+1)  # 3x3 grid
    plt.imshow(img)
    plt.axis('off')

plt.suptitle('Sample of Cancer images', fontsize=16)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 8))

for i, img_name in enumerate(non_cancer_sample_images):
    img_path = f'histo_data/train/{img_name}.tif'
    img = Image.open(img_path)
    
    plt.subplot(3, 3, i+1)  # 3x3 grid
    plt.imshow(img)
    plt.axis('off')

plt.suptitle('Sample of Benign images')
plt.tight_layout()
plt.show()


train_labels['label'].value_counts()


# Firts, let's sample the data to reduce it and balancing it.
sample_size = 80000
random_state = 102
train_labels_1 = train_labels[train_labels['label'] == 1].sample(
    int(sample_size/2), random_state=random_state)
train_labels_0 = train_labels[train_labels['label'] == 0].sample(
    int(sample_size/2), random_state=random_state)


train_labels_sample = pd.concat([train_labels_1, train_labels_0]).reset_index(drop=True)


train_labels_sample['label'].value_counts()


from torch.utils.data import Dataset, DataLoader, random_split


class TrainData(Dataset):
    def __init__(self, train_labels_sample, img_dir, transform=None):
        '''
        train_labels_sample: The training samples id's and their label.
        img_dir: the path in which the images are located.
        transform: The transformation to be applied to the images. (Resize, augmentation, normalization)
        '''
        self.data = train_labels_sample
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        ''' 
        This is the typical template in Pytorch, 
        it needs a __len__ method and __getitem__, to obtain the input and label
        based on index.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0] + ".tif"
        label = int(self.data.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label



# 64x64
train_transforms = transforms.Compose([
    transforms.Resize((96,96)), # Resize the image to ensure that all have 96x96 shape
    transforms.RandomHorizontalFlip(), # Performs horizonal flip in image randomly
    transforms.RandomVerticalFlip(), # Performs vertical flip in image randomly
    transforms.RandomRotation(20), # Rotates the image randomly between (-20, 20) degrees
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2, 
        saturation=0.2, hue=0.05), # transform randomly changes the brightness, contrast, saturation, hue
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# This is for validation/testing data, as it doesn't require to do the part 
# of data augmentation
val_transforms = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])


dataset = TrainData(train_labels_sample, img_dir="histo_data/train", transform=train_transforms)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# For the validation data only use the parte that normalizes and resize the images, not the augmentation
val_dataset.dataset.transform = val_transforms


print('Train data size:')
print(len(train_dataset))
print('Validation data size:')
print(len(val_dataset))



# This is one of the hyperparameters, it controls the batch size in which the data will be partitioned
batch_size = 32
# Now, let's pass each dataset to a DataLoader Pytorch class, this is in charge
# of suffling, creating the batch, and can do many other things.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



images, labels = next(iter(train_loader))
print(f"Train batch images: {images.shape}, labels: {labels.shape}")

images, labels = next(iter(val_loader))
print(f"Validation batch images: {images.shape}, labels: {labels.shape}")



import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # 32 is the number of filters, and kernel size the filter size
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolution 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolution 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected
        self.fc1 = nn.Linear(128 * 12 * 12, 256)  # 96x96 -> 3 poolings => 12x12
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)  # 1 for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 96x96 -> 48x48
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 48x48 -> 24x24
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 24x24 -> 12x12
        x = x.view(-1, 128 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Whithout sigmoid becase we'll use BCEWithLogitsLoss



class CNNAdvanced(nn.Module):
    def __init__(self):
        super(CNNAdvanced, self).__init__()

        # First block
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )

        # Second block
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )

        # Third block
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )

        # Fully connected layer
        # height and weight size after 3 pools: 12x12
        # that's because we have 3 blocks, and each block has a MaxPool of (2,2): 96 -> 48 -> 24 -> 12
        self.flatten_dim = 128 * 12 * 12
        # And note that 128 is the depth size of the 3 block
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # 1 for binary classification
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(-1, self.flatten_dim)  # flatten
        x = self.fc(x)
        return x # Whithout sigmoid becase we'll use BCEWithLogitsLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simple_model = SimpleCNN().to(device)
print('Simple model architecture:')
print(simple_model)
print('*'*100)




print('Advanced model architecture:')
advanced_model = CNNAdvanced().to(device)
print(advanced_model)


import torch.optim as optim
# For binarry classifier, and as I mentioned earlier, this function applies sigmoid while evaluationg the Loss.
loss_function_simple_model = nn.BCEWithLogitsLoss()
loss_function_advanced_model= nn.BCEWithLogitsLoss() 


# I'll use the Adam optimizer because is good optimization method.
optimizer_simple_model = optim.Adam(simple_model.parameters(), lr=0.0001)
optimizer_advanced_model = optim.Adam(advanced_model.parameters(), lr=0.0001)


scheduler_simple_model = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_simple_model, mode='max', factor=0.5, patience=2, min_lr=1e-5
)

scheduler_advanced_model = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_advanced_model, mode='max', factor=0.5, patience=2, min_lr=1e-5
)


from sklearn.metrics import roc_auc_score

def training_model(num_epochs, optimizer, model, loss_function, scheduler, model_save_path):
    # This is the value to store the best validation AUC and save the best model
    best_val_auc = -1.0
    # Iterate all the data in each epoch
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # In this part, it iterates over each batch
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)

            # Set gradients to zero, because Pytorch acumulates the previous ones.
            optimizer.zero_grad()
            outputs = model(images)
            # calculate the loss
            loss = loss_function(outputs, labels)
            # Calculate the gradients respect to the loss
            loss.backward()
            # Finally update the parameters based on the gradients calculated with loss.backward()
            optimizer.step()

            # Calculate the loss
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        # here I'll save the labels and prediction for each batch in the validation set
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = probs >= 0.5
                correct += (preds.float() == labels).sum().item()
                
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        # Concat for all the batches
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
        val_auc = roc_auc_score(all_labels, all_probs)

        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # Save the best model parameters
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved with validation AUC: {best_val_auc:.4f}")

        # Update the learning rate with the scheduler
        scheduler.step(val_auc) # Based on AUC, maximize the AUC




training_model(
    num_epochs=15, optimizer=optimizer_simple_model,
    model=simple_model, loss_function=loss_function_simple_model,
    scheduler=scheduler_simple_model,
    model_save_path='models/best_simple_cnn_model.pth')


simple_model = SimpleCNN().to(device)
simple_model.load_state_dict(torch.load("models/best_simple_cnn_model.pth"))


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def calculate_probability_of_cancer(model, data_loader):
    '''
    Model: The model to calculate the probability
    data_loader: The data batches, train or validation
    '''
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            # This is necessesary because the model outputs the raw after 
            # the conected layer, no the probability
            probs = torch.sigmoid(outputs)
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    return all_labels, all_probs


train_labels, train_probs = calculate_probability_of_cancer(
    simple_model, train_loader)
val_labels, val_probs = calculate_probability_of_cancer(
    simple_model, val_loader)


train_fpr, train_tpr, _ = roc_curve(train_labels, train_probs)
val_fpr, val_tpr, _ = roc_curve(val_labels, val_probs)

train_auc = auc(train_fpr, train_tpr)
val_auc = auc(val_fpr, val_tpr)

# Plot the AUC
plt.figure(figsize=(8,6))
plt.plot(train_fpr, train_tpr, label=f'Train AUC = {train_auc:.4f})')
plt.plot(val_fpr, val_tpr, label=f'Validation AUC = {val_auc:.4f})')
plt.plot([0,1], [0,1], 'k--')  # diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Simple model ROC Curves')
plt.legend()
plt.grid(True)
plt.show()


training_model(
    num_epochs=15, optimizer=optimizer_advanced_model,
    model=advanced_model, loss_function=loss_function_advanced_model,
    scheduler=scheduler_advanced_model,
    model_save_path='models/best_advanced_cnn_model.pth')


advanced_model = CNNAdvanced().to(device)
advanced_model.load_state_dict(torch.load("models/best_advanced_cnn_model.pth"))


train_labels, train_probs = calculate_probability_of_cancer(
    advanced_model, train_loader)
val_labels, val_probs = calculate_probability_of_cancer(
    advanced_model, val_loader)


train_fpr, train_tpr, _ = roc_curve(train_labels, train_probs)
val_fpr, val_tpr, _ = roc_curve(val_labels, val_probs)

train_auc = auc(train_fpr, train_tpr)
val_auc = auc(val_fpr, val_tpr)

# Plot the AUC
plt.figure(figsize=(8,6))
plt.plot(train_fpr, train_tpr, label=f'Tran AUC = {train_auc:.4f})')
plt.plot(val_fpr, val_tpr, label=f'Validation AUC = {val_auc:.4f})')
plt.plot([0,1], [0,1], 'k--')  # diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Advanced model ROC Curves')
plt.legend()
plt.grid(True)
plt.show()


import optuna


def objective(trial):

    # Hyperparameter space
    first_filters = trial.suggest_categorical('first_filters', [16, 32, 64])
    second_filters = trial.suggest_categorical('second_filters', [32, 64, 128])
    third_filters = trial.suggest_categorical('third_filters', [64, 128, 256])
    fc1_units = trial.suggest_categorical('fc1_units', [128, 256, 512])
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.6)
    dropout_rate_fc_layer = trial.suggest_float('dropout_rate_fc_layer', 0.3, 0.6)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    # Model definition based on hyperparameters
    class TunedCNN(nn.Module):
        def __init__(self):
            super(TunedCNN, self).__init__()

            # First block
            self.block1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=first_filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(first_filters, first_filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(first_filters, first_filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout_rate)
            )

            # Second block
            self.block2 = nn.Sequential(
                nn.Conv2d(first_filters, second_filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(second_filters, second_filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(second_filters, second_filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout_rate)
            )

            # Third block
            self.block3 = nn.Sequential(
                nn.Conv2d(second_filters, third_filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(third_filters, third_filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(third_filters, third_filters, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout_rate)
            )

            # Fully connected layer
            # height and weight size after 3 pools: 12x12
            # that's because we have 3 blocks, and each block has a MaxPool of (2,2): 96 -> 48 -> 24 -> 12
            self.flatten_dim = third_filters * 12 * 12
            self.fc = nn.Sequential(
                nn.Linear(self.flatten_dim, fc1_units),
                nn.ReLU(),
                nn.Dropout(dropout_rate_fc_layer),
                nn.Linear(fc1_units, 1)  # 1 for binary classification
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = x.view(-1, self.flatten_dim)  # flatten
            x = self.fc(x)
            return x # Whithout sigmoid becase we'll use BCEWithLogitsLoss
    
    # Define the model, loss and optimizer
    model = TunedCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training the model with only 3 epochs
    for epoch in range(3):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validation set
    # Here we calculate the AUC in validation after the 3 epochs
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    val_auc = roc_auc_score(all_labels, all_probs)

    return val_auc  # Maximize AUC



# Use optuna for hyperparameter tuning
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)


print("Best hyperparameters:", study.best_params)
print("Best AUC:", study.best_value)


class TunedCNN(nn.Module):
    def __init__(self):
        super(TunedCNN, self).__init__()

        # First block
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5106408629195538)
        )

        # Second block
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5106408629195538)
        )

        # Third block
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5106408629195538)
        )

        # Fully connected layer
        # height and weight size after 3 pools: 12x12
        # that's because we have 3 blocks, and each block has a MaxPool of (2,2): 96 -> 48 -> 24 -> 12
        self.flatten_dim = 64 * 12 * 12
        # And note that 128 is the depth size of the 3 block
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4776451070371182),
            nn.Linear(256, 1)  # 1 for binary classification
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(-1, self.flatten_dim)  # flatten
        x = self.fc(x)
        return x # Whithout sigmoid becase we'll use BCEWithLogitsLoss
# Define the model, loss and optimizer
model_tunned = TunedCNN().to(device)
optimizer = optim.Adam(model_tunned.parameters(), lr=0.00034503462288880974)
loss_fn = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5
)



training_model(
    num_epochs=15, optimizer=optimizer,
    model=model_tunned, loss_function=loss_fn,
    scheduler=scheduler,
    model_save_path='models/best_tuned_cnn_model.pth')


model_tunned = TunedCNN().to(device)
model_tunned.load_state_dict(torch.load("models/best_tuned_cnn_model.pth"))


train_labels, train_probs = calculate_probability_of_cancer(
    model_tunned, train_loader)
val_labels, val_probs = calculate_probability_of_cancer(
    model_tunned, val_loader)


train_fpr, train_tpr, _ = roc_curve(train_labels, train_probs)
val_fpr, val_tpr, _ = roc_curve(val_labels, val_probs)

train_auc = auc(train_fpr, train_tpr)
val_auc = auc(val_fpr, val_tpr)

# Plot the AUC
plt.figure(figsize=(8,6))
plt.plot(train_fpr, train_tpr, label=f'Train AUC = {train_auc:.4f})')
plt.plot(val_fpr, val_tpr, label=f'Validation AUC = {val_auc:.4f})')
plt.plot([0,1], [0,1], 'k--')  # diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Tuned model ROC Curves')
plt.legend()
plt.grid(True)
plt.show()


from torch.utils.data import Dataset



# we need to create a custom class with the same template of Training data, but this is simpler.
class TestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = os.listdir(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_name

# The tranfsomation for the test, is equal to the validation set.
test_transforms = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

test_dataset = TestDataset("histo_data/test", transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



def obtain_test_predictions(model):
    model.eval()
    predictions = []

    with torch.no_grad():
        # Iterate over the batches
        for images, file_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()  # Apply the sigmoid for convert to probability

            # save each probability sample in the batch
            for fname, prob in zip(file_names, probs):
                predictions.append((fname, prob[0]))
    return predictions


predictions_simple_model = obtain_test_predictions(simple_model)
submission_simple_model = pd.DataFrame(predictions_simple_model, columns=['id', 'label'])

submission_simple_model['id'] = submission_simple_model['id'].str.replace('.tif', '')
submission_simple_model.to_csv("submission_simple_model.csv", index=False)



predictions_advanced_model = obtain_test_predictions(advanced_model)
submission_advanced_model = pd.DataFrame(predictions_advanced_model, columns=['id', 'label'])

submission_advanced_model['id'] = submission_advanced_model['id'].str.replace('.tif', '')
submission_advanced_model.to_csv("submission_advanced_model.csv", index=False)



predictions_tunned_model = obtain_test_predictions(model_tunned)
submission_tunned_model = pd.DataFrame(predictions_tunned_model, columns=['id', 'label'])

submission_tunned_model['id'] = submission_tunned_model['id'].str.replace('.tif', '')
submission_tunned_model.to_csv("submission_tunned_model.csv", index=False)

