import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchmetrics.classification import BinaryAccuracy
from torchinfo import summary

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import random
import zipfile


# setup the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
!nvidia-smi


# set seed for reproduceability
def manual_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.Generator().manual_seed(seed)

manual_seed(seed=23)


# location of train.zip
path_to_zip_file = "/kaggle/input/dogs-vs-cats/train.zip"

# target directory
directory_to_extract_to = "/kaggle/working/"

# running the following code will extract the "train.zip" into "/kaggle/working/train"
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)


class CatsAndDogsDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)   
    
    def __getitem__(self, index):
        
        # filename and filepath
        filename = self.image_files[index]
        filepath = os.path.join(self.root_dir, filename)

        # load image
        image = Image.open(filepath).convert("RGB")

        # Label: 0 for cat, 1 for dog
        label = 0 if "cat" in filename.lower() else 1

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label


# Data directory
root_dir = "/kaggle/working/train"

# Pretrained model parameters and transformation
weights = models.EfficientNet_B0_Weights.DEFAULT
data_transfoms = weights.transforms()
print(data_transfoms)

# train dataset
full_dataset = CatsAndDogsDataset(root_dir=root_dir, transform=data_transfoms)


# Extract targets (class labels)
targets = np.array([sample[1] for sample in full_dataset])


# Define stratified split
splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.3,
    random_state=23
)


# Get train/val indices
for train_idx, test_idx in splitter.split(np.zeros(len(targets)), targets):
    full_train_data = Subset(
        full_dataset,
        train_idx
    )
    test_data = Subset(
        full_dataset,
        test_idx
    )
    train_targets = targets[train_idx]


# Subsample full_train_data: 90%, 80%, 70%, 60%, 50%

def get_sumsampled_train(full_train_data, train_targets, ratio, seed=23):
    from sklearn.utils import shuffle

    np.random.seed(seed)
    n = int(len(full_train_data) * ratio)
    indices = np.arange(len(full_train_data))

    # stratified shuffle
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=n,
        random_state=seed
    )
    for sub_idx, _ in splitter.split(np.zeros(len(indices)), train_targets):
        return Subset(full_train_data, sub_idx)


train_set_100 = full_train_data
train_set_90 = get_sumsampled_train(full_train_data, train_targets, 0.1)
train_set_80 = get_sumsampled_train(full_train_data, train_targets, 0.2)
train_set_70 = get_sumsampled_train(full_train_data, train_targets, 0.3)
train_set_60 = get_sumsampled_train(full_train_data, train_targets, 0.4)
train_set_50 = get_sumsampled_train(full_train_data, train_targets, 0.5)


def count_class(full_dataset, train_set, test_set):


    targets_full = [sample[1] for sample in full_dataset]
    class_counts_full = Counter(targets_full)

    targets_train = [sample[1] for sample in train_set]
    class_counts_train = Counter(targets_train)

    targets_test = [sample[1] for sample in test_set]
    class_counts_test = Counter(targets_test)

    print(f"full train set: {class_counts_full}")
    print(f"Subset train set: {class_counts_train}")
    print(f"Validation set: {class_counts_test}")

    class_names = ["Cat", "Dog"]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    
    p = axs[0].bar(class_names, [class_counts_full[0], class_counts_full[1]], color=["black", "gray"], label=(260,260))
    axs[0].set_title("Full Dataset")
    axs[0].bar_label(p)
    
    p = axs[1].bar(class_names, [class_counts_train[0], class_counts_train[1]], color=["black", "gray"])
    axs[1].set_title("Train Dataset")
    axs[1].bar_label(p)
    
    p = axs[2].bar(class_names, [class_counts_test[0], class_counts_test[1]], color=["black", "gray"])
    axs[2].set_title("Validation Dataset")
    axs[2].bar_label(p)
    
    plt.show()


count_class(full_dataset, train_set_100, test_data)


count_class(full_dataset, train_set_90, test_data)


count_class(full_train_data, train_set_80, test_data)


# Create DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Visualize transformed image
img_batch, label_batch = next(iter(train_loader))

def show_random_image(img_batch, label_batch):
    random_index = random.randint(0,31)
    img_batch_preview = img_batch[random_index,:,:,:].permute(1,2,0)
    label_batch_preview = label_batch[random_index]
    label = "cat" if label_batch_preview == 0 else "dog"
    plt.imshow(img_batch_preview)
    plt.title(f"label:{label}")
    plt.axis("off")
    plt.show()

show_random_image(img_batch, label_batch)


# download Resnet18 CNN Model
model = models.efficientnet_b0(weights=weights).to(device)


# preview the Resnet18 Architecture: input and ouput of each layers
summary(
    model=model,
    input_size=(32,3,224,224),
    col_names=["input_size","output_size","num_params","trainable"],
    col_width=20,
    row_settings = ['var_names']
)


# first, freeze the model parameter in "features" layer
for param in model.features.parameters():
    param.requires_grad = False

# determine the output shape based on the total number of class
output_shape = 1

# redifine the "classifier" layer
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280,
             out_features=output_shape, # we only change this part
             bias=True)
).to(device)

# preview the model architecture after we modify the "classifier" layer
summary(
    model = model,
    input_size = (32, 3, 224, 224),
    col_names = ["input_size", "output_size", "num_params", "trainable"],
    col_width = 20,
    row_settings = ["var_names"]
)


# Loss, Optimizer and Accuracy
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_acc_metric = BinaryAccuracy().to(device)

val_acc_metric = BinaryAccuracy().to(device)


# Training

# tracking loss and accuracy
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# training loop

num_epochs = 50
for epoch in range(num_epochs):

    #------------TRAINING---------------
    
    model.train()

    train_loss = 0.0
    train_acc_metric.reset()

    for images, labels in train_loader:
        
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_acc_metric.update(torch.sigmoid(outputs), labels)
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = train_acc_metric.compute().item()

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    #----------Validation-----------------
    
    model.eval()

    val_loss = 0.0

    val_acc_metric.reset()

    with torch.inference_mode():
        
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_acc_metric.update(torch.sigmoid(outputs), labels)

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = val_acc_metric.compute().item()

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")


import pandas as pd

result = pd.DataFrame({"train_loss":train_losses,
                       "train_acc":train_accuracies,
                       "val_loss":val_losses,
                       "val_acc":val_accuracies})


result.head(5)


# save to Kaggle working directory
result.to_csv("/kaggle/working/result_20250605_efficientnet.csv")


# ----- Plotting -----
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
plt.plot(epochs, val_losses, 'ro-', label='Val Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'bo-', label='Train Acc')
plt.plot(epochs, val_accuracies, 'ro-', label='Val Acc')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

