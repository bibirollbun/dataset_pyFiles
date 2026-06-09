# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
1


!pip -q install py7zr
import py7zr
import glob
import shutil

# Find archive
candidates = glob.glob('/kaggle/input/**/test.7z', recursive=True)
archive_path = candidates[0]

shutil.rmtree('test', ignore_errors=True)
os.makedirs('test', exist_ok=True)

#extracting all these to /test folder
with py7zr.SevenZipFile(archive_path, mode='r') as z:
    z.extractall(path='test')



#repeat for train set to path /train
candidates = glob.glob('/kaggle/input/**/train.7z', recursive=True)
archive_path = candidates[0]

shutil.rmtree('train', ignore_errors=True)
os.makedirs('train', exist_ok=True)
with py7zr.SevenZipFile(archive_path, mode='r') as z:
    z.extractall(path='train')




import os
import torch

from PIL import Image # For images
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms # For data transformations

#defining tramsforms
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_transforms = transforms.Compose([
    # Pad the image, then take a random 32x32 crop
    transforms.RandomCrop(32, padding=4), 
    # Randomly flip the image horizontally (common for objects like cars, birds)
    transforms.RandomHorizontalFlip(),
    # Slightly change brightness, contrast, saturation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    # Convert image to a PyTorch tensor
    transforms.ToTensor(),
    # Normalize with pre-calculated CIFAR-10 specific values
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# 2. Transforms for the VALIDATION dataset (no augmentation, just normalization)
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

batch_size = 4   

#had to create a custom dataset to handle this data

import os
from PIL import Image
from torch.utils.data import Dataset


labels_df=pd.read_csv("../input/cifar-10/trainLabels.csv", index_col="id")


#for train set into dataset obj
class myDataset(Dataset):
    def __init__(self, labels_df=None, image_dir="train/train", transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.labels_df = labels_df
        if self.labels_df is not None:
            self.label_map = {label: idx for idx, label in enumerate(sorted(labels_df['label'].unique()))}

    def __len__(self):
        return len(self.labels_df) if self.labels_df is not None else len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        filename = f"{idx + 1}.png"
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.labels_df is not None:
            label = self.labels_df.iloc[idx]['label']
            label = torch.tensor(self.label_map[label], dtype=torch.long)
            return image, label
        else:
            return image


train_dataset = myDataset(labels_df=labels_df,transform=train_transforms)
test_dataset = myDataset(image_dir="test/test",transform=transform)



import copy
from torch.utils.data import random_split, DataLoader, Subset

# 1. Load the full dataset (use train transforms as a starting point)
full_dataset = myDataset(labels_df=labels_df, image_dir="train/train", transform=train_transforms)

# 2. Perform the split
total_size = len(full_dataset)
val_size = int(total_size * 0.1) 
train_size = total_size - val_size
lengths = [train_size, val_size]

# This creates two Subset objects
train_subset, val_subset_indices = random_split(
    full_dataset, 
    lengths, 
    generator=torch.Generator().manual_seed(42)
)

# 3. Create a *separate copy* of the dataset for validation and apply the validation transforms
# This is required because both Subset objects initially reference the SAME full_dataset object.
val_dataset_final = copy.deepcopy(full_dataset)
val_dataset_final.transform = val_transforms

# Recreate the validation subset using the *newly copied* and transformed dataset instance
val_subset = Subset(val_dataset_final, val_subset_indices.indices)

# Note: The 'train_subset' still uses the original 'full_dataset' with 'train_transforms'

# 4. Create DataLoaders (using BATCH_SIZE=32 is recommended for modern CNNs with BatchNorm)
BATCH_SIZE = 4

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)




import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def show_image_from_dataset(dataset, idx):
    image, label = dataset[idx]

    # If it's a tensor, convert back to PIL for display
    if isinstance(image, torch.Tensor):
        image = TF.to_pil_image(image)

    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()



#for i in range(50,55):  # Check first 5
#    show_image_from_dataset(train_dataset, i)




image, label = train_dataset[0]
print("Label:", label)
print("Image size:", image.size)  # If it's PIL


#!ls test/test/




import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def view_image(image_path):
    """
    Loads and displays an image in a Kaggle notebook.

    Args:
        image_path (str): The file path to the image.
    """
    try:
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.axis('off')  
        plt.show()
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
    except Exception as e:
        print(f"An error occurred while displaying the image: {e}")
        



import os

image_dir = 'train/train'
files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
for i, f in enumerate(files[:10]):
    print(i, f)



#val_curve = [10,15,16,30,40]

#val_x = np.linspace(1, len(train_losses), len(val_curve))



for x in range(100,106):
    print(labels_df.iloc[x-1])
    view_image(f"train/train/{x}.png")



#the normalization happened as transforms defined as a parameter during downloading the dataset




# Grab one batch from your train loader
dataiter = iter(train_loader)
images, labels = next(dataiter)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)



from torch import nn

#myseqmodel = nn.Sequential(
#    nn.Conv2d(3,32,3),
#    nn.ReLU(),
#    nn.MaxPool2d(2,2),
#    nn.Flatten(),
#    nn.Linear(32*15*15, 128),
#    nn.ReLU(),
#    nn.Linear(128,10),
    #nn.Softmax(dim=1)
#)

myseqmodel = nn.Sequential(
     # Stage 1
    nn.Conv2d(3, 32, 3, padding=1),
    nn.BatchNorm2d(32), # Added Batch Normalization
    nn.ReLU(),
    nn.MaxPool2d(2, 2), # Output size: 16x16

    # Stage 2
    nn.Conv2d(32, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2), # Output size: 8x8

    # Stage 3
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2), # Output size: 4x4
    
    # Stage 4 (Adding another stage for more depth, within sequential limits)
    nn.Conv2d(128, 256, 3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(2, 2), # Output size: 2x2. We must adjust the linear layer input size below.

    nn.Flatten(),
  
    nn.Linear(256 * 2 * 2, 512), 
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)



#defining the above parameters
import torch
from torch import optim
epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(myseqmodel.parameters(), lr=0.005)



train_losses, val_losses = [], []
train_accs, val_accs = [], []
val_aucs = []


device= torch.device("cuda")
myseqmodel.to(device)






#training loop



for epoch in range(epochs):

    myseqmodel.train()
    
    running_loss, correct,total= 0.0,0,0
    val_running_loss, val_correct, val_total = 0.0, 0, 0 
    for  inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device),labels.to(device)

        #zero gradients
        optimizer.zero_grad()

        #forward pass
        outputs = myseqmodel(inputs)

        #compute loss
        loss = criterion(outputs, labels)

        #backward + optimize
        loss.backward()

        #update weights
        optimizer.step()
        
        running_loss += loss.item()

        _,predicted = outputs.max(1) #highest probability class
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()


    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    #validation
    myseqmodel.eval()

    with torch.no_grad():
        for inputs,labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = myseqmodel(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _,predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct+= predicted.eq(labels).sum().item()

    epoch_val_loss= val_running_loss / len(val_loader)
    epoch_val_acc = 100* val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Tr Loss: {train_loss:.4f} - Tr Acc: {train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")


        
print("training finished")
        



#plt.plot(train_accs, label='Train Accuracy')

#plt.plot(val_x, val_curve, label='Validation accuracy', color='red')

#plt.legend(); plt.show()



"""
myseqmodel.eval()   #evaluation mode
running_loss, correct, total, val_loss = 0,0,0,0
all_probs, all_labels=[],[]


with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = myseqmodel(inputs)
        _, predicted = torch.max(outputs, 1)
        
        
        val_loss +=loss.item()
        _,predicted = outputs.max(1)
        total+=labels.size(0)
        correct+= predicted.eq(labels).sum().item()

    val_loss = running_loss / len(test_loader)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    """


print(predicted.shape, labels.shape)



batch = next(iter(test_loader))
print(type(batch))
print(len(batch))





# Inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myseqmodel = myseqmodel.to(device)  # 
myseqmodel.eval()

predictions = []
image_ids = list(range(1, len(predictions) + 1))


with torch.no_grad():
    for images in test_loader:  # only one output
        images = images.to(device)
        outputs = myseqmodel(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

# map back to labels
idx_to_label = {v: k for k, v in train_dataset.label_map.items()}




# Build sequential IDs to match predictions
image_ids = list(range(1, len(predictions) + 1))

# Map prediction indices to class labels
labels = [idx_to_label[int(p)] for p in predictions]

# Create submission DataFrame
submission = pd.DataFrame({
    "id": image_ids,
    "label": labels
})

# Save to CSV
submission.to_csv("submission.csv", index=False)

submission


submission.label.value_counts()


print("Predictions:", len(predictions))
print("Image IDs:", len(image_ids))



import matplotlib.pyplot as plt
import numpy as np 

# --- Start of the corrected plotting script ---

# DYNAMICALLY determine how many epochs actually completed based on the shortest list
actual_epochs_completed = min(len(train_losses), len(val_losses), len(train_accs), len(val_accs))

epochs_range = range(1, actual_epochs_completed + 1)

# Ensure all lists match this length exactly before plotting
train_losses = train_losses[:actual_epochs_completed]
val_losses = val_losses[:actual_epochs_completed]
train_accs = train_accs[:actual_epochs_completed]
val_accs = val_accs[:actual_epochs_completed]


# Plot Loss (this will now work correctly)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy (this will also work correctly now)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label='Training Accuracy')
plt.plot(epochs_range, val_accs, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()




torch.save(myseqmodel.state_dict(), "seqmodelv1.pth")
"model saved"





# MNIST Feedforward Network in PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# 1. Data loading & normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. Model definition
class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

model = FeedForwardNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 5

train_losses, test_losses, train_accs, test_accs = [], [], [], []

# 3. Training loop
for epoch in range(epochs):
    model.train()
    correct, total, train_loss = 0, 0, 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_acc = 100 * correct / total
    train_losses.append(train_loss / len(train_loader))
    train_accs.append(train_acc)

    # Evaluation
    model.eval()
    correct, total, test_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    test_losses.append(test_loss / len(test_loader))
    test_accs.append(test_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {train_losses[-1]:.4f}, Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_losses[-1]:.4f}, Acc: {test_acc:.2f}%")

# 4. Final accuracy
print(f"\nFinal Test Accuracy: {test_accs[-1]:.2f}%")

# 5. Suggestions for improvement
print("""
Model Improvement Suggestions:
- Add more hidden layers or increase neurons.
- Try Adam optimizer for adaptive learning.
- Add Dropout or BatchNorm to prevent overfitting.
- Use more epochs or learning rate scheduling.
- Consider CNNs for spatial pattern recognition.
""")

# 6. Visualization
# Loss and Accuracy
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Curve'); plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.title('Accuracy Curve'); plt.legend()
plt.show()

# Confusion Matrix
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Show sample predictions
fig, axes = plt.subplots(2,5, figsize=(10,4))
for i, ax in enumerate(axes.flat):
    img, label = test_dataset[i]
    with torch.no_grad():
        pred = model(img.unsqueeze(0)).argmax(dim=1).item()
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"T:{label}, P:{pred}")
    ax.axis('off')
plt.show()


