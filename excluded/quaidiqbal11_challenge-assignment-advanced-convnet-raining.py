# Install required package
!pip install py7zr

# -----------------------------
# Required Imports
# -----------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import py7zr
import math

# -----------------------------
# Device Configuration
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Model Definition: ResNet152
# -----------------------------

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Stem layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Final classifier layer
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

# -----------------------------
# Instantiate the Model
# -----------------------------

model = ResNet152().to(device)
print("Model instantiated successfully.")

# -----------------------------
# Data Augmentation: CutOut
# -----------------------------

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w, dtype=img.dtype, device=img.device)

        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()

        y1 = max(0, y - self.length // 2)
        y2 = min(h, y + self.length // 2)
        x1 = max(0, x - self.length // 2)
        x2 = min(w, x + self.length // 2)

        mask[y1:y2, x1:x2] = 0
        img = img * mask.unsqueeze(0)  # Apply mask to all channels
        return img

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------

# CIFAR-10 mean and std for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Training transformations with CutOut
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    Cutout(length=16),
])

# Validation/Test transformations
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Load CIFAR-10 dataset
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# -----------------------------
# Define Loss Function and Optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss()

# AdamW optimizer with weight decay
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)

# -----------------------------
# Cosine Annealing Scheduler with Warmup
# -----------------------------

class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.001, T_up=5, gamma=0.5, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.gamma = gamma
        self.cycle = 0
        self.T_i = T_0
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_up:
            # Warmup phase
            return [(self.base_lrs[i] + (self.eta_max - self.base_lrs[i]) * self.last_epoch / self.T_up) 
                    for i in range(len(self.base_lrs))]
        else:
            # Cosine annealing phase
            cos_inner = (self.last_epoch - self.T_up) / (self.T_i - self.T_up)
            return [self.base_lrs[i] + 0.5 * (self.eta_max - self.base_lrs[i]) * 
                    (1 + math.cos(math.pi * cos_inner)) 
                    for i in range(len(self.base_lrs))]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        pass  # Not implemented

scheduler = CosineAnnealingWarmupRestarts(optimizer, T_0=60, T_mult=1, eta_max=0.001, T_up=10, gamma=0.5)

# -----------------------------
# Training and Validation Functions
# -----------------------------

def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc=f"Training Epoch {epoch+1}")):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:    # Print every 100 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def validate(epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc=f"Validation Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    print(f'Validation Accuracy after Epoch {epoch + 1}: {accuracy:.2f}%')
    return accuracy

# -----------------------------
# Training Loop with Checkpointing
# -----------------------------

best_acc = 0.0
num_epochs = 70  # Total number of epochs

for epoch in range(num_epochs):
    train(epoch)
    acc = validate(epoch)
    scheduler.step()

    # Save the model checkpoint if it's the best so far
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'resnet152_cifar10_weights_best.pt')
        print(f"Best model saved with accuracy: {best_acc:.2f}%")

print("Initial Training completed.")

# -----------------------------
# Fine-Tuning Phase
# -----------------------------

# Load the best model from initial training
model.load_state_dict(torch.load('resnet152_cifar10_weights_best.pt'))
model.eval()

# Fine-Tuning Transformations (additional normalization if needed)
transform_finetune = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    Cutout(length=16),
])

# Update the training dataset with fine-tuning transformations
finetune_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_finetune)
finetune_trainloader = DataLoader(finetune_trainset, batch_size=128, shuffle=True, num_workers=4)

# Re-define optimizer for fine-tuning with lower learning rate
finetune_optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5e-4)

# Re-define scheduler for fine-tuning
finetune_scheduler = CosineAnnealingWarmupRestarts(finetune_optimizer, T_0=30, T_mult=1, eta_max=0.0001, T_up=5, gamma=0.5)

def finetune(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(tqdm(finetune_trainloader, desc=f"Fine-Tuning Epoch {epoch+1}")):
        inputs, targets = inputs.to(device), targets.to(device)
        finetune_optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        finetune_optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print(f'[Fine-Tuning Epoch {epoch + 1}, Batch {batch_idx + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

def finetune_validate(epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc=f"Fine-Tuning Validation Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100. * correct / total
    print(f'Fine-Tuning Validation Accuracy after Epoch {epoch + 1}: {accuracy:.2f}%')
    return accuracy

# Fine-Tuning loop
fine_tune_epochs = 15
for epoch in range(fine_tune_epochs):
    finetune(epoch)
    acc = finetune_validate(epoch)
    finetune_scheduler.step()

    # Save the fine-tuned model checkpoint if it's the best so far
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'resnet152_cifar10_finetuned_best.pt')
        print(f"Fine-Tuned best model saved with accuracy: {best_acc:.2f}%")

print("Fine-Tuning completed.")




# -----------------------------
# Inference on Test Set
# -----------------------------

# Decompressing Test Data (Assuming you have a 7z file)
# Note: Adjust the paths based on your environment
test_archive_path = '/kaggle/input/cifar-10/test.7z'
extracted_path = '/kaggle/working/test'

if not os.path.exists(extracted_path):
    os.makedirs(extracted_path, exist_ok=True)
    with py7zr.SevenZipFile(test_archive_path, mode='r') as z:
        z.extractall(extracted_path)
    print("Test data decompressed successfully.")
else:
    print("Test data already decompressed.")


archive_path = '/kaggle/input/cifar-10/test.7z'




import py7zr, os

extracted_path = '/kaggle/working/test'

if not os.path.exists(extracted_path):
    os.makedirs(extracted_path, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode='r') as z:
        z.extractall(path=extracted_path)



import py7zr
import os

archive_path = '/kaggle/input/cifar-10/test.7z'  # Your CIFAR-10 test.7z file
extracted_path = '/kaggle/working/test'

# Create the extraction folder if it doesn't exist
if not os.path.exists(extracted_path):
    os.makedirs(extracted_path, exist_ok=True)

# Extract the .7z archive
with py7zr.SevenZipFile(archive_path, mode='r') as z:
    z.extractall(path=extracted_path)



!ls /kaggle/working/test



# Install required package
!pip install py7zr

# -----------------------------
# Required Imports
# -----------------------------

import os
import torch
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# -----------------------------
# Model Definition: ResNet152
# -----------------------------

import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # Stem layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Final classifier layer
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

# -----------------------------
# Define CIFAR-10 Classes
# -----------------------------

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# -----------------------------
# Custom Dataset Class
# -----------------------------

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory with test images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Extract image ID from filename (assuming filename starts with an integer)
        img_id_str = os.path.basename(img_name).split('.')[0]
        try:
            img_id = int(img_id_str)
        except ValueError:
            # Handle cases where the filename does not start with an integer
            img_id = img_id_str  # You can choose to skip or handle differently
        return image, img_id

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------

# CIFAR-10 mean and std for normalization (Ensure these match the training phase)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Define the test data transformations (consistent with training)
inference_transforms = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# -----------------------------
# Instantiate the Model
# -----------------------------

model = ResNet152().to(device)
print("Model instantiated successfully.")

# -----------------------------
# Load Model Weights
# -----------------------------

model_path = '/kaggle/working/resnet152_cifar10_finetuned_best.pt'  # Path to the fine-tuned model weights

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model weights loaded successfully.")
else:
    raise FileNotFoundError(f"Model weights file not found at {model_path}. Please check the path.")

model.eval()

# -----------------------------
# Prepare Test DataLoader
# -----------------------------

# Set the path to your test data directory on Kaggle
test_dir = '/kaggle/working/test/test' 
  # Adjust this path based on where your test images are extracted

# Ensure the test directory exists
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found at {test_dir}. Please check the path.")

# Create the dataset and dataloader
test_dataset = CustomImageDataset(test_dir, transform=inference_transforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# -----------------------------
# Perform Inference and Save Predictions
# -----------------------------

# Perform inference and store results
all_predictions = []
all_filenames = []

with torch.no_grad():
    for inputs, img_ids in tqdm(test_loader, desc="Performing Inference"):
        inputs = inputs.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Get predicted class
        _, predicted = torch.max(outputs, 1)
        
        # Move predictions to CPU and convert to numpy
        predicted = predicted.cpu().numpy()
        
        # Append predictions and corresponding image IDs
        for img_id, pred in zip(img_ids, predicted):
            # Handle non-integer img_ids if necessary
            if isinstance(img_id, int):
                label = classes[pred]
                all_predictions.append((img_id, label))
            else:
                # If img_id is not integer, handle accordingly (e.g., skip or assign a default value)
                label = classes[pred]
                all_predictions.append((img_id, label))

# -----------------------------
# Save Predictions to CSV
# -----------------------------

# Convert the list of tuples to a DataFrame
submission_df = pd.DataFrame(all_predictions, columns=['id', 'label'])

# If you need 'id' to be integers and they are currently strings, convert them
# This step assumes that all 'id's are integers
if submission_df['id'].apply(lambda x: isinstance(x, int)).all():
    submission_df['id'] = submission_df['id'].astype(int)
else:
    # Handle cases where 'id's are not all integers
    # For example, you might extract numeric part if 'id's are strings like 'test_1'
    submission_df['id'] = submission_df['id'].apply(lambda x: int(''.join(filter(str.isdigit, x))) if isinstance(x, str) else x)

# Sort the DataFrame by 'id' if necessary
submission_df = submission_df.sort_values(by='id')
submission_df['id'] = submission_df['id'].astype(int)
# Save the DataFrame to a CSV file
submission_df.to_csv('resnet152_cifar10_predictions_updated.csv', index=False)

print("Predictions saved to resnet152_cifar10_predictions.csv")





