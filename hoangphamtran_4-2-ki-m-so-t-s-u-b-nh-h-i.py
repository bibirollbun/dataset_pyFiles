# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


train=pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test=pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")


base_path='/kaggle/input/plant-pathology-2020-fgvc7/images/'
def generate_image_path(image_id):
    return f"{base_path}{image_id}.jpg"

# Apply the function to create the 'img' column
train['img_path'] = train['image_id'].apply(generate_image_path)
test['img_path'] = test['image_id'].apply(generate_image_path)


train.info()


print(train.shape)
train.head()


print(test.shape)
test.head()


test3 = cv2.imread(test['img_path'].iloc[1819])

plt.imshow(test3)
plt.axis('off')
plt.show()


train.iloc[379]


img = cv2.imread(train['img_path'].iloc[379])
plt.imshow(img)
plt.axis('off')
plt.show()


train.iloc[1173]


img = cv2.imread(train['img_path'].iloc[1173])

plt.imshow(img)
plt.axis('off')
plt.show()


img.shape


img


print("\nClass distribution in training set:")
label_counts = train[['healthy', 'multiple_diseases', 'rust', 'scab']].sum()
print(label_counts)

# Vẽ biểu đồ cột
plt.figure(figsize=(8, 6))
label_counts.plot(kind='bar', color=['green', 'orange', 'brown', 'red'])

# Thêm tiêu đề và nhãn trục
plt.title('Tổng số mẫu của từng loại bệnh')
plt.xlabel('Loại bệnh')
plt.ylabel('Số lượng')

# Hiển thị giá trị trên từng cột
for index, value in enumerate(label_counts):
    plt.text(index, value + 10, str(value), ha='center', fontsize=12)

# Hiển thị biểu đồ
plt.show()




# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define class names (target columns)
class_names = ['healthy', 'multiple_diseases', 'rust', 'scab']




# Custom Dataset class
class PlantPathologyDataset(Dataset):
    def __init__(self, dataframe, transform=None, is_test=False):
        self.dataframe = dataframe
        self.transform = transform
        self.is_test = is_test
        self.class_names = ['healthy', 'multiple_diseases', 'rust', 'scab']
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, self.dataframe.iloc[idx]['image_id']
        else:
            labels = torch.tensor(
                self.dataframe.iloc[idx][self.class_names].values.astype(np.float32)
            )
            return image, labels




# Data augmentation transforms
# Training transform with optimal augmentations for plant disease classification
train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation transform - consistent, no augmentations
val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Split train data into train and validation
train_idx, val_idx = train_test_split(
    range(len(train)),
    test_size=0.2,
    random_state=SEED,
    stratify=train[class_names].values.argmax(axis=1)
)

train_dataset = PlantPathologyDataset(
    train.iloc[train_idx].reset_index(drop=True),
    transform=train_transform
)

val_dataset = PlantPathologyDataset(
    train.iloc[val_idx].reset_index(drop=True),
    transform=val_transform
)

test_dataset = PlantPathologyDataset(
    test,
    transform=val_transform,
    is_test=True
)



# Create DataLoaders
BATCH_SIZE = 64
NUM_WORKERS = 4

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)



# Model definition
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(PlantDiseaseModel, self).__init__()
        # Use a pre-trained ResNet50 as the backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        # self.backbone.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(in_features, num_classes)
        # )
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added dropout after first ReLU
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added dropout after second ReLU
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# class PlantDiseaseModel(nn.Module):
#     def __init__(self, num_classes=4):
#         super(PlantDiseaseModel, self).__init__()
        
#         # Convolutional layers using Sequential
#         self.features = nn.Sequential(
#             # First convolutional block
#             nn.Conv2d(3, 8, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Dropout(0.2),
            
#             # Second convolutional block
#             nn.Conv2d(8, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Dropout(0.2),
            
#             # Third convolutional block
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Dropout(0.2)
#         )
        
#         # Calculate input features for the classifier
#         # Input: 224x224 -> After 3 pooling layers: 28x28
#         fc_input_features = 32 * 28 * 28
        
#         # Fully connected layers using Sequential
#         self.classifier = nn.Sequential(
#             nn.Linear(fc_input_features, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )
    
#     def forward(self, x):
#         # Input normalization
#         x = x / 255.0
        
#         # Apply convolutional layers
#         x = self.features(x)
        
#         # Flatten
#         x = x.view(x.size(0), -1)
        
#         # Apply classifier
#         x = self.classifier(x)
        
#         return x

# Initialize model
model = PlantDiseaseModel(num_classes=len(class_names))
model = model.to(device)
print(model)


# Loss function
criterion = nn.BCEWithLogitsLoss()

# Optimizer with learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)



# Training and validation functions
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate loss
            running_loss += loss.item() * inputs.size(0)
            
            # Store predictions and targets
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Calculate validation loss
    val_loss = running_loss / len(dataloader.dataset)
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return val_loss, all_preds, all_targets



# Training loop
NUM_EPOCHS = 25
best_val_loss = float('inf')
best_model_path = 'best_model.pth'

# Lists to store metrics for plotting
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    # Train
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    
    # Validate
    val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    
    print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    
    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Model saved with validation loss: {best_val_loss:.4f}")

# Load the best model
model.load_state_dict(torch.load(best_model_path))



# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS+1), train_losses, label='Train Loss')
plt.plot(range(1, NUM_EPOCHS+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_history.png')
plt.show()


# Make predictions on test set
model.eval()
test_predictions = []
test_ids = []

with torch.no_grad():
    for inputs, image_ids in tqdm(test_loader, desc='Testing'):
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs).cpu().numpy()
        
        test_predictions.append(preds)
        test_ids.extend(image_ids)
# Concatenate all predictions
test_predictions = np.concatenate(test_predictions, axis=0)

# Create submission dataframe
submission_df = pd.DataFrame({'image_id': test_ids})
for i, class_name in enumerate(class_names):
    submission_df[class_name] = test_predictions[:, i]

# Save submission file
submission_df.to_csv('submission.csv', index=False)
print("Submission file created successfully!")


submission_df

