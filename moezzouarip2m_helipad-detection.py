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


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time




BASE_DIR = '/kaggle/input/helipad-detection-challenge-sup-com/helipad_hackathon/'
TRAIN_CSV = os.path.join(BASE_DIR, 'train.csv')
IMAGE_DIR = os.path.join(BASE_DIR, 'images')

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 12
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Training on: {DEVICE}")




class HelipadDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0] 
        label = self.df.iloc[idx, 1]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)



# --- 2. PREPARE DATA & AUGMENTATION ---
full_df = pd.read_csv(TRAIN_CSV)

train_df, val_df = train_test_split(
    full_df, 
    test_size=0.15, 
    stratify=full_df['label'], 
    random_state=42
)

print(f"Training Images:   {len(train_df)}")
print(f"Validation Images: {len(val_df)}")

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



train_loader = DataLoader(
    HelipadDataset(train_df, IMAGE_DIR, train_transforms), 
    batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

val_loader = DataLoader(
    HelipadDataset(val_df, IMAGE_DIR, val_transforms), 
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)



# --- 3. LOAD RESNET50 MODEL ---
from torchvision.models import ResNet50_Weights

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)


num_features = model.fc.in_features 
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 1)
)

model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)



# --- 4. TRAINING LOOP ---
best_val_acc = 0.0
history = {'train_loss': [], 'val_acc': []}

print("\n--- Starting Training ---")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        labels = labels.unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    history['train_loss'].append(epoch_loss)
    
    # VALIDATION
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            labels = labels.unsqueeze(1)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            predicted_labels = (preds > 0.5).float()
            
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
    
    epoch_acc = correct / total
    history['val_acc'].append(epoch_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.4f}")
    
    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        torch.save(model.state_dict(), 'helipad_resnet50.pth')
        print("  >>> Model Saved (New Best)")

total_time = (time.time() - start_time) / 60
print(f"\nTraining Finished in {total_time:.1f} minutes.")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")






# --- 5. PLOT RESULTS ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', color='red')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['val_acc'], label='Validation Accuracy', color='blue')
plt.title('Validation Accuracy')
plt.grid(True)

plt.show()



import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm # Progress bar

# --- CONFIGURATION ---
BASE_DIR = '/kaggle/input/helipad-detection-challenge-sup-com/helipad_hackathon/'
TEST_CSV_PATH = os.path.join(BASE_DIR, 'test.csv') 
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
MODEL_PATH = 'helipad_resnet50.pth' # The file you just trained
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# --- 1. TEST DATASET CLASS ---
# This is different from training because we don't have labels
class TestDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get the Image ID (assuming column 0 is the ID)
        img_name = str(self.df.iloc[idx, 0])
        
        # Handle file paths safely
        img_path = os.path.join(self.root_dir, img_name)
        
        # If extension is missing in CSV, try adding it
        if not os.path.exists(img_path):
            if os.path.exists(img_path + '.jpg'): img_path += '.jpg'
            elif os.path.exists(img_path + '.png'): img_path += '.png'
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # If image is missing, return a black image (rare safety check)
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

# --- 2. SETUP ---
# Load the list of test images
# If test.csv doesn't exist, check for sample_submission.csv
if not os.path.exists(TEST_CSV_PATH):
    TEST_CSV_PATH = os.path.join(BASE_DIR, 'sample_submission.csv')

test_df = pd.read_csv(TEST_CSV_PATH)
print(f"Loaded Test CSV: {len(test_df)} images to predict.")

# Define Transforms (Must be same resizing/normalization as training)
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create Loader
test_dataset = TestDataset(test_df, IMAGE_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# --- 3. LOAD MODEL ARCHITECTURE ---
# We must define the model EXACTLY as we did in training
model = models.resnet50(pretrained=False) # Pretrained=False because we load our own weights

# Re-create the custom head
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 1)
)

# Load the trained weights
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("✅ Model weights loaded successfully!")
else:
    print(f"❌ ERROR: Could not find {MODEL_PATH}. Did you run the training cell?")

model = model.to(DEVICE)
model.eval() # Important: Turns off Dropout and BatchNorm updates

# --- 4. PREDICTION LOOP ---
predictions = []
image_ids = []

print("Starting Prediction...")

with torch.no_grad(): # Disable gradient calculation to save memory
    for images, names in tqdm(test_loader):
        images = images.to(DEVICE)
        
        # Forward pass
        outputs = model(images)
        
        # Convert logits to probabilities (0 to 1)
        probs = torch.sigmoid(outputs)
        
        # Move to CPU and store
        predictions.extend(probs.cpu().numpy().flatten())
        image_ids.extend(names)

# --- 5. CREATE SUBMISSION CSV ---
# Apply Threshold (0.5 is standard)
binary_predictions = [1 if p > 0.5 else 0 for p in predictions]

# Create DataFrame
submission_df = pd.DataFrame({
    'id': image_ids,      # Ensure this column name matches Kaggle's sample_submission
    'label': binary_predictions
})

# Save
submission_df.to_csv('submission.csv', index=False)
print("\n✅ 'submission.csv' created successfully!")
print(submission_df.head())


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# --- CONFIGURATION ---
SUBMISSION_FILE = 'submission.csv'
IMAGE_DIR = '/kaggle/input/helipad-detection-challenge-sup-com/helipad_hackathon/images'

# 1. LOAD RESULTS
if not os.path.exists(SUBMISSION_FILE):
    print("Error: submission.csv not found! Did you run the prediction step?")
else:
    df = pd.read_csv(SUBMISSION_FILE)
    print(f"Loaded predictions for {len(df)} images.")

    # 2. SHOW STATISTICS
    count_0 = df[df['label'] == 0].shape[0]
    count_1 = df[df['label'] == 1].shape[0]

    print(f"\n--- PREDICTION STATS ---")
    print(f"Predicted NO Helipad (0): {count_0}")
    print(f"Predicted Helipad (1):    {count_1}")
    print(f"Helipad Ratio: {count_1 / len(df) * 100:.2f}%")

    plt.figure(figsize=(6, 4))
    sns.countplot(x=df['label'])
    plt.title("Distribution of Predictions")
    plt.show()

    # 3. VISUALIZE ACTUAL IMAGES
    # Helper function to find image paths safely
    def get_image_path(filename, root_dir):
        path = os.path.join(root_dir, str(filename))
        if os.path.exists(path): return path
        if os.path.exists(path + '.jpg'): return path + '.jpg'
        if os.path.exists(path + '.png'): return path + '.png'
        return None

    def show_predictions(label_value, title):
        # Get random samples for this class
        subset = df[df['label'] == label_value]
        if len(subset) == 0:
            print(f"No images predicted as {title}")
            return
        
        samples = subset.sample(min(5, len(subset))) # Show up to 5
        
        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Model Prediction: {title}", fontsize=16)
        
        for i, (_, row) in enumerate(samples.iterrows()):
            img_id = row['id'] # Adjust column name if different (e.g. 'ImageID')
            img_path = get_image_path(img_id, IMAGE_DIR)
            
            plt.subplot(1, 5, i+1)
            if img_path:
                img = Image.open(img_path)
                plt.imshow(img)
                plt.title(img_id)
            else:
                plt.text(0.5, 0.5, "Img Not Found", ha='center')
            plt.axis('off')
        plt.show()

    # Show images the model thinks are Helipads
    show_predictions(1, "Helipad (1)")

    # Show images the model thinks are Empty
    show_predictions(0, "No Helipad (0)")

