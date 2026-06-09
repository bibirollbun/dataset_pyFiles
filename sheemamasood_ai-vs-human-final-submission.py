import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score ,roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torchvision.models as models

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.amp import GradScaler, autocast


# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available, using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU.")



def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set fixed seed
set_seed(42)


# Load the dataset
base_dir = '/kaggle/input/ai-vs-human-generated-dataset'
train_df = pd.read_csv("/kaggle/input/detect-ai-vs-human-generated-images/train.csv")
test_df = pd.read_csv("/kaggle/input/detect-ai-vs-human-generated-images/test.csv")

# Display basic info
print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)
train_df.head()


test_df.head()


train_df = train_df.drop(columns=['Unnamed: 0'])


# Adding the full path to the file_name instead of just "trainORtest_data/xxx.jpg"
test_df['id'] = test_df['id'].apply(lambda x: os.path.join(base_dir, x))
train_df['file_name'] = train_df['file_name'].apply(lambda x: os.path.join(base_dir, x))


print(f"Train Data: {len(train_df)}")
print(f"Test Data: {len(test_df)}")


# Check for missing values
print("Missing values in Train Dataset:\n", train_df.isnull().sum())
print("Missing values in Test Dataset:\n", test_df.isnull().sum())

# Check for duplicates
print("Duplicate entries in Train Dataset:", train_df.duplicated().sum())



# Countplot of AI vs. Human-generated images
plt.figure(figsize=(6,4))
sns.countplot(x="label", data=train_df, palette="coolwarm")
plt.title("Label Distribution")
plt.xticks([0, 1], ["Human-Created", "AI-Generated"])
plt.show()



# Function to display images
def show_images(df, label, num_images=5):
    sample_images = df[df["label"] == label].sample(num_images, random_state=42)["file_name"].values

    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(sample_images):
        img = cv2.imread(img_path)  # Read image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("AI-Generated" if label == 1 else "Human-Created")
        
    plt.show()

# Show AI-Generated Images
show_images(train_df, label=1)

# Show Human-Created Images
show_images(train_df, label=0)


# Function to display random test images
def show_test_images(df, num_images=5):
    sample_images = df.sample(num_images, random_state=42)["id"].values  

    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(sample_images):
        img = cv2.imread(img_path)  # Read image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Image {i+1}")  
    
    plt.show()

# Show random test images
show_test_images(test_df, num_images=5)



# Function to get image dimensions
def get_image_dimensions(image_paths, sample_size=500):
    image_sizes = []
    
    for img_path in image_paths.sample(sample_size, random_state=42):  # Randomly sampling
        img = cv2.imread(img_path)
        if img is not None:
            h, w, _ = img.shape
            image_sizes.append((w, h))
    
    return pd.DataFrame(image_sizes, columns=["Width", "Height"])

# Get dimensions for train and test separately
train_size_df = get_image_dimensions(train_df["file_name"])
test_size_df = get_image_dimensions(test_df["id"])

# Plot Train vs. Test Image Dimension Distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(x=train_size_df["Width"], y=train_size_df["Height"], alpha=0.5, ax=axes[0])
axes[0].set_title("Train Image Dimensions")
axes[0].set_xlabel("Width")
axes[0].set_ylabel("Height")

sns.scatterplot(x=test_size_df["Width"], y=test_size_df["Height"], alpha=0.5, ax=axes[1], color='red')
axes[1].set_title("Test Image Dimensions")
axes[1].set_xlabel("Width")
axes[1].set_ylabel("Height")

plt.tight_layout()
plt.show()



# Function to get image dimensions
def get_image_dimensions(image_paths, sample_size=500):
    image_sizes = []
    
    for img_path in image_paths.sample(sample_size, random_state=42):  # Randomly sampling
        img = cv2.imread(img_path)
        if img is not None:
            h, w, _ = img.shape
            image_sizes.append((w, h))
    
    return pd.DataFrame(image_sizes, columns=["Width", "Height"])

# Separate AI-generated and Human-created images
ai_images = train_df[train_df["label"] == 1]["file_name"]
human_images = train_df[train_df["label"] == 0]["file_name"]

# Get dimensions for both categories
ai_size_df = get_image_dimensions(ai_images)
human_size_df = get_image_dimensions(human_images)

# Plot AI vs. Human Image Dimension Distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(x=ai_size_df["Width"], y=ai_size_df["Height"], alpha=0.5, ax=axes[0], color='red')
axes[0].set_title("AI-Generated Image Dimensions")
axes[0].set_xlabel("Width")
axes[0].set_ylabel("Height")

sns.scatterplot(x=human_size_df["Width"], y=human_size_df["Height"], alpha=0.5, ax=axes[1], color='blue')
axes[1].set_title("Human-Created Image Dimensions")
axes[1].set_xlabel("Width")
axes[1].set_ylabel("Height")

plt.tight_layout()
plt.show()



def plot_pixel_intensity_side_by_side(img_path1, img_path2, title1="AI-Generated", title2="Human-Created"):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    axes[0].hist(img1.ravel(), bins=256, color="gray", alpha=0.7)
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{title1} - Pixel Intensity")

    axes[1].hist(img2.ravel(), bins=256, color="gray", alpha=0.7)
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{title2} - Pixel Intensity")

    plt.tight_layout()
    plt.show()

# Select random AI-generated and human-created images
sample_ai = train_df[train_df["label"] == 1]["file_name"].sample(1).values[0]
sample_human = train_df[train_df["label"] == 0]["file_name"].sample(1).values[0]

# Plot side by side
plot_pixel_intensity_side_by_side(sample_ai, sample_human)



import cv2
import matplotlib.pyplot as plt
import pandas as pd

def plot_color_distribution_side_by_side(img_path1, img_path2, title1="AI-Generated", title2="Human-Created"):
    img1 = cv2.imread(img_path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    img2 = cv2.imread(img_path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Splitting into RGB channels
    r1, g1, b1 = cv2.split(img1)
    r2, g2, b2 = cv2.split(img2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

    # AI-Generated Image Histogram
    axes[0].hist(r1.ravel(), bins=256, color="red", alpha=0.5, label="Red")
    axes[0].hist(g1.ravel(), bins=256, color="green", alpha=0.5, label="Green")
    axes[0].hist(b1.ravel(), bins=256, color="blue", alpha=0.5, label="Blue")
    axes[0].set_xlabel("Pixel Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{title1} - Color Distribution")
    axes[0].legend()

    # Human-Created Image Histogram
    axes[1].hist(r2.ravel(), bins=256, color="red", alpha=0.5, label="Red")
    axes[1].hist(g2.ravel(), bins=256, color="green", alpha=0.5, label="Green")
    axes[1].hist(b2.ravel(), bins=256, color="blue", alpha=0.5, label="Blue")
    axes[1].set_xlabel("Pixel Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{title2} - Color Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# Select random AI-generated and human-created images
sample_ai = train_df[train_df["label"] == 1]["file_name"].sample(1).values[0]
sample_human = train_df[train_df["label"] == 0]["file_name"].sample(1).values[0]

# Plot side by side
plot_color_distribution_side_by_side(sample_ai, sample_human)



# âœ… Split the train_df DataFrame itself, NOT just 'file_name' and 'label' separately
train_df, val_df = train_test_split(
    train_df,  
    test_size=0.05,        
    stratify=train_df['label'],  
    random_state=42
)

# âœ… Print checks
print(f"Train Data: {len(train_df)}")
print(f"Validation Data: {len(val_df)}")
print(f"Test Data: {len(test_df)}")  # Test data remains unchanged



# âœ… Training Transformations (with Augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((232)),                
    transforms.RandomResizedCrop(224),  
    transforms.RandomHorizontalFlip(),      
    transforms.RandomRotation(10),           
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),        
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])                        
    
# âœ… Testing Transformations (NO Augmentation)
test_transforms = transforms.Compose([
    transforms.Resize(232),  
    transforms.CenterCrop(224),             
    transforms.ToTensor(),                        
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   
])



class CustomDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = pd.DataFrame(df)  # Ensure it's a DataFrame
        self.transform = transform
        self.is_test = is_test  # Test dataset flag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["file_name"]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img  # No label for test dataset
        else:
            label = self.df.iloc[idx]["label"]
            return img, label



# âœ… Train & Validation Datasets
train_dataset = CustomDataset(df=train_df, transform=train_transforms)
val_dataset = CustomDataset(df=val_df, transform=test_transforms)

batch_size = 8
# âœ… DataLoaders
#train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")



batch_images, batch_labels = next(iter(train_loader))
print(f"Batch shape: {batch_images.shape}, Labels shape: {batch_labels.shape}")



print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {device}")



#  Load Pretrained ResNet Model
#model = models.resnet18(pretrained=True)

#  Modify Final Layer for Binary Classification (AI vs. Human)
#num_features = model.fc.in_features
#model.fc = nn.Linear(num_features, 2)  

#  Move the model to the correct device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)

#print("Model Loaded Successfully!")
#for name, param in model.named_parameters():
    #print(f"{name} : {param.requires_grad} | {param.shape}")

## Unfreeze some layers (e.g., last 2 blocks of ResNet)
#for name, param in model.named_parameters():
   # if "layer4" in name or "layer3" in name:  # Modify based on model
   #     param.requires_grad = True  # Unfreeze
   # else:
    #    param.requires_grad = False  # Keep frozen



# import torch
# import torchvision.models as models

# model = models.convnext_large(pretrained=True)

# # Freeze layers (optional)
# for param in model.parameters():
#     param.requires_grad = False

# # Modify classifier
# num_ftrs = model.classifier[2].in_features
# num_classes = 2  # Change based on competition
# model.classifier[2] = torch.nn.Linear(num_ftrs, num_classes)

# # Move model to GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # âœ… Step 1: Freeze All Layers
# for param in model.parameters():
#     param.requires_grad = False  

# # âœ… Step 2: Unfreeze Last 10 Layers (Fine-Tuning)
# for param in list(model.parameters())[-10:]:  
#     param.requires_grad = True  

# # âœ… Step 3: Modify Classifier for Binary Classification
# model.classifier = nn.Sequential(
#     nn.AdaptiveAvgPool2d((1, 1)),  
#     nn.Flatten(),                  
#     nn.Linear(1536, 512),  
#     nn.ReLU(),                    
#     nn.Dropout(0.4),              
#     nn.Linear(512, 1)
# )

# âœ… Move Model to GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# print("âœ… ConvNext Model Ready for Binary Classification Training!")



import torch
import torch.nn as nn
from timm import create_model

class ViTConXWithAvgPooling(nn.Module):
    def __init__(self, num_classes=1):
        super(ViTConXWithAvgPooling, self).__init__()

        # Load ConvNeXt Large
        self.convnext = create_model("convnext_large", pretrained=True, num_classes=0)
        convnext_out = self.convnext.num_features

        # Load Swin Transformer
        self.swin = create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)
        swin_out = self.swin.num_features

        # Global Average Pooling for each model
        self.global_avg_pooling_convnext = nn.AdaptiveAvgPool1d(1)
        self.global_avg_pooling_swin = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layers for feature fusion
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(convnext_out + swin_out),
            nn.Linear(convnext_out + swin_out, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        
        # Decoder: Additional layers to output classification results
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)  
        )

    def forward(self, x):
        # Pass through ConvNeXt and Swin Transformer
        x_convnext = self.convnext(x)
        x_swin = self.swin(x)

        # Debug print statements to inspect tensor shapes
        #print("Shape of x_convnext before pooling:", x_convnext.shape)
        #print("Shape of x_swin before pooling:", x_swin.shape)

        # Apply global average pooling
        x_convnext = self.global_avg_pooling_convnext(x_convnext.unsqueeze(2)).view(x_convnext.size(0), -1)
        x_swin = self.global_avg_pooling_swin(x_swin.unsqueeze(2)).view(x_swin.size(0), -1)

        # Debug print statements to inspect tensor shapes after pooling
        #print("Shape of x_convnext after pooling:", x_convnext.shape)
        #print("Shape of x_swin after pooling:", x_swin.shape)

        # Concatenate both feature vectors
        x_combined = torch.cat((x_convnext, x_swin), dim=1)
        x_fused = self.feature_fusion(x_combined)

        # Pass through the decoder to output the final classification result
        decoded_output = self.decoder(x_fused)

        return decoded_output

# Initialize the model with ConvNeXt Large and Swin Transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTConXWithAvgPooling(num_classes=1).to(device)

# Freeze All Layers
for param in model.convnext.parameters():
    param.requires_grad = False

for param in model.swin.parameters():
    param.requires_grad = False

# Unfreeze Last 10 Layers
for param in list(model.convnext.parameters())[-20:]:
    param.requires_grad = True

for param in list(model.swin.parameters())[-20:]:
    param.requires_grad = True

    
# Prepare optimizer
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Define Loss Function
criterion = nn.BCEWithLogitsLoss().to(device)


torch.cuda.empty_cache()



# Initialize GradScaler for mixed precision training
scaler = GradScaler(device='cuda')

# Initialize lists to store training progress
train_losses, train_accuracies = [], []
val_losses, val_accuracies, val_f1_scores, val_roc_aucs = [], [], [], []

epochs = 3
patience = 1
best_val_loss = float("inf")
epochs_without_improvement = 0  

for epoch in range(epochs):
    model.train()
    epoch_loss, epoch_accuracy = 0.0, 0.0

    for data, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        data, label = data.to(device), label.to(device).float()

        optimizer.zero_grad()

        # Mixed precision training
        with autocast(device_type='cuda'):
            output = model(data).squeeze(1)
            loss = criterion(output, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        preds = (torch.sigmoid(output) > 0.5).float()
        epoch_accuracy += (preds == label).float().mean().item()

    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # -- Validation --
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    val_pred_classes, val_labels_list = [], []

    with torch.no_grad():
        for data, label in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            data, label = data.to(device), label.to(device).float()

            with autocast(device_type='cuda'):
                output = model(data).squeeze(1)
                loss = criterion(output, label)

            val_loss += loss.item()
            preds = (torch.sigmoid(output) > 0.5).float()
            val_acc += (preds == label).float().mean().item()

            val_pred_classes.extend(preds.cpu().numpy())
            val_labels_list.extend(label.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_f1 = f1_score(np.array(val_labels_list, dtype=int), 
                      np.array(val_pred_classes, dtype=int))
    val_roc_auc = roc_auc_score(np.array(val_labels_list, dtype=int), 
                                np.array(val_pred_classes, dtype=int))

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_f1_scores.append(val_f1)
    val_roc_aucs.append(val_roc_auc)

    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val ROC AUC: {val_roc_auc:.4f}")

    scheduler.step()

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_weights.pth')  
        torch.save(model, 'best_model.pth')  # Save both model and weights
        print(" Model saved!")
        epochs_without_improvement = 0  # Reset counter after improvement
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"âš ï¸� Early stopping at epoch {epoch+1}")
            break


# Plot Loss and Accuracy

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot Loss
axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red')
axs[0].set_title('Loss per Epoch')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Plot Accuracy
axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='blue')
axs[1].plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='red')
axs[1].set_title('Accuracy per Epoch')

axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

# Show the plots
plt.tight_layout()
plt.savefig('Accuracy per Epoch.png')
plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Compute ROC curve
fpr, tpr, _ = roc_curve(np.array(val_labels_list, dtype=int), np.array(val_pred_classes, dtype=int))

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {val_roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()



# Generate and plot Confusion Matrix
conf_matrix = confusion_matrix(val_labels_list, val_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print Classification Report
print("\nClassification Report:")
print(classification_report(val_labels_list, val_pred_classes))


# Load the full model
#model = torch.load("best_model_full.pth")
#model.to(device)



# âœ… Test Dataset (NO Labels)
class CustomTestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = pd.DataFrame(df)  # Ensure it's a DataFrame
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["id"]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img  # No label for test dataset



# âœ… Test Dataset
test_dataset = CustomTestDataset(df=test_df, transform=test_transforms)

# âœ… Test DataLoader
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f"Test dataset size: {len(test_dataset)}")



# âœ… Set model to evaluation mode
model.eval()

# âœ… List to store predictions
test_predictions = []

# âœ… Disable gradient computation (since we're not training)
with torch.no_grad():
    for data in tqdm(test_loader, desc="Inference Progress", total=len(test_loader)):
        # Move data to device
        data = data.to(device)

        # Forward pass
        output = model(data).squeeze(1)  # Ensure correct dimensions

        # Apply sigmoid activation to convert logits to probabilities
        probs = torch.sigmoid(output)

        # Convert probabilities to binary class (0 or 1) using threshold 0.5
        preds = (probs > 0.5).int()

        # Store predictions
        test_predictions.extend(preds.cpu().numpy())

# Convert predictions to numpy array
test_predictions = np.array(test_predictions, dtype=int)



# âœ… Create Submission DataFrame using `test_df`
submission_df = test_df.copy()  # Copy original test dataframe

# Extract only the filename from the full path
submission_df["id"] = submission_df["id"].apply(lambda x: x.split("/")[-1])  
# Reattach `test_data_v2/` prefix
submission_df["id"] = "test_data_v2/" + submission_df["id"]
# Add predictions
submission_df["label"] = test_predictions  


submission_df["label"].value_counts()



# âœ… Save CSV file
submission_df.to_csv("submission.csv", index=False)










