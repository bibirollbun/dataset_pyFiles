import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os


! python --version


# Load the training data
train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

# Check for missing values in the dataframe
print("Missing values in each column:")
print(train_df.isnull().sum())


rows_with_nulls = train_df[train_df.isnull().any(axis=1)]
sample_of_nulls = rows_with_nulls.sample(527)

# 3. Print the resulting table
print(sample_of_nulls)


train_df.shape


df_cleaned = train_df.dropna()


df_cleaned.shape



# Get the counts of benign vs. malignant cases
class_counts = sample_of_nulls['benign_malignant'].value_counts()

# Get the percentage distribution
class_percentages = sample_of_nulls['benign_malignant'].value_counts(normalize=True) * 100

print("Class Distribution:")
print(class_counts)
print("\nClass Distribution (%):")
print(class_percentages)


# Path to the training images
image_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

# Get the list of benign and malignant image names
benign_images = train_df[train_df['target'] == 0]['image_name'].values
malignant_images = train_df[train_df['target'] == 1]['image_name'].values

# Plot 10 benign images
print("Benign Images")
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = cv2.imread(os.path.join(image_path, benign_images[i] + '.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title('Benign')
    plt.axis('off')
plt.show()

# Plot 10 malignant images
print("Malignant Images")
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = cv2.imread(os.path.join(image_path, malignant_images[i] + '.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title('Malignant')
    plt.axis('off')
plt.show()


# # use pytorch
# knowing the dataset class:
#     the targeted model of multimodal:
#              ann and cnn


import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer




# --- Configuration and Setup ---
# Define paths and parameters
DATA_PATH = '/kaggle/input/siim-isic-melanoma-classification/'
IMAGE_PATH = os.path.join(DATA_PATH, 'jpeg/train')
CSV_PATH = os.path.join(DATA_PATH, 'train.csv')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")




# --- Load and Preprocess Tabular Data ---
print("Loading tabular data...")
df = pd.read_csv(CSV_PATH)
print(f"Original dataset size: {len(df)} samples")




# Drop all rows that have at least one missing value
df.dropna(inplace=True)
print(f"Dataset size after dropping nulls: {len(df)} samples")




# Manually encode the 'sex' column into a binary numerical format
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# **FIX:** Explicitly convert the 'sex' column to a numeric data type
df['sex'] = pd.to_numeric(df['sex'])
print("Manually encoded 'sex' column and converted to numeric type.")


# Define which features are numerical and which are categorical
numerical_features = ['age_approx', 'sex']
categorical_features = ['anatom_site_general_challenge']

# Create a ColumnTransformer to preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])

# Fit the preprocessor on the training data
preprocessor.fit(train_df[numerical_features + categorical_features])

print("\nPart 1 Complete: Data is loaded, preprocessed, and ready.")
print(f"Training set size: {len(train_df)} samples")
print(f"Validation set size: {len(val_df)} samples")


import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# --- Part 2: Custom PyTorch Dataset and DataLoaders ---

# This class will load images and their corresponding tabular data for the model
class MelanomaDataset(Dataset):
    def __init__(self, df, tabular_preprocessor, image_dir, transform=None):
        self.df = df
        self.tabular_preprocessor = tabular_preprocessor
        self.image_dir = image_dir
        self.transform = transform
        
        # Pre-process the tabular data and store it.
        # FIX: Removed .toarray() as the output is already a NumPy array
        self.tabular_data = self.tabular_preprocessor.transform(
            self.df[numerical_features + categorical_features]
        )

    def __len__(self):
        # This method returns the total number of samples in the dataset.
        return len(self.df)

    def __getitem__(self, idx):
        # This method fetches a single sample from the dataset at the given index.
        
        # Get the corresponding row from the dataframe
        row = self.df.iloc[idx]
        
        # Load the image from the file path
        img_name = row['image_name']
        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        # Apply the image transformations (e.g., resize, augment, convert to tensor)
        if self.transform:
            image = self.transform(image)
            
        # Get the pre-processed tabular data for this index
        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float)
        
        # Get the label (target) for this sample
        label = torch.tensor(row['target'], dtype=torch.float)
        
        return image, tabular, label

# --- Define Image Transformations ---
IMG_SIZE = 224 # A standard size for pre-trained models like ResNet/EfficientNet

# Define the transformations for the training set.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the transformations for the validation set.
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Create Datasets and DataLoaders ---
# Instantiate the custom datasets
train_dataset = MelanomaDataset(train_df, preprocessor, IMAGE_PATH, transform=train_transform)
val_dataset = MelanomaDataset(val_df, preprocessor, IMAGE_PATH, transform=val_transform)

# Create the DataLoaders
BATCH_SIZE = 60
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print("Part 2 Complete: Custom Datasets and DataLoaders are ready.")
print(f"One batch of training data will have {BATCH_SIZE} images and {BATCH_SIZE} corresponding tabular data rows.")


import torch
import torch.nn as nn
from torchvision import models

# --- Part 3: Multimodal Model Architecture ---

class MultimodalMelanomaNet(nn.Module):
    def __init__(self, num_tabular_features, pretrained=True):
        super(MultimodalMelanomaNet, self).__init__()
        
        # --- Image Branch (CNN - EfficientNet-B0) ---
        # Load the pre-trained EfficientNet-B0 model
        self.image_branch = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # We need to replace the final classification layer of the pre-trained model.
        # Instead of classifying into 1000 ImageNet classes, we want it to output a feature vector.
        # Let's make it output a vector of size 128.
        num_image_features = self.image_branch.classifier[1].in_features
        self.image_branch.classifier = nn.Linear(num_image_features, 128)

        # --- Tabular Branch (ANN/MLP) ---
        # This is a simple multi-layer perceptron for our metadata.
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout helps prevent overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # --- Fusion and Final Classifier Head ---
        # This part combines the outputs from both branches.
        self.fusion = nn.Linear(128 + 32, 64) # 128 from image branch + 32 from tabular branch
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1) # Final output: a single logit for binary classification
        )

    def forward(self, image, tabular):
        # This defines the data flow through the model.
        
        # 1. Process inputs through their respective branches
        image_features = self.image_branch(image)
        tabular_features = self.tabular_branch(tabular)
        
        # 2. Concatenate (fuse) the feature vectors from both branches
        combined_features = torch.cat([image_features, tabular_features], dim=1)
        
        # 3. Pass the fused features through the final classifier layers
        fused = self.fusion(combined_features)
        output = self.classifier(fused)
        
        return output

# --- Instantiate the Model ---
# To create the model, we first need to know the exact number of features our tabular preprocessor creates.
num_tab_features = preprocessor.transform(train_df.head(1)[numerical_features + categorical_features]).shape[1]

# Now, create an instance of our model and move it to the correct device (CPU or GPU)
model = MultimodalMelanomaNet(num_tab_features).to(DEVICE)

# Optional: Print the model architecture to verify it's correct
print("--- Model Architecture ---")
print(model)

print("\nPart 3 Complete: The multimodal model architecture has been defined.")


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm # Import tqdm for progress bars

# --- Part 4: Training, Evaluation, and Visualization ---

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs):
    """
    Function to handle the training and validation of the model.
    """
    # Dictionary to store metrics for each epoch
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_auc': [], 'val_auc': []
    }

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        # Wrap train_loader with tqdm for a progress bar
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for images, tabular_data, labels in train_loop:
            images, tabular_data, labels = images.to(DEVICE), tabular_data.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images, tabular_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_loop.set_postfix(loss=loss.item())

        epoch_train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # --- Evaluation Phase ---
        model.eval()
        all_preds = {'train': [], 'val': []}
        all_labels = {'train': [], 'val': []}
        val_loss = 0.0

        with torch.no_grad():
            # Evaluate on training set to get training accuracy and AUC
            for images, tabular_data, labels in train_loader:
                images, tabular_data = images.to(DEVICE), tabular_data.to(DEVICE)
                outputs = model(images, tabular_data)
                preds_proba = torch.sigmoid(outputs).cpu().numpy()
                all_preds['train'].extend(preds_proba)
                all_labels['train'].extend(labels.cpu().numpy())

            # Evaluate on validation set
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for images, tabular_data, labels in val_loop:
                images, tabular_data, labels_dev = images.to(DEVICE), tabular_data.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(images, tabular_data)
                loss = criterion(outputs, labels_dev)
                val_loss += loss.item() * images.size(0)
                preds_proba = torch.sigmoid(outputs).cpu().numpy()
                all_preds['val'].extend(preds_proba)
                all_labels['val'].extend(labels.cpu().numpy())
                val_loop.set_postfix(loss=loss.item())

        # Calculate and store metrics for the epoch
        epoch_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)

        train_preds_binary = (np.array(all_preds['train']) > 0.5).astype(int)
        val_preds_binary = (np.array(all_preds['val']) > 0.5).astype(int)

        history['train_acc'].append(accuracy_score(all_labels['train'], train_preds_binary))
        history['val_acc'].append(accuracy_score(all_labels['val'], val_preds_binary))
        history['train_auc'].append(roc_auc_score(all_labels['train'], all_preds['train']))
        history['val_auc'].append(roc_auc_score(all_labels['val'], all_preds['val']))

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | Train Acc: {history['train_acc'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f} | Val AUC: {history['val_auc'][-1]:.4f}")
    
    print("\nFinished Training.")
    return history, all_labels['val'], val_preds_binary

# --- Define Loss Function and Optimizer ---
pos_weight = len(train_df[train_df['target'] == 0]) / len(train_df[train_df['target'] == 1])
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# --- Execute Training ---
EPOCHS = 5
history, final_val_labels, final_val_preds = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=EPOCHS)

# --- Plotting Training Curves ---
def plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    epochs_range = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs_range, history['train_loss'], 'o-', label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], 'o-', label='Validation Loss')
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, history['train_acc'], 'o-', label='Train Accuracy')
    ax2.plot(epochs_range, history['val_acc'], 'o-', label='Validation Accuracy')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.show()

plot_training_curves(history)

# --- Final Evaluation ---
print("\n--- Final Model Evaluation on Validation Set ---")
print("\nClassification Report:")
print(classification_report(final_val_labels, final_val_preds, target_names=['Benign', 'Malignant']))

print("Confusion Matrix:")
cm = confusion_matrix(final_val_labels, final_val_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# --- Exporting the Model ---
MODEL_EXPORT_PATH = 'multimodal_melanoma_model.pth'
torch.save(model.state_dict(), MODEL_EXPORT_PATH)
print(f"\nModel state dictionary saved to: {MODEL_EXPORT_PATH}")







