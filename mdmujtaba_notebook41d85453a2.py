#resnet18+lsb dt+train code
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Custom Dataset for Stego and Cover Images
class StegoDataset(Dataset):
    def __init__(self, cover_dir, stego_dir, transform=None):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.transform = transform
        self.cover_images = [os.path.join(cover_dir, f) for f in os.listdir(cover_dir) if f.endswith(('.png', '.jpg'))]
        self.stego_images = [os.path.join(stego_dir, f) for f in os.listdir(stego_dir) if f.endswith(('.png', '.jpg'))]
        self.all_images = self.cover_images + self.stego_images
        self.labels = [0] * len(self.cover_images) + [1] * len(self.stego_images)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Data Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Dataset and DataLoader
cover_dir = '/kaggle/input/lsb-stego/lsb_stego/lsb_stego/cover'
stego_dir = '/kaggle/input/lsb-stego/lsb_stego/lsb_stego/lsb'
dataset = StegoDataset(cover_dir, stego_dir, transform=data_transforms['train'])

# Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = data_transforms['val']

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Load Pretrained ResNet18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification (cover vs stego)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training Loop
num_epochs = 40
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    scheduler.step()

# Save the Model
model_save_path = '/kaggle/working/stego_model.pth'
torch.save(model.state_dict(), model_save_path)

# Evaluation Metrics
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
cm = confusion_matrix(all_labels, all_preds)

# Generate Report
report = f"""
# Steganalysis Model Report

## Training Summary
- **Epochs Trained**: {num_epochs}
- **Final Training Accuracy**: {train_accuracies[-1]:.2f}%
- **Final Validation Accuracy**: {val_accuracies[-1]:.2f}%
- **Final Training Loss**: {train_losses[-1]:.4f}
- **Final Validation Loss**: {val_losses[-1]:.4f}

## Evaluation Metrics (Validation Set)
- **Accuracy**: {val_accuracies[-1]:.2f}%
- **Precision**: {precision:.4f}
- **Recall**: {recall:.4f}
- **F1 Score**: {f1:.4f}

## Confusion Matrix
|                | Predicted Cover | Predicted Stego |
|----------------|-----------------|-----------------|
| Actual Cover   | {cm[0,0]}       | {cm[0,1]}       |
| Actual Stego   | {cm[1,0]}       | {cm[1,1]}       |

## Model Saved
- **Path**: {model_save_path}
"""

# Plot Training and Validation Metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('/kaggle/working/training_plots.png')

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cover', 'Stego'], yticklabels=['Cover', 'Stego'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('/kaggle/working/confusion_matrix.png')

# Save Report to File
report_file = '/kaggle/working/stego_report.md'
with open(report_file, 'w') as f:
    f.write(report)

print(f"Training completed. Model saved at {model_save_path}. Report saved at {report_file}.")
print(report)


#test code for the lsb trained model 
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define image transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the saved model
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to classify a single image
def classify_image(model, image_path, device):
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item(), image_path

# Main function to process directory and output results
def analyze_directory(input_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path).to(device)
    
    # Get list of images
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))]
    stego_images = []
    cover_images = []
    
    # Classify each image
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        prediction, _ = classify_image(model, img_path, device)
        if prediction == 1:  # Stego
            stego_images.append(img_file)
        else:  # Cover
            cover_images.append(img_file)
    
    # Print results
    total_images = len(image_files)
    print(f"Total Images Processed: {total_images}")

    num_stego = len(stego_images)
    num_cover = len(cover_images)
    
    print(f"Stego Images Detected: {num_stego}")
    print(f"Cover Images Detected: {num_cover}")
    print(f"Stego Detection Rate: {(num_stego / total_images * 100):.2f}%")

# Example usage
input_directory = '/kaggle/input/lsb-stego/lsb_stego/lsb_stego/lsb'  # Replace with your input directory
model_path = '/kaggle/working/stego_model.pth'
analyze_directory(input_directory, model_path)


#Train-->resnet18+lsb dataset+nsgaIII
#Install necessary libraries
!pip install pymoo==0.6.1.1 -q
!pip install tqdm -q

import os
import time
import torch
import gc # Import the garbage collector
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from collections import Counter
from tqdm.auto import tqdm
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback

# --- 1. Configuration ---
print("--- 1. CONFIGURATION ---")
DATA_DIR = "/kaggle/working/data1" # Example Kaggle input path
SAVE_DIR = "/kaggle/working/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_EPOCHS_FINAL, PATIENCE = 20, 5
RANDOM_SEED = 42
POP_SIZE, N_GEN, NUM_EPOCHS_OPT = 12, 10, 5
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"Device: {DEVICE}, Data Dir: {DATA_DIR}, Save Dir: {SAVE_DIR}\n" + "-"*30 + "\n")

# --- 2. Data Preparation ---
# (This section is unchanged)
print("--- 2. DATA PREPARATION ---")
transform_train = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
val_dataset = datasets.ImageFolder(DATA_DIR, transform=transform_train) # Use same transform for simplicity
full_dataset = val_dataset
print(f"Classes found: {full_dataset.classes}")
dataset_size = len(full_dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)
split = int(np.floor(0.2 * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
print(f"Total: {dataset_size} | Train: {len(train_indices)} | Val: {len(val_indices)}\n" + "-"*30 + "\n")


# --- 3. Model & Helper Functions ---
# (This section is unchanged)
print("--- 3. MODEL & HELPER FUNCTIONS ---")
def create_model(dropout_rate=0.5):
    model = models.resnet18(weights="IMAGENET1K_V1"); [p.requires_grad_(False) for p in model.parameters()]; [p.requires_grad_(True) for p in model.layer4.parameters()]
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 512), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(512, 2))
    return model.to(DEVICE)
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience):
    best_val_acc = 0.0; best_model_wts = None; patience_counter = 0; start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE); optimizer.zero_grad()
            outputs = model(inputs); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
        model.eval()
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"); val_correct_preds = 0
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs); _, preds = torch.max(outputs, 1); val_correct_preds += torch.sum(preds == labels.data)
        val_acc = val_correct_preds.double() / len(val_loader.sampler)
        if val_acc > best_val_acc:
            best_val_acc = val_acc; best_model_wts = model.state_dict().copy(); patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience: break
    if best_model_wts: model.load_state_dict(best_model_wts)
    return model, best_val_acc.item(), time.time() - start_time
def evaluate_model(model, loader):
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE); outputs = model(images)
            _, predicted = torch.max(outputs, 1); all_preds.extend(predicted.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
    return accuracy, precision, recall, f1
print("Helpers defined.\n" + "-"*30 + "\n")

# --- 4. Hyperparameter Optimization (NSGA-III) ---
print("--- 4. HYPERPARAMETER OPTIMIZATION (NSGA-III) ---")

class HyperparameterOptimization(ElementwiseProblem):
    def __init__(self, save_dir): # CHANGED: Pass save_dir
        super().__init__(n_var=4, n_obj=3, n_constr=0, xl=np.array([1e-5, 0.1, 1e-6, 16]), xu=np.array([1e-2, 0.6, 1e-3, 64]))
        # NEW: Track best accuracy and provide the save directory
        self.best_acc_so_far = 0.0
        self.save_dir = save_dir

    def _evaluate(self, x, out, *args, **kwargs):
        lr, dropout, weight_decay, batch_size = x
        batch_size = int(round(batch_size))
        print(f"\nEvaluating: LR={lr:.6f}, Dropout={dropout:.3f}, WD={weight_decay:.6f}, BS={batch_size}")
        try:
            train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0)
            model = create_model(dropout_rate=dropout)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_OPT)
            model, val_acc, training_time = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS_OPT, patience=3)
            _, _, _, f1 = evaluate_model(model, val_loader)
            out["F"] = [-val_acc, -f1, training_time]
            print(f"Result: ValAcc={val_acc:.4f}, F1={f1:.4f}, Time={training_time:.2f}s")
            
            # NEW: Check if this is the best model so far and save it
            if val_acc > self.best_acc_so_far:
                self.best_acc_so_far = val_acc
                print(f"ğŸ�‰ New best validation accuracy: {val_acc:.4f}. Saving model to 'best_model_during_opt.pth'")
                torch.save(model.state_dict(), os.path.join(self.save_dir, "best_model_during_opt.pth"))

        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            out["F"] = [1.0, 1.0, 1e6]
        finally:
            if 'model' in locals(): del model
            if 'optimizer' in locals(): del optimizer
            if 'criterion' in locals(): del criterion
            if 'train_loader' in locals(): del train_loader
            if 'val_loader' in locals(): del val_loader
            gc.collect()
            torch.cuda.empty_cache()

# We no longer need the SaveCallback for hyperparameters, but you can keep it if you want
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=3)
algorithm = NSGA3(pop_size=POP_SIZE, ref_dirs=ref_dirs)

# CHANGED: Instantiate the problem with the save directory
problem = HyperparameterOptimization(save_dir=SAVE_DIR)

res = minimize(
    problem, # Use the problem instance
    algorithm,
    termination=('n_gen', N_GEN),
    seed=RANDOM_SEED,
    verbose=True,
    save_history=True
)

print("\nOptimization finished.\n" + "-"*30 + "\n")

# --- 5. Final Analysis and Training ---
# The rest of the script remains the same. It will still train a final model for 20 epochs
# which is the recommended approach. The 'best_model_during_opt.pth' serves as a great backup.
# ... (rest of the script is unchanged) ...
print("--- 5. ANALYSIS & FINAL TRAINING ---")
np.save(os.path.join(SAVE_DIR, 'nsga3_final_X.npy'), res.X)
np.save(os.path.join(SAVE_DIR, 'nsga3_final_F.npy'), res.F)
sorted_indices = np.lexsort((-res.F[:, 1], -res.F[:, 0]))
best_idx = sorted_indices[0]
best_hyperparams = res.X[best_idx]
print(f"Best Hyperparameters Found: {best_hyperparams}")
best_lr, best_dropout, best_wd, best_bs = best_hyperparams
final_train_loader = DataLoader(full_dataset, batch_size=int(round(best_bs)), sampler=train_sampler, num_workers=0)
final_val_loader = DataLoader(val_dataset, batch_size=int(round(best_bs)), sampler=val_sampler, num_workers=0)
final_model = create_model(dropout_rate=best_dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_lr, weight_decay=best_wd)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_FINAL)
final_model, best_val_acc, _ = train_model(final_model, final_train_loader, final_val_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS_FINAL, patience=PATIENCE)
model_save_path = os.path.join(SAVE_DIR, "final_trained_model.pth")
torch.save({'model_state_dict': final_model.state_dict(), 'best_val_acc': best_val_acc, 'hyperparameters': best_hyperparams,}, model_save_path)
print(f"\nFinal model saved to {model_save_path}\n" + "-"*30 + "\n")

# --- 6. Final Evaluation & Visualization ---
print("--- 6. FINAL EVALUATION ---")
accuracy, precision, recall, f1 = evaluate_model(final_model, final_val_loader)
print(f"Accuracy:  {accuracy:.4f}%\nPrecision: {precision:.4f}\nRecall:    {recall:.4f}\nF1-Score:  {f1:.4f}")
def plot_confusion_matrix(model, loader, class_names):
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE); outputs = model(images)
            _, predicted = torch.max(outputs, 1); all_preds.extend(predicted.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.show()
plot_confusion_matrix(final_model, final_val_loader, full_dataset.classes)


#Testing on alaska dt using the previously trained model

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm

# --- 1. CONFIGURATION ---
# IMPORTANT: Update these paths before running!
MODEL_PATH = "/kaggle/input/rasheed-models/final_trained_model.pth"  # Path to your saved model
TEST_DIR = "/kaggle/input/alaska2-image-steganalysis/Cover"  # Path to the folder with images you want to test
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------

# Define the class names based on your training setup
# This assumes 'clean' was the first folder alphabetically, and 'stego' was the second.
CLASS_NAMES = ['clean', 'stego']


# --- 2. MODEL DEFINITION ---
# This function must be identical to the one used for training to ensure the architecture matches the saved weights.
def create_model(dropout_rate=0.5):
    """Creates a ResNet-18 model with a custom classifier head."""
    model = models.resnet18(weights="IMAGENET1K_V1") # Using pre-trained weights is fine, they get overwritten by your state_dict
    
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the final convolutional block (layer4) for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, 2) # 2 classes: clean vs stego
    )
    return model.to(DEVICE)


# --- 3. PREDICTION SCRIPT ---
def test_directory(model_path, test_dir):
    """
    Loads a model and predicts classes for all images in a directory.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    print("--- Starting Prediction ---")
    print(f"Using device: {DEVICE}")

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)    
    # Extract hyperparameters to recreate the model architecture
    # The key here is the dropout rate
    hyperparameters = checkpoint['hyperparameters']
    dropout_rate = hyperparameters[1] 
    
    # Create model instance and load the trained weights
    model = create_model(dropout_rate=dropout_rate)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set model to evaluation mode

    # Define the image transformations (must be same as validation transforms)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize counters
    prediction_counts = {class_name: 0 for class_name in CLASS_NAMES}
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print("No image files found in the test directory.")
        return

    # Loop through all images and predict
    for image_name in tqdm(image_files, desc="Predicting on test images"):
        image_path = os.path.join(test_dir, image_name)
        try:
            # Open and preprocess the image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension and send to device

            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted_idx = torch.max(outputs, 1)
            
            # Record the prediction
            predicted_class_name = CLASS_NAMES[predicted_idx.item()]
            prediction_counts[predicted_class_name] += 1

        except Exception as e:
            print(f"Could not process {image_name}. Error: {e}")

    # --- 4. REPORT RESULTS ---
    print("\n--- Prediction Complete ---")
    total_images = len(image_files)
    print(f"Total images processed: {total_images}")
    print(f"Number of 'clean' images predicted: {prediction_counts['clean']}")
    print(f"Number of 'stego' images predicted: {prediction_counts['stego']}")
    print("---------------------------\n")


# --- RUN THE TEST ---
test_directory(MODEL_PATH, TEST_DIR)


##nsga3-jpg test code
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Custom Dataset for loading images
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, label_fn=None):
        self.image_dir = image_dir
        self.transform = transform
        self.label_fn = label_fn
        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.label_fn:
            label = self.label_fn(img_path)
            return image, label, img_path
        return image, img_path

# CNN Model matching the saved model's architecture
class StegoClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(StegoClassifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        return self.resnet(x)

def evaluate_model(image_dir, model_path, dataset_name, label_fn=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = ImageDataset(image_dir, transform=transform, label_fn=label_fn)
    if len(dataset) == 0:
        print(f"No images found in {image_dir}.")
        return 0, 0, []
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model (dropout set to 0.5; adjust if known from training)
    model = StegoClassifier(dropout=0.5).to(device)
    
    # Load pretrained weights
    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Remap state dictionary keys
            new_state_dict = {}
            for key, value in state_dict.items():
                if 'num_batches_tracked' in key:
                    continue
                # Add 'resnet.' prefix to match StegoClassifier
                if key.startswith('fc.'):
                    new_key = 'resnet.' + key
                else:
                    new_key = 'resnet.' + key
                new_state_dict[new_key] = value
            
            model.load_state_dict(new_state_dict, strict=True)
            print(f"Loaded model from {model_path}.")
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            print("Cannot proceed without a valid model.")
            return 0, 0, []
    else:
        print(f"Model path {model_path} not found. Cannot proceed.")
        return 0, 0, []
    
    model.eval()
    
    stego_count = 0
    clean_count = 0
    all_preds = []
    all_confidences = []
    all_labels = [] if label_fn else None
    
    try:
        with torch.no_grad():
            for batch in dataloader:
                if label_fn:
                    images, labels, _ = batch
                    images, labels = images.to(device, non_blocking=True), labels.to(device)
                    all_labels.extend(labels.cpu().numpy())
                else:
                    images, _ = batch
                    images = images.to(device, non_blocking=True)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                stego_count += (predicted == 1).sum().item()
                clean_count += (predicted == 0).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
    except KeyboardInterrupt:
        print("Inference interrupted. Returning partial results.")
    
    # Print results
    total_images = stego_count + clean_count
    print(f"\n{dataset_name} Results:")
    print(f"Number of stego images detected: {stego_count}")
    print(f"Number of clean images detected: {clean_count}")
    print(f"Percentage of stego images: {(stego_count / total_images * 100) if total_images > 0 else 0:.2f}%")
    print(f"Average confidence for predictions: {np.mean(all_confidences):.4f}")
    
    # Plot confidence distribution
    plt.figure(figsize=(8, 4))
    plt.hist(all_confidences, bins=20, range=(0, 1), color='blue', alpha=0.7)
    plt.title(f'Confidence Distribution - {dataset_name}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.show()
    
    # Plot confusion matrix if labels are available
    if all_labels:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['clean', 'stego'], yticklabels=['clean', 'stego'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
    return stego_count, clean_count, all_preds

# Example label function (modify based on your naming convention)
def label_fn(img_path):
    # Example: Assume filenames contain 'stego' for stego images, else clean
    return 1 if 'stego' in os.path.basename(img_path).lower() else 0

# Example usage
if __name__ == "__main__":
    model_path = "/kaggle/input/rasheed-models/nsga3.pth"
    lsb_test_dir = "/kaggle/input/lsb-stego/lsb_stego/lsb_stego/lsb"
    
    # Evaluate on LSB test set
    if os.path.exists(lsb_test_dir):
        # Set label_fn if you have a way to infer labels (e.g., filename patterns)
        # Otherwise, use label_fn=None
        lsb_stego, lsb_clean, lsb_preds = evaluate_model(lsb_test_dir, model_path, "LSB Test", label_fn=None)
    else:
        print(f"LSB test directory {lsb_test_dir} not found.")
        lsb_stego, lsb_clean, lsb_preds = 0, 0, []


#resnet18+44k png imagedt+train code
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import re
from collections import Counter
from tqdm.auto import tqdm

# --- 1. CONFIGURATION ---
torch.manual_seed(42); torch.cuda.manual_seed_all(42); np.random.seed(42)
data_dir = "/kaggle/input/stegoimagesdataset/train/train"
save_dir = "/kaggle/working/steg_model_from_scratch"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(save_dir, exist_ok=True)
print(f"Device: {device}, Data Dir: {data_dir}, Save Dir: {save_dir}\n" + "-"*30 + "\n")

# --- 2. DATA PREPARATION ---
print("--- 2. DATA PREPARATION ---")
full_dataset = datasets.ImageFolder(data_dir)
def create_disjoint_split(dataset, test_split=0.2):
    base_name_map = {}
    for idx, (path, _) in enumerate(dataset.samples):
        base_name = re.split(r'[._]', os.path.basename(path))[0]
        if base_name not in base_name_map: base_name_map[base_name] = []
        base_name_map[base_name].append(idx)
    unique_base_names = list(base_name_map.keys())
    np.random.shuffle(unique_base_names)
    split_idx = int(np.floor(test_split * len(unique_base_names)))
    val_names, train_names = unique_base_names[:split_idx], unique_base_names[split_idx:]
    train_indices = [idx for name in train_names for idx in base_name_map[name]]
    val_indices = [idx for name in val_names for idx in base_name_map[name]]
    return train_indices, val_indices

train_idx, val_idx = create_disjoint_split(full_dataset)
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

class_counts = Counter(np.array(full_dataset.targets)[train_idx])
weights = torch.tensor([len(train_idx) / class_counts[i] for i in range(len(class_counts))], dtype=torch.float32).to(device)
print(f"Loss weights: {weights}\n" + "-"*30 + "\n")

# --- 3. TRANSFORMS & DATALOADERS ---
print("--- 3. TRANSFORMS & DATALOADERS ---")
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset.transform = transform_train
val_dataset = datasets.ImageFolder(data_dir, transform=transform_val)
train_loader = DataLoader(full_dataset, batch_size=64, sampler=train_sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, sampler=val_sampler, num_workers=2, pin_memory=True)
print("DataLoaders created.\n" + "-"*30 + "\n")

# --- 4. MODEL & TRAINING ---
print("--- 4. MODEL & TRAINING ---")

# --- MODEL SETUP CHANGED ---
# 1. Initialize with weights=None to train from scratch
# 2. Make all parameters trainable by removing the freezing loops
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512), nn.ReLU(),
    nn.Dropout(0.5), nn.Linear(512, 2)
)
model = model.to(device)
# --- END OF CHANGES ---

def train_model_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, patience=5):
    best_val_acc = 0.0
    best_model_wts = model.state_dict()
    for epoch in range(num_epochs):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Get final training accuracy for the epoch
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        train_acc = correct / total

        # Validation
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            print(f"ğŸš€ New best validation accuracy: {best_val_acc:.4f}. Saving model.")
            torch.save({'model_state_dict': best_model_wts}, os.path.join(save_dir, "steganalyzer_from_scratch.pth"))
        
        scheduler.step()
    
    model.load_state_dict(best_model_wts)
    return model, best_val_acc

# --- OPTIMIZER CHANGED ---
# Use a slightly higher learning rate, which is common when training from scratch
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001) # Higher LR
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

model, val_acc = train_model_loop(model, train_loader, val_loader, criterion, optimizer, scheduler)

print("\n--- Training Complete ---")
print(f"Final best validation accuracy: {val_acc:.4f}")
print(f"Model saved at {os.path.join(save_dir, 'steganalyzer_from_scratch.pth')}")


#test code for the resnet18+png trained model
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm.auto import tqdm

# --- 1. CONFIGURATION ---
# IMPORTANT: Update these paths before running
MODEL_PATH = "/kaggle/working/steg_model_from_scratch/steganalyzer_from_scratch.pth"
TEST_DIR   = "/kaggle/input/stegoimagesdataset/train/train/stego"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['clean', 'stego'] # 'clean' is class 0, 'stego' is 1
# -------------------------


# --- 2. MODEL DEFINITION ---
# This function must EXACTLY match the architecture of the saved model
def create_resnet18_model(dropout_rate=0.5):
    """Creates the ResNet-18 architecture used during training."""
    # Start with an untrained ResNet-18 architecture
    model = models.resnet18(weights=None)
    
    # Re-create the custom fully-connected (fc) layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, 2)
    )
    return model.to(DEVICE)


# --- 3. PREDICTION SCRIPT ---
def analyze_directory(model_path, test_dir):
    """
    Loads a model and predicts the percentage of stego images in a directory.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return

    print("--- Starting Analysis ---")

    # --- Load Model ---
    model = create_resnet18_model(dropout_rate=0.5)
    # The saved file is a dictionary, so we load the 'model_state_dict' key
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set model to evaluation mode
    print(f"Model loaded successfully from {model_path}")

    # --- Define Image Transformations ---
    # These must match the validation transforms from your training script
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Predict on all images ---
    stego_count = 0
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)

    if total_images == 0:
        print("No valid image files found in the directory.")
        return

    for fname in tqdm(image_files, desc=f"Analyzing images in {os.path.basename(test_dir)}"):
        image_path = os.path.join(test_dir, fname)
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(tensor)
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # Get the prediction index (0 for 'clean', 1 for 'stego')
                _, pred_idx = torch.max(outputs, 1)

            if pred_idx.item() == 1: # Class 1 is 'stego'
                stego_count += 1
        except Exception as e:
            print(f"Skipping {fname} due to error: {e}")
    
    # --- Report Results ---
    stego_percentage = (stego_count / total_images) * 100 if total_images > 0 else 0
    
    print("\n--- Analysis Complete ---")
    print(f"Total images processed: {total_images}")
    print(f"Images predicted as 'stego': {stego_count}")
    print(f"Percentage of stego images: {stego_percentage:.2f}%")
    print("--------------------------\n")

# --- RUN THE ANALYSIS ---
analyze_directory(MODEL_PATH, TEST_DIR)


##png model train code

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

# Config
data_dir = "/kaggle/input/stegoimagesdataset/train/train"
save_dir = "/kaggle/working/train_stego_pvd"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(save_dir, exist_ok=True)

# Verify dataset integrity
def verify_images(data_dir):
    corrupt_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()
            except Exception:
                corrupt_files.append(os.path.join(root, file))
    return corrupt_files

print("Checking dataset integrity...")
corrupt_files = verify_images(data_dir)
print(f"Corrupt files found: {len(corrupt_files)}")
if corrupt_files:
    print(corrupt_files[:5])
else:
    print("No corrupt files found.")

# Verify dataset
dataset = datasets.ImageFolder(data_dir)
print("Dataset:", os.listdir(data_dir))
print("Classes:", dataset.class_to_idx)

# Visualize sample images
def denormalize(image):
    image = image.clone()
    for i in range(3):
        image[i] = image[i] * 0.229 + 0.485
    return image

transform_vis = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset_vis = datasets.ImageFolder(data_dir, transform=transform_vis)
vis_loader = DataLoader(dataset_vis, batch_size=4, shuffle=True, num_workers=0)
for images, labels in vis_loader:
    plt.figure(figsize=(8, 3))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        img = denormalize(images[i])
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f"{'stego' if labels[i].item() == 1 else 'cover'}")
        plt.axis('off')
    plt.show()
    plt.close()
    break

# Transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset setup
dataset = datasets.ImageFolder(data_dir, transform=transform_train)
dataset_val = datasets.ImageFolder(data_dir, transform=transform_val)
dataset_size = len(dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)
split = int(np.floor(0.2 * dataset_size))
train_idx, val_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler, num_workers=0)
val_loader = DataLoader(dataset_val, batch_size=64, sampler=val_sampler, num_workers=0)

# Model setup
model = models.resnet18(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)
model = model.to(device)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, patience=10):
    best_val_acc = 0.0
    best_model_wts = None
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_loss = val_loss / total
        val_acc = correct / total
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    model.load_state_dict(best_model_wts)
    return model, best_val_acc

# Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
model, val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc}, os.path.join(save_dir, "stego_pvd_trained_resnet18.pth"))

# Verify
print(f"Trained model saved at {os.path.join(save_dir, 'stego_pvd_trained_resnet18.pth')}")




#test code for png
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Custom Dataset for loading images
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

# CNN Model matching the saved model's architecture
class StegoClassifier(nn.Module):
    def __init__(self):
        super(StegoClassifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        return self.resnet(x)

def evaluate_model(image_dir, model_path, dataset_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = ImageDataset(image_dir, transform=transform)
    if len(dataset) == 0:
        print(f"No images found in {image_dir}.")
        return 0, 0, []
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = StegoClassifier().to(device)
    
    # Load pretrained weights
    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Remap state dictionary keys to match model structure
            new_state_dict = {}
            for key, value in state_dict.items():
                if 'num_batches_tracked' in key:
                    continue  # Skip non-parameter keys
                if key.startswith('fc.0.'):
                    new_key = key.replace('fc.0.', 'resnet.fc.0.')
                elif key.startswith('fc.3.'):
                    new_key = key.replace('fc.3.', 'resnet.fc.3.')
                else:
                    new_key = 'resnet.' + key
                new_state_dict[new_key] = value
            
            model.load_state_dict(new_state_dict, strict=True)
            print(f"Loaded model from {model_path}.")
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            print("Cannot proceed without a valid model.")
            return 0, 0, []
    else:
        print(f"Model path {model_path} not found. Cannot proceed.")
        return 0, 0, []
    
    model.eval()
    
    stego_count = 0
    clean_count = 0
    all_preds = []
    all_confidences = []
    
    try:
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                stego_count += (predicted == 1).sum().item()
                clean_count += (predicted == 0).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
    except KeyboardInterrupt:
        print("Inference interrupted. Returning partial results.")
    
    # Print results
    total_images = stego_count + clean_count
    print(f"\n{dataset_name} Results:")
    print(f"Number of stego images detected: {stego_count}")
    print(f"Number of clean images detected: {clean_count}")
    print(f"Percentage of stego images: {(stego_count / total_images * 100) if total_images > 0 else 0:.2f}%")
    print(f"Average confidence for predictions: {np.mean(all_confidences):.4f}")
    
    # Plot confidence distribution
    plt.figure(figsize=(8, 4))
    plt.hist(all_confidences, bins=20, range=(0, 1), color='blue', alpha=0.7)
    plt.title(f'Confidence Distribution - {dataset_name}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.show()
    
    return stego_count, clean_count, all_preds

# Example usage
if __name__ == "__main__":
    model_path = "/kaggle/input/rasheed-models/stego_pvd_trained_resnet18.pth"
    lsb_test_dir = "/kaggle/input/stego-pvd-dataset/Stego-pvd-dataset/test/stegoTest"
    
    # Evaluate on LSB test set
    if os.path.exists(lsb_test_dir):
        lsb_stego, lsb_clean, lsb_preds = evaluate_model(lsb_test_dir, model_path, "LSB Test")
    else:
        print(f"LSB test directory {lsb_test_dir} not found.")
        lsb_stego, lsb_clean, lsb_preds = 0, 0, []



#Dataset Aggregation and Organization Script

import os
import shutil
from pathlib import Path

# Define output directory structure
output_base_dir = '/kaggle/working/stego_dataset'
clean_base_dir = os.path.join(output_base_dir, 'clean')
stego_base_dir = os.path.join(output_base_dir, 'stego')
subfolders = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

# Create directories
for subfolder in subfolders:
    os.makedirs(os.path.join(clean_base_dir, subfolder), exist_ok=True)
    os.makedirs(os.path.join(stego_base_dir, subfolder), exist_ok=True)

# Function to copy images to a specific target directory
def copy_images(src_path, dst_dir, num_images, start_idx=0, match_filenames=None):
    copied = 0
    image_files = sorted([f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # If matching filenames are provided, filter by those
    if match_filenames:
        image_files = [f for f in image_files if f in match_filenames]
    
    # Select images starting from start_idx
    for img_file in image_files[start_idx:start_idx + num_images]:
        if copied >= num_images:
            break
        src_file = os.path.join(src_path, img_file)
        dst_file = os.path.join(dst_dir, img_file)
        shutil.copy2(src_file, dst_file)
        copied += 1
    
    return copied

# 1. LSB Dataset: 100 cover and 100 stego with matching filenames to clean/a and stego/a
lsb_cover_dir = '/kaggle/input/lsb-stego/lsb_stego/lsb_stego/cover'
lsb_stego_dir = '/kaggle/input/lsb-stego/lsb_stego/lsb_stego/lsb'
lsb_clean_dst = os.path.join(clean_base_dir, 'a')
lsb_stego_dst = os.path.join(stego_base_dir, 'a')

# Get 100 cover images
cover_files = sorted([f for f in os.listdir(lsb_cover_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:100]
copied_cover = copy_images(lsb_cover_dir, lsb_clean_dst, 100, match_filenames=cover_files)
copied_stego = copy_images(lsb_stego_dir, lsb_stego_dst, 100, match_filenames=cover_files)

print(f"Copied {copied_cover} cover images to {lsb_clean_dst}")
print(f"Copied {copied_stego} stego images to {lsb_stego_dst}")

# 2. StegoImagesDataset: 300 clean and 300 stego to clean/b and stego/b
val_clean_dir = '/kaggle/input/stegoimagesdataset/val/val/clean'
val_stego_dir = '/kaggle/input/stegoimagesdataset/val/val/stego'
val_clean_dst = os.path.join(clean_base_dir, 'b')
val_stego_dst = os.path.join(stego_base_dir, 'b')

copied_clean = copy_images(val_clean_dir, val_clean_dst, 300)
copied_stego = copy_images(val_stego_dir, val_stego_dst, 300)

print(f"Copied {copied_clean} clean images to {val_clean_dst}")
print(f"Copied {copied_stego} stego images to {val_stego_dst}")

# 3. Stego-PVD Dataset: 300 clean and 300 stego to clean/c and stego/c
pvd_clean_dir = '/kaggle/input/stego-pvd-dataset/Stego-pvd-dataset/train/cleanTrain'
pvd_stego_dir = '/kaggle/input/stego-pvd-dataset/Stego-pvd-dataset/train/stegoTrain'
pvd_clean_dst = os.path.join(clean_base_dir, 'c')
pvd_stego_dst = os.path.join(stego_base_dir, 'c')

copied_clean = copy_images(pvd_clean_dir, pvd_clean_dst, 300)
copied_stego = copy_images(pvd_stego_dir, pvd_stego_dst, 300)

print(f"Copied {copied_clean} clean images to {pvd_clean_dst}")
print(f"Copied {copied_stego} stego images to {pvd_stego_dst}")

# 4. ALASKA2 Dataset: First 2000 cover and JMiPOD to clean/d and stego/d
alaska_cover_dir = '/kaggle/input/alaska2-image-steganalysis/Cover'
alaska_jmipod_dir = '/kaggle/input/alaska2-image-steganalysis/JMiPOD'
alaska_clean_dst_d = os.path.join(clean_base_dir, 'd')
alaska_stego_dst_d = os.path.join(stego_base_dir, 'd')

copied_cover = copy_images(alaska_cover_dir, alaska_clean_dst_d, 2000, start_idx=0)
copied_stego = copy_images(alaska_jmipod_dir, alaska_stego_dst_d, 2000, start_idx=0)

print(f"Copied {copied_cover} cover images to {alaska_clean_dst_d}")
print(f"Copied {copied_stego} JMiPOD stego images to {alaska_stego_dst_d}")

# 5. ALASKA2 Dataset: Third 2000 cover and JUNIWARD to clean/e and stego/e
alaska_juniward_dir = '/kaggle/input/alaska2-image-steganalysis/JUNIWARD'
alaska_clean_dst_e = os.path.join(clean_base_dir, 'e')
alaska_stego_dst_e = os.path.join(stego_base_dir, 'e')

copied_cover = copy_images(alaska_cover_dir, alaska_clean_dst_e, 2000, start_idx=4000)
copied_stego = copy_images(alaska_juniward_dir, alaska_stego_dst_e, 2000, start_idx=4000)

print(f"Copied {copied_cover} cover images to {alaska_clean_dst_e}")
print(f"Copied {copied_stego} JUNIWARD stego images to {alaska_stego_dst_e}")

# 6. ALASKA2 Dataset: Fourth 2000 cover and UERD to clean/f and stego/f
alaska_uerd_dir = '/kaggle/input/alaska2-image-steganalysis/UERD'
alaska_clean_dst_f = os.path.join(clean_base_dir, 'f')
alaska_stego_dst_f = os.path.join(stego_base_dir, 'f')

copied_cover = copy_images(alaska_cover_dir, alaska_clean_dst_f, 2000, start_idx=6000)
copied_stego = copy_images(alaska_uerd_dir, alaska_stego_dst_f, 2000, start_idx=6000)

print(f"Copied {copied_cover} cover images to {alaska_clean_dst_f}")
print(f"Copied {copied_stego} UERD stego images to {alaska_stego_dst_f}")

# 7. Stego-Dataset: 400 cover and 400 F5 to clean/g and stego/g
stego_dataset_cover_dir = '/kaggle/input/stego-dataset/data_Set/train/cover'
stego_dataset_f5_dir = '/kaggle/input/stego-dataset/data_Set/train/f5'
stego_clean_dst = os.path.join(clean_base_dir, 'g')
stego_stego_dst = os.path.join(stego_base_dir, 'g')

copied_cover = copy_images(stego_dataset_cover_dir, stego_clean_dst, 400, start_idx=0)
copied_stego = copy_images(stego_dataset_f5_dir, stego_stego_dst, 400, start_idx=0)

print(f"Copied {copied_cover} cover images to {stego_clean_dst}")
print(f"Copied {copied_stego} F5 stego images to {stego_stego_dst}")

# Verify total images
total_clean = sum(len(os.listdir(os.path.join(clean_base_dir, sub))) for sub in subfolders)
total_stego = sum(len(os.listdir(os.path.join(stego_base_dir, sub))) for sub in subfolders)

print(f"\nTotal Clean Images Copied: {total_clean}")
print(f"Total Stego Images Copied: {total_stego}")
print(f"Images saved in: {output_base_dir}/clean/[a-g] and {output_base_dir}/stego/[a-g]")


#resnet18+ using the massive, combined dataset train code
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Custom Dataset for Clean and Stego Images
class StegoDataset(Dataset):
    def __init__(self, clean_dirs, stego_dirs, transform=None):
        self.clean_dirs = clean_dirs
        self.stego_dirs = stego_dirs
        self.transform = transform
        self.all_images = []
        self.labels = []

        # Load clean images (label 0)
        for clean_dir in clean_dirs:
            clean_images = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.all_images.extend(clean_images)
            self.labels.extend([0] * len(clean_images))

        # Load stego images (label 1)
        for stego_dir in stego_dirs:
            stego_images = [os.path.join(stego_dir, f) for f in os.listdir(stego_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.all_images.extend(stego_images)
            self.labels.extend([1] * len(stego_images))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Data Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Define directory paths
base_dir = '/kaggle/working/stego_dataset'
subfolders = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
clean_dirs = [os.path.join(base_dir, 'clean', sub) for sub in subfolders]
stego_dirs = [os.path.join(base_dir, 'stego', sub) for sub in subfolders]

# Create dataset
dataset = StegoDataset(clean_dirs, stego_dirs, transform=data_transforms['train'])

# Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = data_transforms['val']

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Load Pretrained ResNet18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification (cover vs stego)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training Loop
num_epochs = 32
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print("Starting Training...")
print("-" * 50)

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
    print("-" * 50)

    scheduler.step()

# Save the Model
model_save_path = '/kaggle/working/stego_model.pth'
torch.save(model.state_dict(), model_save_path)

# Evaluation Metrics
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
cm = confusion_matrix(all_labels, all_preds)

# Display Final Report
print("\nTraining Completed!")
print("=" * 50)
print("Final Steganalysis Model Report")
print("=" * 50)
print(f"Total Epochs Trained: {num_epochs}")
print(f"Total Images: {len(dataset)} (Train: {train_size}, Validation: {val_size})")
print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
print(f"Final Training Loss: {train_losses[-1]:.4f}")
print(f"Final Validation Loss: {val_losses[-1]:.4f}")
print("\nEvaluation Metrics (Validation Set):")
print(f"Accuracy: {val_accuracies[-1]:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(f"{'':<15} | {'Predicted Cover':<15} | {'Predicted Stego':<15}")
print(f"{'-'*15}-+-{'-'*15}-+-{'-'*15}")
print(f"{'Actual Cover':<15} | {cm[0,0]:<15} | {cm[0,1]:<15}")
print(f"{'Actual Stego':<15} | {cm[1,0]:<15} | {cm[1,1]:<15}")
print(f"\nModel Saved at: {model_save_path}")

# Plot Training and Validation Metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='#1f77b4')
plt.plot(val_losses, label='Val Loss', color='#ff7f0e')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='#1f77b4')
plt.plot(val_accuracies, label='Val Accuracy', color='#ff7f0e')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cover', 'Stego'], yticklabels=['Cover', 'Stego'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



#resnet18+alaska 2 dt train code
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torch.amp import GradScaler, autocast  # Updated for PyTorch >= 2.0

# Custom Dataset for ALASKA2 Images
class StegoDataset(Dataset):
    def __init__(self, cover_dir, stego_dirs, num_images_per_subset, transform=None):
        self.cover_dir = cover_dir
        self.stego_dirs = stego_dirs
        self.num_images_per_subset = num_images_per_subset
        self.transform = transform
        self.all_images = []
        self.labels = []

        # Load cover images (label 0) from Cover directory
        cover_files = sorted([f for f in os.listdir(cover_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # First 10k Cover
        self.all_images.extend([os.path.join(cover_dir, f) for f in cover_files[0:num_images_per_subset]])
        self.labels.extend([0] * min(num_images_per_subset, len(cover_files[0:num_images_per_subset])))
        
        # Second 10k Cover
        self.all_images.extend([os.path.join(cover_dir, f) for f in cover_files[25000:25000 + num_images_per_subset]])
        self.labels.extend([0] * min(num_images_per_subset, len(cover_files[25000:25000 + num_images_per_subset])))
        
        # Third 10k Cover
        self.all_images.extend([os.path.join(cover_dir, f) for f in cover_files[50000:50000 + num_images_per_subset]])
        self.labels.extend([0] * min(num_images_per_subset, len(cover_files[50000:50000 + num_images_per_subset])))

        # Load stego images (label 1) from JMiPOD, JUNIWARD, UERD
        for stego_dir, start_idx in zip(stego_dirs, [0, 25000, 50000]):
            stego_files = sorted([f for f in os.listdir(stego_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.all_images.extend([os.path.join(stego_dir, f) for f in stego_files[start_idx:start_idx + num_images_per_subset]])
            self.labels.extend([1] * min(num_images_per_subset, len(stego_files[start_idx:start_idx + num_images_per_subset])))

        # Verify balance
        print(f"Dataset: {len(self.all_images)} images ({sum(self.labels)} stego, {len(self.all_images) - sum(self.labels)} cover)")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image and label to avoid crashing
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            label = 0
        if self.transform:
            image = self.transform(image)
        return image, label

# Data Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Define directory paths
cover_dir = '/kaggle/input/alaska2-image-steganalysis/Cover'
stego_dirs = [
    '/kaggle/input/alaska2-image-steganalysis/JMiPOD',
    '/kaggle/input/alaska2-image-steganalysis/JUNIWARD',
    '/kaggle/input/alaska2-image-steganalysis/UERD'
]
num_images_per_subset = 10000  # 10k per subset for speed

# Create dataset
dataset = StegoDataset(cover_dir, stego_dirs, num_images_per_subset, transform=data_transforms['train'])

# Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_dataset.dataset.transform = data_transforms['val']

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Define Model
model = models.resnet18(pretrained=True)
# Unfreeze last two layer blocks (layer3 and layer4) and fc layer
for param in model.parameters():
    param.requires_grad = False
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification (cover vs stego)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
scaler = GradScaler('cuda')  # Updated for PyTorch >= 2.0

# Training Loop
num_epochs = 18
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print("Starting Training...")
print("-" * 50)

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast('cuda'):  # Updated for PyTorch >= 2.0
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast('cuda'):  # Updated for PyTorch >= 2.0
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
    print("-" * 50)

    scheduler.step()

# Save the Model
model_save_path = '/kaggle/working/stego_model.pth'
torch.save(model.state_dict(), model_save_path)

# Evaluation Metrics
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
cm = confusion_matrix(all_labels, all_preds)

# Display Final Report
print("\nTraining Completed!")
print("=" * 50)
print("Final Steganalysis Model Report")
print("=" * 50)
print(f"Total Epochs Trained: {num_epochs}")
print(f"Total Images: {len(dataset)} (Train: {train_size}, Validation: {val_size})")
print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Final Validation Accuracy: {val_accuracies[-1]:.2f}%")
print(f"Final Training Loss: {train_losses[-1]:.4f}")
print(f"Final Validation Loss: {val_losses[-1]:.4f}")
print("\nEvaluation Metrics (Validation Set):")
print(f"Accuracy: {val_accuracies[-1]:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(f"{'':<15} | {'Predicted Cover':<15} | {'Predicted Stego':<15}")
print(f"{'-'*15}-+-{'-'*15}-+-{'-'*15}")
print(f"{'Actual Cover':<15} | {cm[0,0]:<15} | {cm[0,1]:<15}")
print(f"{'Actual Stego':<15} | {cm[1,0]:<15} | {cm[1,1]:<15}")
print(f"\nModel Saved at: {model_save_path}")

# Plot Training and Validation Metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='#1f77b4')
plt.plot(val_losses, label='Val Loss', color='#ff7f0e')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='#1f77b4')
plt.plot(val_accuracies, label='Val Accuracy', color='#ff7f0e')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cover', 'Stego'], yticklabels=['Cover', 'Stego'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


#lr train code
import numpy as np
import os
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from PIL import Image
from tqdm import tqdm
import pywt


def extract_hog(img):
    return hog(img, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

def extract_lbp(img):
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
    return hist

def extract_bitplane_stats(img):
    bit_planes = [(img >> i) & 1 for i in range(8)]
    return np.array([np.mean(p) for p in bit_planes] + [np.std(p) for p in bit_planes])

def extract_color_stats(img_rgb):
    means = np.mean(img_rgb, axis=(0, 1))
    stds = np.std(img_rgb, axis=(0, 1))
    return np.concatenate([means, stds])

def extract_wavelet(img):
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    return np.concatenate([cA.ravel()[:100], cH.ravel()[:100], cV.ravel()[:100], cD.ravel()[:100]])

clean_path = "/kaggle/input/stegoimagesdataset/test/test/clean"
stego_path = "/kaggle/input/stegoimagesdataset/test/test/stego"
clean_images = sorted(os.listdir(clean_path))
stego_images = sorted(os.listdir(stego_path))

features_clean = []
features_stego = []

print("Extracting clean image features...")
for img in tqdm(clean_images):
    feat = extract_all_features(os.path.join(clean_path, img))
    if feat is not None:
        features_clean.append(feat)
features_clean = np.array(features_clean)
np.save("/kaggle/working/features_clean.npy", features_clean)
print("âœ… Features saved. Clean:", features_clean.shape)

print("Extracting stego image features...")
for img in tqdm(stego_images):
    feat = extract_all_features(os.path.join(stego_path, img))
    if feat is not None:
        features_stego.append(feat)

features_stego = np.array(features_stego)
np.save("/kaggle/working/features_stego.npy", features_stego)

print("âœ… Features saved. Stego:", features_stego.shape)


import numpy as np

# Load feature file to verify shape
features_clean = np.load("/kaggle/input/features-for-stego/features_clean.npy")
print("Feature vector shape:", features_clean.shape)
print("Number of features per image:", features_clean.shape[1])


#lr test code
import os
import numpy as np
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed

import pywt
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern
from skimage.io import imread
from PIL import Image

# â”€â”€ 1. Feature Extraction Function (same as used in training) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def extract_all_features(image_path):
    try:
        img = imread(image_path)
        if img is None:
            return None
        if img.ndim == 2:
            gray = img
            rgb = np.stack((img,) * 3, axis=-1)
        elif img.shape[2] == 4:
            img = np.array(Image.open(image_path).convert("RGB"))
            gray = rgb2gray(img)
            rgb = img
        else:
            gray = rgb2gray(img)
            rgb = img

        gray = (gray * 255).astype(np.uint8)

        # HOG
        hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

        # LBP
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_feat, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)

        # Bit-planes
        bitplanes = [(gray >> i) & 1 for i in range(8)]
        bp_feat = np.array([np.mean(b) for b in bitplanes] +
                           [np.std(b) for b in bitplanes])

        # Color stats
        means = np.mean(rgb, axis=(0, 1))
        stds = np.std(rgb, axis=(0, 1))
        color_feat = np.concatenate([means, stds])

        # Wavelet
        cA, (cH, cV, cD) = pywt.dwt2(gray, 'haar')
        wav_feat = np.concatenate([
            cA.ravel()[:100], cH.ravel()[:100],
            cV.ravel()[:100], cD.ravel()[:100]
        ])

        return np.concatenate([hog_feat, lbp_feat, bp_feat, color_feat, wav_feat])
    except:
        return None

# â”€â”€ 2. Load Model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
model_path = "/kaggle/input/batch-70/alaska_sgd_lr_pipeline1.pkl"  # âœ… Change if needed
pipeline = joblib.load(model_path)
print(f"âœ… Loaded trained model: {model_path}")

# â”€â”€ 3. Predict on a Given Directory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def test_on_directory(folder_path, n_jobs=4):
    print(f"\nğŸ”� Scanning directory: {folder_path}")
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.pgm')
    image_paths = [os.path.join(folder_path, f)
                   for f in os.listdir(folder_path)
                   if f.lower().endswith(valid_exts)]
    
    if not image_paths:
        print("âš ï¸� No valid images found.")
        return

    print(f"ğŸ“¸ Found {len(image_paths)} images")

    # Extract features
    features = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(extract_all_features)(img_path) for img_path in tqdm(image_paths)
    )
    features = [f for f in features if f is not None]
    print(f"âœ… Extracted features from {len(features)} images")

    # Predict
    preds = pipeline.predict(np.vstack(features))
    stego = np.sum(preds == 1)
    clean = np.sum(preds == 0)

    print("\nğŸ“Š Result Summary:")
    print(f"Total Images       : {len(preds)}")
    print(f"Predicted as Stego : {stego}")
    print(f"Predicted as Clean : {clean}")

# â”€â”€ 4. Call with Your Directory Path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# âš ï¸� Replace this with your real test directory path
test_dir = "/kaggle/input/stegoimagesdataset/train/train/stego"  
test_on_directory(test_dir)


