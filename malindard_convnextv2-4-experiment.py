import os
import copy
import timm
import random
import time
import torch
import cv2
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.optim.optimizer
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from PIL import Image
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.cuda import amp
from torchvision import transforms as T
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from tqdm import tqdm

print(torch.__version__)


def seed_everything(seed):
    """
    Sets seeds for reproducibility in training.

    Args:
        seed (int): Seed value to ensure determinism.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # Seed for hash-based operations
    np.random.seed(seed)  # Seed for NumPy
    torch.manual_seed(seed)  # Seed for PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Seed for PyTorch (GPU)
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic
    torch.backends.cudnn.benchmark = False  # Enable benchmark mode for CuDNN

seed_everything(42)


# Path to the image directory and load the CSV file
image_dir = '../input/aptos2019-blindness-detection/train_images'
data = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

# Add .png extension to the ID column to get the actual filenames
data['file_name'] = data['id_code'] + '.png'

# 1. Check for missing files
missing_files = []
for fname in data['file_name']:
    if not os.path.exists(os.path.join(image_dir, fname)):
        missing_files.append(fname)

if missing_files:
    print(f"{len(missing_files)} file hilang dan akan dihapus dari data:")
    data = data[~data['file_name'].isin(missing_files)]
else:
    print("✅ Semua file gambar ada.")

# 2. Remove/flag corrupted images & 3. Validate resolution + resize to 384x384
valid_images = []
invalid_images = []
resized_dir = './resized_cleaned_images_384'
os.makedirs(resized_dir, exist_ok=True)

for idx, row in tqdm(data.iterrows(), total=len(data)):
    path = os.path.join(image_dir, row['file_name'])
    try:
        img = Image.open(path)
        img.verify()  # validate file content
        img = Image.open(path).convert('RGB')  # reopen for resizing
        img = img.resize((384, 384))
        img.save(os.path.join(resized_dir, row['file_name']))  # save to new folder
        valid_images.append(row['file_name'])
    except Exception as e:
        print(f"[X] Rusak: {row['file_name']} - {e}")
        invalid_images.append(row['file_name'])

# Keep only valid images in the DataFrame
data = data[data['file_name'].isin(valid_images)]

# 4. Remove duplicate images based on hash comparison
hash_dict = {}
duplicate_files = []

def get_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

for fname in tqdm(data['file_name']):
    path = os.path.join(resized_dir, fname)
    try:
        img_hash = get_hash(path)
        if img_hash in hash_dict:
            duplicate_files.append(fname)
        else:
            hash_dict[img_hash] = fname
    except Exception as e:
        print(f"Hashing error: {fname} - {e}")

if duplicate_files:
    print(f"Menghapus {len(duplicate_files)} duplikat gambar.")
    data = data[~data['file_name'].isin(duplicate_files)]

# Reset index and save to a new CSV
data = data.reset_index(drop=True)
data.to_csv('cleaned_train.csv', index=False)

print(f"✅ Total data bersih: {len(data)}")
print("DataFrame disimpan ke 'cleaned_train.csv'")


raw_data = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
cleaned_data = pd.read_csv('/kaggle/working/cleaned_train.csv')


print('Number of samples Before Data Cleaning: ', raw_data.shape[0])
display(raw_data.head())


print('Number of samples After Data Cleaning: ', cleaned_data.shape[0])
display(cleaned_data.head())


raw_data['diagnosis'].value_counts()


cleaned_data['diagnosis'].value_counts()


# Mapping label numeric to text
label_names = {
    0: "No DR/Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Copy data and change diagnosis to string label
data_viz = raw_data.copy()
data_viz["diagnosis"] = data_viz["diagnosis"].map(label_names)

# Set label order manually
label_order = [
    "No DR/Normal",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]
data_viz["diagnosis"] = pd.Categorical(data_viz["diagnosis"], categories=label_order, ordered=True)

# Set display
plt.figure(figsize=(12, 7), facecolor="white")
sns.set_style("whitegrid")

# Barplot
ax = sns.countplot(x="diagnosis", data=data_viz, palette="Blues_d")

# Data count
for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom', fontsize=12, color='black')

# Title & Label
ax.set_title("Distribusi Kelas Retinopati Diabetik Sebelum Data Cleaning", fontsize=16, weight='bold')
ax.set_xlabel("Kategori Diagnosis", fontsize=13)
ax.set_ylabel("Jumlah Gambar", fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.despine()

# Save to file
plt.tight_layout()
plt.savefig("distribusi_kelas_before.png", dpi=300, facecolor='white')
plt.show()


# Mapping label numeric to text
label_names = {
    0: "No DR/Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Copy data and change diagnosis to string label
data_viz = cleaned_data.copy()
data_viz["diagnosis"] = data_viz["diagnosis"].map(label_names)

# Set label order manually
label_order = [
    "No DR/Normal",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]
data_viz["diagnosis"] = pd.Categorical(data_viz["diagnosis"], categories=label_order, ordered=True)

# Set display
plt.figure(figsize=(12, 7), facecolor="white")
sns.set_style("whitegrid")

# Barplot
ax = sns.countplot(x="diagnosis", data=data_viz, palette="Blues_d")

# Data count
for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom', fontsize=12, color='black')

# Title & Label
ax.set_title("Distribusi Kelas Retinopati Diabetik Sesudah Data Cleaning", fontsize=16, weight='bold')
ax.set_xlabel("Kategori Diagnosis", fontsize=13)
ax.set_ylabel("Jumlah Gambar", fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.despine()

# Save to file
plt.tight_layout()
plt.savefig("distribusi_kelas_after.png", dpi=300, facecolor='white')
plt.show()


# Setting the style for the plot
sns.set_style("white")

# Mapping class labels to their corresponding categories
level_to_category = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

# Plotting the first 15 images along with their labels
count = 1
plt.figure(figsize=[20, 20])

for img_name in data['id_code'][:15]:  # Assuming 'train' contains the dataset
    img = cv2.imread(f"/kaggle/working/resized_cleaned_images_384/{img_name}.png")[..., [2, 1, 0]]  # Reading the image
    
    # Getting the label (class) for the image
    label = data[data['id_code'] == img_name]['diagnosis'].values[0]  # Assuming 'diagnosis' is the label column
    
    # Setting up the subplot with image and label
    plt.subplot(5, 5, count)
    plt.imshow(img)
    plt.title(f"Citra {count}: {level_to_category[label]}")  # Display the class label
    count += 1
    
# Display the plot
plt.savefig('/kaggle/working/imagebeforepreprecssing.png')
plt.show()


# Set input and output directories
input_dir = '/kaggle/working/resized_cleaned_images_384'
output_dir = '/kaggle/working/processed_images/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the CSV containing cleanded image names and labels
df = pd.read_csv('/kaggle/working/cleaned_train.csv')


# Function to crop the image based on grayscale threshold
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark so that we crop out everything
            return img  # Return original image
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def process_image(row, sigmaX=10):
    sample_image_id = row['id_code']
    sample_image_file = sample_image_id + '.png'
    sample_image_path = os.path.join(input_dir, sample_image_file)
    
    if os.path.exists(sample_image_path):
        # Ben Graham Preprocessing
        image = cv2.imread(sample_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (384, 384))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
        
        # Save the processed image to the output directory
        output_path = os.path.join(output_dir, sample_image_file)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# Using ThreadPoolExecutor to process images in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(process_image, [row for _, row in df.iterrows()]), total=df.shape[0], desc="Processing images", unit="image"))

print("Processing complete for all images.")


# Setting the style for the plot
sns.set_style("white")

# Mapping class labels to their corresponding categories`
level_to_category = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

# Plotting the first 15 images along with their labels
count = 1
plt.figure(figsize=[20, 20])

for img_name in data['id_code'][:15]:  # Assuming 'train' contains the dataset
    img = cv2.imread(f"/kaggle/working/processed_images/{img_name}.png")[..., [2, 1, 0]]  # Reading the image
    
    # Getting the label (class) for the image
    label = data[data['id_code'] == img_name]['diagnosis'].values[0]  # Assuming 'diagnosis' is the label column
    
    # Setting up the subplot with image and label
    plt.subplot(5, 5, count)
    plt.imshow(img)
    plt.title(f"Citra {count}: {level_to_category[label]}")  # Display the class label
    count += 1

# Display the plot
plt.show()


DATA_DIR = "/kaggle/working/processed_images"
CSV_PATH = "/kaggle/working/cleaned_train.csv"
MODEL_PATH = "./kaggle/working/"
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
NUM_WORKERS = 4
USE_AMP = True
EPOCHS = 20


class RetinopathyDataset(Dataset):
    def __init__(self, image_dir, csv_file, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.loc[idx, 'id_code'] + '.png')

        tensor_image = read_image(img_name)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'], dtype=torch.long)

        if self.transforms is not None:
            tensor_image = self.transforms(tensor_image)

        return (tensor_image, label)


# Train Augmentation
train_transforms = T.Compose([
    T.RandomResizedCrop(size=384, scale=(0.8, 1.2)),          # Zoom (scaling)
    T.RandomHorizontalFlip(p=0.5),                            # Flip horizontal
    T.RandomVerticalFlip(p=0.5),                              # Flip vertical
    T.RandomRotation(degrees=25),                             # Rotation
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),          # Height & width shift (translation)
    T.ConvertImageDtype(torch.float32),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Val & Test Augementation
val_test_transforms = T.Compose([
    T.Resize((384, 384)),
    T.ConvertImageDtype(torch.float32),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Dataset
train_base_dataset = RetinopathyDataset(DATA_DIR, CSV_PATH, transforms=train_transforms)
val_base_dataset = RetinopathyDataset(DATA_DIR, CSV_PATH, transforms=val_test_transforms)
test_base_dataset = RetinopathyDataset(DATA_DIR, CSV_PATH, transforms=val_test_transforms)
labels = train_base_dataset.data['diagnosis'].values

# Split index
train_idx, test_idx = train_test_split(np.arange(len(train_base_dataset)), test_size=TEST_SPLIT, 
                                       stratify=labels, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=VAL_SPLIT/(1 - TEST_SPLIT), 
                                      stratify=labels[train_idx], random_state=42)

# Use Subset to divide the dataset
train_dataset = Subset(train_base_dataset, train_idx)
val_dataset = Subset(val_base_dataset, val_idx)
test_dataset = Subset(test_base_dataset, test_idx)

# DataLoader for each dataset
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, drop_last=True, pin_memory=True,
                          prefetch_factor=2)

val_loader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, 
                        num_workers=NUM_WORKERS, drop_last=False, pin_memory=True,
                          prefetch_factor=2)

test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, 
                         num_workers=NUM_WORKERS, drop_last=False, pin_memory=True,
                          prefetch_factor=2)

from collections import Counter
# Jumlah per kelas
def count_labels(indices, label_array):
    counter = Counter(label_array[indices])
    return dict(sorted(counter.items()))

print(f"\n Distribusi Dataset:")
print(f"Total Dataset: {len(train_base_dataset)}")
print(f"Train Set: {len(train_dataset)} samples")
print(f"Validation Set: {len(val_dataset)} samples")
print(f"Test Set: {len(test_dataset)} samples")

print("\n\n Jumlah Data per Kelas:")
print("Train:", count_labels(train_idx, labels))
print("Validation:", count_labels(val_idx, labels))
print("Test:", count_labels(test_idx, labels))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# List of model
MODEL_NAME = "convnextv2_base.fcmae_ft_in22k_in1k_384"
MODEL_SAVE = "/kaggle/working/deit_training_results.csv"
CHECKPOINT_PATH = f"/kaggle/working/{MODEL_NAME}_best.pt"

# Load Model
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=5)
model.to(device)


# Calculate class weights
labels = data['diagnosis'].values
classes = sorted(data['diagnosis'].unique())
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"Class Weights: {class_weights_tensor}")


# Loss function & optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None


# Training Function
def train_step(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total_samples = 0, 0, 0

    with tqdm(train_loader, desc="Training", unit="batch") as pbar:
        for inputs, target in pbar:
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(inputs)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            _, predicted = output.max(1)
            total_loss += loss.item() * inputs.size(0)
            correct += predicted.eq(target).sum().item()
            total_samples += inputs.size(0)

            pbar.set_postfix({"Loss": f"{total_loss / total_samples:.4f}", "Accuracy": f"{100.0 * correct / total_samples:.2f}%"})

    return {"loss": total_loss / total_samples, "accuracy": 100.0 * correct / total_samples}

# Validation Function
@torch.no_grad()
def val_step(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0

    with tqdm(val_loader, desc="Validation", unit="batch") as pbar:
        for inputs, target in pbar:
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            loss = criterion(output, target)

            _, predicted = output.max(1)
            total_loss += loss.item() * inputs.size(0)
            correct += predicted.eq(target).sum().item()
            total_samples += inputs.size(0)

            pbar.set_postfix({"Loss": f"{total_loss / total_samples:.4f}", "Accuracy": f"{100.0 * correct / total_samples:.2f}%"})

    return {"loss": total_loss / total_samples, "accuracy": 100.0 * correct / total_samples}


# Training Loop
best_val_acc = 0
best_model_wts = copy.deepcopy(model.state_dict())

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_loss, train_acc, val_loss, val_acc = [], [], [], []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_metrics = train_step(model, train_loader, criterion, optimizer, device, scaler)
    val_metrics = val_step(model, val_loader, criterion, device)

    scheduler.step()

    train_loss.append(train_metrics["loss"])
    train_acc.append(train_metrics["accuracy"])
    val_loss.append(val_metrics["loss"])
    val_acc.append(val_metrics["accuracy"])

    if val_metrics["accuracy"] > best_val_acc:
        best_val_acc = val_metrics["accuracy"]
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, CHECKPOINT_PATH)
        print(f"=> Model saved at {CHECKPOINT_PATH}")

# Save Metrics
metrics_df = pd.DataFrame({
    "epoch": range(1, len(train_loss) + 1),
    "train_loss": train_loss,
    "train_accuracy": train_acc,
    "val_loss": val_loss,
    "val_accuracy": val_acc
})
metrics_df.to_csv(MODEL_SAVE, index=False)

# Plot Training Results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.lineplot(x='epoch', y='train_loss', data=metrics_df, label='Train Loss')
sns.lineplot(x='epoch', y='val_loss', data=metrics_df, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()

plt.subplot(1, 2, 2)
sns.lineplot(x='epoch', y='train_accuracy', data=metrics_df, label='Train Accuracy')
sns.lineplot(x='epoch', y='val_accuracy', data=metrics_df, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()
plt.show()

# Load Best Model for Testing
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()


# Function to Evaluate Test Set
@torch.no_grad()
def evaluate_test_set(model, test_loader, device):
    y_true, y_pred = [], []

    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    return np.array(y_true), np.array(y_pred)

# Evaluate on Test Set
y_true_test, y_pred_test = evaluate_test_set(model, test_loader, device)

# Calculate Test Accuracy
test_accuracy = accuracy_score(y_true_test, y_pred_test) * 100
print(f"\n Test Accuracy: {test_accuracy:.3f}%")

# Generate Classification Report
class_report = classification_report(y_true_test, y_pred_test, digits=3)
print("\n Classification Report:\n", class_report)

# Label DR
class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Calculate confusion matrix
cm = confusion_matrix(y_true_test, y_pred_test)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Klasifikasi Retinopati Diabetik')
plt.tight_layout()
plt.show()


# Save Weights
save_weight = "best_weights.pt"
torch.save(model.state_dict(), save_weight)
print(f"Model weights saved to {save_weight}")


plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
sns.lineplot(x='epoch', y='train_loss', data=metrics_df, label='Train Loss')
sns.lineplot(x='epoch', y='val_loss', data=metrics_df, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
sns.lineplot(x='epoch', y='train_accuracy', data=metrics_df, label='Train Accuracy')
sns.lineplot(x='epoch', y='val_accuracy', data=metrics_df, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.close()


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Skenario 4')
plt.tight_layout()
plt.savefig("confusion_matrix.png")




