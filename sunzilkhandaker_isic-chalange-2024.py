# Quick Kaggle environment overview
import os
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")

print("Top-level datasets in /kaggle/input:")
for item in sorted(INPUT_ROOT.iterdir()):
    print(" -", item.name)


required_paths = [
    "/kaggle/input/isic-2024-challenge/train-image/image",
    "/kaggle/input/meddatacsv-of-multimodal"
 ]
for path in required_paths:
    if os.path.exists(path):
        print(f"âœ… Found {path}")
    else:
        print(f"â�Œ Missing {path}")

print("\nWritable directories:")
for work_dir in ["/kaggle/working", "/kaggle/temp"]:
    if os.path.exists(work_dir):
        stats = os.statvfs(work_dir)
        free_gb = stats.f_bavail * stats.f_frsize / 1e9
        print(f" - {work_dir} (~{free_gb:.1f} GB free)")
    else:
        print(f" - {work_dir} (not available in this session)")
        


!pip install --quiet "protobuf<5.0"  # 4.25.3 is safe with TF 2.15


import tensorflow as tf, google.protobuf
print(tf.__version__)
print(google.protobuf.__version__)  # should now read 4.25.x (or 3.20.x)


import os


image_dir_path = '/kaggle/input/isic-2024-challenge/train-image/image'

if os.path.exists(image_dir_path):
    # Get all entries in the directory (files and subdirectories)
    all_entries = os.listdir(image_dir_path)
    total_entries = len(all_entries)
    print(f"Total number of entries (files and directories) in '{image_dir_path}': {total_entries}")

    # Count only image files (e.g., .jpg)
    image_files = [entry for entry in all_entries if entry.lower().endswith(('.jpg', '.jpeg'))]
    total_images = len(image_files)
    print(f"Total number of image files (.jpg, .jpeg) in '{image_dir_path}': {total_images}")
else:
    print(f"Error: Image directory '{image_dir_path}' does not exist.")


import pandas as pd

DATA_ROOT = "/kaggle/input/isic-2024-challenge/train-image/image"
GROUND_TRUTH_CSV = "/kaggle/input/meddatacsv-of-multimodal/ISIC_2024_Training_GroundTruth.csv"
METADATA_CSV = "/kaggle/input/meddatacsv-of-multimodal/metadata.csv"
SUPPLEMENT_CSV = "/kaggle/input/meddatacsv-of-multimodal/ISIC_2024_Training_Supplement.csv"

# Load core CSVs with consistent dtypes
groundtruth_df = pd.read_csv(GROUND_TRUTH_CSV)
metadata_df = pd.read_csv(METADATA_CSV, low_memory=False)
supplement_df = pd.read_csv(SUPPLEMENT_CSV, low_memory=False)

# Align with original notebook expectations by ensuring a `target` column
if 'malignant' in groundtruth_df.columns and 'target' not in groundtruth_df.columns:
    groundtruth_df = groundtruth_df.rename(columns={'malignant': 'target'})

if 'target' not in groundtruth_df.columns:
    raise KeyError("Expected a 'target' or 'malignant' column in the ground-truth CSV.")

if 'isic_id' not in metadata_df.columns:
    raise KeyError("Metadata CSV must contain 'isic_id' to merge with ground truth.")

# Merge metadata (left join keeps original behaviour of working off metadata table)
df = metadata_df.merge(
    groundtruth_df[['isic_id', 'target']],
    on='isic_id',
    how='left'
)


# Drop rows where target is still missing (common in supplementary rows without labels)
df = df.dropna(subset=['target']).reset_index(drop=True)

print("Metadata preview:")
df.head()


# Check basic information about the dataset
print("Metadata Overview:")
print(df.info())


# Display first few rows
print(df.head())


print("Available Columns in Metadata:")
print(df.columns)


print("\nUnique Diagnosis Labels:")
print(df["target"].unique())  # Checking lesion classification labels



print("\nMissing Values Count:")
print(df.isnull().sum())  # Check for missing data


# Ensure "age_approx" is numeric before filling missing values
df["age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce")
df.loc[:, "age_approx"] = df["age_approx"].fillna(df["age_approx"].median())

# Fill missing values for categorical columns with "unknown"
df.loc[:, "sex"] = df["sex"].fillna("unknown")
df.loc[:, "anatom_site_general"] = df["anatom_site_general"].fillna("unknown")

# Drop columns that are mostly empty (no inplace=True to avoid warnings)
columns_to_drop = ["lesion_id", "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5", "mel_mitotic_index", "mel_thick_mm"]
df = df.drop(columns=columns_to_drop, errors="ignore")  # Avoids error if column doesn't exist

# Confirm there are no more missing values
print("Missing Values After Cleaning:")
print(df.isnull().sum())



from sklearn.preprocessing import LabelEncoder

# Convert string labels to integers
label_encoder = LabelEncoder()
df["target"] = label_encoder.fit_transform(df["target"])

# Force them to integers (critical for PyTorch compatibility)
df["target"] = df["target"].astype(int)


# (Optional) Print class mapping for reference
class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Class Mapping:", class_mapping)


# Confirm data type
print(df["target"].dtype)

# Confirm fix worked
print("\nSample Encoded Labels:")
print(df["target"].value_counts())



import os

# Define the path where images are stored (update this based on your dataset location)
image_dir = DATA_ROOT

# Check if some images exist in the folder
sample_images = os.listdir(image_dir)[:10]  # Get 10 sample images
print("Sample Image Filenames:", sample_images)



import cv2
import numpy as np
import os  # Make sure this is imported if you're using os.path.join

# Function to load and preprocess an image
def preprocess_image(image_path, size=(128, 128)):  # âœ… Changed to 128x128
    img = cv2.imread(image_path)  # Load image
    img = cv2.resize(img, size)   # Resize to 128x128
    img = img / 255.0             # Normalize pixel values to 0â€“1
    return img

# Test on a sample image
sample_image_path = os.path.join(image_dir, sample_images[0])  # sample_images must be defined
processed_img = preprocess_image(sample_image_path)

print("Processed Image Shape:", processed_img.shape)  # Should be (128, 128, 3)



import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# âœ… Convert labels to string for categorical mode
df["target"] = df["target"].astype(int).astype(str)

# âœ… Add .jpg extension to filenames
df["image_path"] = df["isic_id"].astype(str) + ".jpg"

# âœ… Correct image directory
image_dir = DATA_ROOT

# âœ… Build a set of files once instead of hitting the filesystem per row
listing_start = time.time()
available_files = {name for name in os.listdir(image_dir) if name.lower().endswith((".jpg", ".jpeg"))}
print(
    f"âœ… Indexed {len(available_files):,} image files in {time.time() - listing_start:.1f}s",
	)

filter_start = time.time()
df_filtered = df[df["image_path"].isin(available_files)].reset_index(drop=True)
dropped = len(df) - len(df_filtered)
if dropped:
    print(f"âš ï¸� Dropping {dropped} rows whose images were not found in DATA_ROOT.")
print(
    f"âœ… Filtering complete in {time.time() - filter_start:.1f}s â€” using {len(df_filtered)} images",
 )

if df_filtered.empty:
    raise RuntimeError("No matching image files were found in DATA_ROOT. Check that the dataset is available in Kaggle.")

# âœ… Define data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
 )

# âœ… Use 128x128 image size instead of 224x224
batch_size = 32
generator_start = time.time()
train_generator = datagen.flow_from_dataframe(
    dataframe=df_filtered,
    directory=image_dir,
    x_col="image_path",
    y_col="target",
    target_size=(128, 128),  # âœ… Updated here
    batch_size=batch_size,
    class_mode="categorical",
    validate_filenames=False  # redundant after manual filtering
 )
print(
    f"âœ… ImageDataGenerator ready in {time.time() - generator_start:.1f}s",
 )


import os
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from PIL import Image

MAX_SAMPLES = 6

# Reset the generator so we know which filenames correspond to this batch
train_generator.reset()
preview_images, preview_labels = next(train_generator)
batch_size = preview_images.shape[0]

# Look up the original filepaths for the sampled batch
batch_indices = train_generator.index_array[:batch_size]
batch_paths = [train_generator.filepaths[idx] for idx in batch_indices]

samples = []
for idx in range(min(MAX_SAMPLES, batch_size)):
    label_idx = int(np.argmax(preview_labels[idx]))
    samples.append({
        "label": label_idx,
        "original_path": batch_paths[idx],
        "original_img": Image.open(batch_paths[idx]).convert("RGB"),
        "augmented_img": preview_images[idx],
    })

# Ensure at least one label "1" example is displayed
if not any(sample["label"] == 1 for sample in samples):
    label_one_rows = df_filtered[df_filtered["target"] == "1"]
    if not label_one_rows.empty:
        label_one_row = label_one_rows.sample(1, random_state=42).iloc[0]
        label_one_path = os.path.join(image_dir, label_one_row["image_path"])
        original_img = Image.open(label_one_path).convert("RGB")
        img_array = np.array(original_img).astype(np.float32)
        augmented_img = datagen.random_transform(img_array)
        augmented_img = datagen.standardize(augmented_img)
        label_one_sample = {
            "label": 1,
            "original_path": label_one_path,
            "original_img": original_img,
            "augmented_img": augmented_img,
        }
        if len(samples) >= MAX_SAMPLES:
            samples = samples[: MAX_SAMPLES - 1]
        samples.append(label_one_sample)
    else:
        print("âš ï¸� No label '1' samples available in df_filtered; showing default batch.")

rows = ceil(len(samples))
plt.figure(figsize=(12, rows * 3))
for idx, sample in enumerate(samples):
    label_idx = sample["label"]

    # Original image
    plt.subplot(rows, 2, idx * 2 + 1)
    plt.imshow(sample["original_img"])
    plt.title(f"Original (label {label_idx})")
    plt.axis("off")

    # Augmented version (already scaled by the generator/datagen)
    plt.subplot(rows, 2, idx * 2 + 2)
    plt.imshow(np.clip(sample["augmented_img"], 0, 1))
    plt.title("Augmented")
    plt.axis("off")

plt.suptitle("Original vs. Augmented samples", y=1.02)
plt.tight_layout()
plt.show()


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os
from torch.utils.data import Subset


import time

# âœ… Ensure image metadata from augmentation step is available
if "df_filtered" not in globals():
    raise RuntimeError("Run the data-augmentation cell (5.3) first so df_filtered is defined.")

image_dir = DATA_ROOT

prep_start = time.time()
torch_df = df_filtered.copy()
torch_df["target_int"] = torch_df["target"].astype(int)
image_paths = torch_df["image_path"].apply(lambda name: os.path.join(image_dir, name)).to_numpy()
image_labels = torch_df["target_int"].to_numpy(dtype=int)
print(f"âœ… Prepared {len(image_paths):,} image paths in {time.time() - prep_start:.1f}s")

if image_paths.size == 0:
    raise RuntimeError("No images found after filtering. Please verify DATA_ROOT and df_filtered.")

split_start = time.time()
train_ratio, val_ratio, test_ratio = 0.70, 0.10, 0.20
holdout_ratio = val_ratio + test_ratio

train_paths, holdout_paths, train_labels, holdout_labels = train_test_split(
    image_paths,
    image_labels,
    test_size=holdout_ratio,
    stratify=image_labels,
    random_state=42,
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    holdout_paths,
    holdout_labels,
    test_size=test_ratio / holdout_ratio,
    stratify=holdout_labels,
    random_state=42,
)
print(
    f"âœ… Completed stratified splits in {time.time() - split_start:.1f}s â€” "
    f"{len(train_paths):,} train / {len(val_paths):,} val / {len(test_paths):,} test"
)

feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

class SkinDataset(Dataset):
    def __init__(self, image_paths, labels, feature_extractor):
        self.image_paths = list(image_paths)
        self.labels = labels.astype(int)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return inputs["pixel_values"].squeeze(0), label

train_dataset = SkinDataset(train_paths, train_labels, feature_extractor)
val_dataset = SkinDataset(val_paths, val_labels, feature_extractor)
test_dataset = SkinDataset(test_paths, test_labels, feature_extractor)

print(
    f"âœ… Dataset objects ready in {time.time() - split_start:.1f}s total (including splits)"
)

def build_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

train_loader = build_loader(train_dataset, shuffle=True)
val_loader = build_loader(val_dataset, shuffle=False)
test_loader = build_loader(test_dataset, shuffle=False)

print(
    f"âœ… Dataloaders ready â€” {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test"
)


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

split_counts = pd.DataFrame({
    'split': ['train'] * len(train_labels) + ['val'] * len(val_labels) + ['test'] * len(test_labels),
    'label': np.concatenate([train_labels, val_labels, test_labels])
})

plt.figure(figsize=(8, 5))
sns.countplot(data=split_counts, x='split', hue='label', palette='viridis')
plt.title('Label distribution per split')
plt.xlabel('Dataset split')
plt.ylabel('Count')
plt.legend(title='Encoded label')
plt.tight_layout()
plt.show()



import random

path_to_label = {path: label for path, label in zip(train_paths, train_labels)}
sample_paths = random.sample(list(train_paths), k=min(6, len(train_paths)))
plt.figure(figsize=(12, 6))
for idx, img_path in enumerate(sample_paths, 1):
    label = path_to_label[img_path]
    img = Image.open(img_path)
    plt.subplot(2, 3, idx)
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis('off')
plt.tight_layout()
plt.show()



# Ensure correct number of unique classes using the filtered dataset
import numpy as np
unique_labels = np.unique(image_labels)
num_classes = len(unique_labels)
assert num_classes > 1, "Your dataset must contain at least 2 classes for classification."

# Load Pretrained Vision Transformer with Correct Output Size
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

# Move model to GPU (if available)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
if multi_gpu:
    model = torch.nn.DataParallel(model)
    print(f"ğŸ”Œ Multi-GPU enabled â€” using {torch.cuda.device_count()} GPUs")
else:
    print(f"ğŸ’» Using device: {device}")
model.to(device)

def get_base_model(model_instance: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model, unwrapping DataParallel if needed."""
    return model_instance.module if isinstance(model_instance, torch.nn.DataParallel) else model_instance

# Define Optimizer & Loss Function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

print(f"âœ… Model Loaded Successfully with {num_classes} Classes ({unique_labels})")


for idx in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(idx) / 1024**3
    reserved = torch.cuda.memory_reserved(idx) / 1024**3
    print(f"GPU{idx}: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")


import os
import torch
from tqdm import tqdm

CHECKPOINT_DIR = "/content/drive/MyDrive/ISIC/ISIC 2024/Code"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
checkpoint_path = os.path.join(CHECKPOINT_DIR, "vit_checkpoint.pth")
best_model_path = os.path.join(CHECKPOINT_DIR, "vit_best_model.pth")
start_epoch = 0
best_accuracy = 0.0

if isinstance(model, torch.nn.DataParallel):
    print(f"ğŸš€ Training with DataParallel across {torch.cuda.device_count()} GPUs")
else:
    print(f"ğŸš€ Training on device: {device}")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    get_base_model(model).load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint.get('best_accuracy', 0.0)
    print(f"âœ… Loaded checkpoint â€” Resuming from epoch {start_epoch + 1}")
else:
    print("ğŸ†• No checkpoint found â€” Starting from scratch")

num_epochs = 5
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total if total > 0 else 0

        progress_bar.set_postfix({
            "Loss": f"{avg_loss:.4f}",
            "Acc": f"{accuracy:.2f}%"
        })

    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total if total > 0 else 0

    # âœ… Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total if val_total > 0 else 0

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(
        f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}% | "
        f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%"
    )

    # âœ… Save checkpoint after each epoch
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': get_base_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy
    }
    torch.save(checkpoint, checkpoint_path)

    # âœ… Save best model (based on validation accuracy)
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(get_base_model(model).state_dict(), best_model_path)
        print(f"ğŸ“Œ New Best Validation Accuracy: {val_acc:.2f}% â€” Model Saved!")

print("âœ… Training Complete & Checkpoints Saved!")


if not history['train_loss']:
    print("Run the training cell before plotting diagnostics.")
else:
    epochs_ran = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_ran, history['train_loss'], label='Train Loss')
    plt.plot(epochs_ran, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_ran, history['train_acc'], label='Train Acc')
    plt.plot(epochs_ran, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()



# Load the best saved model
state_dict = torch.load(best_model_path, map_location=device)
get_base_model(model).load_state_dict(state_dict)
model.to(device)
model.eval()
print("âœ… Best model loaded for final evaluation.")


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images).logits
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# âœ… Report
print("ğŸ“Š Final Evaluation on Test Set:")
print(classification_report(all_labels, all_preds))



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


