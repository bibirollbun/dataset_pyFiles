!pip install -r ../input/fathomnet-2025/requirements.txt


from collections import defaultdict, OrderedDict
import contextlib
import io
import json
import os
import random
import sys
import time
from typing import Dict, Tuple, List, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import requests
from PIL import Image
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms


sys.path.append("/kaggle/input/taxonomic-hierarchical-distance")
from metric import hierarchical_distance, score


def create_subset_json(input_json_path: str, output_json_path: str, subset_size: int) -> None:
    """
    Create a subset of a COCO-style dataset by randomly sampling a fixed number of images.

    Args:
        input_json_path (str): Path to input JSON file in COCO format.
        output_json_path (str): Path to write the subset JSON file.
        subset_size (int): Number of images to sample from the original dataset.

    The function maintains the dataset structure including 'info', 'licenses', 'categories',
    and filters 'images' and 'annotations' accordingly.
    """
    with open(input_json_path, "r") as f:
        full_data = json.load(f)

    # Randomly sample images
    sampled_images = random.sample(full_data["images"], subset_size)
    sampled_image_ids = {img["id"] for img in sampled_images}

    # Filter annotations corresponding to sampled images
    sampled_annotations = [
        ann for ann in full_data["annotations"] if ann["image_id"] in sampled_image_ids
    ]

    subset_data = {
        "info": full_data.get("info", {}),
        "licenses": full_data.get("licenses", []),
        "images": sampled_images,
        "annotations": sampled_annotations,
        "categories": full_data["categories"]
    }

    with open(output_json_path, "w") as f:
        json.dump(subset_data, f, indent=4)


# Enable downsampling training data to be downloaded to speed up experimentation and testing
downsampling = False
if downsampling:
    full_json = '/kaggle/input/fathomnet-2025/dataset_train.json'  # Path to dataset_train.json
    subset_json = '../../dataset_train_subset.json'  # Path to subset json
    subset_size = 1000  # Number of images to sample (adjust this as needed)    
    create_subset_json(full_json, subset_json, subset_size)


!python /kaggle/input/fathomnet-2025/download.py /kaggle/input/fathomnet-2025/dataset_train.json ../../train/ -n 1
#!python /kaggle/input/fathomnet-2025/download.py "$subset_json" ../../train/


!ls ../../train


!ls ../../train/rois | wc -l


!python /kaggle/input/fathomnet-2025/download.py /kaggle/input/fathomnet-2025/dataset_test.json ../../test/


def plot_random_images(image_dir: str, num_images: int = 16) -> None:
    """
    Display a random grid of images from a directory.

    Args:
        image_dir (str): Path to the directory containing images.
        num_images (int): Number of images to display (ideally a perfect square like 16).
    """
    supported_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(supported_exts)]

    if len(image_files) < num_images:
        print(f"Only found {len(image_files)} images, plotting those.")
        selected_images = image_files
    else:
        selected_images = random.sample(image_files, num_images)

    grid_size = int(num_images ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for ax, img_path in zip(axes, selected_images):
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(os.path.basename(img_path), fontsize=8)
            ax.axis("off")
        except Exception:
            ax.axis("off")
            ax.set_title("Error loading image")

    for ax in axes[len(selected_images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


plot_random_images("../../train/images")


# Load and encode the labels
train_annotations_df = pd.read_csv("../../train/annotations.csv")
label_encoder = LabelEncoder().fit(train_annotations_df["label"].dropna())
train_annotations_df.head()


# Get the number of classes
num_classes = len(label_encoder.classes_)
num_classes


# Inspect labels
pd.set_option('display.max_rows', None)
train_annotations_df["label"].value_counts()


def get_classification(base_url: str, name: str) -> Optional[Dict]:
    """
    Query WoRMS API to retrieve the full classification hierarchy of a taxon by name.

    Args:
        base_url (str): Base URL of the WoRMS REST API.
        name (str): Scientific name to query.

    Returns:
        Optional[Dict]: Nested classification dictionary if successful, otherwise None.
    """
    try:
        response = requests.get(f"{base_url}/AphiaRecordsByName/{name}?like=false&marine_only=false")
        data = response.json()
        if isinstance(data, list) and data:
            aphia_id = data[0].get("AphiaID")
            class_response = requests.get(f"{base_url}/AphiaClassificationByAphiaID/{aphia_id}")
            return class_response.json()
    except Exception as e:
        print(f"Error for {name}: {e}")
    return None

def extract_ranks(classification: Dict) -> Dict[str, Optional[str]]:
    """
    Extract standard taxonomic ranks from a nested classification dict.

    Args:
        classification (Dict): Classification response from WoRMS.

    Returns:
        Dict[str, Optional[str]]: Dictionary mapping rank to scientific name.
    """
    taxon = {rank: None for rank in ["species", "genus", "family", "order", "class", "phylum", "kingdom"]}
    current = classification
    while current and isinstance(current, dict):
        rank = current.get("rank", "").lower()
        name = current.get("scientificname")
        if rank in taxon:
            taxon[rank] = name
        current = current.get("child")
    return taxon


# Create list of labels
labels = train_annotations_df["label"].unique()

# Base URL for WoRMS API
base_url = "https://www.marinespecies.org/rest"

# Build taxonomy records
taxonomy_records = []
for label in tqdm(labels):
    # Query WoRMS API for classification
    classification = get_classification(base_url, label)
    taxon = {"label": label}
    # Extract taxonomic hierarchy from nested dict
    if classification:
        taxon.update(extract_ranks(classification))
    taxonomy_records.append(taxon)

# Create DataFrame and standardize column names
taxonomy_df = pd.DataFrame(taxonomy_records)
taxonomy_df.columns = [col.capitalize() if col != "label" else col for col in taxonomy_df.columns]
taxonomy_df.head()


# Check unique values
for col in taxonomy_df.columns[1:]:
    print(f"Unique values in column {col}: {taxonomy_df[[col]].nunique().iloc[0]}")


# Check null values
n_total = taxonomy_df.shape[0]
for col in taxonomy_df.columns[1:]:
    n_null_values = taxonomy_df[[col]].isna().sum().iloc[0]
    share_null_values = n_null_values / n_total
    print(f"Share of null values in column {col}: {share_null_values:.2f}")


# Define levels and fill missing values
levels = ["Phylum", "Class", "Order", "Family", "Genus", "Species"]
taxonomy_df[levels] = taxonomy_df[levels].fillna("Unknown")

encoders = {}
unknown_class_indices = {}

# Encode labels and make sure that Unknown is always the last index
for lvl in levels:
    unique_vals = taxonomy_df[lvl].unique().tolist()
    if "Unknown" not in unique_vals:
        unique_vals.append("Unknown")

    ordered = sorted([val for val in unique_vals if val != "Unknown"]) + ["Unknown"]

    le = LabelEncoder()
    le.classes_ = np.array(ordered)  # Set class order manually
    taxonomy_df[f"{lvl}_encoded"] = le.transform(taxonomy_df[lvl])  # Apply to data

    encoders[lvl] = le
    unknown_class_indices[lvl] = le.transform(["Unknown"])[0]

# Merge encoded taxonomy into train_annotations_df
train_annotations_df_extended = train_annotations_df.merge(taxonomy_df, on="label", how="left")

# Drop unnecessary columns
train_annotations_df_extended = train_annotations_df_extended.drop(["Kingdom"], axis=1)
train_annotations_df_extended.head()


# Step 1: Extract original image ID from the filename
train_annotations_df_extended["orig_image_id"] = train_annotations_df_extended["path"].apply(lambda p: int(p.split("/")[-1].split("_")[0]))

# Step 2: Get all unique original image IDs
unique_orig_ids = train_annotations_df_extended["orig_image_id"].unique()

# Step 3: Split original images into train and val sets
seed = 42
train_ids, val_ids = train_test_split(
    unique_orig_ids, test_size=0.1, random_state=seed
)

# Step 4: Assign split label based on original image ID
train_annotations_df_extended["split"] = train_annotations_df_extended["orig_image_id"].apply(
    lambda x: "val" if x in val_ids else "train"
)
train_annotations_df_extended["split"].value_counts()


class FathomNetDataset(Dataset):
    """
    A custom PyTorch Dataset for loading images and pre-encoded multi-level taxonomic labels from a DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame with image paths and *_encoded taxonomy columns.
        levels (list): List of taxonomic levels to consider.
        transform (callable, optional): Optional image transformations.

    Expected columns:
        - "path"
        - For training/validation: "Phylum_encoded", ..., "Species_encoded"
    """
    def __init__(self, dataframe, levels, transform=None):
        self.data = dataframe
        self.levels = levels
        self.transform = transform

        # Define taxonomy levels and their encoded counterparts
        self.encoded_columns = [f"{lvl}_encoded" for lvl in self.levels]

        # Determine if we're in test mode
        self.is_test = not all(col in self.data.columns for col in self.encoded_columns)
        self.image_paths = self.data["path"].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, self.image_paths[idx]
        else:
            # Extract pre-encoded labels
            label_dict = {
                lvl: torch.tensor(self.data.iloc[idx][f"{lvl}_encoded"], dtype=torch.long)
                for lvl in self.levels
            }
            return image, label_dict


# Define data transformations for preprocessing
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
    ], p=0.8),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),

    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.2),

    transforms.ToTensor(),

    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# For validation/test (no augmentations, just resizing and normalization)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


train_df = train_annotations_df_extended[train_annotations_df_extended["split"] == "train"].reset_index(drop=True)
val_df = train_annotations_df_extended[train_annotations_df_extended["split"] == "val"].reset_index(drop=True)

# Initialize datasets with appropriate transforms
train_dataset = FathomNetDataset(train_df, levels, transform=train_transforms)
val_dataset = FathomNetDataset(val_df, levels, transform=val_transforms)

# Create the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=3)


class FathomNetClassifier(pl.LightningModule):
    """
    A PyTorch Lightning module for classifying marine species using a pretrained EfficientNetV2 model.
    This version supports hierarchical multi-output classification across multiple taxonomic levels 
    (e.g., kingdom → species), each with its own classifier head.
    """
    def __init__(self, num_classes, unknown_class_indices):
        """
        Args:
            num_classes (dict): Dictionary mapping each taxonomic level to its number of classes.
                                Example: {"Phylum": 5, ..., "Species": 200}
        """
        super().__init__()
        self.save_hyperparameters()

        self.levels = list(num_classes.keys())
        self.unknown_class_indices = unknown_class_indices

        # Load a pretrained EfficientNetV2 model
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

        # Extract the number of features before classification
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # Remove the final classification layer

        # Create a separate classifier head for each taxonomic level
        self.classifier_heads = nn.ModuleDict({
            level: nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(in_features // 2, num_classes[level])
            )
            for level in self.levels
        })

        # Create a separate CrossEntropyLoss per level with ignore_index
        self.criterions = {
            level: nn.CrossEntropyLoss(
                label_smoothing=0.1,
                ignore_index=self.unknown_class_indices[level]
            )
            for level in self.levels
        }

        self.val_outputs = []

    def forward(self, x):
        """Forward pass returns a dict of predictions for each taxonomic level."""
        features = self.backbone(x)
        return {level: head(features) for level, head in self.classifier_heads.items()}

    def training_step(self, batch, batch_idx):
        """Training step with deepest known level loss."""
        x, y = batch
        outputs = self(x)
    
        total_loss = torch.tensor(0.0, device=x.device)
        valid_losses = 0
    
        for level in self.levels:
            targets = y[level]
            preds = outputs[level]
    
            known_mask = targets != self.unknown_class_indices[level]
            if known_mask.any():
                loss = self.criterions[level](preds[known_mask], targets[known_mask])
                total_loss += loss
                valid_losses += 1

        if valid_losses > 0:
            total_loss = total_loss / valid_losses
        else:
            total_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
    
        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step with deepest known level loss."""
        x, y = batch
        outputs = self(x)
    
        total_loss = torch.tensor(0.0, device=x.device)
        valid_losses = 0
    
        for level in self.levels:
            targets = y[level]
            preds = outputs[level]
    
            known_mask = targets != self.unknown_class_indices[level]
            if known_mask.any():
                loss = self.criterions[level](preds[known_mask], targets[known_mask])
                total_loss += loss
                valid_losses += 1
    
        if valid_losses > 0:
            total_loss = total_loss / valid_losses
        else:
            total_loss = torch.tensor(0.0, device=x.device)
    
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        """AdamW optimizer with ReduceLROnPlateau scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                verbose=True,
                min_lr=1e-6
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# Get num classes per level
num_classes = {
    level: len(encoders[level].classes_)
    for level in levels
}

# Ensure order
num_classes = OrderedDict(sorted(num_classes.items(), key=lambda x: levels.index(x[0])))
num_classes


# Check unknow class indices
unknown_class_indices


# Initialize the model
model = FathomNetClassifier(num_classes=num_classes, unknown_class_indices=unknown_class_indices)


# Setup PyTorch Lightning Trainer with CSV Logger, Early Stopping, Model Checkpoint and TQDM Progress Bar
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    filename="best_model"
)
early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=True,
    mode="min",
    min_delta=0.001
)
progress_bar = TQDMProgressBar(refresh_rate=1)
csv_logger = CSVLogger("logs/", name="fathomnet")
trainer = pl.Trainer(
    logger=csv_logger,
    max_epochs=50,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[checkpoint_callback, early_stopping_callback, progress_bar],
    precision="16-mixed",
    fast_dev_run=False
)


# Train the model
trainer.fit(model, train_loader, val_loader)


print(f"Training completed after {trainer.current_epoch} epochs.")


log_df = pd.read_csv("logs/fathomnet/version_0/metrics.csv")
train_loss = log_df[log_df["train_loss"].notna()][["epoch", "train_loss"]]
val_loss = log_df[log_df["val_loss"].notna()][["epoch", "val_loss"]]

plt.plot(train_loss["epoch"], train_loss["train_loss"], label="Train Loss")
plt.plot(val_loss["epoch"], val_loss["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curves")
plt.show()


# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the best model and move it to the correct device
best_model = FathomNetClassifier.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    num_classes=num_classes,
    unknown_class_indices=unknown_class_indices
)

# Send model to the device
best_model = best_model.to(device)


def prepare_probs_and_lookup(model,
                             data_loader,
                             levels: List[str],
                             taxonomy_df: pd.DataFrame) -> Tuple[Dict[str, List[float]], Dict[Tuple[str, str], int]]:
    """
    Compute softmax probabilities per level and build fallback lookup.

    Args:
        model: Trained multi-head classification model.
        data_loader: Torch DataLoader containing validation or test data.
        levels (List[str]): List of taxonomic levels used for encoding.
        taxonomy_df (pd.DataFrame): Taxonomy metadata dataframe containing encoded labels.

    Returns:
        Tuple of:
            - prob_distributions (dict): Max predicted probs for each taxonomic level.
            - lookup (dict): Map from (level, class_name) → flat label.
    """
    model.eval()
    prob_distributions = {level: [] for level in levels}

    with torch.no_grad():
        for batch in data_loader:
            images, _ = batch
            images = images.to(device)
            logits_all_levels = model(images)

            for level, logits in logits_all_levels.items():
                probs = F.softmax(logits, dim=1)
                max_probs = probs.max(dim=1).values
                prob_distributions[level].extend(max_probs.cpu().tolist())

    label_to_taxonomy = taxonomy_df.set_index("label")
    lookup = {}
    for _, row in label_to_taxonomy.iterrows():
        for level in levels:
            name = row.get(level)
            if pd.notna(name):
                lookup[(level, name)] = row.name

    return prob_distributions, lookup


prob_distributions, lookup = prepare_probs_and_lookup(best_model, val_loader, levels, taxonomy_df)


def plot_prob_distributions(prob_distributions: Dict[str, List[float]], levels: List[str]) -> None:
    """
    Plot histogram of max predicted probabilities per taxonomic level.

    Args:
        prob_distributions (dict): Dictionary mapping level → list of max probabilities.
        levels (List[str]): List of taxonomic levels used for encoding.
    """
    warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")
    sns.set(style="whitegrid")

    n_levels = len(prob_distributions)
    ncols = 3
    nrows = int(np.ceil(n_levels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, level in enumerate(prob_distributions):
        ax = axes[i]
        sns.histplot(prob_distributions[level], kde=True, binwidth=0.05, ax=ax, color="steelblue")
        ax.set_title(f"{level} Level")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Max Predicted Probability")
        ax.set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribution of Max Predicted Probabilities per Taxonomic Rank", fontsize=16)
    plt.show()


plot_prob_distributions(prob_distributions, levels)


taxonomic_mappings = {}

# Function to create child->parent mapping array for adjacent levels
def create_taxonomic_mapping(df, child_lvl, parent_lvl, encoders, unknown_idx):
    # Get unique child classes
    unique_children = df[f"{child_lvl}_encoded"].unique()
    max_child_idx = encoders[child_lvl].classes_.shape[0]
    
    # Initialize mapping array with unknown parent index (default fallback)
    mapping = np.full(max_child_idx, unknown_idx, dtype=np.int32)
    
    # For each unique child, find the most common parent and map child idx -> parent idx
    for child_idx in unique_children:
        # Filter rows for this child_idx
        parent_vals = df.loc[df[f"{child_lvl}_encoded"] == child_idx, f"{parent_lvl}_encoded"]
        if parent_vals.empty:
            continue
        
        # Find the most common parent class index (mode)
        parent_mode = parent_vals.mode()
        if not parent_mode.empty:
            mapping[child_idx] = parent_mode.iloc[0]
    
    return mapping

# Fill unknown indices from your dict
unknown_idx = unknown_class_indices

# Create mappings for adjacent levels (child -> parent)
taxonomic_mappings["species_to_genus"] = create_taxonomic_mapping(
    train_annotations_df_extended, "Species", "Genus", encoders, unknown_idx["Genus"]
)
taxonomic_mappings["genus_to_family"] = create_taxonomic_mapping(
    train_annotations_df_extended, "Genus", "Family", encoders, unknown_idx["Family"]
)
taxonomic_mappings["family_to_order"] = create_taxonomic_mapping(
    train_annotations_df_extended, "Family", "Order", encoders, unknown_idx["Order"]
)
taxonomic_mappings["order_to_class"] = create_taxonomic_mapping(
    train_annotations_df_extended, "Order", "Class", encoders, unknown_idx["Class"]
)
taxonomic_mappings["class_to_phylum"] = create_taxonomic_mapping(
    train_annotations_df_extended, "Class", "Phylum", encoders, unknown_idx["Phylum"]
)


def predict_with_fallback(model,
                          data_loader,
                          levels: List[str],
                          thresholds: Dict[str, float],
                          lookup: Dict[Tuple[str, str], int],
                          silent=False,
                          taxonomic_mappings=taxonomic_mappings,
                          encoders=encoders) -> pd.DataFrame:
    """
    Predict using fixed threshold + prefer deepest valid level strategy.
    Added taxonomy consistency check between child and parent predictions.

    Args:
        model: Trained multi-head classification model.
        data_loader: DataLoader for test/validation data.
        levels (List[str]): List of taxonomic levels used for encoding.
        thresholds (dict): Confidence thresholds per taxonomic level.
        lookup (dict): (level, predicted name) → final flat label.
        silent (bool): Defaults to False. Suppress print statements.
        taxonomic_mappings (dict): Optional dict like
            {
                "species_to_genus": np.array([...]),
                "genus_to_family": np.array([...]),
                ...
            }
        encoders (dict): level -> LabelEncoder used for inverse transforming indices to names.

    Returns:
        pd.DataFrame with ['annotation_id', 'concept_name'].
    """
    model.eval()
    submission_labels = []
    fallback_count = 0
    consistency_fail_count = 0
    level_counts = defaultdict(int)
    
    # Map each level to its parent level, for easy lookup
    parent_level_map = {}
    for i, lvl in enumerate(levels):
        if i < len(levels) - 1:
            parent_level_map[lvl] = levels[i + 1]
        else:
            parent_level_map[lvl] = None  # highest level has no parent

    with torch.no_grad():
        for batch in data_loader:
            images, ids_batch = batch
            images = images.to(device)
            logits_all_levels = model(images)
            batch_size = images.shape[0]

            for i in range(batch_size):
                best_fallback = None
                max_prob_overall = -1
                valid_predictions = []

                preds_info = {}  # store pred idx and prob for all levels for this sample

                for level in levels:
                    logits = logits_all_levels[level][i]
                    probs = F.softmax(logits, dim=0)
                    max_prob, pred_idx = torch.max(probs, dim=0)
                    pred_name = encoders[level].inverse_transform([pred_idx.item()])[0]

                    preds_info[level] = {
                        "idx": pred_idx.item(),
                        "prob": max_prob.item(),
                        "name": pred_name,
                    }

                    # Track best fallback in case none exceed threshold
                    if max_prob.item() > max_prob_overall:
                        max_prob_overall = max_prob.item()
                        best_fallback = (level, pred_name)

                    # Track valid predictions that pass the level-specific threshold
                    if max_prob.item() >= thresholds[level]:
                        valid_predictions.append(level)

                # Now filter valid_predictions by taxonomy consistency, deepest level first
                final_level = None
                for lvl in reversed(valid_predictions):  # start from deepest
                    pred = preds_info[lvl]["idx"]
                    prob = preds_info[lvl]["prob"]

                    parent_lvl = parent_level_map[lvl]
                    if parent_lvl is None:
                        # No parent level, accept
                        final_level = lvl
                        break

                    parent_pred = preds_info[parent_lvl]["idx"]
                    parent_prob = preds_info[parent_lvl]["prob"]

                    # Check taxonomy consistency if mapping is available
                    map_key = f"{lvl.lower()}_to_{parent_lvl.lower()}"
                    if taxonomic_mappings and map_key in taxonomic_mappings:
                        mapped_parent = taxonomic_mappings[map_key][pred]
                        if mapped_parent == parent_pred and parent_prob >= thresholds[parent_lvl]:
                            # Consistent with parent
                            final_level = lvl
                            break
                        else:
                            # Consistency failed, try fallback level (go to next shallower valid prediction)
                            consistency_fail_count += 1
                            continue
                    else:
                        # No mapping provided, accept prediction
                        final_level = lvl
                        break

                # If no consistent level found, fallback to highest prob overall
                if final_level is None:
                    final_level, pred_name = best_fallback
                    fallback_count += 1
                else:
                    pred_name = preds_info[final_level]["name"]

                label = lookup.get((final_level, pred_name))
                if label is not None:
                    submission_labels.append(label)
                    level_counts[final_level] += 1
                else:
                    # True fallback: random if all else fails
                    random_label = random.choice(list(lookup.values()))
                    submission_labels.append(random_label)
                    level_counts["random"] += 1
                    fallback_count += 1

    if not silent:
        print(f"\nFallback to random label occurred {fallback_count} times out of {len(submission_labels)} predictions.")
        print(f"Taxonomy consistency check failed {consistency_fail_count} times.")
        print("\nPrediction counts by taxonomic level:")
        for level, count in level_counts.items():
            print(f"{level:>10}: {count}")

    submission_df = pd.DataFrame({
        "annotation_id": range(1, len(submission_labels) + 1),
        "concept_name": submission_labels
    })

    return submission_df


def create_solution_df_from_val_dataset(val_dataset, taxonomy_df: pd.DataFrame, levels: List[str]) -> pd.DataFrame:
    """
    Generate solution dataframe from a validation dataset and taxonomy.

    Args:
        val_dataset: Dataset with __getitem__ returning (image, label_dict).
        taxonomy_df (pd.DataFrame): Taxonomy dataframe with encoded labels.
        levels (List[str]): List of taxonomic levels used for encoding.

    Returns:
        pd.DataFrame: Ground-truth DataFrame with columns ['annotation_id', 'concept_name'].
    """
    rows = []
    taxonomy_df = taxonomy_df.fillna("Unknown")
    key_cols = [f"{level}_encoded" for level in levels]
    mapping = {
        tuple(row[key_cols].astype(int)): row["label"]
        for _, row in taxonomy_df.iterrows()
    }

    for i in range(len(val_dataset)):
        _, label_dict = val_dataset[i]
        key = tuple(int(label_dict.get(level, -1)) for level in levels)
        concept_name = mapping.get(key, "Unknown")
        rows.append({"annotation_id": i + 1, "concept_name": concept_name})

    return pd.DataFrame(rows)


solution_df = create_solution_df_from_val_dataset(val_dataset, taxonomy_df, levels)
solution_df.head()


# Search space: discrete thresholds from 0.1 to 0.9 in steps of 0.1
threshold_options = [round(x, 2) for x in np.arange(0.1, 1.0, 0.1)]

# To collect trial results
trial_results = []

# Create study
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=30, seed=seed)
)

# Objective function
def objective(trial):

    # Get thresholds to try out
    thresholds = {
        level: trial.suggest_categorical(level, threshold_options)
        for level in levels
    }

    # Get score
    submission_df = predict_with_fallback(best_model, val_loader, levels, thresholds, lookup, silent=True)
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        trial_score = score(solution_df, submission_df, row_id_column_name="annotation_id")

    # Log trial result
    trial_result = {**thresholds, "trial_score": trial_score}
    trial_results.append(trial_result)

    return trial_score

# Run study
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, timeout=600)

# Convert results to DataFrame
trial_results_df = pd.DataFrame(trial_results)


print(f"Trials run: {len(study.trials)}")


# Average thresholds of top k trials
top_k = max(10, int(0.1 * len(study.trials)))  # Use top 10% of trials, at least 10
top_thresholds_df = trial_results_df.sort_values("trial_score").head(top_k)
top_thresholds_df


# Calculate average per level
tuned_thresholds = {
    level: round(top_thresholds_df[level].mean(), 2)
    for level in levels
}
tuned_thresholds


submission_df = predict_with_fallback(best_model, val_loader, levels, tuned_thresholds, lookup)
submission_df.head()


# Create a dummy stream to capture excessive prints
f = io.StringIO()
with contextlib.redirect_stdout(f):
    overall_score = score(solution_df, submission_df, row_id_column_name="annotation_id")


print(f"Overall taxonomic hierarchical distance for val dataset: {overall_score:.2f}")


test_annotations_df = pd.read_csv("../../test/annotations.csv")
test_dataset = FathomNetDataset(test_annotations_df, levels, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=3)


prob_distributions, lookup = prepare_probs_and_lookup(best_model, test_loader, levels, taxonomy_df)


plot_prob_distributions(prob_distributions, levels)


# Make thresholds for test set more permissive
adjusted_tuned_thresholds = {
    "Phylum": 0.25,
    "Class": 0.60,
    "Order": 0.35,
    "Family": 0.45,
    "Genus": 0.65,
    "Species": 0.70
}


submission_df = predict_with_fallback(best_model, test_loader, levels, adjusted_tuned_thresholds, lookup)
submission_df.head()


submission_df["concept_name"].value_counts()


submission_df.to_csv("submission.csv", index=False)

