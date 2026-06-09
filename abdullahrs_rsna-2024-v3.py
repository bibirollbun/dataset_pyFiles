import os
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import json
import numpy as np
import torch
import pydicom
import cv2
from tqdm.auto import tqdm
import torchvision.transforms as T
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision.models import resnet50, efficientnet_b0
import torch.optim as optim


# Define paths
base_path = Path("../input/rsna-2024-lumbar-spine-degenerative-classification")
output_path = Path("../working")
train_csv_path = base_path / "train.csv"
label_coordinates_csv_path = base_path / "train_label_coordinates.csv"
series_descriptions_csv_path = base_path / "train_series_descriptions.csv"
train_images_path = base_path / "train_images"

# Load CSV files
train_df = pd.read_csv(train_csv_path)
label_coords_df = pd.read_csv(label_coordinates_csv_path)
series_desc_df = pd.read_csv(series_descriptions_csv_path)

# Display a few rows from each CSV
print("Train CSV Sample:")
display(train_df.head())

print("Label Coordinates CSV Sample:")
display(label_coords_df.head())

print("Series Descriptions CSV Sample:")
display(series_desc_df.head())

print("Coordinates nan counts")
display(label_coords_df.isna().sum())
# Check class distribution
severity_cols = [
    col for col in train_df.columns if "stenosis" in col or "narrowing" in col
]
severity_counts = train_df[severity_cols].apply(pd.Series.value_counts).sum(axis=1)
print("\nClass Distribution Across Severity Levels:")
print(severity_counts)

# Visualize class distribution
severity_counts.plot(kind="bar", figsize=(10, 6), title="Severity Class Distribution")
plt.show()

# Explore a single study's image files
sample_study_id = train_df.iloc[0]["study_id"]
sample_study_path = train_images_path / str(sample_study_id)
sample_series = os.listdir(sample_study_path)
print(f"\nSample Study ID: {sample_study_id}")
print(f"Available Series in the Study: {sample_series}")

# Visualize a single image
def show_dicom_image(series_path):
    dicom_files = list(Path(series_path).glob("*.dcm"))
    if not dicom_files:
        print(f"No DICOM files found in {series_path}")
        return
    sample_dicom = pydicom.dcmread(dicom_files[0])
    image = sample_dicom.pixel_array
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.title(f"DICOM Image from {series_path}")
    plt.axis("off")
    plt.show()

for series_id in sample_series:
    print(f"\nDisplaying series: {series_id}")
    show_dicom_image(sample_study_path / series_id)

del sample_study_id, sample_study_path, sample_series, severity_counts, train_df, label_coords_df, series_desc_df


train = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv')
figure, axis = plt.subplots(1,3, figsize=(20,5)) 
for idx, d in enumerate(['foraminal', 'subarticular', 'canal']):
    diagnosis = list(filter(lambda x: x.find(d) > -1, train.columns))
    dff = train[diagnosis]
    
    value_counts = dff.apply(pd.value_counts).fillna(0).T
    value_counts.plot(kind='bar', stacked=True, ax=axis[idx])
    axis[idx].set_title(f'{d} distribution')
del train


class SpineDataProcessor:

    def __init__(self, base_path, output_path, roi_size=224):

        self.base_path = Path(base_path)

        self.output_path = Path(output_path)

        self.roi_size = roi_size

        self.processed_data = {"samples": {}, "processed_rois": {}}

        # Load metadata

        self.train_df = pd.read_csv(self.base_path / "train.csv")

        self.coords_df = pd.read_csv(self.base_path / "train_label_coordinates.csv")

        self.series_df = pd.read_csv(self.base_path / "train_series_descriptions.csv")

    def extract_roi(self, image, points):
        """
        Extract ROI that contains all points while maintaining aspect ratio
        points: list of (x,y) coordinates for conditions in the same image
        """
        h, w = image.shape

        # Get bounding box that contains all points
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        # Add padding to ensure we capture enough context
        padding = 32
        min_x = max(0, int(min(x_coords) - padding))
        max_x = min(w, int(max(x_coords) + padding))
        min_y = max(0, int(min(y_coords) - padding))
        max_y = min(h, int(max(y_coords) + padding))

        # Extract region containing all points
        roi = image[min_y:max_y, min_x:max_x]

        # Calculate padding needed to maintain aspect ratio
        roi_h, roi_w = roi.shape
        if roi_h > roi_w:
            # Add padding to width
            target_w = int(roi_h * (224 / 224))  # maintain square aspect ratio
            pad_w = target_w - roi_w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            roi = np.pad(roi, ((0, 0), (pad_left, pad_right)), mode="constant")
        else:
            # Add padding to height
            target_h = int(roi_w * (224 / 224))
            pad_h = target_h - roi_h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            roi = np.pad(roi, ((pad_top, pad_bottom), (0, 0)), mode="constant")

        # Resize to 224x224 while maintaining aspect ratio
        roi_resized = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)

        # Calculate scaled coordinates for all points
        scaled_points = []
        for x, y in points:
            # Adjust for padding and scaling
            if roi_h > roi_w:
                scaled_x = ((x - min_x + pad_left) * 224) / roi.shape[1]
                scaled_y = ((y - min_y) * 224) / roi.shape[0]
            else:
                scaled_x = ((x - min_x) * 224) / roi.shape[1]
                scaled_y = ((y - min_y + pad_top) * 224) / roi.shape[0]
            scaled_points.append((scaled_x, scaled_y))

        return roi_resized, scaled_points

    def augment_image(self, image, num_augmentations=4):
        """Apply basic augmentations for moderate/severe cases"""
        augmented = []
        for _ in range(num_augmentations):
            # Convert to tensor for torchvision transforms
            img_tensor = torch.from_numpy(image).unsqueeze(0)

            # Apply random transformations
            transforms = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.GaussianBlur(kernel_size=3),
                    T.RandomRotation(15),
                    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                ]
            )

            aug_tensor = transforms(img_tensor)
            augmented.append(aug_tensor.squeeze(0).numpy())

        return augmented

    def load_dicom(
        self, study_id: str, series_id: str, instance_number: int
    ) -> np.ndarray:
        """Load and preprocess DICOM image"""

        dicom_path = (
            self.base_path
            / "train_images"
            / str(study_id)
            / str(series_id)
            / f"{instance_number}.dcm"
        )

        try:

            dcm = pydicom.dcmread(str(dicom_path))

            image = dcm.pixel_array

            # Convert to float and normalize

            image = image.astype(float)

            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(
                np.uint8
            )

            # Convert to grayscale if needed

            if len(image.shape) > 2:

                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            return image

        except Exception as e:

            print(f"Error loading DICOM {dicom_path}: {e}")

            return None

    def save_processed_data(self):
        """Save processed data to disk with type conversion"""

        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            return obj

        # Convert and save metadata
        converted_samples = convert_numpy_types(self.processed_data["samples"])
        self.output_path.mkdir(parents=True, exist_ok=True)

        with open(self.output_path / "metadata.json", "w") as f:
            json.dump(converted_samples, f)

        # Save ROIs
        # Convert instance numbers to strings and ensure all numpy arrays are properly handled
        processed_rois = {}
        for study_id, study_data in self.processed_data["processed_rois"].items():
            processed_rois[str(study_id)] = {}
            for instance_num, instance_data in study_data.items():
                processed_rois[str(study_id)][str(instance_num)] = {
                    "rois": [
                        {
                            "series_id": roi["series_id"],
                            "condition": roi["condition"],
                            "level": roi["level"],
                            "image": roi["image"],  # Keep as numpy array
                            "original_coords": convert_numpy_types(
                                roi["original_coords"]
                            ),
                            "scaled_coords": convert_numpy_types(roi["scaled_coords"]),
                        }
                        for roi in instance_data["rois"]
                    ]
                }

        # Save ROIs using numpy's save function
        np.save(self.output_path / "processed_rois.npy", processed_rois)

    def process_study(self, study_id: int):
        """Process a single study with all its series"""
        try:
            study_id_str = str(study_id)
            # Pre-allocate dictionaries with expected structure
            processed_samples = {"series_info": [], "conditions": {}}
            processed_rois = {}

            # Filter dataframes once using boolean indexing instead of multiple times
            study_series = self.series_df[self.series_df["study_id"] == study_id]
            study_coords = self.coords_df[self.coords_df["study_id"] == study_id]

            # Batch process series info
            processed_samples["series_info"].extend(
                [
                    {
                        "series_id": series["series_id"],
                        "series_description": series["series_description"],
                    }
                    for _, series in study_series.iterrows()
                ]
            )

            # Create temporary data structure for instance grouping
            instance_data = {}

            # Process coordinates in batch
            for _, coord_row in study_coords.iterrows():
                instance_num = coord_row["instance_number"]
                series_id = coord_row["series_id"]
                condition = coord_row["condition"]
                level = coord_row["level"].lower().replace("/", "_")
                x, y = coord_row["x"], coord_row["y"]

                # Get label once per condition/level pair
                label = self.get_label(study_id, condition, level)

                # Initialize nested dictionaries if needed using dict.setdefault
                level_dict = processed_samples["conditions"].setdefault(level, {})
                condition_list = level_dict.setdefault(condition, [])

                # Append coordinate information
                coord_info = {
                    "series_id": series_id,
                    "instance_number": instance_num,
                    "x": x,
                    "y": y,
                    "label": label,
                }
                condition_list.append(coord_info)

                # Group by instance number for ROI processing
                instance_key = (instance_num, series_id)
                if instance_key not in instance_data:
                    instance_data[instance_key] = {
                        "coords": [],
                        "conditions": set(),
                        "levels": set(),
                        "label": label,
                    }
                instance_data[instance_key]["coords"].append((x, y))
                instance_data[instance_key]["conditions"].add(condition)
                instance_data[instance_key]["levels"].add(level)

            # Process ROIs in batch
            for (instance_num, series_id), data in instance_data.items():
                # Load DICOM image once per instance
                image = self.load_dicom(study_id, series_id, instance_num)
                if image is None:
                    continue

                # Extract ROI once per instance
                roi, scaled_coords = self.extract_roi(image, data["coords"])

                # Initialize ROI storage using dict.setdefault
                instance_rois = processed_rois.setdefault(instance_num, {"rois": []})

                # Create base ROI data
                for condition in data["conditions"]:
                    for level in data["levels"]:
                        roi_data = {
                            "series_id": series_id,
                            "condition": condition,
                            "level": level,
                            "image": roi,
                            "original_coords": data["coords"],
                            "scaled_coords": scaled_coords,
                        }
                        instance_rois["rois"].append(roi_data)

                        # Apply augmentation if needed
                        if data["label"] in ["Moderate", "Severe"]:
                            augmented_rois = self.augment_image(roi)
                            for aug_roi in augmented_rois:
                                aug_data = roi_data.copy()
                                aug_data["image"] = aug_roi
                                instance_rois["rois"].append(aug_data)

            # Update final processed data
            self.processed_data["samples"][study_id_str] = processed_samples
            self.processed_data["processed_rois"][study_id_str] = processed_rois

            return True

        except Exception as e:
            print(f"Error processing study {study_id}: {e}")
            return False

    def get_label(self, study_id: str, condition: str, level: str) -> str:
        """Get label (Normal/Mild, Moderate, Severe) for a specific condition and level"""
        condition_col = f"{condition.replace(' ', '_').lower()}_{level}"
        study_row = self.train_df[self.train_df["study_id"] == study_id]
        if not study_row.empty and condition_col in study_row.columns:

            try:
                return study_row[condition_col].iloc[0]
            except:
                pass

        return "Normal/Mild"  # Default case


processor = SpineDataProcessor(
    base_path=base_path,
    output_path=output_path
)

# Get unique study IDs
study_ids = processor.train_df['study_id'].unique()
# Process each study with progress bar
for study_id in tqdm(study_ids, desc="Processing studies"):
    processor.process_study(study_id)

processor.save_processed_data()
del processor


class DataVisualizer:
    def __init__(self, base_path, processed_path):
        self.base_path = Path(base_path)
        self.processed_path = Path(processed_path)
        
        print("Loading metadata and processed ROIs...")
        # Load processed data
        with open(self.processed_path / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.processed_rois = np.load(self.processed_path / "processed_rois.npy", allow_pickle=True).item()
        
        print(f"Found {len(self.metadata)} studies in metadata")
        print(f"Found {len(self.processed_rois)} studies in processed ROIs")

    def visualize_study(self, study_id: str):
        """Simple visualization of original and processed images with coordinates."""
        metadata = self.metadata[study_id]
        processed_rois = self.processed_rois[study_id]
        
        for instance_num, roi_data in processed_rois.items():
            for roi_info in roi_data['rois']:
                # Get series ID from ROI info
                roi_series_id = roi_info['series_id']
                
                # Load original image with matching series ID
                original_img = self.load_original_dicom(study_id, roi_series_id, instance_num)
                if original_img is None:
                    continue
                
                # Create figure with subplots
                fig, (ax1, ax2) = plt.subplots(1, 2)
                
                # Plot original
                ax1.imshow(original_img, cmap='gray')
                ax1.set_title('Original')
                if roi_info['original_coords']:
                    coords = np.array(roi_info['original_coords'])
                    ax1.scatter(coords[:, 0], coords[:, 1], c='red', marker='x', s=100)
                ax1.axis('off')
                
                # Plot processed
                ax2.imshow(roi_info['image'], cmap='gray')
                ax2.set_title('Processed ROI')
                if roi_info['scaled_coords']:
                    coords = np.array(roi_info['scaled_coords'])
                    ax2.scatter(coords[:, 0], coords[:, 1], c='red', marker='x', s=100)
                ax2.axis('off')
                
                plt.show()
            
    def load_original_dicom(self, study_id, series_id, instance_number):
        """Load original DICOM image with error handling"""
        try:
            dicom_path = self.base_path / 'train_images' / str(study_id) / str(series_id) / f"{instance_number}.dcm"
            if not dicom_path.exists():
                return None
                
            print(f"Loading DICOM from: {dicom_path}")
            dcm = pydicom.dcmread(dicom_path)
            image = dcm.pixel_array
            
            # Normalize to 0-255
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return image
            
        except Exception as e:
            print(f"Error loading DICOM: {e}")
            return None

    def get_condition_counts(self, num_samples=3):
        """Count occurrences of each condition-level pair and their severity"""
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for study_data in self.metadata.values():
            for level, conditions in study_data['conditions'].items():
                for condition, instances in conditions.items():
                    for instance in instances:
                        counts[condition][level][instance['label']] += 1
        
        # Print counts in a structured way
        for condition in counts:
            print(f"\n{condition}:")
            for level in counts[condition]:
                print(f"  {level}:")
                for severity, count in counts[condition][level].items():
                    print(f"    {severity}: {count}")

        """Visualize original and processed images for given severity"""
        plt.figure(figsize=(15, 5*num_samples))
        sample_count = 0
        
        for study_id, study_data in self.processed_rois.items():
            for instance_num, instance_data in study_data.items():
                for roi_data in instance_data['rois']:
                    if roi_data.get('label') == severity and sample_count < num_samples:
                        # Get original image
                        original_img = self.load_original_dicom(
                            study_id, 
                            roi_data['series_id'],
                            roi_data['instance_number']
                        )
                        
                        # Plot original image
                        plt.subplot(num_samples, 2, sample_count*2 + 1)
                        plt.imshow(original_img, cmap='gray')
                        plt.plot(roi_data['original_coords']['x'], 
                               roi_data['original_coords']['y'], 
                               'r.', markersize=10)
                        circle = plt.Circle((roi_data['original_coords']['x'], 
                                          roi_data['original_coords']['y']), 
                                         radius=112,  # half of 224 for ROI visualization
                                         color='r', 
                                         fill=False)
                        plt.gca().add_patch(circle)
                        plt.title(f"Original - {roi_data['condition']} {roi_data['level']}")
                        
                        # Plot processed ROI
                        plt.subplot(num_samples, 2, sample_count*2 + 2)
                        plt.imshow(roi_data['image'], cmap='gray')
                        plt.plot(roi_data['scaled_coords']['x'], 
                               roi_data['scaled_coords']['y'], 
                               'r.', markersize=10)
                        plt.title(f"Processed ROI - {severity}")
                        
                        sample_count += 1
                        if sample_count >= num_samples:
                            break
            if sample_count >= num_samples:
                break
                
        plt.tight_layout()
        plt.show()



visualizer = DataVisualizer(
    base_path=base_path,
    processed_path=output_path
)
print("Condition-Level Pair Counts:")
visualizer.get_condition_counts()
visualizer.visualize_study("4003253")
del visualizer
# print(visualizer.processed_rois['4003253'].keys())
# print(visualizer.processed_rois['4003253']['8'].keys())
# print(visualizer.processed_rois['4003253']['8']['rois'][0].keys())
# print(visualizer.processed_rois['4003253']['8']['rois'][0].values())
# print(visualizer.processed_rois['4003253'].values())


class SpineDataset(Dataset):
    def __init__(self, processed_rois, metadata, study_ids):
        self.processed_rois = processed_rois
        self.metadata = metadata
        self.study_ids = study_ids
        self.series_type_mapping = {
            "Sagittal T1": 0,
            "Axial T2": 1,
            "Sagittal T2/STIR": 2,
        }

        # Define all possible conditions and levels
        self.conditions = [
            "Spinal Canal Stenosis",
            "Left Neural Foraminal Narrowing",
            "Right Neural Foraminal Narrowing",
            "Left Subarticular Stenosis",
            "Right Subarticular Stenosis",
        ]
        self.levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

        # Filter out studies with no ROIs and preprocess
        self.valid_study_ids = []
        self.cached_data = {}
        self._preprocess_data()

    def __len__(self):
        return len(self.valid_study_ids)

    def get_series_type_encoding(self, series_id, study_id):
        series_desc = next(
            s["series_description"]
            for s in self.metadata[study_id]["series_info"]
            if s["series_id"] == series_id
        )
        return self.series_type_mapping[series_desc]

    def _preprocess_data(self):
        """Preprocess and cache all data during initialization"""
        print("Preprocessing and caching dataset...")

        for study_id in tqdm(self.study_ids):
            try:
                study_rois = self.processed_rois[study_id]
                study_metadata = self.metadata[study_id]

                # Process images and series types
                all_images = []
                all_series_types = []
                all_conditions = []

                # Check if study has any ROIs
                has_rois = False
                for instance_num, instance_data in study_rois.items():
                    if "rois" in instance_data and instance_data["rois"]:
                        has_rois = True
                        for roi in instance_data["rois"]:
                            # Normalize and convert to tensor (keeping in CPU)
                            image = torch.FloatTensor(roi["image"]) / 255.0
                            series_type = self.get_series_type_encoding(
                                roi["series_id"], study_id
                            )

                            all_images.append(image)
                            all_series_types.append(series_type)
                            all_conditions.append((roi["condition"], roi["level"]))

                # Skip studies with no ROIs
                if not has_rois or not all_images:
                    print(f"Warning: Study {study_id} has no ROIs, skipping...")
                    continue

                # Stack images and convert series types to tensor
                images = torch.stack(all_images)
                series_types = torch.tensor(all_series_types)

                # Create and fill labels tensor
                labels = torch.zeros(25, 3)

                for i, condition in enumerate(self.conditions):
                    for j, level in enumerate(self.levels):
                        idx = i * 5 + j
                        if level in study_metadata["conditions"]:
                            if condition in study_metadata["conditions"][level]:
                                label = study_metadata["conditions"][level][condition][
                                    0
                                ]["label"]
                                if label == "Normal/Mild":
                                    labels[idx] = torch.tensor([1, 0, 0])
                                elif label == "Moderate":
                                    labels[idx] = torch.tensor([0, 1, 0])
                                elif label == "Severe":
                                    labels[idx] = torch.tensor([0, 0, 1])

                # Cache processed data
                self.cached_data[study_id] = {
                    "images": images,
                    "series_types": series_types,
                    "labels": labels,
                }
                self.valid_study_ids.append(study_id)

            except Exception as e:
                print(f"Error processing study {study_id}: {str(e)}")
                continue

        print(
            f"Dataset preprocessing completed! Valid studies: {len(self.valid_study_ids)}"
        )

    def __getitem__(self, idx):
        """Get preprocessed data from cache"""
        study_id = self.valid_study_ids[idx]
        cached_item = self.cached_data[study_id]

        return {
            "images": cached_item["images"],
            "series_types": cached_item["series_types"],
            "labels": cached_item["labels"],
            "study_id": study_id,
        }


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    
    def forward(self, x):
        # x: [batch_size, input_dim]
        weights = self.attention(x)  # [batch_size, 1]
        weights = torch.softmax(weights, dim=1)
        return x * weights  # [batch_size, input_dim]

class SpineModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initialize EfficientNet-B0 without pretrained weights
        self.efficient_net = efficientnet_b0(weights=None)
        
        # Modify first conv layer for single channel
        self.efficient_net.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Get feature dimensions (1280 for EfficientNet-B0)
        self.feature_dim = 1280
        
        # Series type embedding
        self.series_embedding = nn.Embedding(3, 32)
        
        # Classification head with attention
        self.attention1 = MultiHeadAttention(self.feature_dim + 32)
        self.fc1 = nn.Linear(self.feature_dim + 32, 256)
        self.attention2 = MultiHeadAttention(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 25 * 3)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images, series_types):
        batch_size = images.size(0)
        num_rois = images.size(1)
        
        # Process each image through EfficientNet
        images = images.view(-1, 1, images.size(-2), images.size(-1))
        features = self.efficient_net.features(images)
        features = self.efficient_net.avgpool(features)
        features = torch.flatten(features, 1)  # [batch_size * num_rois, feature_dim]
        
        # Get series type embeddings
        series_embeddings = self.series_embedding(series_types.view(-1))  # [batch_size * num_rois, 32]
        
        # Concatenate features and embeddings
        combined = torch.cat([features, series_embeddings], dim=1)  # [batch_size * num_rois, feature_dim + 32]
        
        # Apply attention and classification
        x = self.attention1(combined)
        x = self.fc1(x)
        x = torch.relu(x)  # Use ReLU as in original
        x = self.attention2(x)
        x = self.dropout1(x)
        predictions = self.fc2(x)
        
        # Reshape to [batch_size, num_rois, 25 * 3]
        predictions = predictions.view(batch_size, num_rois, -1)
        
        # Create a mask for valid ROIs (non-zero images)
        roi_mask = (images.view(batch_size, num_rois, -1).sum(dim=-1) != 0).float()
        roi_mask = roi_mask.unsqueeze(-1)  # Add dimension for broadcasting
        
        # Apply mask and average predictions across valid ROIs only
        masked_predictions = predictions * roi_mask
        predictions = masked_predictions.sum(dim=1) / (roi_mask.sum(dim=1) + 1e-6)
        
        # Reshape to [batch_size, 25, 3] and apply softmax
        predictions = predictions.view(batch_size, 25, 3)
        predictions = torch.softmax(predictions, dim=-1)
        
        return predictions


class SpineTrainer:
    def __init__(self, processed_rois, metadata, device="cuda"):
        self.processed_rois = processed_rois
        self.metadata = metadata
        self.device = device

        # Create model
        self.model = SpineModel().to(device)

        # Calculate class weights
        self.class_weights = self.calculate_class_weights()
        print(f"Class weights device: {self.class_weights.device}")
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scaler = torch.amp.GradScaler()

    def calculate_class_weights(self):
        # Count occurrences of each class
        class_counts = defaultdict(int)
        total_samples = 0

        for study_data in self.metadata.values():
            for level_data in study_data["conditions"].values():
                for condition_data in level_data.values():
                    label = condition_data[0]["label"]
                    class_counts[label] += 1
                    total_samples += 1

        # Calculate weights
        weights = torch.zeros(3)
        weights[0] = total_samples / (3 * class_counts["Normal/Mild"])
        weights[1] = total_samples / (3 * class_counts["Moderate"])
        weights[2] = total_samples / (3 * class_counts["Severe"])

        return weights.to(self.device)

    def train_epoch(self, train_loader):
        print("Train epoch started")
        self.model.train()
        print("Model is training...")
        total_loss = 0
        print(f"\nStarting to iterate through {len(train_loader)} batches...")
        for batch in tqdm(
            train_loader, total=len(train_loader), desc="iterate through train batches"
        ):
            try:
                images = batch["images"].to(self.device)
                series_types = batch["series_types"].to(self.device)
                labels = batch["labels"].to(self.device)

                self.optimizer.zero_grad()

                with torch.amp.autocast(device.type):
                    outputs = self.model(images, series_types)
                    # Calculate loss for each condition
                    loss = 0
                    for i in range(25):
                        loss += self.criterion(outputs[:, i], labels[:, i])
                    loss = loss / 25
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
            except Exception as err:
                print(err)

        print(f"\nEpoch completed. Average loss: {total_loss / len(train_loader)}")
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        print("\nStarting validation...")
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                total=len(val_loader),
                desc="iterate through validation batches",
            ):
                images = batch["images"].to(self.device)
                series_types = batch["series_types"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(images, series_types)

                loss = 0
                for i in range(25):
                    loss += self.criterion(outputs[:, i], labels[:, i])
                loss = loss / 25

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, num_epochs=10):
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")

    def predict(self, test_loader):
        self.model.eval()
        predictions = {}

        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader), desc="Prediction"):
                images = batch["images"].to(self.device)
                series_types = batch["series_types"].to(self.device)
                study_ids = batch["study_id"]

                outputs = self.model(images, series_types)

                # Store predictions
                for i, study_id in enumerate(study_ids):
                    predictions[study_id] = outputs[i].cpu().numpy()

        return predictions


def load_data(processed_rois_path, metadata_path):
    """Load the processed ROIs and metadata"""
    # Load processed ROIs
    processed_rois = np.load(processed_rois_path, allow_pickle=True).item()

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return processed_rois, metadata

def custom_collate(batch):
    """Custom collate function to handle variable number of ROIs per study"""
    # Get max number of ROIs in this batch
    max_rois = max([b["images"].size(0) for b in batch])

    # Get other dimensions from first item
    first = batch[0]
    img_h, img_w = first["images"].size(-2), first["images"].size(-1)

    # Initialize tensors for the batch
    batch_size = len(batch)
    batched_images = torch.zeros(batch_size, max_rois, img_h, img_w)
    batched_series_types = torch.zeros(batch_size, max_rois, dtype=torch.long)
    batched_labels = torch.stack([b["labels"] for b in batch])
    study_ids = [b["study_id"] for b in batch]

    # Fill in the batched tensors
    for i, item in enumerate(batch):
        num_rois = item["images"].size(0)
        batched_images[i, :num_rois] = item["images"]
        batched_series_types[i, :num_rois] = item["series_types"]

    return {
        "images": batched_images,
        "series_types": batched_series_types,
        "labels": batched_labels,
        "study_id": study_ids,
    }


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, processed_rois, study_series, target_size=(256, 256)):
        self.processed_rois = processed_rois
        self.study_series = {s["series_id"]: s for s in study_series}
        self.target_size = target_size
        self.series_type_mapping = {
            "Sagittal T1": 0,
            "Axial T2": 1,
            "Sagittal T2/STIR": 2,
        }

        # Prepare instance list
        self.instances = []
        for instance_num, instance_data in processed_rois.items():
            self.instances.append((instance_num, instance_data))

    def get_series_type(self, series_id):
        series_info = self.study_series.get(series_id)
        if series_info:
            return self.series_type_mapping[series_info["series_description"]]
        return 0  # Default value if not found

    def resize_image(self, image):
        """Resize image to target size"""
        h, w = image.shape
        if (h, w) != self.target_size:
            # Convert to PIL Image for resizing
            from PIL import Image
            import numpy as np

            img_pil = Image.fromarray(image)
            img_pil = img_pil.resize(self.target_size, Image.Resampling.BILINEAR)
            return np.array(img_pil)
        return image

    def __getitem__(self, idx):
        instance_num, instance_data = self.instances[idx]

        # Process all ROIs in this instance
        all_images = []
        all_series_types = []

        for roi in instance_data["rois"]:
            # Resize image before converting to tensor
            resized_image = self.resize_image(roi["image"])
            image = torch.FloatTensor(resized_image) / 255.0
            series_type = self.get_series_type(roi["series_id"])

            all_images.append(image)
            all_series_types.append(series_type)

        try:
            # Stack tensors
            images = torch.stack(all_images)
            series_types = torch.tensor(all_series_types)

            return {
                "images": images,
                "series_types": series_types,
                "instance_num": instance_num,
            }
        except Exception as e:
            print(f"Error stacking tensors for instance {instance_num}:")
            print(f"Image shapes: {[img.shape for img in all_images]}")
            raise e

    def __len__(self):
        return len(self.instances)


def load_test_metadata(csv_path):
    """Load and structure test series descriptions"""
    df = pd.read_csv(csv_path)

    # Group by study_id
    study_metadata = defaultdict(list)
    for _, row in df.iterrows():
        study_metadata[str(row["study_id"])].append(
            {
                "series_id": str(row["series_id"]),
                "series_description": row["series_description"],
            }
        )

    return study_metadata

def process_test_image(dcm_path):
    """Process a single DICOM image"""
    try:
        # Read DICOM
        dcm = pydicom.dcmread(dcm_path)
        image = dcm.pixel_array.astype(float)

        # Normalize image
        if image.max() != image.min():
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(
                np.uint8
            )
        else:
            image = np.zeros_like(image, dtype=np.uint8)

        # print(f"Processed image shape: {image.shape}")  # Debug info
        return image

    except Exception as e:
        print(f"Error processing DICOM {dcm_path}: {str(e)}")
        raise e
    
def process_test_study(test_path, study_id, study_series):
    """Process all images for a study"""
    processed_rois = {}

    # Process each series
    for series_info in study_series:
        series_id = series_info["series_id"]
        series_path = test_path / str(study_id) / str(series_id)

        if not series_path.exists():
            print(f"Warning: Series path {series_path} does not exist")
            continue

        # Process each DICOM in series
        for dcm_path in series_path.glob("*.dcm"):
            instance_num = int(dcm_path.stem)
            try:
                processed_image = process_test_image(dcm_path)

                if instance_num not in processed_rois:
                    processed_rois[instance_num] = {"rois": []}

                processed_rois[instance_num]["rois"].append(
                    {"series_id": series_id, "image": processed_image}
                )

            except Exception as e:
                print(f"Error processing {dcm_path}: {str(e)}")
                continue

    return processed_rois


def predict_study(model, test_loader, device):
    """Generate predictions for a study"""
    model.eval()  # Set model to evaluation mode
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            series_types = batch['series_types'].to(device)
            
            try:
                outputs = model(images, series_types)
                # Ensure outputs are the right shape (batch_size, 25, 3)
                if len(outputs.shape) != 3 or outputs.shape[1:] != (25, 3):
                    print(f"WARNING: Unexpected output shape: {outputs.shape}")
                    continue
                    
                predictions.append(outputs.cpu().numpy())
                
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                print(f"Images shape: {images.shape}")
                print(f"Series types shape: {series_types.shape}")
                continue
    
    if not predictions:
        raise RuntimeError("No valid predictions were generated for this study")
    
    # Stack all predictions
    predictions = np.concatenate(predictions, axis=0)
    
    # Average predictions across all instances
    final_prediction = np.mean(predictions, axis=0)
    
    # Ensure final prediction has shape (25, 3)
    if final_prediction.shape != (25, 3):
        print(f"WARNING: Final prediction has unexpected shape: {final_prediction.shape}")
        if len(final_prediction.shape) == 3 and final_prediction.shape[0] == 1:
            final_prediction = final_prediction.squeeze(0)
    
    # Ensure probabilities sum to 1
    final_prediction = final_prediction / final_prediction.sum(axis=1, keepdims=True)
    
    return final_prediction

def process_and_predict(test_path, model, series_csv_path, device):
    """Main function to process test data and generate predictions"""
    study_metadata = load_test_metadata(series_csv_path)

    test_path = Path(test_path)
    predictions = {}

    # Process each study
    for study_id in study_metadata.keys():
        print(f"Processing study {study_id}")

        study_series = study_metadata[study_id]

        # Process study images
        processed_rois = process_test_study(test_path, study_id, study_series)

        if not processed_rois:
            print(f"Warning: No ROIs processed for study {study_id}")
            continue

        # Create dataset and dataloader
        test_dataset = TestDataset(processed_rois, study_series)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=2
        )

        # Generate predictions
        study_predictions = predict_study(model, test_loader, device)
        predictions[study_id] = study_predictions

    return predictions


def create_submission(predictions, output_path):
    """Create submission file from model predictions"""
    rows = []
    conditions = [
        "spinal_canal_stenosis",
        "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        "left_subarticular_stenosis",
        "right_subarticular_stenosis",
    ]
    levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

    for study_id, study_preds in predictions.items():

        # Reshape predictions if necessary
        if len(study_preds.shape) == 3:  # If shape is (1, 25, 3)
            study_preds = study_preds.squeeze(0)  # Convert to (25, 3)

        if len(study_preds.shape) != 2 or study_preds.shape != (25, 3):
            print(
                f"WARNING: Unexpected prediction shape for study {study_id}: {study_preds.shape}"
            )
            continue

        for condition_idx, condition in enumerate(conditions):
            for level_idx, level in enumerate(levels):
                pred_idx = condition_idx * len(levels) + level_idx
                probs = study_preds[pred_idx]

                row_id = f"{study_id}_{condition}_{level}"
                rows.append(
                    {
                        "row_id": row_id,
                        "normal_mild": float(probs[0]),
                        "moderate": float(probs[1]),
                        "severe": float(probs[2]),
                    }
                )

    if not rows:
        raise ValueError("No predictions were processed successfully!")

    submission_df = pd.DataFrame(rows)

    # Verify probabilities sum to approximately 1
    prob_sum = submission_df[["normal_mild", "moderate", "severe"]].sum(axis=1)
    if not np.allclose(prob_sum, 1.0, atol=1e-5):
        print("WARNING: Some probability rows don't sum to 1!")
        print("Probability sums range:", prob_sum.min(), "to", prob_sum.max())

    submission_df.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
test_base_path = (
    base_path / "test_images"
)
# # Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# # Hyperparameters
BATCH_SIZE = 1
NUM_EPOCHS = 30
VAL_SIZE = 0.2
NUM_WORKERS = 0

# # Load data
processed_rois, metadata = load_data(
    output_path /"processed_rois.npy", output_path / "metadata.json"
)

# Get all study IDs
study_ids = list(metadata.keys())

# Split into train and validation sets
train_ids, val_ids = train_test_split(
    study_ids, test_size=VAL_SIZE, random_state=42
)

print(f"Number of training studies: {len(train_ids)}")
print(f"Number of validation studies: {len(val_ids)}")

# Create datasets
train_dataset = SpineDataset(processed_rois, metadata, train_ids)
val_dataset = SpineDataset(processed_rois, metadata, val_ids)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=custom_collate,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=custom_collate,
)

print("Length of the train data :", len(train_loader))
print("Length of the val data :", len(val_loader))

# # Initialize trainer
trainer = SpineTrainer(
    processed_rois=processed_rois, metadata=metadata, device=device
)

# # Training loop with checkpointing
best_val_loss = float("inf")
save_dir = output_path
save_dir.mkdir(exist_ok=True)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 20)

    # Train and validate
    train_loss = trainer.train_epoch(train_loader)
    val_loss = trainer.validate(val_loader)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

    # Save checkpoint if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = (
            save_dir / f"model_epoch_{epoch+1}_valloss_{val_loss:.4f}.pth"
        )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
            },
            checkpoint_path,
        )

        print(f"Saved checkpoint to {checkpoint_path}")

print("Training completed!")

# # Load best model for predictions
best_model_path = sorted(
    save_dir.glob("*.pth"),
    key=lambda x: float(str(x).split("valloss_")[1].split(".pth")[0]),
)[0]
print(f"Loading best model from {best_model_path}")

checkpoint = torch.load(best_model_path)
trainer.model.load_state_dict(checkpoint["model_state_dict"])

# Generate predictions for validation set
# val_predictions = trainer.predict(val_loader)

test_predictions = process_and_predict(
    test_base_path,
    trainer.model,
    base_path /"test_series_descriptions.csv",
    device=device,
)

# Create submission file for validation set
# create_submission(val_predictions, "validation_predictions.csv")
create_submission(test_predictions, output_path / "submission.csv")

print("Created validation predictions file!")


df = pd.read_csv("../working/submission.csv")

display(df)

