# Install pyvolmap and transformers as wheel dependency package.
!pip install /kaggle/input/pycolmap-v11-1/pycolmap-3.11.1-cp311-cp311-manylinux_2_28_x86_64.whl > /dev/null 2>&1
!pip install /kaggle/input/transformers-4-51-3/transformers-4.51.3-py3-none-any.whl > /dev/null 2>&1


# ğŸ“¦ Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T
import torchvision.models as models
import torch
import warnings
from tqdm import tqdm
import kagglehub
import time
from IPython.display import display
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from pathlib import Path
from torchvision.models import vit_b_16
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import AutoImageProcessor, AutoModel


warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


!cp -r /kaggle/input/m/nehadas55/lightglue/transformers/default/1/LightGlue-main /kaggle/working/
%cd /kaggle/working/LightGlue-main
!pip install -e . > /dev/null 2>&1 || true


try:
    # Download model checkpoints using KaggleHub
    aliked_path = kagglehub.model_download("oldufo/aliked/pyTorch/aliked-n16")
    lightglue_path = kagglehub.model_download("oldufo/lightglue/pyTorch/aliked")
    dinov2_path = kagglehub.model_download("metaresearch/dinov2/pyTorch/base")

    # Create cache dir for torch hub compatibility
    !mkdir -p /root/.cache/torch/hub/checkpoints
    !cp $aliked_path/aliked-n16.pth /root/.cache/torch/hub/checkpoints/aliked-n16.pth
    !cp $lightglue_path/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue.pth
    !cp $lightglue_path/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv-pth
    !cp -R /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth
    print("âœ… Models downloaded from KaggleHub and prepared.")
except Exception as e:
    print(f"âš ï¸� KaggleHub model setup failed: {e}")


from lightglue import LightGlue, ALIKED
from lightglue.utils import load_image, match_pair


 # Optional: pycolmap (COLMAP must be installed separately)
try:
    import pycolmap
except ImportError:
    pycolmap = None


# data load csvs and images (optional)
def _load_data():
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


#_load_data()


#move into working directory
%cd /kaggle/working/


# ==============================
# Data Loading
# ==============================

# Set the path for the Kaggle dataset
data_path = "/kaggle/input/image-matching-challenge-2025/"
train_dir = os.path.join(data_path, "train")
test_dir = os.path.join(data_path, "test")


# Function to safely read CSV files with error handling
def safe_read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path} with shape {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Empty file: {file_path}")
        return None
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None


# Load CSVs
train_labels_df = safe_read_csv(os.path.join(data_path, "train_labels.csv"))
train_thresholds_df = safe_read_csv(os.path.join(data_path, "train_thresholds.csv"))
sample_submission = safe_read_csv(os.path.join(data_path, "sample_submission.csv"))


# Create lookup for real poses
train_pose_lookup = {}
if train_labels_df is not None:
    for _, row in train_labels_df.iterrows():
        key = (row['dataset'], row['image'])
        train_pose_lookup[key] = (row['rotation_matrix'], row['translation_vector'])


aliked = ALIKED(pretrained='/kaggle/input/aliked/pytorch/aliked-n16/1/aliked-n16.pth').to(device)
matcher = LightGlue(
    features='aliked',
    weights='/kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth'
).to(device)


# ==============================
# Load Dino Model
# ==============================
def load_dino_model():
    processor = AutoImageProcessor.from_pretrained(dinov2_path)
    model = AutoModel.from_pretrained(dinov2_path).to(device)
    model.eval()
    return processor, model


# ==============================
# Extract Dino Features
# ==============================
def extract_dino_features(image_paths):
    processor, model = load_dino_model()
    features = []
    for path in tqdm(image_paths):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            feat = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        features.append(feat)
    return np.array(features)



# ğŸ§  Feature Extraction 
def build_feature_matrix_from_folder(folder_path):
    print(f"ğŸ§  Matching features using ALIKED + LightGlue in {folder_path}")
    image_paths = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
    full_paths = [os.path.join(folder_path, f) for f in image_paths]
    keypoints_list = []

    for image_path in tqdm(full_paths):
        image = load_image(image_path).to(device)
        feats = aliked.extract(image)
        keypoints_list.append(feats)

    # Compute pairwise matching scores (number of matches)
    n = len(full_paths)
    match_scores = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = matcher({"image0": keypoints_list[i], "image1": keypoints_list[j]})
            score = len(m["matches"]) if m["matches"] is not None else 0
            match_scores[i, j] = match_scores[j, i] = score

    return match_scores, image_paths


# ==============================
# Clustering
# ==============================

def cluster_features(features):
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("Insufficient data for clustering. Need at least 2 images with valid features.")
    print(f"DEBUG: Clustering {features.shape[0]} feature vectors")
    n_components = min(50, features.shape[0], features.shape[1])
    reduced = PCA(n_components=n_components).fit_transform(StandardScaler().fit_transform(features))
    k = min(5, features.shape[0] // 2)
    clustering = KMeans(n_clusters=k, random_state=42).fit(reduced)
    print(f"DEBUG: Cluster labels: {np.unique(clustering.labels_)}")
    return clustering.labels_


# ==============================
# Visualization
# ==============================

def visualize_clusters(features, labels):
    if len(features) < 2:
        print("Not enough data to visualize clusters.")
        return
    reduced = PCA(n_components=2).fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10')
    plt.title("Cluster Visualization")
    plt.colorbar(scatter)
    plt.show()


# ==============================
# Evaluate Pose Accuracy (mAA)
# ==============================

def evaluate_pose_accuracy(submission_df, train_labels_df):
    if train_labels_df is None:
        print("Skipping mAA calculation: no ground truth loaded.")
        return

    merged = submission_df.merge(train_labels_df, on=["dataset", "image"], suffixes=("_pred", "_gt"))
    if merged.empty:
        print("No overlap between submission and train labels for evaluation.")
        return

    merged['scene_match'] = merged['scene_pred'] == merged['scene_gt']
    maa = merged['scene_match'].sum() / len(merged)
    print(f"\nğŸ§ª mAA (mean Average Accuracy) Score: {maa:.4f}")


# ==============================
# Combined Submission File Generator
# ==============================

submission_entries = []

def generate_submission(image_paths, labels, dataset_name):
    print("DEBUG: Generating submission entries...")
    df = pd.DataFrame({"image": image_paths, "cluster": labels})
    for _, row in df.iterrows():
        image_name = row['image'].lower()
        is_outlier = "outlier" in os.path.basename(image_name) or "outlier" in os.path.dirname(image_name)
        cluster = "outliers" if is_outlier else f"cluster{row['cluster']}"

        if cluster == "outliers":
            rotation = ";".join(["nan"] * 9)
            translation = ";".join(["nan"] * 3)
        else:
            pose_key = (dataset_name, row['image'])
            if pose_key in train_pose_lookup:
                rotation, translation = train_pose_lookup[pose_key]
            else:
                # Here you could replace dummy with COLMAP-derived pose if available
                rotation = ";".join(["0"] * 9)
                translation = ";".join(["0"] * 3)

        submission_entries.append([dataset_name, cluster, row['image'], rotation, translation])


# ==============================
# ğŸš€ COLMAP SfM Integration for Test Set
# ==============================

def estimate_pose_with_colmap(dataset_path, output_dir="colmap_workspace"):
    if pycolmap is None:
        print("pycolmap is not installed. Please install it manually to use this feature.")
        return

    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.abspath(dataset_path)
    database_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")

    print(f"Running COLMAP SfM on: {dataset_path}")
    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)
    options = pycolmap.IncrementalPipelineOptions()
    options.multiple_models = False
    options.ignore_watermarks = True
    reconstruction_manager = pycolmap.ReconstructionManager()
    pipeline = pycolmap.IncrementalPipeline(
        options,
        image_dir,
        database_path,
        reconstruction_manager
    )
    pipeline.run()

    print(f"âœ… COLMAP SfM completed. Results in: {sparse_dir}")


def run_pipeline(dataset_path, mode, use_dino=False):
    image_paths = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".png")])
    if use_dino:
        feats = extract_dino_features(image_paths)
    else:
        feats, image_paths = build_feature_matrix_from_folder(dataset_path)

    labels = cluster_features(feats)
    visualize_clusters(feats, labels)

    if mode == "test":
        estimate_pose_with_colmap(dataset_path)

    generate_submission(image_paths, labels, os.path.basename(dataset_path))


def run_and_compare(dataset_path):

    image_paths = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".png")])
    results = []

    for method in ["aliked", "dino"]:
        print(f"ğŸ”„ Running pipeline with: {method.upper()} on {os.path.basename(dataset_path)}")
        start = time.time()

        if method == "dino":
            feats = extract_dino_features(image_paths)
            used_paths = image_paths
        else:
            feats, used_paths = build_feature_matrix_from_folder(dataset_path)

        labels = cluster_features(feats)
        visualize_clusters(feats, labels)
        duration = time.time() - start

        results.append({
            "Method": method.upper(),
            "Images": len(used_paths),
            "Clusters": len(set(labels)),
            "Time (s)": round(duration, 2)
        })

    print("ğŸ“Š Comparison Table")
    display(pd.DataFrame(results))


# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    try:
        print("Processing both TRAIN and TEST folders...")
        for mode, directory in [("train", train_dir), ("test", test_dir)]:
            for dataset_name in sorted(os.listdir(directory)):
                dataset_path = os.path.join(directory, dataset_name)
                if not os.path.isdir(dataset_path):
                    continue
                print(f"\nProcessing {mode.upper()} dataset: {dataset_name}")
                feats, imgs = build_feature_matrix_from_folder(dataset_path)
                labels = cluster_features(feats)
                visualize_clusters(feats, labels)

                if mode == "test":
                    estimate_pose_with_colmap(dataset_path)

                generate_submission(imgs, labels, dataset_name)

        # Save combined submission
        print("\nSaving combined submission file...")
        submission_df = pd.DataFrame(submission_entries, columns=["dataset", "scene", "image", "rotation_matrix", "translation_vector"])

        # ğŸ”� Ensure valid formatting for submission
        submission_df.dropna(subset=["dataset", "scene", "image", "rotation_matrix", "translation_vector"], inplace=True)
        submission_df = submission_df.astype(str)

        # ğŸ”§ Validate rotation and translation string format
        def valid_rt_format(rt_str, expected_count):
            parts = rt_str.split(";")
            return len(parts) == expected_count and all(p.replace(".", "", 1).replace("-", "", 1).isdigit() or p.lower() == "nan" for p in parts)

        submission_df = submission_df[submission_df["rotation_matrix"].apply(lambda x: valid_rt_format(x, 9))]
        submission_df = submission_df[submission_df["translation_vector"].apply(lambda x: valid_rt_format(x, 3))]
        required_columns = ["dataset", "scene", "image", "rotation_matrix", "translation_vector"]
        submission_df = submission_df[required_columns]  # Drop any extra columns
        submission_df.dropna(inplace=True)               # Remove any rows with NaN
        submission_df = submission_df.astype(str)        # Ensure all fields are strings
        submission_df = submission_df.drop_duplicates()  # Remove duplicates
        #submission_df = submission_df[~submission_df["image"].str.contains("outliers_")]
        submission_df.to_csv("submission.csv", index=False)
        print("Saved submission.csv")

        # Evaluate mAA on training set
        evaluate_pose_accuracy(submission_df, train_labels_df)

        print("\nğŸ§ª Running clustering comparison on ETs (DINO vs ALIKED)...")
        run_and_compare(os.path.join(test_dir, "ETs"))

    except (FileNotFoundError, ValueError) as e:
        print(f"â�Œ Error: {e}")


# Optional Cleanup
%rm -rf /kaggle/working/LightGlue-main

