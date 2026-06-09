import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_curve, auc, cohen_kappa_score)
from sklearn.preprocessing import StandardScaler  
warnings.filterwarnings('ignore')

import cv2
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from collections import defaultdict
from tqdm import tqdm
# from easyfsl.samplers import TaskSampler


# Directories
# MY_DIR = "/home/rs/20CS91P02/projects/22CS10066_deepak/"
BASE_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
PREPROCESSED_DIR = "/kaggle/working/preprocessed/"
os.makedirs(PREPROCESSED_DIR, exist_ok=True)


df= pd.read_csv(f"{BASE_DIR}train.csv")
df.head()

df_org = pd.read_csv(f"{BASE_DIR}train.csv")
# Print the total number of rows in the dataset
print(f"Total rows in the dataset: {len(df)}")

#Randomly select 10,000 rows for a quick training check
df_subset = df_org.sample(n=14286, random_state=42)
# df_subset = df_org.sample(n=1000, random_state=42)
print(f"Total rows in the dataset: {len(df_subset)}")
# # Display the first few rows of the sampled dataframe
df_subset.head()


#code to split df in train,test val 70, 15,15 
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
test_df, val_df = train_test_split(temp_df, test_size=0.50, random_state=42)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# Save the datasets
train_csv = "/kaggle/working/train10K_70.csv"
val_csv = "/kaggle/working/val10K_15.csv"
test_csv = "/kaggle/working/test10K_15.csv"

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f"Train CSV saved to: {train_csv}")
print(f"Validation CSV saved to: {val_csv}")
print(f"Test CSV saved to: {test_csv}")





# Extract EEGid, labels, and offsets
# EEGid_label_list = df_subset[["eeg_id", "expert_consensus", "eeg_label_offset_seconds"]].values.tolist()

# X = []
# y = []
# prev_eegId = ""

brain_activities = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']
activity_mapping = {activity: idx for idx, activity in enumerate(brain_activities)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OptimizedBrainDataset(Dataset):
    def __init__(self, csv_file, base_dir, activity_mapping, preprocessed_dir="preprocessed"):
        self.df = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.activity_mapping = activity_mapping
        self.preprocessed_dir = preprocessed_dir
        self.resize_transform = transforms.Resize((224, 224))
        
        # Create directory if needed
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        
        # Memory map preprocessed files
        self.spect_mmaps = {}
        spect_ids = self.df["spectrogram_id"].unique()
        for spect_id in spect_ids:
            npy_path = f"{self.preprocessed_dir}/{spect_id}.npy"
            if not os.path.exists(npy_path):
                self._preprocess_and_save(spect_id)
            self.spect_mmaps[spect_id] = np.load(npy_path, mmap_mode='r')

    def __len__(self):
        return len(self.df)

    def _preprocess_and_save(self, spect_id):
        """Batch process and save spectrogram once"""
        parquet_path = f'{self.base_dir}/train_spectrograms/{spect_id}.parquet'
        temp_df = pd.read_parquet(parquet_path).drop('time', axis=1)
        
        # Process entire spectrogram
        arr = temp_df.to_numpy()
        arr = np.log1p(arr)
        arr /= arr.max()
        arr = np.nan_to_num(arr, nan=1e-4)
        arr_uint8 = (255 * arr).astype(np.uint8)
        
        np.save(f"{self.preprocessed_dir}/{spect_id}.npy", arr_uint8)

    def __getitem__(self, idx):
        spect_id, label, offset = self.df.iloc[idx][["spectrogram_id", "expert_consensus", "spectrogram_label_offset_seconds"]]
        start = int(offset) // 2
        
        # Direct memory access
        spectrogram = self.spect_mmaps[spect_id]
        segment = spectrogram[start:start+300]
        
        # Convert to RGB
        rgb_image = cv2.applyColorMap(segment, cv2.COLORMAP_JET)
        rgb_image = rgb_image.astype(np.float32) / 255.0
        tensor_image = torch.tensor(rgb_image).permute(2, 0, 1)
        return self.resize_transform(tensor_image), torch.tensor(self.activity_mapping[label])



class EpisodeGenerator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_to_indices = defaultdict(list)
        
        # Precompute once
        for idx, (_, label) in enumerate(dataset):
            self.class_to_indices[label.item()].append(idx)

        # Ensure all classes have sufficient samples
        min_samples = min(len(indices) for indices in self.class_to_indices.values())
        # print(f"Minimum samples per class: {min_samples}")
            
    def get_episode(self, n_way=6, k_shot=3, q_queries=3):
        all_classes = list(self.class_to_indices.keys())
        if len(all_classes) < n_way:
            raise ValueError(f"Dataset only has {len(all_classes)} classes, need {n_way}")
        # selected_classes = random.sample(list(self.class_to_indices.keys()), n_way)
        selected_classes = all_classes[:n_way]
        support, query = [], []
        
        for cls in selected_classes:
            available_indices = self.class_to_indices[cls]
            if len(available_indices) < (k_shot + q_queries):
                # Handle insufficient samples
                indices = np.random.choice(available_indices, 
                                         size=k_shot+q_queries, replace=True)
            else:
                indices = np.random.choice(available_indices, 
                                         size=k_shot+q_queries, replace=False)
            # indices = np.random.choice(self.class_to_indices[cls], 
            #                          size=k_shot+q_queries, replace=False)
            support.extend(indices[:k_shot])
            query.extend(indices[k_shot:])
        
        # Batch loading
        sX = torch.stack([self.dataset[i][0] for i in support])
        qX = torch.stack([self.dataset[i][0] for i in query])
        sy = torch.tensor([self.dataset[i][1] for i in support])
        qy = torch.tensor([self.dataset[i][1] for i in query])
        
        return sX, sy, qX, qy



def pregenerate_episodes(dataset, cache_size=1000):
    import gc
    generator = EpisodeGenerator(dataset)
    episode_cache = []
    
    # Warmup GPU
    _ = torch.rand(1).to(device)
    
    print(f"=> Pre-generating {cache_size} episodes")
    for i in tqdm(range(cache_size)):
        sX, sy, qX, qy = generator.get_episode()
        
        # Async GPU transfer with pinned memory
        sX = sX.pin_memory().to(device, non_blocking=True)
        sy = sy.pin_memory().to(device, non_blocking=True)
        qX = qX.pin_memory().to(device, non_blocking=True)
        qy = qy.pin_memory().to(device, non_blocking=True)
        
        episode_cache.append((sX, sy, qX, qy))

        # Clear CUDA cache every 25 episodes to prevent OOM
        if i % 25 == 0 and i > 0:
            torch.cuda.empty_cache()
            
        # Force garbage collection every 50 episodes
        if i % 50 == 0 and i > 0:
            gc.collect()
            torch.cuda.empty_cache()
            
        # Print memory usage every 100 episodes
        if i % 100 == 0 and i > 0:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    return episode_cache



class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()  # Required for PyTorch modules
        self.backbone = backbone  # Essential for parameter storage

    def forward(self, sX, sy, qX):
        z_support = self.backbone(sX)
        z_query = self.backbone(qX)
        
        prototypes = torch.stack([
            z_support[sy == cls].mean(0) 
            for cls in torch.unique(sy)
        ])
        
        return -torch.cdist(z_query, prototypes)



# Run once before training
train_dataset = OptimizedBrainDataset(train_csv, BASE_DIR, activity_mapping)
test_dataset =  OptimizedBrainDataset(test_csv, BASE_DIR, activity_mapping)
val_dataset =  OptimizedBrainDataset(val_csv, BASE_DIR, activity_mapping)
print("Data Processing done")


# Initialize backbone
# efficientnet = models.efficientnet_b3(pretrained=True)
efficientnet = models.efficientnet_v2_s(pretrained=True)
efficientnet.classifier = nn.Identity()  # Keep only feature extractor
model = PrototypicalNetworks(efficientnet).cuda()


# from collections import defaultdict
# import random

def get_image_episode(dataset, n_way=6, k_shot=3, q_queries=3, return_original=False):
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_idx = label.item() if torch.is_tensor(label) else label
        class_to_indices[class_idx].append(idx)

    selected_classes = random.sample(list(class_to_indices.keys()), n_way)
    support_images, query_images = [], []
    support_labels, query_labels = [], []
    label_map = {cls: i for i, cls in enumerate(selected_classes)}

    for cls in selected_classes:
        indices = class_to_indices[cls]
        selected = random.sample(indices, k_shot + q_queries)
        s_idx, q_idx = selected[:k_shot], selected[k_shot:]

        for idx in s_idx:
            img, _ = dataset[idx]
            support_images.append(img)
            support_labels.append(label_map[cls])
        for idx in q_idx:
            img, _ = dataset[idx]
            query_images.append(img)
            query_labels.append(label_map[cls])

    if return_original:
        return (torch.stack(support_images),
                torch.tensor(support_labels),
                torch.stack(query_images),
                torch.tensor(query_labels),
                selected_classes)  # Return original class IDs
    else:
        return (torch.stack(support_images),
                torch.tensor(support_labels),
                torch.stack(query_images),
                torch.tensor(query_labels))


# Task sampling configuration
N_WAY = 6
N_SHOT = 10
N_QUERY = 10

# Initial evaluation before training
model.eval()
sX_test, sy_test, qX_test, qy_test = get_image_episode(test_dataset, n_way=N_WAY, k_shot=N_SHOT, q_queries=N_QUERY)
sX_test, sy_test = sX_test.to(device), sy_test.to(device)
qX_test, qy_test = qX_test.to(device), qy_test.to(device)

with torch.no_grad():
    scores = model(sX_test, sy_test, qX_test)
    class_ids, _ = torch.unique(sy_test, sorted=True, return_inverse=True)
    class_to_index = {cls.item(): idx for idx, cls in enumerate(class_ids)}
    qy_indices = torch.tensor([class_to_index[c.item()] for c in qy_test], device=device)
    preds = torch.argmax(scores, dim=1)
    initial_acc = (preds == qy_indices).float().mean().item()

print(f"Initial Test Accuracy: {initial_acc:.4f}")


# Training loop
embedding_model = model  # Use your PrototypicalNetworks model
optimizer = torch.optim.Adam(embedding_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

n_episodes = 1000  # or whatever you prefer
patience = 10
# patience_counter = 0
# best_val_acc = 0.0

# Pre-generate episode cache
# EPISODE_CACHE_SIZE = 1000  # Adjust based on memory constraints
EPISODE_CACHE_SIZE = min(300, len(train_dataset)//(N_WAY*(N_SHOT+N_QUERY)))
print(f"episode cache size: {EPISODE_CACHE_SIZE}")
episode_cache = []

print("==> Pre-generating training episodes...")
episode_cache = pregenerate_episodes(train_dataset, EPISODE_CACHE_SIZE)
print("Done")

# Modified training loop with cached episodes
best_val_acc = 0.0
patience_counter = 0

for episode in range(n_episodes):
    # Get pre-generated episode
    sX, sy, qX, qy = episode_cache[episode % EPISODE_CACHE_SIZE]
    
    optimizer.zero_grad()
    
    # Forward pass
    scores = embedding_model(sX, sy, qX)
    
    # Create episode-specific class mapping
    class_ids, _ = torch.unique(sy, sorted=True, return_inverse=True)
    class_to_index = {cls.item(): idx for idx, cls in enumerate(class_ids)}
    qy_indices = torch.tensor([class_to_index[c.item()] for c in qy], device=device)
    
    # Calculate loss
    loss = criterion(scores, qy_indices)
    loss.backward()
    optimizer.step()

    # Training metrics
    preds = torch.argmax(scores, dim=1)
    acc = (preds == qy_indices).float().mean().item()
    
    if episode % 10 == 0:
        print(f"[Episode {episode}] Loss: {loss.item():.4f} | Train Acc: {acc:.4f}")

    # Memory management during training
    if episode % 50 == 0 and episode > 0:
        torch.cuda.empty_cache()

    # Validation (unchanged)
    if episode % 50 == 0:
        embedding_model.eval()
        with torch.no_grad():
            val_sX, val_sy, val_qX, val_qy = get_image_episode(val_dataset, n_way=N_WAY, k_shot=N_SHOT, q_queries=N_QUERY)
            val_sX, val_sy = val_sX.to(device), val_sy.to(device)
            val_qX, val_qy = val_qX.to(device), val_qy.to(device)
            
            val_scores = embedding_model(val_sX, val_sy, val_qX)
            val_class_ids, _ = torch.unique(val_sy, sorted=True, return_inverse=True)
            val_class_to_index = {cls.item(): idx for idx, cls in enumerate(val_class_ids)}
            val_qy_indices = torch.tensor([val_class_to_index[c.item()] for c in val_qy], device=device)
            
            val_preds = torch.argmax(val_scores, dim=1)
            val_acc = (val_preds == val_qy_indices).float().mean().item()
            
            print(f"--> [Validation] Episode {episode} | Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(embedding_model.state_dict(), "best_embedding_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f" No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(" Early stopping triggered.")
                    break
            # Clear validation tensors to free memory
            del val_sX, val_sy, val_qX, val_qy, val_scores
            torch.cuda.empty_cache()
            
        embedding_model.train()



# Classification metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ... [your existing evaluation code] ...
# Final evaluation on test set with comprehensive metrics
print("\n=== Final Evaluation ===")
embedding_model.load_state_dict(torch.load("best_embedding_model.pt"))
embedding_model.eval()

test_accuracies = []
all_true = []
all_preds = []
n_test_episodes = 100  # Standard practice for few-shot evaluation

with torch.no_grad():
    for episode_idx in tqdm(range(n_test_episodes), desc="Test Episodes"):
        # Generate episode ensuring valid class distribution
        while True:
            try:
                sX_test, sy_test, qX_test, qy_test = get_image_episode(
                    test_dataset, 
                    n_way=N_WAY, 
                    k_shot=N_SHOT, 
                    q_queries=N_QUERY
                )
                # Validate episode structure
                assert len(torch.unique(sy_test)) == N_WAY
                break
            except (ValueError, AssertionError):
                continue

        # Device transfer
        sX_test, sy_test = sX_test.to(device), sy_test.to(device)
        qX_test, qy_test = qX_test.to(device), qy_test.to(device)

        # Forward pass
        scores = embedding_model(sX_test, sy_test, qX_test)

        # Create episode-specific mapping
        class_ids, _ = torch.unique(sy_test, sorted=True, return_inverse=True)
        class_to_index = {cls.item(): idx for idx, cls in enumerate(class_ids)}
        
        # Convert query labels using episode mapping
        qy_indices = torch.tensor([class_to_index[c.item()] for c in qy_test], device=device)
        
        # Calculate metrics
        preds = torch.argmax(scores, dim=1)
        
        # Store original class labels for comprehensive metrics
        true_labels = [class_ids[idx].item() for idx in qy_indices]
        pred_labels = [class_ids[idx].item() for idx in preds]
        
        all_true.extend(true_labels)
        all_preds.extend(pred_labels)

        # Calculate episode accuracy
        episode_acc = (preds == qy_indices).float().mean().item()
        test_accuracies.append(episode_acc)

# Calculate final statistics
mean_acc = np.mean(test_accuracies)
std_acc = np.std(test_accuracies)
confidence_interval = 1.96 * std_acc / np.sqrt(n_test_episodes)

# Classification metrics
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

print("\n=== Comprehensive Metrics ===")
print(f"Final Test Accuracy over {n_test_episodes} episodes:")
print(f"Mean Accuracy: {mean_acc:.4f} ± {confidence_interval:.4f} (95% CI)")
print(f"Standard Deviation: {std_acc:.4f}")

# Generate classification report
report = classification_report(
    all_true, 
    all_preds, 
    labels=list(range(len(brain_activities))),
    target_names=brain_activities,
    zero_division=0,
    output_dict=True
)

# Calculate overall accuracy separately
overall_accuracy = accuracy_score(all_true, all_preds)

# Macro-averaged metrics
print("\nMacro-Averaged Scores:")
print(f"Precision: {report['macro avg']['precision']:.4f}")
print(f"Recall: {report['macro avg']['recall']:.4f}")
print(f"F1-Score: {report['macro avg']['f1-score']:.4f}")

# Confusion matrix with heatmap visualization
cm = confusion_matrix(all_true, all_preds, labels=list(range(len(brain_activities))))

# Create heatmap
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)

# Create DataFrame for better labeling
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {c}" for c in brain_activities],
    columns=[f"Predicted {c}" for c in brain_activities]
)

# Plot heatmap
sns.heatmap(cm_df, 
           annot=True,           # Show numbers in cells
           fmt='d',              # Format as integers
           cmap='Blues',         # Color scheme
           square=True,          # Square cells
           linewidths=0.5,       # Add grid lines
           cbar_kws={'label': 'Number of Samples'})

plt.title('Confusion Matrix - Brain Activity Classification', fontsize=16, pad=20)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('Actual Labels', fontsize=14)
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

# Also print the numerical matrix for reference
print("\nConfusion Matrix (Numerical):")
print(cm_df)

# Per-class metrics
print("\nDetailed Class Performance:")
for class_name in brain_activities:
    if class_name in report:
        print(f"\n{class_name}:")
        print(f"  Precision: {report[class_name]['precision']:.4f}")
        print(f"  Recall:    {report[class_name]['recall']:.4f}")
        print(f"  F1-Score:  {report[class_name]['f1-score']:.4f}")
        print(f"  Support:   {report[class_name]['support']}")

# Additional metrics
print("\nAdditional Metrics:")
print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
print(f"Overall Accuracy: {overall_accuracy:.4f}")





