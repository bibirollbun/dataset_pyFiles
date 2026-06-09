!pip install /kaggle/input/faiss-cpu/faiss_cpu-1.10.0-cp311-cp311-manylinux_2_28_x86_64.whl


import os
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import csv
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.models import efficientnet_b0
from collections import defaultdict

# --- Classifier setup ---
class_map = {
    "CurveFault_A": 0, "CurveFault_B": 1, "CurveVel_A": 2, "CurveVel_B": 3,
    "FlatFault_A": 4, "FlatFault_B": 5, "FlatVel_A": 6, "FlatVel_B": 7,
    "Style_A": 8, "Style_B": 9,
}
inv_class_map = {v: k for k, v in class_map.items()}

class SeismicEfficientNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        if self.backbone.features[0][0].in_channels != 1:
            self.backbone.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = SeismicEfficientNetClassifier(num_classes=10)
classifier.load_state_dict(torch.load('/kaggle/input/waveform-inversion-models/classifier.pth', map_location=device))
classifier.to(device)
classifier.eval()

# --- Submission setup ---
test_files = list(Path('/kaggle/input/waveform-inversion/test').glob('*.npy'))
x_cols = [f'x_{i}' for i in range(1, 70, 2)]
fieldnames = ['oid_ypos'] + x_cols

def get_class_train_files(category):
    # Define all data sources to search
    data_sources = [
        # Original data source
        # {'base_path': '/kaggle/input/waveform-inversion/train_samples', 'subdir': True},
        # Additional data sources - add more as needed
        {'base_path': '/kaggle/input/waveform-inversion-1', 'subdir': False},
        {'base_path': '/kaggle/input/waveform-inversion-2', 'subdir': False},
        # {'base_path': '/kaggle/input/waveform-inversion-3', 'subdir': False},
        # Add more sources as needed
    ]
    
    all_input_files = []
    all_output_files = []
    
    for source in data_sources:
        base_path = source['base_path']
        if source['subdir']:
            # Original structure with train_samples subdirectory
            folder = Path(f"{base_path}/{category}")
        else:
            # Direct category structure without train_samples
            folder = Path(f"{base_path}/{category}")
        
        # Skip if folder doesn't exist
        if not folder.exists():
            continue
            
        # Find input files
        files = [f for f in folder.rglob('*.npy') if ('seis' in f.stem) or ('data' in f.stem)]
        # Generate corresponding output files
        out_files = [Path(str(f).replace('seis', 'vel').replace('data', 'model')) for f in files]
        # Only keep pairs where both input and output exist
        valid_pairs = [(f, o) for f, o in zip(files, out_files) if o.exists()]
        
        if valid_pairs:
            files, out_files = zip(*valid_pairs)
            all_input_files.extend(files)
            all_output_files.extend(out_files)
    
    print(f"Found {len(all_input_files)} training samples for category {category}")
    return all_input_files, all_output_files

def load_train_batch(inputs_files, outputs_files, start_idx, batch_size):
    """Load a batch of training samples starting from start_idx"""
    end_idx = min(start_idx + batch_size, len(inputs_files))
    x_batch = []
    y_batch = []
    for i in range(start_idx, end_idx):
        x = np.load(inputs_files[i])  # Shape: [500, 5, 1000, 70]
        y = np.load(outputs_files[i])
        x_batch.append(x)
        y_batch.append(y)
    return torch.tensor(np.concatenate(x_batch), dtype=torch.float32), np.concatenate(y_batch)

# Step 1: Classify all test files efficiently
print("Classifying all test files...")
class TestClassificationDataset(Dataset):
    def __init__(self, test_files):
        self.test_files = test_files


    def __len__(self):
        return len(self.test_files)


    def __getitem__(self, i):
        test_file = self.test_files[i]

        return np.load(test_file), test_file.stem

# Create dataset and dataloader for classification
classification_dataset = TestClassificationDataset(test_files)
classification_loader = DataLoader(
    classification_dataset, 
    batch_size=128,  # Adjust based on GPU memory
    num_workers=6,
    pin_memory=True
)

# Simple dictionary to store file → category mapping
test_file_to_category = {}

# Classify all test files
for batch_tensors, batch_files in tqdm(classification_loader, desc="Classifying test files"):
    batch_tensors = batch_tensors.to(device)
    
    with torch.no_grad():
        x_for_class = batch_tensors.mean(dim=1, keepdim=True)
        class_logits = classifier(x_for_class)
        pred_classes = class_logits.argmax(dim=1).cpu().numpy()
    
    # Store results in the dictionary
    for file_path, pred_class in zip(batch_files, pred_classes):
        test_file_to_category[file_path] = inv_class_map[pred_class]

# Group files by category for efficient processing
test_files_by_category = defaultdict(list)
for test_file, category in test_file_to_category.items():
    test_files_by_category[category].append(test_file)

# Print summary of classification
for category, files in test_files_by_category.items():
    print(f"Category {category}: {len(files)} test files")

# Free classification resources
del classifier
torch.cuda.empty_cache()



import numpy as np
import csv
from tqdm import tqdm
import faiss
import gc
import torch

with open('submission.csv', 'wt', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # Process each category one by one
    for category, category_test_files in test_files_by_category.items():
        print(f"Processing category: {category} with {len(category_test_files)} files")
        
        # Load training data for this category only once
        train_inputs, train_outputs = get_class_train_files(category)
        
        # Preprocess all training data for this category
        print(f"Preprocessing training data for {category}...")
        all_train_data = []
        file_positions = []  # To keep track of which file and position each sample is from
        
        for file_idx, input_file in enumerate(tqdm(train_inputs, desc="Building index")):
            # Load training file
            train_data = np.load(input_file)
            num_samples = train_data.shape[0]
            
            # Flatten the dimensions (keeping batch dimension)
            flattened_data = train_data.reshape(num_samples, -1).astype(np.float32)
            
            # Add to our dataset
            all_train_data.append(flattened_data)
            
            # Keep track of file index and position within file
            for pos in range(num_samples):
                file_positions.append((file_idx, pos))
        
        # Concatenate all data
        all_train_data = np.vstack(all_train_data)

        # Use more efficient index types
        dimension = all_train_data.shape[1]
        
        # For large datasets (>1M vectors), use IVF index
        nlist = min(4096, max(int(len(file_positions)/50), 256))  # Rule of thumb: nlist ≈ sqrt(N)
        quantizer = faiss.IndexFlat(dimension)
        quantizer.metric_type = faiss.METRIC_L1
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(all_train_data)
        index.add(all_train_data)
        index.nprobe = min(nlist//4, 256)
        
        # For even better speed, consider:
        # index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_L1)  # HNSW with 32 neighbors
        
        # Use GPU if available and dataset fits
        if torch.cuda.is_available() and all_train_data.shape[0] < 10_000_000:
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
        
        # # Create FAISS index with L1 distance (MAE) metric
        # print("Creating FAISS index...")
        
        # # Use METRIC_L1 (value=1) for L1/Manhattan distance
        # index = faiss.IndexFlat(dimension, faiss.METRIC_L1)
        
        # # Add all training vectors to the index
        # index.add(all_train_data)
        
        # Free memory
        del all_train_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Process each test file in this category
        for test_file in tqdm(category_test_files, desc=f"Processing {category} test files"):
            # Load the test sample
            test_sample = np.load(f'/kaggle/input/waveform-inversion/test/{test_file}.npy')
            
            # Reshape to match the indexing dimensions
            test_flat = test_sample.reshape(1, -1).astype(np.float32)
            
            # Search for nearest neighbor
            distances, indices = index.search(test_flat, 1)  # find 1 nearest neighbor
            
            # Get the original file index and position
            file_idx, pos_in_file = file_positions[indices[0][0]]
            
            # Load the best matching velocity map
            vel_file = train_outputs[file_idx]
            all_velocities = np.load(vel_file)
            best_vel = all_velocities[pos_in_file]
                        
            # Write to CSV - fixed to ensure one value per column
            test_id = test_file
            for y_pos in range(best_vel.shape[1]):
                # Create a dictionary with 'oid_ypos' and individual x_col values
                row = {'oid_ypos': f"{test_id}_y_{y_pos}"}
                
                # Add each x column value individually - use item() to extract scalar
                for i, x_pos in enumerate(range(1, 70, 2)):
                    col_name = f'x_{x_pos}'
                    # Use .item() to convert NumPy array element to Python scalar
                    row[col_name] = best_vel[0, y_pos, x_pos].item()
                
                writer.writerow(row)
        
        # After processing all files for this category, free memory
        del index, file_positions, train_inputs, train_outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("Submission complete!")




