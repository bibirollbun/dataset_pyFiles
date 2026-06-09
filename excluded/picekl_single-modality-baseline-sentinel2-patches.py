!pip install rasterio


import os
import torch
import tqdm
import rasterio
import numpy as np
import pandas as pd
import albumentations as A
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support


def construct_patch_path(data_path, survey_id):
    """Construct the patch file path based on plot_id as './CD/AB/XXXXABCD.jpeg'"""
    path = data_path
    for d in (str(survey_id)[-2:], str(survey_id)[-4:-2]):
        path = os.path.join(path, d)

    path = os.path.join(path, f"{survey_id}.tiff")

    return path

def quantile_normalize(band, low=2, high=98):
    sorted_band = np.sort(band.flatten())
    quantiles = np.percentile(sorted_band, np.linspace(low, high, len(sorted_band)))
    normalized_band = np.interp(band.flatten(), sorted_band, quantiles).reshape(band.shape)
    
    min_val, max_val = np.min(normalized_band), np.max(normalized_band)
    
    # Prevent division by zero if min_val == max_val
    if max_val == min_val:
        return np.zeros_like(normalized_band, dtype=np.float32)  # Return an array of zeros

    # Perform normalization (min-max scaling)
    return ((normalized_band - min_val) / (max_val - min_val)).astype(np.float32)

class TrainDataset(Dataset):
    def __init__(self, data_dir, metadata, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = metadata
        self.metadata = self.metadata.dropna(subset="speciesId").reset_index(drop=True)
        self.metadata['speciesId'] = self.metadata['speciesId'].astype(int)
        self.label_dict = self.metadata.groupby('surveyId')['speciesId'].apply(list).to_dict()
        
        self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        
        survey_id = self.metadata.surveyId[idx]
        species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
        label = torch.zeros(num_classes)  # Initialize label tensor
        for species_id in species_ids:
            label_id = species_id
            label[label_id] = 1  # Set the corresponding class index to 1 for each species
        
        # Read TIFF files (multispectral bands)
        tiff_path = construct_patch_path(self.data_dir, survey_id)
        with rasterio.open(tiff_path) as dataset:
            image = dataset.read(out_dtype=np.float32)  # Read all bands
            image = np.array([quantile_normalize(band) for band in image])  # Apply quantile normalization

        image = np.transpose(image, (1, 2, 0))  # Convert to HWC format
        image = self.transform(image)

        return image, label, survey_id
    
class TestDataset(TrainDataset):
    def __init__(self, data_dir, metadata, transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = metadata
        
    def __getitem__(self, idx):
        
        survey_id = self.metadata.surveyId[idx]
        
        # Read TIFF files (multispectral bands)
        tiff_path = construct_patch_path(self.data_dir, survey_id)
        with rasterio.open(tiff_path) as dataset:
            image = dataset.read(out_dtype=np.float32)  # Read all bands
            image = np.array([quantile_normalize(band) for band in image])  # Apply quantile normalization

        image = np.transpose(image, (1, 2, 0))  # Convert to HWC format
        
        image = self.transform(image)
        return image, survey_id


# Dataset and DataLoader
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)),
])

# Load Training metadata
train_data_path = "/kaggle/input/geoplant-at-paiss/SatelitePatches/PA-train"
train_metadata_path = "/kaggle/input/geoplant-at-paiss/GLC25_PA_metadata_train.csv"
train_metadata = pd.read_csv(train_metadata_path)
train_dataset = TrainDataset(train_data_path, train_metadata, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load Test metadata
test_data_path = "/kaggle/input/geoplant-at-paiss/SatelitePatches/PA-test/"
test_metadata_path = "/kaggle/input/geoplant-at-paiss/GLC25_PA_metadata_test.csv"
test_metadata = pd.read_csv(test_metadata_path)
test_dataset = TestDataset(test_data_path, test_metadata, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """2Ã—(3x3 Conv + BN + ReLU) with optional downsampling on the skip path."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)
        return out

class ResNet6(nn.Module):
    """
    ResNet-6 for 4Ã—64Ã—64 Sentinel-2 patches.
      - Input: [B, 4, 64, 64]
      - Stem: 3Ã—3 conv with stride=2 -> 32Ã—32
      - Blocks:
          * Block1: 64 â†’ 64, stride=1 (32Ã—32)
          * Block2: 64 â†’ 128, stride=2 (16Ã—16)
          * Block3: 128 â†’ 128, stride=1 (16Ã—16)
      - GAP + MLP head -> logits [B, num_classes]
    """
    def __init__(self, num_classes: int, stem_channels: int = 64, mlp_hidden: int = 512, p_drop: float = 0.1):
        super().__init__()

        # Normalize the 4Ã—64Ã—64 tensor per sample (robust for small batches)
        self.norm_input = nn.LayerNorm([4, 64, 64])

        # Stem: light downsampling to 32Ã—32
        self.stem = nn.Sequential(
            nn.Conv2d(4, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),  # 64->32
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True)
        )

        # 3 residual blocks (6 conv layers total)
        self.block1 = BasicBlock(stem_channels, stem_channels, stride=1)   # 32Ã—32
        self.block2 = BasicBlock(stem_channels, stem_channels*2, stride=2) # 32->16
        self.block3 = BasicBlock(stem_channels*2, stem_channels*2, stride=1) # 16Ã—16

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),                                # [B, C, 1, 1] -> [B, C]
            nn.LayerNorm(stem_channels*2),
            nn.Linear(stem_channels*2, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(mlp_hidden, num_classes)          # logits, use BCEWithLogitsLoss
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 4, 64, 64]
        x = self.norm_input(x)
        x = self.stem(x)        # [B, 64, 32, 32]

        x = self.block1(x)      # [B, 64, 32, 32]
        x = self.block2(x)      # [B, 128, 16, 16]
        x = self.block3(x)      # [B, 128, 16, 16]

        x = self.gap(x)         # [B, 128, 1, 1]
        x = self.head(x)        # [B, num_classes] (logits)
        return x


# Check if cuda is available
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("DEVICE = CUDA")

num_classes = 11255 # Number of all unique classes within the PO and PA data.
model = ResNet6(num_classes).to(device)


# Hyperparameters
learning_rate = 0.0002
num_epochs = 4

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=True)


def set_seed(seed):
    # Set seed for Python's built-in random number generator
    torch.manual_seed(seed)
    # Set seed for numpy
    np.random.seed(seed)
    # Set seed for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Set cuDNN's random number generator seed for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(77)


import time
from tqdm import tqdm
import torch

print(f"Training for {num_epochs} epochs started.")
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()

    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (data, targets, _) in pbar:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)

        # Show loss in the tqdm bar
        pbar.set_postfix({
            "batch_loss": f"{loss.item():.4f}",
            "avg_loss": f"{avg_loss:.4f}"
        })

    scheduler.step()
    epoch_time = time.time() - epoch_start
    print(f"\nEpoch {epoch+1} finished in {epoch_time:.2f} seconds")
    print("Scheduler:", scheduler.state_dict())

# Save the trained model
model.eval()
torch.save(model.state_dict(), "resnet6-with-sentinel2-cubes.pth")

total_time = time.time() - start_time
print(f"Training completed in {total_time/60:.2f} minutes")


with torch.no_grad():
    all_predictions = []
    surveys = []
    top_k_indices = None
    for data, surveyID in tqdm(test_loader, total=len(test_loader)):

        data = data.to(device)
        
        outputs = model(data)
        predictions = torch.sigmoid(outputs).cpu().numpy()

        # Sellect top-25 values as predictions
        top_25 = np.argsort(-predictions, axis=1)[:, :25] 
        if top_k_indices is None:
            top_k_indices = top_25
        else:
            top_k_indices = np.concatenate((top_k_indices, top_25), axis=0)

        surveys.extend(surveyID.cpu().numpy())


data_concatenated = [' '.join(map(str, row)) for row in top_k_indices]

pd.DataFrame(
    {'surveyId': surveys,
     'predictions': data_concatenated,
    }).to_csv("submission.csv", index = False)




