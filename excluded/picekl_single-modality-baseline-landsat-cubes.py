import os
import torch
import tqdm
import numpy as np
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support


class TrainDataset(Dataset):
    def __init__(self, data_dir, metadata, subset, transform=None):
        self.subset = subset
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
        sample = torch.nan_to_num(torch.load(os.path.join(self.data_dir, f"GLC25-PA-{self.subset}-landsat-time-series_{survey_id}_cube.pt"), weights_only=True))

        species_ids = self.label_dict.get(survey_id, [])  # Get list of species IDs for the survey ID
        label = torch.zeros(num_classes)  # Initialize label tensor
        for species_id in species_ids:
            label_id = species_id
            label[label_id] = 1  # Set the corresponding class index to 1 for each species

        # Ensure the sample is in the correct format for the transform
        if isinstance(sample, torch.Tensor):
            sample = sample.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
            sample = sample.numpy()  # Convert tensor to numpy array

        if self.transform:
            sample = self.transform(sample)

        return sample, label, survey_id
    
class TestDataset(TrainDataset):
    def __init__(self, data_dir, metadata, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.data_dir = data_dir
        self.metadata = metadata
        
    def __getitem__(self, idx):
        
        survey_id = self.metadata.surveyId[idx]
        sample = torch.nan_to_num(torch.load(os.path.join(self.data_dir, f"GLC25-PA-{self.subset}-landsat_time_series_{survey_id}_cube.pt"), weights_only=True))

        if isinstance(sample, torch.Tensor):
            sample = sample.permute(1, 2, 0)  # Change tensor shape from (C, H, W) to (H, W, C)
            sample = sample.numpy()

        if self.transform:
            sample = self.transform(sample)

        return sample, survey_id


# Dataset and DataLoader
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load Training metadata


train_data_path = "/kaggle/input/geoplant-at-paiss/SateliteTimeSeries-Landsat/cubes/PA-train/"
train_metadata_path = "/kaggle/input/geoplant-at-paiss/GLC25_PA_metadata_train.csv"
train_metadata = pd.read_csv(train_metadata_path)
train_dataset = TrainDataset(train_data_path, train_metadata, subset="train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load Test metadata
test_data_path = "/kaggle/input/geoplant-at-paiss/SateliteTimeSeries-Landsat/cubes/PA-test/"
test_metadata_path = "/kaggle/input/geoplant-at-paiss/GLC25_PA_metadata_test.csv"
test_metadata = pd.read_csv(test_metadata_path)
test_dataset = TestDataset(test_data_path, test_metadata, subset="test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """2Ã—(Conv3Ã—3 + BN + ReLU) with identity skip; stride=1 (no downsample)."""
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c)

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = F.relu(out, inplace=True)
        out = self.conv2(out); out = self.bn2(out)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out

class ResNet6(nn.Module):
    """
    ResNet-6 for Landsat cubes:
      - Input: [B, 6, 4, 21]  (bands, quarters, years)
      - Stem: 1Ã—1 conv (channel mixing) + 3Ã—3 conv (spatial-temporal)
      - 3 residual blocks (6 conv layers total)
      - GAP + MLP head -> logits for multi-label BCEWithLogitsLoss
    """
    def __init__(self, num_classes: int, stem_channels: int = 64, mlp_hidden: int = 512, p_drop: float = 0.1):
        super().__init__()
        # Per-sample normalization over [C,H,W]
        self.norm_input = nn.LayerNorm([6, 4, 21])

        # Two-step stem to (i) mix spectral bands, then (ii) capture local 2D structure
        self.stem1 = nn.Conv2d(6, stem_channels, kernel_size=1, stride=1, padding=0, bias=False)  # channel mixing
        self.stem2 = nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(stem_channels)

        # 3 residual blocks (no downsampling; preserve 4Ã—21)
        self.block1 = BasicBlock(stem_channels)
        self.block2 = BasicBlock(stem_channels)
        self.block3 = BasicBlock(stem_channels)

        # Global average pooling and head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),                       # [B, C, 1, 1] -> [B, C]
            nn.LayerNorm(stem_channels),
            nn.Linear(stem_channels, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(mlp_hidden, num_classes)  # logits
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
        # x: [B, 6, 4, 21]
        x = self.norm_input(x)
        x = self.stem1(x)                  # [B, C=stem_channels, 4, 21]
        x = self.stem2(x)
        x = self.stem_bn(x)
        x = F.relu(x, inplace=True)

        x = self.block1(x)                 # 3 blocks Ã— 2 conv = 6 conv layers
        x = self.block2(x)
        x = self.block3(x)

        x = self.gap(x)                    # [B, C, 1, 1]
        x = self.head(x)                   # [B, num_classes] (logits)
        return x


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

set_seed(69)


# Check if cuda is available
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("DEVICE = CUDA")

num_classes = 11255 # Number of all unique classes within the PO and PA data.
model = ResNet6(num_classes).to(device)


# Hyperparameters
learning_rate = 0.0002
num_epochs = 10

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=True)


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
torch.save(model.state_dict(), "resnet6-with-landsat-cubes.pth")

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




