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
batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)),
])

# Load Training metadata
train_data_path = "/kaggle/input/geolifeclef-2025/SatelitePatches/PA-train"
train_metadata_path = "/kaggle/input/geolifeclef-2025/GLC25_PA_metadata_train.csv"
train_metadata = pd.read_csv(train_metadata_path)
train_dataset = TrainDataset(train_data_path, train_metadata, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load Test metadata
test_data_path = "/kaggle/input/geolifeclef-2025/SatelitePatches/PA-test/"
test_metadata_path = "/kaggle/input/geolifeclef-2025/GLC25_PA_metadata_test.csv"
test_metadata = pd.read_csv(test_metadata_path)
test_dataset = TestDataset(test_data_path, test_metadata, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# Check if cuda is available
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("DEVICE = CUDA")

# Hyperparameters
learning_rate = 0.0001
num_epochs = 25
positive_weigh_factor = 1.0
num_classes = 11255 # Number of all unique classes within the PO and PA data.


model = models.resnet18(weights="IMAGENET1K_V1")
model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2))
model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=25, verbose=True)


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


print(f"Training for {num_epochs} epochs started.")

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets, _) in enumerate(train_loader):

        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        pos_weight = targets*positive_weigh_factor  # All positive weights are equal to 10
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if batch_idx % 348 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

    scheduler.step()
    print("Scheduler:",scheduler.state_dict())

# Save the trained model
model.eval()
torch.save(model.state_dict(), "resnet18-with-sentinel2.pth")


with torch.no_grad():
    all_predictions = []
    surveys = []
    top_k_indices = None
    for data, surveyID in tqdm.tqdm(test_loader, total=len(test_loader)):

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

