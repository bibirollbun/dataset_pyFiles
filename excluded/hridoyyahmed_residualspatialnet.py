import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from scipy.stats import spearmanr
import warnings
warnings.simplefilter('ignore')


# Open the file in read-only mode
file_path = '/content/eh2025/elucidata_ai_challenge_data.h5'


# Visualize Training slides with spot overlays
with h5py.File(file_path, "r") as h5file:
    train_images = h5file["images/Train"]
    train_spots = h5file["spots/Train"]

    num_train_slides = len(train_images)
    fig, ax = plt.subplots(1, num_train_slides, figsize=(14, 3))
    for i, slide_name in enumerate(train_images.keys()):
        image = np.array(train_images[slide_name])
        spots = np.array(train_spots[slide_name])
        x, y = spots["x"], spots["y"]

        ax[i].imshow(image, aspect="auto")
        ax[i].scatter(x, y, color="red", s=1, alpha=0.4)  # Overlay spot locations
        ax[i].set_title(slide_name)
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()



with h5py.File(file_path, "r") as h5file:
    test_images = h5file["images/Test"]
    test_spots = h5file["spots/Test"]
    sample = 'S_7'
    image = np.array(test_images[sample])
    spots = np.array(test_spots[sample])
    x, y = spots["x"], spots["y"]
    plt.figure(figsize=(6,6))
    plt.imshow(image, aspect="auto")
    plt.scatter(x, y, color="red", s=1, alpha=0.4)
    plt.axis('off')
    plt.title(sample)
    plt.show()


# Load and display (x,y) spot locations and cell type annotation table for Train slides
with h5py.File(file_path, "r") as f:
    train_spots = f["spots/Train"]

    # Dictionary to store DataFrames for each slide
    train_spot_tables = {}

    for slide_name in train_spots.keys():
        # Load dataset as NumPy structured array
        spot_array = np.array(train_spots[slide_name])

        # Convert to DataFrame
        df = pd.DataFrame(spot_array)

        # Store in dictionary
        train_spot_tables[slide_name] = df


# Display spot table for Test slide (only the spot coordinates on 2D array)
with h5py.File(file_path, "r") as f:
    test_spots = f["spots/Test"]
    spot_array = np.array(test_spots['S_7'])
    test_spot_table = pd.DataFrame(spot_array)

test_spot_table = test_spot_table.drop(columns=['Test_Set'])


def load_data(file_path):
    with h5py.File(file_path, "r") as h5file:
        train_images = {k: np.array(v) for k, v in h5file["images/Train"].items()}
        test_images = {k: np.array(v) for k, v in h5file["images/Test"].items()}
    return train_images, test_images


train_images, test_images = load_data(file_path)


class SpatialSpotDataset(Dataset):
    def __init__(self, image, spot_table, patch_size=128):
        self.image = image
        self.spot_table = spot_table.reset_index(drop=True)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.spot_table)

    def __getitem__(self, idx):
        x = int(self.spot_table.iloc[idx]['x'])
        y = int(self.spot_table.iloc[idx]['y'])

        half = self.patch_size // 2
        x1, x2 = max(0, x - half), min(self.image.shape[1], x + half)
        y1, y2 = max(0, y - half), min(self.image.shape[0], y + half)

        patch = self.image[y1:y2, x1:x2]

        patch_tensor = torch.tensor(patch.transpose(2,0,1), dtype=torch.float32)

        return patch_tensor


class SpotDataset(Dataset):
    def __init__(self, image, spots_df, patch_size=64, transform=None):

        self.image = image
        self.spots = spots_df.reset_index(drop=True)
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.spots)

    def __getitem__(self, idx):
        x = int(self.spots.loc[idx, 'x'])
        y = int(self.spots.loc[idx, 'y'])
        half = self.patch_size // 2

        x1, x2 = max(0, x - half), min(self.image.shape[1], x + half)
        y1, y2 = max(0, y - half), min(self.image.shape[0], y + half)

        patch = self.image[y1:y2, x1:x2]


        h, w, c = patch.shape
        if h != self.patch_size or w != self.patch_size:
            pad_h = self.patch_size - h
            pad_w = self.patch_size - w
            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')


        patch = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1)

        if self.transform:
            patch = self.transform(patch)


        target_cols = [f'C{i}' for i in range(1, 36)]
        target = torch.tensor(self.spots.loc[idx, target_cols].values, dtype=torch.float32)

        return patch, target



class CellTypeModel(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, num_outputs=35):
        super().__init__()
        self.backbone = getattr(models, backbone_name)(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_outputs)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features)


def spearman_corr_batch(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    corrs = []
    for i in range(y_true.shape[0]):
        corr, _ = spearmanr(y_true[i], y_pred[i])
        if np.isnan(corr):
            corr = 0
        corrs.append(corr)
    return np.mean(corrs)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for patches, targets in loader:
        patches, targets = patches.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * patches.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for patches, targets in loader:
            patches, targets = patches.to(device), targets.to(device)
            outputs = model(patches)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * patches.size(0)
            all_preds.append(outputs)
            all_targets.append(targets)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    spearman = spearman_corr_batch(all_targets, all_preds)
    return total_loss / len(loader.dataset), spearman



def train(train_images, train_spot_tables, val_slide_ids, patch_size=64,
               batch_size=32, epochs=20, lr=1e-4, device='cuda'):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    train_datasets = []
    val_datasets = []

    for slide_id in train_images.keys():
        dataset = SpotDataset(train_images[slide_id], train_spot_tables[slide_id], patch_size)
        if slide_id in val_slide_ids:
            val_datasets.append(dataset)
        else:
            train_datasets.append(dataset)

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = CellTypeModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_spearman = -np.inf
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_spearman = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Spearman: {val_spearman:.4f}")

        if val_spearman > best_spearman:
            best_spearman = val_spearman
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model.")

    return model


def predict_on_test(model, test_images, test_spot_table, patch_size=128, batch_size=64):
    test_image = test_images['S_7']
    test_dataset = SpatialSpotDataset(test_image, test_spot_table, patch_size=patch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            preds.append(outputs.cpu())

    preds = torch.cat(preds, dim=0).numpy()


    df_preds = pd.DataFrame(preds, columns=[f'C{i}' for i in range(1, 36)])
    df_preds.insert(0, 'ID', test_spot_table.index)
    return df_preds


val_slides = ['S_6']
model = train(train_images, train_spot_tables, val_slide_ids=val_slides, epochs=15)
model.load_state_dict(torch.load('best_model.pth'))
test_predictions = predict_on_test(model, test_images, test_spot_table)
print(test_predictions.head())


submission_file = 'sample_submission.csv'
test_predictions.to_csv(submission_file, index=False)

print(f"Saved predictions to {submission_file}")

