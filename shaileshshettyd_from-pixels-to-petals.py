!pip install -q folium matplotlib mapclassify


# ğŸ“š Core Libraries
import os, random, gc, math
import numpy as np
import pandas as pd
from tqdm import tqdm

# ğŸ“Š Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import folium
import geopandas as gpd
from shapely.geometry import Point

# ğŸ”¥ Torch Setup
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

# âš™ï¸� Configs
SEED = 42
BATCH_SIZE = 64
NUM_CLASSES = 11255
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/kaggle/input/geolifeclef-2025"
torch.manual_seed(SEED)
np.random.seed(SEED)


# âœ… Load metadata
train_meta = pd.read_csv("/kaggle/input/geolifeclef-2025/GLC25_PA_metadata_train.csv")
test_meta  = pd.read_csv("/kaggle/input/geolifeclef-2025/GLC25_PA_metadata_test.csv")

# âœ… Convert to GeoDataFrames
train_meta = train_meta.drop_duplicates("surveyId").copy()
train_meta["geometry"] = train_meta.apply(lambda x: Point(x["lon"], x["lat"]), axis=1)
train_gdf = gpd.GeoDataFrame(train_meta, geometry="geometry", crs="EPSG:4326")

test_meta = test_meta.drop_duplicates("surveyId").copy()
test_meta["geometry"] = test_meta.apply(lambda x: Point(x["lon"], x["lat"]), axis=1)
test_gdf = gpd.GeoDataFrame(test_meta, geometry="geometry", crs="EPSG:4326")


import geopandas as gpd
from shapely.geometry import Point

train_meta = train_meta.copy()
train_meta["geometry"] = train_meta.apply(lambda x: Point(x["lon"], x["lat"]), axis=1)
train_gdf = gpd.GeoDataFrame(train_meta, geometry="geometry", crs="EPSG:4326")

# Plot interactive map
m = train_gdf.explore(
    color="green",
    tiles="Stamen Terrain",
    attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
)
test_gdf.explore(m=m, color="red")


class GeoLifeDataset(Dataset):
    def __init__(self, meta_df, cube_path, subset, transform=None):
        self.meta = meta_df.dropna(subset=['speciesId']).drop_duplicates("surveyId")
        self.label_dict = self.meta.groupby("surveyId")["speciesId"].apply(list).to_dict()
        self.cube_path = cube_path
        self.transform = transform
        self.subset = subset

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        survey_id = self.meta.iloc[idx].surveyId
        cube = torch.nan_to_num(torch.load(f"{self.cube_path}/GLC25-PA-{self.subset}-landsat_time_series_{survey_id}_cube.pt", weights_only=True))
        cube = cube.permute(1, 2, 0).numpy()  # HWC
        label = torch.zeros(NUM_CLASSES)
        for sid in self.label_dict.get(survey_id, []):
            label[sid] = 1
        if self.transform: cube = self.transform(cube)
        return cube, label, survey_id


class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.ln = nn.LayerNorm(1000)
        self.fc = nn.Sequential(
            nn.Linear(1000, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float()  # NHWC -> NCHW
        x = self.resnet(x)
        x = self.ln(x)
        return self.fc(x)


def get_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def get_loss(targets, logits):
    pos_weight = targets * 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion(logits, targets)


from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0
    for data, target, _ in tqdm(loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = get_loss(target, output)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def train_model(model, train_loader):
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    scaler = GradScaler()
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, scaler)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")
        scheduler.step()
    return model


class GeoLifeDataset(Dataset):
    def __init__(self, meta_df, cube_path, subset, transform=None):
        self.meta = meta_df.dropna(subset=['speciesId']).drop_duplicates("surveyId")
        self.label_dict = self.meta.groupby("surveyId")["speciesId"].apply(list).to_dict()
        self.cube_path = cube_path
        self.transform = transform
        self.subset = subset

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        survey_id = self.meta.iloc[idx].surveyId
        cube = torch.nan_to_num(torch.load(f"{self.cube_path}/GLC25-PA-{self.subset}-landsat_time_series_{survey_id}_cube.pt", weights_only=True))
        cube = cube.permute(1, 2, 0).numpy()  # HWC
        label = torch.zeros(11255)
        for sid in self.label_dict.get(survey_id, []):
            label[sid] = 1
        if self.transform:
            cube = self.transform(cube)
        return cube, label, survey_id

