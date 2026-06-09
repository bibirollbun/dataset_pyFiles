!pip install torch_geometric


import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
import copy
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.manual_seed(25)
np.random.seed(25)


h5_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"

with h5py.File(h5_path, "r") as f:
    train_spots = f["spots/Train"]
    test_spots = f["spots/Test"]
    train_images = f["images/Train"]
    test_images = f["images/Test"]

    train_spot_tables = {
        slide_id: pd.DataFrame(np.array(train_spots[slide_id]))
        for slide_id in train_spots.keys()
    }
    test_spot_table = pd.DataFrame(np.array(test_spots['S_7']))

    train_image_arrays = {
        slide_id: np.array(train_images[slide_id])
        for slide_id in train_images.keys()
    }
    test_image_array = np.array(test_images['S_7']) 


def rename_columns(df):
    df.columns = ['x', 'y'] + [f'C{i+1}' for i in range(df.shape[1] - 2)]
    return df

train_spot_tables = {k: rename_columns(df) for k, df in train_spot_tables.items()}
test_spot_table = rename_columns(test_spot_table)



class HECellTypeDataset(Dataset):
    def __init__(self, image, coords_table, patch_size=64, transform=None):
        self.image = image
        self.coords = coords_table[["x", "y"]].values.astype(int)
        self.labels = coords_table.drop(columns=["x", "y"]).values.astype(np.float32)
        self.patch_size = patch_size
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        patch = self.extract_patch(self.image, x, y, self.patch_size)
        patch = Image.fromarray(patch.astype(np.uint8))
        patch = self.transform(patch)
        label = torch.tensor(self.labels[idx])
        return patch, label, torch.tensor([x, y], dtype=torch.float32)

    def extract_patch(self, image, x, y, size):
        h, w = image.shape[:2]
        x1 = max(0, x - size // 2)
        y1 = max(0, y - size // 2)
        x2 = min(w, x1 + size)
        y2 = min(h, y1 + size)
        patch = image[y1:y2, x1:x2]
        if patch.shape[0] != size or patch.shape[1] != size:
            patch = np.pad(patch, ((0, size - patch.shape[0]), (0, size - patch.shape[1]), (0, 0)), mode='reflect')
        return patch


def build_edge_index(coords, k=6):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    edge_index = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edge_index.append((i, j))
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

def negative_pearson_loss(preds, targets):
    preds = preds - preds.mean(dim=1, keepdim=True)
    targets = targets - targets.mean(dim=1, keepdim=True)
    num = (preds * targets).sum(dim=1)
    den = torch.norm(preds, dim=1) * torch.norm(targets, dim=1)
    corr = num / (den + 1e-8)
    corr = torch.clamp(corr, -1.0, 1.0)  
    return 1 - corr.mean()

def spearman_surrogate_loss(preds, targets):
    preds_rank = preds.argsort(dim=1).argsort(dim=1).float()
    targets_rank = targets.argsort(dim=1).argsort(dim=1).float()
    return negative_pearson_loss(preds_rank, targets_rank)



class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()  
        y = self.avg_pool(x).view(batch, channels)  
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(batch, channels, 1, 1)
        return x * y  

class CellTypeCNN(nn.Module):
    def __init__(self, num_classes=35, backbone_type='resnet34', pretrained=True, fine_tune=True):
        super().__init__()
        
        if backbone_type == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif backbone_type == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        elif backbone_type == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError("Invalid backbone_type. Choose 'resnet18', 'resnet34', or 'resnet50'.")

        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.se_block = SEBlock(in_channels=512)  
        self.global_pool = nn.AdaptiveAvgPool2d(1) 
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)  
        x = self.se_block(x)  
        x = self.global_pool(x).view(x.size(0), -1)  
        out = self.head(x)
        return out

class GCNRefiner(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 64, heads=4, concat=True)  
        self.bn1 = nn.BatchNorm1d(64 * 4)  
        self.conv2 = GATConv(64 * 4, out_channels, heads=1, concat=False)  

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(self.bn1(x))
        x = self.conv2(x, edge_index)
        return x


alpha = 1.0  
beta = 1.0   
gamma = 0.01

def normalized_mse(pred, target):
    pred_norm = (pred - pred.mean(dim=1, keepdim=True)) / (pred.std(dim=1, keepdim=True) + 1e-6)
    target_norm = (target - target.mean(dim=1, keepdim=True)) / (target.std(dim=1, keepdim=True) + 1e-6)
    return nn.functional.mse_loss(pred_norm, target_norm)


def train_model(cnn, gnn, train_dataset, val_dataset, epochs=10, batch_size=32, lr=1e-4, patience=5):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(list(cnn.parameters()) + list(gnn.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    
    best_loss = float('inf')
    early_stop_counter = 0

    train_losses, val_losses = [], []
    train_spearman_scores, val_spearman_scores = [], []
    train_pearson_losses, val_pearson_losses = [], []
    lrs = []
    epoch_results = []

    for epoch in range(epochs):
        cnn.train()
        gnn.train()
        running_loss, running_pearson, running_spearman = 0.0, 0.0, 0.0
        
        for imgs, targets, coords in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            edge_index = build_edge_index(coords.numpy())
            preds = cnn(imgs)
            preds_refined = gnn(preds, edge_index)
            
            loss1 = negative_pearson_loss(preds_refined, targets)
            loss2 = spearman_surrogate_loss(preds_refined, targets)
            loss3 = normalized_mse(preds_refined, targets)
            loss = alpha*loss1 + beta*loss2 + gamma*loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_pearson += loss1.item()
            running_spearman += loss2.item()

        train_loss = running_loss / len(train_loader)
        train_pearson = running_pearson / len(train_loader)
        train_spearman = running_spearman / len(train_loader)

        train_losses.append(train_loss)
        train_pearson_losses.append(train_pearson)
        train_spearman_scores.append(train_spearman)

        cnn.eval()
        gnn.eval()
        val_loss, val_pearson, val_spearman = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, targets, coords in val_loader:
                edge_index = build_edge_index(coords.numpy())
                preds = cnn(imgs)
                preds_refined = gnn(preds, edge_index)

                loss1 = negative_pearson_loss(preds_refined, targets)
                loss2 = spearman_surrogate_loss(preds_refined, targets)
                loss3 = normalized_mse(preds_refined, targets)
                loss = alpha*loss1 + beta*loss2 + gamma*loss3

                val_loss += loss.item()
                val_pearson += loss1.item()
                val_spearman += loss2.item()

        val_loss /= len(val_loader)
        val_pearson /= len(val_loader)
        val_spearman /= len(val_loader)

        val_losses.append(val_loss)
        val_pearson_losses.append(val_pearson)
        val_spearman_scores.append(val_spearman)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        epoch_results.append([
            epoch + 1,
            round(train_loss, 4), round(val_loss, 4),
            round(train_pearson, 4), round(val_pearson, 4),
            round(train_spearman, 4), round(val_spearman, 4),
            alpha, beta, gamma, current_lr
        ])

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'cnn': cnn.state_dict(), 'gnn': gnn.state_dict()}, "best_model.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for i, (ax, train_vals, val_vals, title) in enumerate(zip(
        axs,
        [train_losses, train_pearson_losses, train_spearman_scores],
        [val_losses, val_pearson_losses, val_spearman_scores],
        ["Total Loss", "Negative Pearson Loss", "Spearman Surrogate Loss"]
    )):
        ax.plot(train_vals, label="Train")
        ax.plot(val_vals, label="Val")
        ax.set_title(title)
        ax.legend(loc='upper right')

        ax2 = ax.twinx()
        ax2.plot(lrs, color='gray', linestyle='--', label="LR")
        ax2.set_ylabel("Learning Rate", color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

    plt.tight_layout()
    plt.show()

    results_df = pd.DataFrame(epoch_results, columns=[
        "Epoch", "Train Loss", "Val Loss",
        "Train Pearson", "Val Pearson",
        "Train Spearman", "Val Spearman",
        "α", "β", "γ", "LR"
    ])
    print(results_df)


def predict(cnn, gnn, test_dataset, test_spot_table, output_path="/kaggle/working/submission.csv"):
    test_loader = DataLoader(test_dataset, batch_size=32)
    cnn.eval()
    gnn.eval()
    all_preds = []

    with torch.no_grad():
        for imgs, _, coords in tqdm(test_loader, desc="Predicting"):  
            edge_index = build_edge_index(coords.numpy())
            preds = cnn(imgs)
            preds_refined = gnn(preds, edge_index)
            all_preds.append(preds_refined.cpu().numpy())

    test_preds = np.concatenate(all_preds, axis=0)
    submission_df = pd.DataFrame(test_preds, columns=[f"C{i+1}" for i in range(35)])
    submission_df.insert(0, 'ID', test_spot_table.index)
    submission_df.to_csv(output_path, index=False)

    print(f"Submission file '{output_path}' created!")
    return submission_df.head()


cnn = CellTypeCNN()
gnn = GCNRefiner(35, 35)

train_dataset = HECellTypeDataset(train_image_arrays['S_1'], train_spot_tables['S_1'])
val_dataset = HECellTypeDataset(train_image_arrays['S_2'], train_spot_tables['S_2'])
test_dataset = HECellTypeDataset(test_image_array, test_spot_table)

train_model(cnn, gnn, train_dataset, val_dataset, epochs=50)



predict(cnn, gnn, test_dataset, test_spot_table)

