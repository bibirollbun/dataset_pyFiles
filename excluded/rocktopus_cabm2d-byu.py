import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from mpl_toolkits.mplot3d import Axes3D

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, channels, height, width = x.size()
        n_pixels = height * width
        
        proj_query = self.query(x).view(batch, self.num_heads, self.head_dim, n_pixels).permute(0, 1, 3, 2)
        proj_key = self.key(x).view(batch, self.num_heads, self.head_dim, n_pixels)
        proj_value = self.value(x).view(batch, self.num_heads, self.head_dim, n_pixels).permute(0, 1, 3, 2)
        
        # Compute attention scores with clipping
        energy = torch.matmul(proj_query, proj_key) / (self.head_dim ** 0.5)
        energy = torch.clamp(energy, min=-10, max=10)  # Prevent extreme values
        attention = self.softmax(energy)
        
        out = torch.matmul(attention, proj_value)        
        out = out.permute(0, 1, 3, 2).contiguous().view(batch, self.in_channels, height, width)
        out = self.out_proj(out)
        
        return self.gamma * out + x

# Replace the SelfAttention in EnhancedCABM2D with MultiHeadSelfAttention
class EnhancedCABM2D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(EnhancedCABM2D, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.conv_spatial1 = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, dilation=1, bias=False)
        self.conv_spatial2 = nn.Conv2d(in_channels, 1, kernel_size=7, padding=9, dilation=3, bias=False)
        self.conv_refine = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.self_attention = MultiHeadSelfAttention(in_channels, num_heads=8)  # Updated to MultiHeadSelfAttention

    def forward(self, x):
        channel_avg = self.global_pool(x)
        channel_att = self.fc1(channel_avg)
        channel_att = self.relu(channel_att)
        channel_att = self.fc2(channel_att)
        channel_att = self.sigmoid(channel_att)
        x_channel = x * channel_att
        spatial_att1 = self.conv_spatial1(x_channel)
        spatial_att2 = self.conv_spatial2(x_channel)
        spatial_att = self.sigmoid(spatial_att1 + spatial_att2)
        x_spatial = x_channel * spatial_att
        x_self_att = self.self_attention(x_spatial)
        x_refined = self.conv_refine(x_self_att)
        x_refined = self.bn(x_refined)
        x_refined = self.relu(x_refined)
        return x + x_refined

class OptimizedCenterNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(OptimizedCenterNet2D, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.attention = EnhancedCABM2D(in_channels=256)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.size_head = nn.Conv2d(32, 2, kernel_size=1)
        self.offset_head = nn.Conv2d(32, 2, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e4_att = self.attention(e4)
        d4 = self.dec4(e4_att) + e3
        d3 = self.dec3(d4) + e2
        d2 = self.dec2(d3) + e1
        heatmap = self.heatmap_head(d2)
        size = self.size_head(d2)
        offset = self.offset_head(d2)
        return heatmap, size, offset

class FlagellarDataset(Dataset):
    def __init__(self, csv_file=None, root_dir=None, new_size=(256, 256), trust_region=4, is_test=False):
        self.root_dir = Path(root_dir)
        self.new_size = new_size
        self.trust_region = trust_region
        self.is_test = is_test
        self.data = []
        self.spatial_augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),  # Reduced from 30
            T.RandomAffine(degrees=0, scale=(0.95, 1.05))  # Narrowed from (0.8, 1.2)
        ])
        self.image_augment = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1)], p=0.2)  # Reduced intensity, dropped blur
        ])

        if csv_file and os.path.exists(csv_file) and not is_test:
            labels = pd.read_csv(csv_file)
            self.tomo_ids = labels['tomo_id'].unique().tolist()
            self.motors_map = labels.groupby('tomo_id').apply(
                lambda g: np.array(g[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values[0], dtype=np.float32),
                include_groups=False
            ).to_dict()
            self.size_map = labels.groupby('tomo_id').agg({
                'Array shape (axis 1)': 'first',
                'Array shape (axis 2)': 'first',
                'Array shape (axis 0)': 'first',
                'Voxel spacing': 'first'
            }).to_dict('index')

            for tomo_id in tqdm(self.tomo_ids, desc="Loading data"):
                motor = self.motors_map[tomo_id]
                files = sorted(self.root_dir.joinpath(tomo_id).glob('*.jpg'))
                num_slices = len(files)

                if np.all(motor == -1):
                    center_z = num_slices // 2
                else:
                    center_z = int(motor[0])

                z_start = max(0, center_z - self.trust_region)
                z_end = min(num_slices, center_z + self.trust_region + 1)

                for z in range(z_start, z_end):
                    img = Image.open(files[z]).convert('L')
                    img = T.functional.resize(img, new_size)
                    img = T.functional.to_tensor(img)
                    img = (img - img.mean()) / (img.std() + 1e-8)

                    motor_norm = torch.tensor([
                        motor[1] / self.size_map[tomo_id]['Array shape (axis 1)'],
                        motor[2] / self.size_map[tomo_id]['Array shape (axis 2)']
                    ], dtype=torch.float32) if np.all(motor != -1) else torch.tensor([-1, -1], dtype=torch.float32)

                    heatmap = self.generate_gaussian_heatmap(motor_norm, self.new_size) if np.all(motor != -1) else torch.zeros(self.new_size)
                    if np.all(motor != -1):
                        voxel_spacing = self.size_map[tomo_id]['Voxel spacing']
                        orig_height = self.size_map[tomo_id]['Array shape (axis 1)']
                        orig_width = self.size_map[tomo_id]['Array shape (axis 2)']
                        target_size_pixels_y = 1000.0 / voxel_spacing
                        target_size_pixels_x = 1000.0 / voxel_spacing
                        size_target = torch.tensor([
                            target_size_pixels_y / orig_height,
                            target_size_pixels_x / orig_width
                        ], dtype=torch.float32)
                    else:
                        size_target = torch.zeros(2, dtype=torch.float32)
                    offset_target = torch.zeros(2, dtype=torch.float32)

                    if not self.is_test and np.all(motor != -1):
                        img_aug = self.image_augment(img)
                        stacked = torch.stack([img_aug[0], heatmap], dim=0)
                        stacked_aug = self.spatial_augment(stacked)
                        img = stacked_aug[0].unsqueeze(0)
                        heatmap = stacked_aug[1]
                        size_map = torch.full(self.new_size, size_target[0], dtype=torch.float32)
                        size_map2 = torch.full(self.new_size, size_target[1], dtype=torch.float32)
                        offset_map = torch.full(self.new_size, offset_target[0], dtype=torch.float32)
                        offset_map2 = torch.full(self.new_size, offset_target[1], dtype=torch.float32)

                        peak_idx = heatmap.view(-1).argmax()
                        y_new = peak_idx // self.new_size[1]
                        x_new = peak_idx % self.new_size[1]
                        motor_norm = torch.tensor([y_new / self.new_size[0], x_new / self.new_size[1]], dtype=torch.float32)

                    self.data.append({
                        'tomo_id': tomo_id,
                        'slice': img,
                        'heatmap': heatmap,
                        'size': size_target,
                        'offset': offset_target,
                        'center': motor_norm,
                        'z': z,
                        'orig_shape': torch.tensor([self.size_map[tomo_id]['Array shape (axis 1)'],
                                                   self.size_map[tomo_id]['Array shape (axis 2)'],
                                                   num_slices], dtype=torch.float32),
                        'voxel_spacing': self.size_map[tomo_id]['Voxel spacing'],
                        'motor': torch.tensor(motor, dtype=torch.float32)
                    })
        else:
            self.tomo_ids = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            for tomo_id in tqdm(self.tomo_ids, desc="Loading test data"):
                files = sorted(self.root_dir.joinpath(tomo_id).glob('*.jpg'))
                num_slices = len(files)
                orig_shape = torch.tensor([Image.open(files[0]).size[1], Image.open(files[0]).size[0], num_slices], dtype=torch.float32)
                for z, file in enumerate(files):
                    self.data.append({
                        'tomo_id': tomo_id,
                        'slice_path': str(file),
                        'z': z,
                        'orig_shape': orig_shape
                    })

    def generate_gaussian_heatmap(self, center, xy_size, sigma=4.0):
        heatmap = torch.zeros(xy_size)
        yc, xc = center
        yc = yc * xy_size[0]
        xc = xc * xy_size[1]
        y_coords, x_coords = torch.meshgrid(
            torch.arange(xy_size[0], dtype=torch.float32),
            torch.arange(xy_size[1], dtype=torch.float32),
            indexing='ij'
        )
        dist = ((y_coords - yc) ** 2 + (x_coords - xc) ** 2) / (2.0 * sigma ** 2)
        heatmap = torch.exp(-dist)
        return heatmap

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.is_test:
            return {
                'tomo_id': item['tomo_id'],
                'slice_path': item['slice_path'],
                'z': item['z'],
                'orig_shape': item['orig_shape']
            }
        return {
            'tomo_id': item['tomo_id'],
            'slice': item['slice'],
            'heatmap': item['heatmap'].unsqueeze(0),
            'size': item['size'],
            'offset': item['offset'],
            'center': item['center'],
            'z': item['z'],
            'orig_shape': item['orig_shape'],
            'voxel_spacing': item['voxel_spacing'],
            'motor': item['motor']
        }

def custom_collate_fn(batch):
    if 'slice' in batch[0]:
        return {
            'tomo_id': [item['tomo_id'] for item in batch],
            'slice': torch.stack([item['slice'] for item in batch]),
            'heatmap': torch.stack([item['heatmap'] for item in batch]),
            'size': torch.stack([item['size'] for item in batch]),
            'offset': torch.stack([item['offset'] for item in batch]),
            'center': torch.stack([item['center'] for item in batch]),
            'z': torch.tensor([item['z'] for item in batch], dtype=torch.long),
            'orig_shape': torch.stack([item['orig_shape'] for item in batch]),
            'voxel_spacing': torch.tensor([item['voxel_spacing'] for item in batch], dtype=torch.float32),
            'motor': torch.stack([item['motor'] for item in batch])
        }
    else:
        return {
            'tomo_id': [item['tomo_id'] for item in batch],
            'slice_path': [item['slice_path'] for item in batch],
            'z': torch.tensor([item['z'] for item in batch], dtype=torch.long),
            'orig_shape': torch.stack([item['orig_shape'] for item in batch])
        }

class CenterNetLoss(nn.Module):
    def __init__(self, gamma=2.0, size_weight=0.5, offset_weight=0.1):
        super(CenterNetLoss, self).__init__()
        self.gamma = gamma  # Focusing parameter
        self.size_weight = size_weight
        self.offset_weight = offset_weight

    def gaussian_focal_loss(self, pred_heatmap, target_heatmap):
        # pred_heatmap: Predicted probabilities (after sigmoid)
        # target_heatmap: Ground truth Gaussian heatmap (0 to 1)
        
        # Positive and negative terms weighted by target heatmap values
        pos_loss = -target_heatmap * (1 - pred_heatmap) ** self.gamma * torch.log(pred_heatmap + 1e-6)
        neg_loss = -(1 - target_heatmap) * pred_heatmap ** self.gamma * torch.log(1 - pred_heatmap + 1e-6)
        
        # Sum over all pixels and normalize by batch size
        loss = (pos_loss + neg_loss).sum() / pred_heatmap.size(0)
        return loss

    def forward(self, pred_heatmap, pred_size, pred_offset, target_heatmap, target_size, target_offset):
        # Ensure pred_heatmap is in probability space (since heatmap_head has Sigmoid)
        gfl = self.gaussian_focal_loss(pred_heatmap, target_heatmap)

        # Size and offset losses (unchanged)
        batch_size = pred_heatmap.size(0)
        pred_size_at_centers = torch.zeros(batch_size, 2, device=pred_heatmap.device)
        pred_offset_at_centers = torch.zeros(batch_size, 2, device=pred_heatmap.device)
        for i in range(batch_size):
            heatmap = pred_heatmap[i].squeeze()
            peak_idx = heatmap.view(-1).argmax()
            y = peak_idx // 256
            x = peak_idx % 256
            pred_size_at_centers[i] = pred_size[i, :, y, x]
            pred_offset_at_centers[i] = pred_offset[i, :, y, x]

        size_loss = F.mse_loss(pred_size_at_centers, target_size, reduction='mean') * self.size_weight
        offset_loss = F.mse_loss(pred_offset_at_centers, target_offset, reduction='mean') * self.offset_weight

        return gfl + size_loss + offset_loss

def extract_centroid(heatmap, size, offset, xy_size=(256, 256), threshold=0.1):  # Updated for 256x256
    heatmap = heatmap.squeeze()
    if heatmap.max() < threshold:
        return torch.tensor([-1, -1], dtype=torch.float32, device=heatmap.device), torch.zeros(2, device=heatmap.device), torch.zeros(2, device=heatmap.device)
    
    peak_value, peak_idx = heatmap.view(-1).topk(1)
    if peak_value < threshold:
        return torch.tensor([-1, -1], dtype=torch.float32, device=heatmap.device), torch.zeros(2, device=heatmap.device), torch.zeros(2, device=heatmap.device)
    
    y = peak_idx // xy_size[1]
    x = peak_idx % xy_size[1]
    
    y_norm = torch.clamp(y.float() / xy_size[0], 0, 1)
    x_norm = torch.clamp(x.float() / xy_size[1], 0, 1)
    
    center = torch.tensor([y_norm, x_norm], dtype=torch.float32, device=heatmap.device)
    pred_size = size[:, y, x].squeeze()
    pred_offset = offset[:, y, x].squeeze()
    center = center + pred_offset
    return center, pred_size, pred_offset

def denormalize_predictions(pred_center, pred_size, z, orig_shape):
    pred_center_denorm = torch.zeros(3, dtype=torch.float32, device=pred_center.device)
    pred_center_denorm[0] = z
    pred_center_denorm[1] = pred_center[0] * orig_shape[0]
    pred_center_denorm[2] = pred_center[1] * orig_shape[1]
    pred_size_denorm = pred_size * torch.tensor([orig_shape[0], orig_shape[1]], dtype=torch.float32, device=pred_size.device)
    return pred_center_denorm, pred_size_denorm

def calculate_fbeta_score(pred_centers, true_centers, voxel_spacings, threshold_angstroms=1000, beta=2.0):
    TP, TN, FP, FN = 0, 0, 0, 0
    for pred_center, true_center, voxel_spacing in zip(pred_centers, true_centers, voxel_spacings):
        voxel_spacing = voxel_spacing.item()
        pred_array = pred_center.detach().cpu().numpy()
        true_array = true_center.detach().cpu().numpy()

        if np.all(true_array == -1):
            if np.all(pred_array == -1):
                TN += 1
            else:
                FP += 1
            continue
        if np.all(pred_array == -1):
            FN += 1
            continue

        distance = np.linalg.norm((true_array - pred_array) * voxel_spacing)
        if distance <= threshold_angstroms:
            TP += 1
        else:
            FN += 1

    if TP + FP + FN == 0:
        fbeta = 0.0
    else:
        beta2 = beta ** 2
        fbeta = (1 + beta2) * TP / ((1 + beta2) * TP + beta2 * FN + FP)
    return fbeta, TP, TN, FP, FN

def plot_slices(model, example, device):
    model.eval()
    with torch.no_grad():
        slice_data = example['slice'].unsqueeze(0).to(device)
        heatmap_true = example['heatmap'].unsqueeze(0)
        size_true = example['size']
        offset_true = example['offset']
        center = example['center']
        orig_shape = example['orig_shape'].to(device)
        tomo_id = example['tomo_id']
        z = example['z']

        heatmap_pred, size_pred, offset_pred = model(slice_data)
        heatmap_pred, size_pred, offset_pred = heatmap_pred[0], size_pred[0], offset_pred[0]
        heatmap_true = heatmap_true[0]

        pred_center_norm, pred_size_norm, pred_offset_norm = extract_centroid(heatmap_pred, size_pred, offset_pred, threshold=0.3)  # Higher threshold
        pred_center, pred_size = denormalize_predictions(pred_center_norm, pred_size_norm, z, orig_shape[:2])
        center_denorm, _ = denormalize_predictions(center, size_true, z, orig_shape[:2]) if torch.all(center >= 0) else (torch.tensor([-1, -1, -1], device=device), torch.zeros(2, device=device))

        slice_data = slice_data[0, 0].cpu().numpy()
        heatmap_pred_slice = heatmap_pred[0].cpu().numpy()
        heatmap_true_slice = heatmap_true[0].cpu().numpy()

        distance = np.linalg.norm(center_denorm.cpu().numpy() - pred_center.cpu().numpy()) if torch.all(center >= 0) else float('inf')
        distance_text = f"Distance: {distance:.2f}" if distance != float('inf') else "No GT Motor"

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        ax.imshow(slice_data, cmap='gray')
        ax.set_title(f"Tomo: {tomo_id}\nZ: {z}\n{distance_text}")
        ax.axis('off')

        scale_y = orig_shape[0].item() / 256  # Updated for 256x256
        scale_x = orig_shape[1].item() / 256

        if torch.all(center >= 0):
            true_y_display = center_denorm[1].item() / scale_y
            true_x_display = center_denorm[2].item() / scale_x
            ax.plot(true_x_display, true_y_display, 'go', label='GT Center')
            rect_gt = patches.Rectangle(
                (max(0, true_x_display - size_true[1].item() * scale_x / 2), max(0, true_y_display - size_true[0].item() * scale_y / 2)),
                size_true[1].item() * scale_x, size_true[0].item() * scale_y, linewidth=2, edgecolor='g', facecolor='none', label='GT Box'
            )
            ax.add_patch(rect_gt)

        if not torch.all(pred_center == -1):
            pred_y_display = pred_center[1].item() / scale_y
            pred_x_display = pred_center[2].item() / scale_x
            ax.plot(pred_x_display, pred_y_display, 'ro', label='Pred Center')
            rect_pred = patches.Rectangle(
                (max(0, pred_x_display - pred_size[1].item() / 2), max(0, pred_y_display - pred_size[0].item() / 2)),
                pred_size[1].item(), pred_size[0].item(), linewidth=2, edgecolor='r', facecolor='none', label='Pred Box'
            )
            ax.add_patch(rect_pred)

        ax.legend()

        axes[1].imshow(heatmap_pred_slice, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title("Predicted Heatmap")
        axes[1].axis('off')

        axes[2].imshow(heatmap_true_slice, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title("Ground Truth Heatmap")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

def train_model(model, train_loader, val_loader, num_epochs=150, device='cuda', accum_steps=2, max_grad_norm=1.0):
    criterion = CenterNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()

    best_val_fbeta = 0.0
    patience = 50
    epochs_no_improve = 0
    best_model_path = 'best_model.pth'
    example_to_plot = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_trues, train_voxel_spacings = [], [], []

        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train")):
            slices = batch['slice'].to(device)
            heatmaps = batch['heatmap'].to(device)
            sizes = batch['size'].to(device)
            offsets = batch['offset'].to(device)
            centers = batch['center'].to(device)
            orig_shapes = batch['orig_shape'].to(device)
            voxel_spacings = batch['voxel_spacing'].to(device)
            motors = batch['motor'].to(device)
            zs = batch['z'].to(device)

            with torch.cuda.amp.autocast():
                pred_heatmaps, pred_sizes, pred_offsets = model(slices)
                loss = criterion(pred_heatmaps, pred_sizes, pred_offsets, heatmaps, sizes, offsets)
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            train_loss += loss.item() * slices.size(0) * accum_steps

            if (i + 1) % accum_steps == 0:
                # Apply gradient clipping before unscaling and stepping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pred_centers_norm, pred_sizes_norm, _ = zip(*[extract_centroid(pred_heatmaps[i], pred_sizes[i], pred_offsets[i], threshold=0.3) 
                                                          for i in range(slices.size(0))])
            pred_centers, _ = zip(*[denormalize_predictions(pred_centers_norm[i], pred_sizes_norm[i], zs[i], orig_shapes[i, :2])
                                    for i in range(slices.size(0))])
            train_preds.extend(pred_centers)
            train_trues.extend(motors)
            train_voxel_spacings.extend(voxel_spacings)

            if example_to_plot is None:
                example_to_plot = {
                    'slice': slices[0].cpu(),
                    'heatmap': heatmaps[0].cpu(),
                    'size': sizes[0].cpu(),
                    'offset': offsets[0].cpu(),
                    'center': centers[0].cpu(),
                    'orig_shape': orig_shapes[0].cpu(),
                    'tomo_id': batch['tomo_id'][0],
                    'z': zs[0].cpu()
                }

        if (i + 1) % accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        model.eval()
        val_loss = 0.0
        val_preds, val_trues, val_voxel_spacings = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                slices = batch['slice'].to(device)
                heatmaps = batch['heatmap'].to(device)
                sizes = batch['size'].to(device)
                offsets = batch['offset'].to(device)
                centers = batch['center'].to(device)
                orig_shapes = batch['orig_shape'].to(device)
                voxel_spacings = batch['voxel_spacing'].to(device)
                motors = batch['motor'].to(device)
                zs = batch['z'].to(device)

                with torch.cuda.amp.autocast():
                    pred_heatmaps, pred_sizes, pred_offsets = model(slices)
                    loss = criterion(pred_heatmaps, pred_sizes, pred_offsets, heatmaps, sizes, offsets)
                val_loss += loss.item() * slices.size(0)

                pred_centers_norm, pred_sizes_norm, _ = zip(*[extract_centroid(pred_heatmaps[i], pred_sizes[i], pred_offsets[i], threshold=0.3) 
                                                              for i in range(slices.size(0))])
                pred_centers, _ = zip(*[denormalize_predictions(pred_centers_norm[i], pred_sizes_norm[i], zs[i], orig_shapes[i, :2])
                                        for i in range(slices.size(0))])
                val_preds.extend(pred_centers)
                val_trues.extend(motors)
                val_voxel_spacings.extend(voxel_spacings)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        train_fbeta, train_TP, train_TN, train_FP, train_FN = calculate_fbeta_score(train_preds, train_trues, train_voxel_spacings)
        val_fbeta, val_TP, val_TN, val_FP, val_FN = calculate_fbeta_score(val_preds, val_trues, val_voxel_spacings)
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.6f}, Train F2: {train_fbeta:.4f}, TP: {train_TP}, TN: {train_TN}, FP: {train_FP}, FN: {train_FN}")
        print(f"Val Loss: {avg_val_loss:.6f}, Val F2: {val_fbeta:.4f}, TP: {val_TP}, TN: {val_TN}, FP: {val_FP}, FN: {val_FN}")

        if example_to_plot is not None:
            plot_slices(model, example_to_plot, device)
            example_to_plot = None

        if val_fbeta > best_val_fbeta:
            best_val_fbeta = val_fbeta
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with Val F2: {best_val_fbeta:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in Val F2. Epochs without improvement: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(torch.load(best_model_path))
                print(f"Loaded best model with Val F2: {best_val_fbeta:.4f}")
                break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Training completed. Best Val F2: {best_val_fbeta:.4f}")

    return best_val_fbeta

def preprocess_batch(slice_paths, new_size=(256, 256)):  # Updated for 256x256
    images = []
    for path in slice_paths:
        img = Image.open(path).convert('L')
        img = T.functional.resize(img, new_size)
        img = T.functional.to_tensor(img)
        img = (img - img.mean()) / (img.std() + 1e-8)
        images.append(img)
    return torch.stack(images)

def process_tomogram(tomo_id, model, test_dataset, device, index=0, total=1, confidence_threshold=0.3, nms_threshold=0.1):  # Tighter thresholds
    print(f"Processing tomogram {tomo_id} ({index}/{total})")
    tomo_dir = os.path.join('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test', tomo_id)
    slice_files = sorted([os.path.join(tomo_dir, f) for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    num_slices = len(slice_files)

    tomo_data = next(item for item in test_dataset.data if item['tomo_id'] == tomo_id)
    orig_shape = tomo_data['orig_shape']

    all_detections = []
    batch_size = 4 if device.startswith('cuda') else os.cpu_count() * 2

    if device.startswith('cuda'):
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem = gpu_mem - torch.cuda.memory_allocated(0) / 1e9
        batch_size = max(4, min(16, int(free_mem / 2)))

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, num_slices, batch_size):
            batch_end = min(batch_start + batch_size, num_slices)
            batch_paths = slice_files[batch_start:batch_end]
            batch_slices = preprocess_batch(batch_paths).to(device)

            with torch.cuda.amp.autocast():
                pred_heatmaps, pred_sizes, pred_offsets = model(batch_slices)

            for i, (heatmap, size, offset) in enumerate(zip(pred_heatmaps, pred_sizes, pred_offsets)):
                confidence = heatmap.max().item()
                if confidence >= confidence_threshold:
                    pred_center_norm, pred_size_norm, _ = extract_centroid(heatmap, size, offset, threshold=confidence_threshold)
                    z = batch_start + i
                    pred_center, pred_size = denormalize_predictions(pred_center_norm, pred_size_norm, z, orig_shape[:2])
                    all_detections.append({
                        'z': z,
                        'y': pred_center[1].item(),
                        'x': pred_center[2].item(),
                        'confidence': confidence,
                        'width': pred_size[1].item(),
                        'height': pred_size[0].item()
                    })

    final_detections = perform_3d_nms(all_detections, nms_threshold)

    if not final_detections:
        return {'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1}

    final_detections.sort(key=lambda x: x['confidence'], reverse=True)
    best_detection = final_detections[0]

    return {
        'tomo_id': tomo_id,
        'Motor axis 0': round(best_detection['z']),
        'Motor axis 1': round(best_detection['y']),
        'Motor axis 2': round(best_detection['x'])
    }

def perform_3d_nms(detections, distance_threshold=0.1):  # Tighter NMS
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    final_detections = []

    def distance_3d(d1, d2):
        return np.sqrt((d1['z'] - d2['z'])**2 + (d1['y'] - d2['y'])**2 + (d1['x'] - d2['x'])**2)

    trust_region = 4
    threshold = trust_region * distance_threshold

    while detections:
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        detections = [d for d in detections if distance_3d(d, best_detection) > threshold]

    return final_detections

def process_tomogram_wrapper(args):
    tomo_id, model, test_dataset, device, index, total = args
    return process_tomogram(tomo_id, model, test_dataset, device, index, total)

def generate_submission(test_dataset, model, device):
    total_tomos = len(test_dataset.tomo_ids)
    model.to(device)
    if device.startswith('cuda'):
        try:
            model.half()
            print("Using FP16 for inference")
        except:
            print("FP16 not supported")

    results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        args_list = [(tomo_id, model, test_dataset, device, i + 1, total_tomos) 
                     for i, tomo_id in enumerate(test_dataset.tomo_ids)]
        results = list(executor.map(process_tomogram_wrapper, args_list))

    submission_df = pd.DataFrame(results, columns=['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2'])
    submission_df.to_csv('/kaggle/working/submission.csv', index=False)
    motors_found = sum(1 for r in results if r['Motor axis 0'] != -1)
    print(f"Submission saved. Motors detected: {motors_found}/{total_tomos}")
    return submission_df

def plot_test_predictions(submission_df, test_dataset):
    valid_preds = submission_df[submission_df['Motor axis 0'] != -1]
    shape_map = {item['tomo_id']: item['orig_shape'] for item in test_dataset.data}

    for idx, row in valid_preds.iterrows():
        tomo_id = row['tomo_id']
        z = row['Motor axis 0']
        y = row['Motor axis 1']
        x = row['Motor axis 2']

        slice_path = os.path.join('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test', tomo_id, f'slice_{z:04d}.jpg')
        if not os.path.exists(slice_path):
            print(f"Slice {slice_path} not found, skipping.")
            continue

        img = Image.open(slice_path).convert('L')
        img_array = np.array(img)

        orig_shape = shape_map[tomo_id]
        orig_height, orig_width = orig_shape[0].item(), orig_shape[1].item()

        plt.figure(figsize=(8, 8))
        plt.imshow(img_array, cmap='gray')
        plt.plot(x, y, 'ro', label='Predicted Motor', markersize=10)
        plt.title(f"Tomogram: {tomo_id}, Z: {z}, Shape: {int(orig_height)}x{int(orig_width)}")
        plt.legend()
        plt.axis('off')
        plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    zs = valid_preds['Motor axis 0']
    ys = valid_preds['Motor axis 1']
    xs = valid_preds['Motor axis 2']

    ax.scatter(xs, ys, zs, c='r', marker='o', label='Predicted Motors')
    ax.set_xlabel('X (Motor axis 2)')
    ax.set_ylabel('Y (Motor axis 1)')
    ax.set_zlabel('Z (Motor axis 0)')
    ax.set_title('3D Distribution of Predicted Motors in Test Set')
    ax.legend()
    plt.show()

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    model = OptimizedCenterNet2D().to(device)

    trust_region = 4
    full_train_dataset = FlagellarDataset(
        csv_file='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv',
        root_dir='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train',
        new_size=(256, 256),
        trust_region=trust_region
    )

    # Get unique tomo_ids and their corresponding indices
    tomo_id_to_indices = {}
    for idx, item in enumerate(full_train_dataset.data):
        tomo_id = item['tomo_id']
        if tomo_id not in tomo_id_to_indices:
            tomo_id_to_indices[tomo_id] = []
        tomo_id_to_indices[tomo_id].append(idx)

    # Split tomo_ids into train and validation sets
    unique_tomo_ids = list(tomo_id_to_indices.keys())
    train_tomo_ids, val_tomo_ids = train_test_split(
        unique_tomo_ids, test_size=0.2, random_state=42
    )

    # Map tomo_ids back to dataset indices
    train_idx = []
    val_idx = []
    for tomo_id in train_tomo_ids:
        train_idx.extend(tomo_id_to_indices[tomo_id])
    for tomo_id in val_tomo_ids:
        val_idx.extend(tomo_id_to_indices[tomo_id])

    # Create train and validation subsets
    train_dataset = torch.utils.data.Subset(full_train_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_train_dataset, val_idx)

    # Verify no overlap in tomo_ids
    train_tomo_set = set(train_dataset.dataset.data[i]['tomo_id'] for i in train_idx)
    val_tomo_set = set(val_dataset.dataset.data[i]['tomo_id'] for i in val_idx)
    overlap = train_tomo_set.intersection(val_tomo_set)
    assert len(overlap) == 0, f"Overlap detected in tomo_ids: {overlap}"
    print(f"Train tomo_ids: {len(train_tomo_set)}, Val tomo_ids: {len(val_tomo_set)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True,
        collate_fn=custom_collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=0, pin_memory=True
    )

    test_dataset = FlagellarDataset(
        csv_file=None,
        root_dir='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test',
        new_size=(256, 256),
        trust_region=trust_region,
        is_test=True
    )

    train_model(model, train_loader, val_loader, num_epochs=150, device=device)
    submission_df = generate_submission(test_dataset, model, device)
    plot_test_predictions(submission_df, test_dataset)
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {(time.time() - start_time)/60:.2f} minutes")

