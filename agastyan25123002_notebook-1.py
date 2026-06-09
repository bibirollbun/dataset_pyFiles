# Imports
import json
import math
import heapq
from collections import deque
from pathlib import Path
from typing import List, Tuple, Dict

from pathlib import Path
import numpy as np
from PIL import Image
import random

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.nn as nn
import torchvision.transforms.functional as TF


GRID_SIZE = 20
IMG_SIZE = 320
NUM_CLASSES = 5  # 0..4
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-4


TRAIN_IMAGES_DIR = Path("/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/train/images")
TRAIN_LABELS_DIR = Path("/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/train/labels")

TEST_IMAGES_DIR = Path("/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/test/images")

SUBMISSION_PATH = Path("/kaggle/working/submission_baseline.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class-weighting to focus on walls/start/goal
# idx: class_id -> weight
CLASS_WEIGHTS = torch.tensor(
    [1.0,  # 0 = walkable
     2.0,  # 1 = wall
     5.0,  # 2 = hazard
     50.0,  # 3 = start
     50.0], # 4 = goal
    dtype=torch.float32
)

CLASS_WALL = 1
CLASS_START = 3
CLASS_GOAL = 4


class SemanticGridDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, augment: bool = False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.augment = augment
        
        self.image_ids = [
            p.stem for p in sorted(labels_dir.glob("*.json")) 
            if (images_dir / f"{p.stem}.png").exists()
        ]
        
        self.to_tensor = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.images_dir / f"{image_id}.png"
        label_path = self.labels_dir / f"{image_id}.json"

        img = Image.open(img_path).convert("RGB")
        with label_path.open("r") as f:
            data = json.load(f)
        grid = np.array(data["grid"], dtype=np.int64)
        
        x = self.to_tensor(img)
        y = torch.from_numpy(grid).long().unsqueeze(0)

        if self.augment:
            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                x = torch.rot90(x, k=rot_k, dims=[1, 2])
                y = torch.rot90(y, k=rot_k, dims=[1, 2])
            if random.random() > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)
            if random.random() > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)
                
        return x, y.squeeze(0)


class DoubleConv(nn.Module):
    """Helper: (Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetGrid(nn.Module):
    def __init__(self, n_channels=3, n_classes=5):
        super(UNetGrid, self).__init__()
        
        # --- ENCODER ---
        self.inc = DoubleConv(n_channels, 64)         
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))   
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)) 
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # --- DECODER ---
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256) 
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128) 
        

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)  

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.grid_pool = nn.AdaptiveAvgPool2d((20, 20))

    def forward(self, x):
        x1 = self.inc(x)       
        x2 = self.down1(x1)    
        x3 = self.down2(x2)    
        x4 = self.down3(x3)    

        x = self.up1(x4)                
        x = torch.cat([x3, x], dim=1)  
        x = self.conv1(x)

        x = self.up2(x)                
        x = torch.cat([x2, x], dim=1)  
        x = self.conv2(x)

        x = self.up3(x)                
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        
        logits_high_res = self.outc(x)
        logits_grid = self.grid_pool(logits_high_res)
        
        return logits_grid


def train_model(model: nn.Module, loader: DataLoader, epochs: int = EPOCHS):
    model.to(DEVICE)
    model.train()

    class_weights = CLASS_WEIGHTS.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE)               
            yb = yb.to(DEVICE)               

            optimizer.zero_grad()
            logits = model(xb)               

            B, C, G, _ = logits.shape
            logits_flat = logits.view(B, C, G * G)   
            y_flat = yb.view(B, G * G)              

            loss = criterion(logits_flat, y_flat)
            if torch.isnan(loss):
                print("Error: Loss turned to NaN! Stopping training.")
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch:02d} | loss = {avg_loss:.4f}")


def pick_start_goal_from_logits(logits):

    b, c, h, w = logits.shape
    flat_logits = logits.view(c, -1)
    
    values_s, indices_s = torch.topk(flat_logits[CLASS_START], k=2)
    s1_idx, s2_idx = indices_s[0].item(), indices_s[1].item()
    s1_score = values_s[0].item()
    
    values_g, indices_g = torch.topk(flat_logits[CLASS_GOAL], k=2)
    g1_idx, g2_idx = indices_g[0].item(), indices_g[1].item()
    g1_score = values_g[0].item()

    start_idx = s1_idx
    goal_idx = g1_idx
    
    if start_idx == goal_idx:
        score_a = s1_score + values_g[1].item()
        score_b = values_s[1].item() + g1_score
        
        if score_a > score_b:
            goal_idx = g2_idx
        else:
            start_idx = s2_idx 
    start_pos = (start_idx // w, start_idx % w)
    goal_pos = (goal_idx // w, goal_idx % w)
    
    return start_pos, goal_pos

import heapq

# --- NEW COST LOGIC & TERRAIN DETECTION ---

# 1. Define Base Costs from Problem Statement (PS1)
TERRAIN_COSTS = {
    "LAB":    {0: 1.0, 1: 9999.0, 2: 3.0, 3: 1.0, 4: 2.0},
    "FOREST": {0: 1.5, 1: 9999.0, 2: 2.8, 3: 1.5, 4: 2.5},
    "DESERT": {0: 1.2, 1: 9999.0, 2: 3.7, 3: 1.2, 4: 2.2}
}

def detect_terrain(img: Image.Image) -> str:
    """
    Determines if map is Lab, Forest, or Desert based on average color of the WHOLE image.
    """
    mean_color = np.array(img).mean(axis=(0, 1)) 
    r, g, b = mean_color[0], mean_color[1], mean_color[2]
    
    if g > r and g > b:
        return "FOREST"  
    elif r > b and g > b and r > 100: 
        return "DESERT"  
    else:
        return "LAB"  

def build_cost_matrix(grid_classes, terrain_type, boost_matrix):
    """
    Calculates the actual Step Cost for every cell.
    Formula: step_cost = base_cost - boost
    """
    costs = np.zeros((20, 20), dtype=np.float32)
    base_map = TERRAIN_COSTS[terrain_type]
    
    for r in range(20):
        for c in range(20):
            class_id = grid_classes[r, c]
            base = base_map.get(class_id, 1.0) 
            
            # Apply Boost
            boost = boost_matrix[r][c]
            
            # Final calculation
            step_cost = base - boost
            
            # Safety clamp
            if step_cost <= 0.01: step_cost = 0.01
            
            costs[r, c] = step_cost
            
    return costs

def astar_exact(cost_matrix, start, goal):
    """
    A* that finds the path with minimum Total Cost.
    """
    rows, cols = cost_matrix.shape
    pq = []
    heapq.heappush(pq, (0, 0, start))
    came_from = {}
    g_score = {start: 0}
    while pq:
        _, current_g, current = heapq.heappop(pq)
        if current == goal:
            break
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                step_cost = cost_matrix[nr, nc]
                if step_cost > 1000:
                    continue 
                new_g = current_g + step_cost  
                if (nr, nc) not in g_score or new_g < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = new_g
                    h = (abs(nr - goal[0]) + abs(nc - goal[1])) * 0.01
                    f = new_g + h           
                    heapq.heappush(pq, (f, new_g, (nr, nc)))
                    came_from[(nr, nc)] = current
    if goal not in came_from:
        return [] 
    path = []
    curr = goal
    while curr != start:
        path.append(curr)
        curr = came_from[curr]
    path.append(start)
    return path[::-1]

def fallback_manhattan_path(start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Simple L-shaped deterministic path (ignores walls).
    start->goal by row, then by col.
    """
    sr, sc = start
    gr, gc = goal
    path = []
    r, c = sr, sc
    path.append((r, c))

    # move vertically
    step_r = 1 if gr > r else -1
    while r != gr:
        r += step_r
        path.append((r, c))

    # move horizontally
    step_c = 1 if gc > c else -1
    while c != gc:
        c += step_c
        path.append((r, c))

    return path


def path_to_lrud(path: List[Tuple[int, int]]) -> str:
    """
    Convert list of (i,j) positions to lrud sequence.
    i = row (down), j = col (right).
    """
    moves = []
    for (i1, j1), (i2, j2) in zip(path[:-1], path[1:]):
        di, dj = i2 - i1, j2 - j1
        if di == 1 and dj == 0:
            moves.append("d")
        elif di == -1 and dj == 0:
            moves.append("u")
        elif di == 0 and dj == 1:
            moves.append("r")
        elif di == 0 and dj == -1:
            moves.append("l")
        else:
            moves.append("x")  # unexpected step
    return "".join(moves)


def predict_logits_and_grid(model: nn.Module, img_path: Path) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Runs inference on a single image using the U-Net.
    1. Resizes image to 320x320 (same as training).
    2. Feeds to model.
    3. Model outputs (1, 5, 20, 20).
    4. Returns grid prediction (20, 20) and logits.
    """
    model.eval()
    
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), 
        transforms.ToTensor(),
    ])
    x = transform(img).unsqueeze(0).to(DEVICE)  
    with torch.no_grad():
        logits = model(x)                  
        preds = torch.argmax(logits, dim=1)    

    grid_pred = preds.squeeze(0).cpu().numpy().astype(np.int64)
    logits_cpu = logits.cpu()
    return grid_pred, logits_cpu

def run_inference_on_test(model: nn.Module):
    print("Running inference with Physics-Aware A*...")
    model.to(DEVICE)
    model.eval()
    VELOCITY_DIR = TEST_IMAGES_DIR.parent / "velocities"
    image_paths = sorted(TEST_IMAGES_DIR.glob("*.png"))
    records = [("image_id", "path")]
    
    for idx, img_path in enumerate(image_paths):
        image_id = img_path.stem
        print(f"Processing {image_id}...")
        # 1. Load Image & Predict Grid
        img = Image.open(img_path).convert("RGB")
        grid_pred, logits = predict_logits_and_grid(model, img_path)
        # 2. Pick Start/Goal
        start, goal = pick_start_goal_from_logits(logits)
        if 0 <= start[0] < 20 and 0 <= start[1] < 20:
            grid_pred[start[0], start[1]] = 0
        if 0 <= goal[0] < 20 and 0 <= goal[1] < 20:
            grid_pred[goal[0], goal[1]] = 0
        # 3. Detect Terrain
        terrain = detect_terrain(img)
        # 4. Load Velocity Boost
        boost_path = VELOCITY_DIR / f"{image_id}.json"
        if boost_path.exists():
            with open(boost_path, 'r') as f:
                boost_data = json.load(f)
            boost_matrix = np.array(boost_data["boost"])
        else:
            boost_matrix = np.zeros((20, 20))
        # 5. Build Cost Matrix & Pathfind
        cost_matrix = build_cost_matrix(grid_pred, terrain, boost_matrix)
        path = astar_exact(cost_matrix, start, goal)
        # 6. Fallback
        if not path:
            path = fallback_manhattan_path(start, goal)
        # 7. Convert to Moves
        if len(path) < 2:
             moves = "r" 
        else:
             moves = path_to_lrud(path)
        records.append((image_id, moves))
        
    with SUBMISSION_PATH.open("w", encoding="utf-8") as f:
        for image_id, path_str in records:
            f.write(f"{image_id},{path_str}\n")

    print(f"Submission saved to: {SUBMISSION_PATH}")


train_dataset = SemanticGridDataset(
    TRAIN_IMAGES_DIR, 
    TRAIN_LABELS_DIR, 
    augment=True
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNetGrid(n_channels=3, n_classes=5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 3. Train model
print("Training baseline model (focus: start/goal/walls)...")
train_model(model, train_loader, epochs=EPOCHS)

# 4. Inference + CSV
print("Running inference on test set...")
run_inference_on_test(model)

