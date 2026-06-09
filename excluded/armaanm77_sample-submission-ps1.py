# Imports
import json
import math
import heapq
from collections import deque
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


GRID_SIZE = 20
NUM_CLASSES = 5  # 0..4
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-3


TRAIN_IMAGES_DIR = Path("/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/train/images")
TRAIN_LABELS_DIR = Path("/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/train/labels")

TEST_IMAGES_DIR = Path("/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/test/images")

SUBMISSION_PATH = Path("/kaggle/working/submission_baseline.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class-weighting to focus on walls/start/goal
# idx: class_id -> weight
CLASS_WEIGHTS = torch.tensor(
    [0.5,  # 0 = walkable
     3.0,  # 1 = wall
     0.7,  # 2 = hazard
     4.0,  # 3 = start
     4.0], # 4 = goal
    dtype=torch.float32
)

CLASS_WALL = 1
CLASS_START = 3
CLASS_GOAL = 4


def load_label_grid(json_path: Path) -> np.ndarray:
    """Load true class grid from label JSON (expects key 'grid')."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "grid" not in data:
        raise KeyError(
            f"'grid' not found in {json_path}. "
            "Make sure your generator writes 'grid': grid.tolist() into labels."
        )
    grid = np.array(data["grid"], dtype=np.int64)
    assert grid.shape == (GRID_SIZE, GRID_SIZE), f"Expected {GRID_SIZE}x{GRID_SIZE}, got {grid.shape}"
    return grid


def compute_cell_tensor(img: Image.Image, grid_size: int) -> torch.Tensor:
    """
    Convert full map image into a (3, grid_size, grid_size) tensor by
    averaging RGB values in each cell.

    Very crude, but enough for a baseline.
    """
    img = img.convert("RGB")
    w, h = img.size
    img_arr = np.array(img, dtype=np.float32) / 255.0  # (H,W,3) in [0,1]

    cell_w = w // grid_size
    cell_h = h // grid_size

    cells = np.zeros((grid_size, grid_size, 3), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            x0 = j * cell_w
            x1 = (j + 1) * cell_w if j < grid_size - 1 else w
            y0 = i * cell_h
            y1 = (i + 1) * cell_h if i < grid_size - 1 else h

            patch = img_arr[y0:y1, x0:x1, :]
            if patch.size == 0:
                continue
            cells[i, j, :] = patch.mean(axis=(0, 1))

    # (G,G,3) -> (3,G,G)
    cells = np.transpose(cells, (2, 0, 1))
    return torch.from_numpy(cells)  # float32


class GridDataset(Dataset):
    """
    Each sample:
      X: (3, GRID_SIZE, GRID_SIZE) tensor
      y: (GRID_SIZE, GRID_SIZE) long tensor with values 0..4
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        grid_size: int,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.grid_size = grid_size

        self.image_ids: List[str] = []
        for p in sorted(labels_dir.glob("*.json")):
            image_id = p.stem  # "0001"
            img_path = images_dir / f"{image_id}.png"
            if img_path.is_file():
                self.image_ids.append(image_id)

        if not self.image_ids:
            raise RuntimeError(f"No training labels/images found in {labels_dir}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_path = self.images_dir / f"{image_id}.png"
        label_path = self.labels_dir / f"{image_id}.json"

        img = Image.open(img_path)
        x = compute_cell_tensor(img, self.grid_size).float()  # (3, G, G)

        grid = load_label_grid(label_path)  # (G, G)
        y = torch.from_numpy(grid).long()

        return x, y


class SimpleGridNet(nn.Module):
    """
    Simple per-grid classifier:
      - Flatten (3,G,G) -> FC -> ReLU -> FC -> reshape to (C,G,G)
    This is intentionally basic, but with class-weighted loss we bias
    it to learn walls/start/goal reasonably.
    """

    def __init__(self, grid_size: int, num_classes: int):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        in_features = 3 * grid_size * grid_size
        hidden = 256

        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, num_classes * grid_size * grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, G, G)
        b = x.shape[0]
        x = x.reshape(b, -1)            # (B, 3*G*G)
        x = torch.relu(self.fc1(x))     # (B, hidden)
        x = self.fc2(x)                 # (B, num_classes*G*G)
        x = x.view(b, self.num_classes, self.grid_size, self.grid_size)
        return x


def train_model(model: nn.Module, loader: DataLoader, epochs: int = EPOCHS):
    model.to(DEVICE)
    model.train()

    class_weights = CLASS_WEIGHTS.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE)               # (B, 3, G, G)
            yb = yb.to(DEVICE)               # (B, G, G)

            optimizer.zero_grad()
            logits = model(xb)               # (B, C, G, G)

            B, C, G, _ = logits.shape
            logits_flat = logits.view(B, C, G * G)   # (B, C, G*G)
            y_flat = yb.view(B, G * G)              # (B, G*G)

            loss = criterion(logits_flat, y_flat)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch:02d} | loss = {avg_loss:.4f}")


def pick_start_goal_from_logits(logits: torch.Tensor) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    logits: (1, C, G, G)
    Returns:
      start = (row, col) with highest logit for START class
      goal  = (row, col) with highest logit for GOAL class
    """
    assert logits.shape[0] == 1
    _, C, G, _ = logits.shape

    # class index 3 = start, 4 = goal
    logit_start = logits[0, CLASS_START, :, :]  # (G,G)
    logit_goal = logits[0, CLASS_GOAL, :, :]    # (G,G)

    # Flatten then argmax -> index -> (i,j)
    start_idx = torch.argmax(logit_start).item()
    goal_idx = torch.argmax(logit_goal).item()

    start_row = start_idx // G
    start_col = start_idx % G

    goal_row = goal_idx // G
    goal_col = goal_idx % G

    return (int(start_row), int(start_col)), (int(goal_row), int(goal_col))


def bfs_path(grid_pred: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    BFS ignoring cost, just avoiding walls (class 1).
    Returns path as list of (i,j), or [] if no path found.
    """
    G = grid_pred.shape[0]
    sr, sc = start
    gr, gc = goal

    visited = np.zeros((G, G), dtype=bool)
    prev: Dict[Tuple[int, int], Tuple[int, int]] = {}

    q = deque()
    q.append((sr, sc))
    visited[sr, sc] = True

    def neighbours(i, j):
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < G and 0 <= nj < G:
                yield ni, nj

    while q:
        i, j = q.popleft()
        if (i, j) == (gr, gc):
            break
        for ni, nj in neighbours(i, j):
            if visited[ni, nj]:
                continue
            if grid_pred[ni, nj] == CLASS_WALL:
                continue  # block walls
            visited[ni, nj] = True
            prev[(ni, nj)] = (i, j)
            q.append((ni, nj))

    if (gr, gc) not in prev and (gr, gc) != (sr, sc):
        # No path found
        return []

    # Reconstruct
    path: List[Tuple[int, int]] = []
    cur = (gr, gc)
    path.append(cur)
    while cur != (sr, sc):
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


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
    Returns:
      grid_pred: (G,G) np.int64 argmax classes
      logits:    (1,C,G,G) torch.Tensor (on CPU) for start/goal picking
    """
    img = Image.open(img_path)
    x = compute_cell_tensor(img, GRID_SIZE).unsqueeze(0).float()  # (1,3,G,G)
    x = x.to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(x)                       # (1,C,G,G)
        preds = torch.argmax(logits, dim=1)     # (1,G,G)

    grid_pred = preds.squeeze(0).cpu().numpy().astype(np.int64)
    logits_cpu = logits.cpu()
    return grid_pred, logits_cpu


def run_inference_on_test(model: nn.Module):
    """
    For each test image:
      - Predict grid + logits
      - Pick start/goal from logits
      - BFS path avoiding walls, else Manhattan fallback
      - Write submission CSV
    """
    model.to(DEVICE)
    image_paths = sorted(TEST_IMAGES_DIR.glob("*.png"))

    records = [("image_id", "path")]

    for img_path in image_paths:
        image_id = img_path.stem
        print(f"Inference on {image_id}...")

        grid_pred, logits = predict_logits_and_grid(model, img_path)

        # Start/goal via logits (most confident cell for class 3/4)
        start, goal = pick_start_goal_from_logits(logits)

        # BFS with walls from predicted grid
        path = bfs_path(grid_pred, start, goal)
        if not path:
            # BFS failed -> fallback ignoring walls
            path = fallback_manhattan_path(start, goal)

        moves = path_to_lrud(path)
        records.append((image_id, moves))

    # Write CSV
    with SUBMISSION_PATH.open("w", encoding="utf-8") as f:
        for image_id, path_str in records:
            f.write(f"{image_id},{path_str}\n")

    print(f"Baseline submission written to: {SUBMISSION_PATH}")



train_dataset = GridDataset(
        images_dir=TRAIN_IMAGES_DIR,
        labels_dir=TRAIN_LABELS_DIR,
        grid_size=GRID_SIZE,
    )

train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # 2. Create model
model = SimpleGridNet(grid_size=GRID_SIZE, num_classes=NUM_CLASSES)

# 3. Train model
print("Training baseline model (focus: start/goal/walls)...")
train_model(model, train_loader, epochs=EPOCHS)

# 4. Inference + CSV
print("Running inference on test set...")
run_inference_on_test(model)

