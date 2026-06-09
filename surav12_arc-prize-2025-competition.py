# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Imports
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# File paths (auto-locate in Kaggle environment)
DATA_DIR = Path("/kaggle/input/arc-prize-2025")
TRAIN_FILE = DATA_DIR / "arc-agi_training_challenges.json"
TEST_FILE = DATA_DIR / "arc-agi_test_challenges.json"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.json"

# Load JSON
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

train_data = load_json(TRAIN_FILE)
test_data = load_json(TEST_FILE)
print(f"Loaded {len(train_data)} training tasks and {len(test_data)} test tasks.")


# Visualize a grid with matplotlib
def show_grid(grid, title=None):
    arr = np.array(grid)
    cmap = plt.cm.get_cmap('tab20', 10)  # 10 discrete colors
    plt.imshow(arr, cmap=cmap, vmin=0, vmax=9)
    plt.xticks([]), plt.yticks([])
    if title: plt.title(title)
    plt.show()

# Visualize a task (input/output pairs)
def visualize_task(task, task_id=None):
    print(f"Task ID: {task_id}")
    for i, pair in enumerate(task['train']):
        print(f"Train Example {i+1}")
        show_grid(pair['input'], "Input")
        show_grid(pair['output'], "Output")

    for i, pair in enumerate(task['test']):
        print(f"Test Input {i+1}")
        show_grid(pair['input'], "Test Input")



# Example: Visualize a random training task
import random
random_task_id = random.choice(list(train_data.keys()))
visualize_task(train_data[random_task_id], random_task_id)


# Example solver: copies the input as output (not correct but valid format)
def identity_solver(input_grid):
    return input_grid  # Just returns the same grid

# Generate two attempts (for now both are the same)
def solve_task(task):
    predictions = []
    for test_pair in task['test']:
        input_grid = test_pair['input']
        attempt_1 = identity_solver(input_grid)
        attempt_2 = identity_solver(input_grid)  # You can try a variation
        predictions.append({
            "attempt_1": attempt_1,
            "attempt_2": attempt_2
        })
    return predictions



import numpy as np

# Rotate grid 90 * k degrees
def rotate(grid, k=1):
    return np.rot90(grid, k)

# Flip grid: axis = 0 (vertical), 1 (horizontal)
def flip(grid, axis=0):
    return np.flip(grid, axis=axis)

# Replace all occurrences of from_color with to_color
def color_replace(grid, from_color, to_color):
    grid = np.array(grid)
    grid[grid == from_color] = to_color
    return grid

# Crop grid from (x1,y1) to (x2,y2) inclusive
def crop(grid, x1, y1, x2, y2):
    return grid[y1:y2+1, x1:x2+1]

# Pad grid with specified padding amounts
def pad(grid, top=0, bottom=0, left=0, right=0, fill=0):
    return np.pad(grid, ((top, bottom), (left, right)), constant_values=fill)

# Flood fill from (x, y)
def flood_fill(grid, x, y, new_color):
    grid = grid.copy()
    target_color = grid[y][x]
    if target_color == new_color:
        return grid

    h, w = grid.shape
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue
        if grid[cy][cx] != target_color:
            continue
        grid[cy][cx] = new_color
        stack += [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]
    return grid

# Get bounding box (x1, y1, x2, y2) of a color
def bounding_box(grid, color):
    grid = np.array(grid)
    yx = np.argwhere(grid == color)
    if len(yx) == 0:
        return None
    ys, xs = yx[:, 0], yx[:, 1]
    return np.min(xs), np.min(ys), np.max(xs), np.max(ys)



# Example usage
grid = np.array([[1, 1, 0],
                 [1, 0, 0],
                 [0, 0, 2]])

print("Original Grid:")
print(grid)

print("\nRotated:")
print(rotate(grid, 1))

print("\nFlipped:")
print(flip(grid, 1))

print("\nColor Replace (1â†’3):")
print(color_replace(grid, 1, 3))

print("\nFlood Fill (0,0) with 9:")
print(flood_fill(grid, 0, 0, 9))

print("\nBounding Box of color 1:")
print(bounding_box(grid, 1))



# List of available DSL operations with parameters to try
def get_dsl_operation_candidates(grid):
    return [
        lambda g: rotate(g, 1),
        lambda g: rotate(g, 2),
        lambda g: rotate(g, 3),
        lambda g: flip(g, 0),
        lambda g: flip(g, 1),
        lambda g: color_replace(g, 1, 2),
        lambda g: color_replace(g, 2, 1),
        lambda g: color_replace(g, 1, 3),
        lambda g: color_replace(g, 3, 1),
        lambda g: color_replace(g, 2, 3),
        lambda g: color_replace(g, 3, 2),
    ]


def run_program(grid, program_steps):
    result = np.array(grid)
    try:
        for op in program_steps:
            result = op(result)
        return result
    except Exception:
        return None



def program_matches(train_examples, program_steps):
    for pair in train_examples:
        input_grid = np.array(pair["input"])
        target_output = np.array(pair["output"])
        predicted_output = run_program(input_grid, program_steps)
        if predicted_output is None or not np.array_equal(predicted_output, target_output):
            return False
    return True



# DFS search for a working program of max depth D
def search_program(train_examples, max_depth=3):
    def dfs(current_steps):
        if len(current_steps) > max_depth:
            return None
        if len(current_steps) > 0 and program_matches(train_examples, current_steps):
            return current_steps
        for op in get_dsl_operation_candidates(train_examples[0]['input']):
            next_steps = current_steps + [op]
            result = dfs(next_steps)
            if result is not None:
                return result
        return None

    return dfs([])



# Solve ARC task using program search
def solve_task_symbolic(task, max_depth=3):
    train = task["train"]
    test = task["test"]
    solution_program = search_program(train, max_depth=max_depth)
    
    predictions = []
    for test_example in test:
        input_grid = np.array(test_example["input"])
        if solution_program:
            output = run_program(input_grid, solution_program)
            if output is None:
                output = input_grid  # fallback
        else:
            output = input_grid  # fallback
        predictions.append({
            "attempt_1": output.tolist(),
            "attempt_2": output.tolist()
        })
    return predictions


# Try symbolic solver on 1 training task
example_id = list(train_data.keys())[0]
example_task = train_data[example_id]
visualize_task(example_task, task_id=example_id)

predictions = solve_task_symbolic(example_task, max_depth=2)
print("Predicted test outputs:")
for p in predictions:
    show_grid(p["attempt_1"])



# Define DSL operation tokens
DSL_TOKENS = [
    "rotate_90", "rotate_180", "rotate_270",
    "flip_v", "flip_h",
    "color_replace_1_2", "color_replace_2_1",
    "color_replace_1_3", "color_replace_3_1",
    "color_replace_2_3", "color_replace_3_2"
]

TOKEN_TO_FUNC = {
    "rotate_90": lambda g: rotate(g, 1),
    "rotate_180": lambda g: rotate(g, 2),
    "rotate_270": lambda g: rotate(g, 3),
    "flip_v": lambda g: flip(g, 0),
    "flip_h": lambda g: flip(g, 1),
    "color_replace_1_2": lambda g: color_replace(g, 1, 2),
    "color_replace_2_1": lambda g: color_replace(g, 2, 1),
    "color_replace_1_3": lambda g: color_replace(g, 1, 3),
    "color_replace_3_1": lambda g: color_replace(g, 3, 1),
    "color_replace_2_3": lambda g: color_replace(g, 2, 3),
    "color_replace_3_2": lambda g: color_replace(g, 3, 2)
}


import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_TOKENS = len(DSL_TOKENS)
NUM_COLORS = 10  # 0â€“9

# One-hot encoding for grid
def one_hot_encode_grid(grid, num_colors=10):
    h, w = len(grid), len(grid[0])
    encoded = np.zeros((h, w, num_colors), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            color = grid[i][j]
            if 0 <= color < num_colors:
                encoded[i, j, color] = 1.0

    return encoded

# ARC Sketch Predictor Model
class SketchPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(40, 32, kernel_size=3, padding=1),  # 40 channels = inp + out
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, NUM_TOKENS)

    def forward(self, x):  # x: (B, 40, H, W)
        x = self.cnn(x)     # -> (B, 64, 1, 1)
        x = x.view(x.size(0), -1)  # -> (B, 64)
        logits = self.fc(x)        # -> (B, NUM_TOKENS)
        return logits



# Dummy dataset generation (replace with real ARC DSL annotations later)
def generate_training_pairs(train_data, n=100):
    data = []
    for task in train_data.values():
        for pair in task["train"]:
            inp = np.array(pair["input"])
            out = np.array(pair["output"])
            # âš ï¸� For now, randomly assign a fake token label
            label = np.random.randint(0, NUM_TOKENS)
            data.append((inp, out, label))
            if len(data) >= n:
                return data
    return data



def to_tensor_batch(batch, max_channels=40, max_size=30):
    def one_hot(grid, num_channels=10):
        h, w = len(grid), len(grid[0])
        arr = np.zeros((num_channels, h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                arr[grid[i][j], i, j] = 1.0
        return arr

    def pad_channels(g, target_c):
        pad_c = target_c - g.shape[0]
        return np.pad(g, ((0, pad_c), (0, 0), (0, 0)), mode='constant')

    def pad_size(g, target_h, target_w):
        pad_h = target_h - g.shape[1]
        pad_w = target_w - g.shape[2]
        return np.pad(g, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')

    xs, ys = [], []
    for example in batch:
        inp, out = example[:2]
    
        inp_1hot = one_hot(inp)
        out_1hot = one_hot(out)
    
        inp_1hot = pad_channels(inp_1hot, max_channels)
        out_1hot = pad_channels(out_1hot, max_channels)
    
        # âœ… Make sure height and width match before stacking
        h = max(inp_1hot.shape[1], out_1hot.shape[1])
        w = max(inp_1hot.shape[2], out_1hot.shape[2])
    
        inp_1hot = pad_size(inp_1hot, h, w)
        out_1hot = pad_size(out_1hot, h, w)
    
        x = inp_1hot  # now (2*C, H, W)
        x = pad_size(x, target_h=max_size, target_w=max_size)  # final shape
        xs.append(x)
        ys.append(0)  # or real label

    x_tensor = torch.tensor(np.array(xs))  # shape: (B, C, H, W)
    y_tensor = torch.tensor(ys)
    return x_tensor, y_tensor



model = SketchPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

data = generate_training_pairs(train_data, n=500)
batch_size = 32

for epoch in range(5):
    np.random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        x, y = to_tensor_batch(batch)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")



def predict_dsl_tokens(model, input_grid, output_grid, top_k=5):
    x_in = one_hot_encode_grid(input_grid)
    x_out = one_hot_encode_grid(output_grid)

    # âœ… Pad both to same shape
    def pad_to_shape(a, shape):
        pad_h = shape[0] - a.shape[0]
        pad_w = shape[1] - a.shape[1]
        return np.pad(a, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    target_shape = (
        max(x_in.shape[0], x_out.shape[0]),
        max(x_in.shape[1], x_out.shape[1])
    )

    x_in = pad_to_shape(x_in, target_shape)
    x_out = pad_to_shape(x_out, target_shape)

    # âœ… Now safe to concatenate
    x = torch.tensor([np.concatenate([x_in, x_out], axis=0)])  # shape: (1, H+H, W, C)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        top_tokens = torch.topk(probs, top_k, dim=-1).indices.squeeze(0).tolist()

    return top_tokens



# Search constrained to predicted tokens only
def search_program_guided(train_examples, predicted_tokens, max_depth=2):
    allowed_ops = [TOKEN_TO_FUNC[tok] for tok in predicted_tokens]

    def dfs(current_steps):
        if len(current_steps) > max_depth:
            return None
        if len(current_steps) > 0 and program_matches(train_examples, current_steps):
            return current_steps
        for op in allowed_ops:
            next_steps = current_steps + [op]
            result = dfs(next_steps)
            if result is not None:
                return result
        return None

    return dfs([])



from glob import glob

all_jsons = glob("**/*.json", recursive=True)
print(f"Total JSON files found: {len(all_jsons)}")
print("\n".join(all_jsons[:10]))  # Show first 10



import os
from glob import glob

test_dir = "data/test"
print("Test folder exists:", os.path.exists(test_dir))
print("Test JSON files found:", glob(os.path.join(test_dir, "*.json")))



import json

test_file = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"

with open(test_file, "r") as f:
    data = json.load(f)

# Print type and sample
print("Type of data:", type(data))

if isinstance(data, list):
    print("List item sample keys:", list(data[0].keys()))
    print("Sample content:", data[0])
elif isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))
    for k, v in list(data.items())[:1]:
        print(f"Sample key: {k}, value type: {type(v)}, sample value: {v}")
else:
    print("Unknown structure")



import json

TRAIN_PATH = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"

with open(TRAIN_PATH, "r") as f:
    data = json.load(f)

print(type(data))
print(data[:1] if isinstance(data, list) else list(data.items())[:1])



import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# âœ… Dataset class
class ARCTaskDataset(Dataset):
    def __init__(self, tasks, labels):
        self.samples = []
        for task_id, task in tasks.items():
            label = labels[task_id]
            for pair in task["train"]:
                x = self.extract_rgb_features(pair["input"])
                self.samples.append((x, label))

    def extract_rgb_features(self, grid):
        arr = np.array(grid)
        r = np.sum(arr == 2)   # Red
        g = np.sum(arr == 3)   # Green
        b = np.sum(arr == 1)   # Blue
        return torch.tensor([r, g, b], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# âœ… Task type map (binary labels for example)
TASK_TYPE_MAP = {
    "00576224": 1,
    "007bbfb7": 0,
    "009d5c81": 1,
    "00d62c1b": 0,
    "00dbd492": 1,
    "017c7c7b": 0,
    "0354d160": 1,
    "0372dc99": 0,
    "03b5d8f3": 1,
    "042ee4c2": 0,
    # Add more if needed
}

# âœ… Load ARC tasks
def load_arc_tasks(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data  # it's already a dict of tasks

# âœ… Define file path
TRAIN_PATH = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"
arc_tasks = load_arc_tasks(TRAIN_PATH)

# âœ… Filter tasks and labels
task_ids = list(TASK_TYPE_MAP.keys())
arc_tasks = {k: arc_tasks[k] for k in task_ids if k in arc_tasks}
labels = {k: TASK_TYPE_MAP[k] for k in arc_tasks}

print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# âœ… Build Dataset and DataLoader
dataset = ARCTaskDataset(arc_tasks, labels)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# âœ… Preview 3 samples
for i, (features, label) in enumerate(loader):
    print("Features (RGB):", features)
    print("Labels:", label)
    if i == 2:
        break



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json

# âœ… Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# âœ… Task mapping (your own set)
TASK_TYPE_MAP = {
    "00576224": 0,
    "007bbfb7": 0,
    "009d5c81": 1,
}

# âœ… Load JSON as dict
def load_arc_tasks(path):
    with open(path, "r") as f:
        tasks = json.load(f)
    return tasks

# âœ… RGB feature extractor
def extract_rgb_features(grid):
    red, green, blue = 0, 0, 0
    for row in grid:
        for val in row:
            if val == 2: red += 1
            elif val == 3: green += 1
            elif val == 5: blue += 1
    return torch.tensor([red, green, blue], dtype=torch.float32)

# âœ… Dataset class
class ARCTaskDataset(torch.utils.data.Dataset):
    def __init__(self, tasks, task_type_map):
        self.samples = []
        for task_id, task_data in tasks.items():
            label = task_type_map[task_id]
            for pair in task_data["train"]:
                x = extract_rgb_features(pair["input"])
                self.samples.append((x, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# âœ… Define file path
TRAIN_PATH = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"
all_tasks = load_arc_tasks(TRAIN_PATH)

# âœ… Filter only available task_ids
arc_tasks = {k: v for k, v in all_tasks.items() if k in TASK_TYPE_MAP}
labels = list(TASK_TYPE_MAP.values())

# âœ… Prepare dataset and dataloader
dataset = ARCTaskDataset(arc_tasks, TASK_TYPE_MAP)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# âœ… Define a simple MLP on RGB features
class RGBClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, len(set(TASK_TYPE_MAP.values())))
        )

    def forward(self, x):
        return self.model(x)

# âœ… Initialize model, loss, optimizer
model = RGBClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# âœ… Train loop
EPOCHS = 20
for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")



import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib  # for saving scaler

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Load ARC Tasks ---
def load_arc_tasks(path):
    with open(path, 'r') as f:
        task_dict = json.load(f)
    return task_dict

# --- Feature Engineering ---
def extract_features(grid):
    grid = np.array(grid)
    height, width = grid.shape
    unique_colors, counts = np.unique(grid, return_counts=True)
    color_counts = np.zeros(10)
    for color, count in zip(unique_colors, counts):
        if color < 10:
            color_counts[color] = count

    probs = counts / counts.sum()
    color_entropy = -np.sum(probs * np.log2(probs + 1e-9))
    sym_h = int(np.array_equal(grid, np.flipud(grid)))
    sym_v = int(np.array_equal(grid, np.fliplr(grid)))

    def is_checkerboard(g):
        return int(np.all(g[::2, ::2] == g[0, 0]))

    checker = is_checkerboard(grid)
    repetition = int(np.any(np.diff(grid, axis=0).std() == 0)) + int(np.any(np.diff(grid, axis=1).std() == 0))

    features = [height, width, len(unique_colors), color_entropy, sym_h, sym_v, checker, repetition]
    return features + color_counts.tolist()

# --- Dataset Creation ---
def prepare_dataset(arc_tasks, task_type_map):
    features, labels = [], []
    for task_id, task in arc_tasks.items():
        if task_id not in task_type_map:
            continue
        label = task_type_map[task_id]
        for pair in task['train']:
            feats = extract_features(pair['input'])
            features.append(feats)
            labels.append(label)
    return np.array(features, dtype=np.float32), np.array(labels)

# --- Neural Classifier ---
class ARCClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- Training Function ---
def train_model(X, y, num_classes=3, epochs=20, batch_size=8):
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = ARCClassifier(input_dim=X.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    print(f"\U0001F3AF Validation Accuracy: {correct / total * 100:.2f}%")
    return model, scaler

# --- Inference Function ---
def predict_task_type(model, scaler, task):
    model.eval()
    inputs = torch.tensor([extract_features(pair['input']) for pair in task['train']], dtype=torch.float32)
    inputs = scaler.transform(inputs)
    inputs = torch.tensor(inputs).to(device)
    with torch.no_grad():
        logits = model(inputs)
        preds = torch.softmax(logits, dim=1).mean(dim=0).argmax().item()
    return preds

# === Full Pipeline ===
TRAIN_PATH = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"
arc_tasks = load_arc_tasks(TRAIN_PATH)

# âœ… Label your training tasks manually (extend as needed)
TASK_TYPE_MAP = {
    "00576224": 0,
    "007bbfb7": 0,
    "009d5c81": 1,
    "00d62c1b": 2,
    "00dbd492": 1,
    "017c7c7b": 2,
    # ... extend with more task_id: label
}

X, y = prepare_dataset(arc_tasks, TASK_TYPE_MAP)
model, scaler = train_model(X, y, num_classes=3)

# Save model & scaler
torch.save(model.state_dict(), "task_classifier.pth")
joblib.dump(scaler, "task_scaler.pkl")
print("âœ… Model and scaler saved!")



# import json
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm

# # === Load trained PyTorch model weights ===
# torch.save(model.state_dict(), "task_classifier_model.pt")
# joblib.dump(scaler, "task_scaler.pkl")


# model.eval()

# # === Load the saved scaler ===
# scaler = joblib.load("task_scaler.pkl")

# # === Load test tasks ===
# TEST_PATH = "/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json"

# def load_arc_tasks(path):
#     with open(path, 'r') as f:
#         return json.load(f)


# def extract_features(task):
#     def get_color_stats(grid):
#         colors = [cell for row in grid for cell in row]
#         counts = pd.Series(colors).value_counts()
#         return counts.to_dict(), len(set(colors))

#     features = []
#     for io in task["train"] + task["test"]:
#         grid = io["input"]
#         color_counts, n_colors = get_color_stats(grid)
#         height = len(grid)
#         width = len(grid[0])
#         color_entropy = pd.Series(color_counts).apply(lambda x: -x * np.log2(x)).sum()
#         sym_h = int(grid == grid[::-1])
#         sym_v = int(all([row == row[::-1] for row in grid]))
#         checker = int(all([(i+j)%2==grid[i][j]%2 for i in range(len(grid)) for j in range(len(grid[0]))]))
#         repetition = int(any(grid[i] == grid[i+1] for i in range(len(grid)-1)))
        
#         feature_vector = [height, width, n_colors, color_entropy, sym_h, sym_v, checker, repetition]
#         for i in range(10):
#             feature_vector.append(color_counts.get(i, 0))
#         features.append(feature_vector)

#     return np.mean(features, axis=0)

# # === Generate predictions and save submission ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# test_tasks = load_arc_tasks(TEST_PATH)
# submission = []

# for task_id, task in tqdm(test_tasks.items(), desc="Generating predictions"):
#     feat = extract_features(task)
#     feat_scaled = scaler.transform([feat])  # scale with saved StandardScaler
#     X_tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(device)
#     with torch.no_grad():
#         pred = model(X_tensor).argmax(dim=1).item()
#     submission.append({"id": task_id, "outputs": [pred]})

# # Save to submission.json
# with open("submission.json", "w") as f:
#     json.dump(submission, f)

# print("âœ… submission.json created successfully!")



# import json
# import os

# task_dir = "/kaggle/input/arc-dataset/"  # Adjust as needed
# valid_tasks = []

# for filename in os.listdir(task_dir):
#     if not filename.endswith(".json"):
#         continue

#     path = os.path.join(task_dir, filename)
#     with open(path) as f:
#         try:
#             task = json.load(f)
#             if "train" in task and "test" in task:
#                 valid_tasks.append((filename, task))
#         except json.JSONDecodeError:
#             continue  # skip corrupt or non-JSON files

# print(f"âœ… Loaded {len(valid_tasks)} valid task files")



# import json
# from tqdm import tqdm

# # File paths
# TEST_CHALLENGES_PATH = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
# SUBMISSION_OUTPUT_PATH = "submission.json"

# # âœ… Load test challenge dictionary
# with open(TEST_CHALLENGES_PATH, "r") as f:
#     test_challenges = json.load(f)

# print(f"âœ… Loaded {len(test_challenges)} test challenges")

# # âœ… Generate predictions
# predictions = []

# for challenge_id, challenge in tqdm(test_challenges.items()):
#     test_cases = challenge.get("test", [])

#     # âš ï¸� Handle missing or empty test sections
#     if not test_cases:
#         print(f"âš ï¸� Skipping {challenge_id} (missing test cases)")
#         continue

#     # ğŸ”� Dummy model: Predict 0 for all outputs
#     # Replace with your model logic
#     output = [0 for _ in test_cases]

#     predictions.append({
#         "id": challenge_id,
#         "outputs": output
#     })

# # âœ… Write submission
# with open(SUBMISSION_OUTPUT_PATH, "w") as f:
#     json.dump(predictions, f)

# print(f"âœ… submission.json saved with {len(predictions)} entries")



import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm

# === Define model class (based on saved state_dict) ===
class ARCClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(18, 64),  # model.0
            nn.ReLU(),          # model.1
            nn.Linear(64, 3)    # model.2
        )

    def forward(self, x):
        return self.model(x)

# === Define 18-feature extractor ===
def extract_features(task):
    def grid_stats(grid):
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        flat = grid_tensor.flatten()
        return [
            flat.mean().item(),
            flat.std().item(),
            flat.min().item(),
            flat.max().item(),
            float(len(grid)),
            float(len(grid[0])),
            float((flat == 0).sum()) / flat.numel(),
            float((flat == 1).sum()) / flat.numel(),
            float((flat == 2).sum()) / flat.numel()
        ]

    input_grids = [pair["input"] for pair in task.get("train", []) + task.get("test", [])]
    features = []
    for grid in input_grids[:2]:  # Limit to first 2 grids
        features += grid_stats(grid)

    while len(features) < 18:
        features.append(0.0)
    features = features[:18]

    return torch.tensor(features).float().unsqueeze(0)  # Shape: [1, 18]


# === Load test challenges ===
test_path = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
with open(test_path, "r") as f:
    test_challenges = json.load(f)

print(f"âœ… Loaded {len(test_challenges)} test challenges")

# === Load trained classifier with strict=False ===
model = ARCClassifier()
model.load_state_dict(torch.load("task_classifier.pth", map_location="cpu"), strict=False)
model.eval()

# === Predict and create submission ===
submission = []

for challenge_id, task in tqdm(test_challenges.items()):
    if "train" not in task or "test" not in task:
        print(f"âš ï¸� Skipping invalid task: {challenge_id}")
        continue

    x = extract_features(task)
    with torch.no_grad():
        logits = model(x)
        task_class = torch.argmax(logits, dim=1).item()

    outputs = [task_class] * len(task["test"])  # Dummy output (replaced later)
    submission.append({"id": challenge_id, "outputs": outputs})

# === Save submission ===
with open("submission.json", "w") as f:
    json.dump(submission, f)

print("âœ… Saved submission.json with", len(submission), "entries")



# import shutil
# import os

# # Confirm current directory
# working_dir = "/kaggle/working"

# # Loop through and delete all files and folders
# for filename in os.listdir(working_dir):
#     file_path = os.path.join(working_dir, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)  # Remove file or link
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)  # Remove directory
#     except Exception as e:
#         print(f"Failed to delete {file_path}. Reason: {e}")

# print("âœ… /kaggle/working directory cleared.")


